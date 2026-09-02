# SIGReg parameter search — plan

Status **2026-07-28: PLAN ONLY, nothing submitted.** Target arm:

```
deltatok_surround_layer12_bsize2_dtok64_tc768_cos1e3_1e5_b95_gradskip_noaff_prenorm_sigreg0.05_pool_nozn
```

Coordinate descent over `occrae/sigreg.py`'s three estimator knobs, in this order:
**`num_slices` → `t_max` → `n_points`(knots)**. One axis per wave; the winner of each wave is
frozen into the next.

## Where the arm stands

`Eval/Loss` best, from TensorBoard under `$TRG_WORK/../deltatok_log` (dumped 2026-07-28):

| arm | best `Eval/Loss` | epochs | final `Train/LossSIGReg` |
|---|---|---|---|
| `_prenorm` (no sigreg, z-norm) | 0.04967 | 53 | — |
| `_prenorm_nozn` (no sigreg) | 0.04989 | 64 | — |
| `_prenorm_mlp` (no sigreg) | 0.08071 | 33 | — |
| **`_prenorm_sigreg0.05_pool_nozn`** | **0.03346** | 45 | 0.0017 ↓ |
| `_prenorm_sigreg1.0_pool` (z-norm) | 0.11215 | 27 | 0.0059 |
| `_prenorm_mlp_sigreg1.0` (old estimator) | 0.12157 | 19 | 0.0080 flat |

The target arm is the best by a wide margin — ~33% under its no-sigreg `_nozn` twin — and was still
descending when it stopped (ep44: KITTI `LossRecon_AR` 0.03893, nuScenes 0.02989). So this is a
**refinement around a known-good point**, not a rescue.

It also settles a contradiction: `slurm/deltatok/archive/train_deltatok_multitoken_mlp_sigreg_jz.slurm` advises that
`SIGREG_WEIGHT` "is a knob to RAISE". Measured, the opposite holds — 1.0 gives `Eval/Loss` 0.112,
0.05 gives 0.033. That comment should be corrected when the script is next touched.

## Reference guidance (`/mnt/d/code/lejepa`)

- `scripts/launch_epps_ablation.md` is the authors' own estimator ablation and sweeps exactly our
  three axes: `num_slices ∈ {512, 1024, 4096}`, `t_max ∈ {1, 3, 5}`, `n_points ∈ {5, 17, 41}` — a
  full 3×3×3 grid at fixed `lambda=0.05`, 100 epochs on inet1k/ViT-L. Our `2048 / 3.0 / 17` sits
  mid-grid. The wave values below are taken from this grid.
- `scripts/launch_inet10.py` sweeps `bstat_lambda ∈ {0.01, 0.02, 0.05, 0.1}`; `0.05` is the default
  in every other launch script. **Our `sigreg_weight=0.05` matching that number is a coincidence,
  not a match**: their loss is a convex combination `sigreg*λ + inv*(1-λ)` and their statistic keeps
  the `* N * world_size` factor that `occrae/sigreg.py` deliberately drops. The two scales are not
  comparable.
- `lejepa/multivariate/slicing.py:104,146` has a `clip_value` knob we do not implement: it zeroes any
  per-slice statistic below a threshold. That is a direct remedy for the finite-sample bias floor in
  `../analysis/2026-07-25_sigreg_debug.md`. **Out of scope here** — it is a code change, not a parameter — but it is
  the most promising follow-up.

## Objective

Score each arm on **epoch 10** (TB step `1125 * 11 = 12375`; the run does 1125 iters/epoch):

- `Eval/206 @ KittiSeqMultiView/LossRecon_AR`
- `Eval/206 @ Occ3dNuscenesSeqMultiView/LossRecon_AR`

**Decision rule.** Promote a value only if it beats the incumbent on **both** datasets. If the two
disagree, or the change is within noise, keep the incumbent — that favours the reference default and
costs nothing.

## Fixed across every arm

Everything that defines the target arm, unchanged:

| knob | value |
|---|---|
| `NUM_DELTA_TOKENS` | 64 |
| `TARGET_CHANNELS` | 768 |
| `BOTTLENECK_MLP` | `false` (no `_mlp` tag in the run name) |
| `Z_NORM` | `false` (`_nozn`) |
| `SIGREG_WEIGHT` | 0.05 |
| `SIGREG_WARMUP` | 2000 |
| `PEAK_LR` | 1e-3 |
| `GRAD_CLIP` | 0.1 |
| `training.epoch` | 100 (**do not shorten** — the cosine LR spans it; changing it changes the schedule and breaks comparability) |

`sigreg_weight` is held at 0.05 for the whole search, mirroring the reference ablation which fixes
`λ` and sweeps only the estimator.

## Wave 1 — `num_slices` (3 arms)

| arm | `SIGREG_NUM_SLICES` | `RUN_SUFFIX` |
|---|---|---|
| incumbent | 2048 | — (read epoch 10 from the existing TB, free) |
| A | 512 | `_sl512` |
| B | 1024 | `_sl1024` |
| C | 4096 | `_sl4096` |

No code change needed — `sigreg_num_slices` is already plumbed
(`configs/train_deltatok.yaml:124` → `deltatok_trainer.py:593`).

**What to expect.** `num_slices` lowers the *variance* of the statistic, not its finite-sample bias
floor — that floor is set by S, the sample count, measured at 0.0085 for S=128 in
`../analysis/2026-07-25_sigreg_debug.md`. So this axis buys gradient reproducibility (which was `cos=0.08` between two
independent draws at 256 slices) with diminishing returns above `2*Cz = 1536`, the floor the
`SIGReg` docstring already states.

**Also record cost.** The projection intermediate is `(S, K, knots)` fp32 ≈ 570 MB at K=4096. Log
`Train/SpeedSamplesPerSec` and peak memory per arm; a sub-noise accuracy gain that costs 2× step
time is not a win.

## Wave 2 — `t_max` (3 arms, at the wave-1 winner)

| arm | `t_max` | `RUN_SUFFIX` |
|---|---|---|
| incumbent | 3.0 | carried from wave 1 |
| D | 1.0 | `_tmax1` |
| E | 2.0 | `_tmax2` |
| F | 5.0 | `_tmax5` |

**What to expect.** `t_max` sets which characteristic-function frequencies are compared. The window
`exp(-t²/2)` gives `t > 3` roughly 1% of the total quadrature weight, so 5 ≈ 3 is likely; the real
question is whether `t_max=1` — essentially a mean/variance match — is already sufficient.

## Wave 3 — `n_points` / knots (3 arms, at the wave-1+2 winner)

| arm | `knots` | `RUN_SUFFIX` |
|---|---|---|
| incumbent | 17 | carried |
| G | 5 | `_k5` |
| H | 9 | `_k9` |
| I | 41 | `_k41` |

Pure quadrature resolution of the same integral — cheapest axis, expected smallest effect. `knots`
must stay **odd** (the reference asserts it; our trapezoid endpoints assume it).

## Code change required before wave 2

`SIGReg.__init__` already accepts `knots` and `t_max` (`occrae/sigreg.py:46`), but the trainer passes
only `num_slices`. Four surgical edits:

1. `configs/train_deltatok.yaml` (after line 125) — add `sigreg_knots: 17` and `sigreg_t_max: 3.0`.
2. `occrae/deltatok_trainer.py:592-594` — pass both through to the `SIGReg(...)` constructor.
3. `slurm/deltatok/archive/train_deltatok_multitoken_mlp_sigreg_jz.slurm` — add `SIGREG_KNOTS` / `SIGREG_TMAX` env
   defaults and the matching `EXTRA_CFG_ARGS` entries.
4. Same script — extend the `RUN_NAME` tag so a non-default value gets its own directory.

Wave 1 needs none of this.

## Run mechanics

Per repo policy: **verify the cluster copy first, ask the user to sync, never sync from here.**

```bash
ssh jean-zay "bash -lc 'grep -n \"SIGREG_NUM_SLICES\|RUN_SUFFIX\" \$TRG_WORK/code/deltatok/slurm/deltatok/archive/train_deltatok_multitoken_mlp_sigreg_jz.slurm'"
```

Then one job per arm, e.g. wave 1 arm A:

```bash
ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch \
  --export=ALL,BOTTLENECK_MLP=false,Z_NORM=false,SIGREG_WEIGHT=0.05,SIGREG_NUM_SLICES=512,RUN_SUFFIX=_sl512,RESUME=0 \
  slurm/deltatok/archive/train_deltatok_multitoken_mlp_sigreg_jz.slurm'"
```

- **`RUN_SUFFIX` is mandatory on every arm.** Without it the job resolves to the incumbent's
  `RUN_NAME`, and with `RESUME=0` it overwrites that run's `current.pth` (only one backup deep — see
  the `deltatok-relaunch-resume-flag` note).
- Keep the script's default `--time=20:00:00`. The arms cannot be told to stop at epoch 10 without
  changing `training.epoch` and therefore the LR schedule, so instead **`scancel` each arm once its
  epoch-10 eval appears in TB**. Epoch wall-time is uncertain (the dev run suggests ~0.5 h/epoch, the
  incumbent's chained history ~0.9 h/epoch including queue gaps) — confirm from a running arm's log
  before assuming a shorter `--time` would reach epoch 10.
- All arms in a wave submit together and run concurrently; waves are sequential.

## Known limitations of this plan

Recorded because they bound what the result can mean.

1. **No noise floor.** A seed-replicate arm (identical config, `training.seed=1`) was proposed and
   **declined** to save budget. Consequence: there is no measured run-to-run spread, so a wave-1
   difference cannot be distinguished from noise, and waves 2–3 are built on top of whatever wave 1
   promotes. Mitigation if a wave comes back tight: treat differences under ~5% relative as no
   decision and keep the incumbent.
2. **The 10-epoch horizon is unvalidated.** The check — comparing the ep-10 ranking against the
   ep-44 ranking on the arms already in TB — is free and read-only, but was **declined**. A
   10-epoch winner may not be the 100-epoch winner, especially since the incumbent was still
   descending at ep44 with its cosine LR only ~half-decayed.
3. **Coordinate descent ≠ the reference's grid.** They ran the full 3×3×3. This finds a local
   optimum along three separate axes and cannot see interactions between them.

Both declined checks can be added later at low cost; item 2 costs nothing but a TB read.
