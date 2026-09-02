# pointdit — does 10% forced `t=0` alone lift the 1-step readout?

Created 2026-09-02 · thread `pointdit` · prior cycle: `../analysis/2026-09-02_pointdit_vs_deltatok.md`
· init ckpt: `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit` iter 200000
· arm: `df_ctx3fwd2_tc128mg9s005compose_forcet0_xxl` · control: `df_ctx3fwd2_tc128mg9s005compose_ft20k_xxl`
· jobs: _pending_ · deck: _pending_

Mitigation 1 of the three in the analysis, **alone**. Split from the combined recipe on 2026-09-02; mitigation 2
is now `2026-09-02_pointdit_xloss_additive.md`.

## 1 Hypothesis

Fine-tuning the trained compose arm with 10% of each batch pinned to exactly `t=0` — pointdit's
`--force_zero_t`, nothing else changed — cuts 1-step `MSEToken` below **0.7417** and `LossDepth` below
**3.6563** on the 128 held-out Waymo val sequences
(`../results/2026-09-01_flow_numsteps_tc128compose_slides.html`, same ckpt, N=1, `ode`, `linear`), and
below the matched-compute `ft20k` control read at the same epoch 110.

**This arm is predicted to be near-flat, and that is the point.** Under this arm's `t_dist=uniform` the
`1/clamp(1-t,0.05)²` weight has mean 39.0, so pinning 10% of samples at `t=0` (weight 1) buys them only
**0.28%** of the gradient mass. Sub-`t=0.1` mass goes 0.286% → 0.566% — a 2× lift, not the 18× the combined
framing implied. pointdit reaches 3.6% at `t=0` only because its logit-normal `t` keeps the flow term's mean
weight at 2.96, 13× smaller. Matching pointdit's 3.6% on a uniform-`t` arm would need ~59% of the batch pinned,
which is no longer a flow model. So this run measures whether the trick has any effect at all when transplanted
verbatim, and its result routes the thread to the knob that actually carries low-`t` mass.

**Falsifiers.**

- **1-step MSEToken drops.** A 0.28% mass share is enough, which means the `t=0` point was starved to the
  degree that almost any gradient there fixes it. Then raise the ratio, and the forced-`t=0` trick is cheap and
  becomes default.
- **1-step MSEToken flat vs the `ft20k` control.** The expected outcome, and consistent with the mass
  arithmetic. It does *not* refute the starvation story — it says forced `t=0` is the wrong instrument for it
  on a uniform-`t` arm. Routes to `t_dist`, where the mass actually lives.
- **1-step improves but N=20 degrades.** The two regimes trade off even at this tiny mass share, which would
  make the trade-off far steeper than the 0.28% suggests and worth a dedicated cycle.
- **Training destabilises.** `t=0` gives `v = x - z` with no `1/(1-t)` blow-up, the *stable* end of the
  schedule. Instability here would mean the bug is in the mask, not the objective.

**Not doing.**

- **No x-MSE term.** That is `2026-09-02_pointdit_xloss_additive.md`, run as its own arm off the same init so the
  two knobs are separable. The combined recipe is a follow-up only if both move alone.
- **No `t_dist` change.** Keeping `uniform` is what makes this a clean read on the one knob. Moving to
  logit-normal was already measured a loss (`../results/2026-08-20_flow_sampler_ablation.html`), but at
  `mu=-0.7` on the *noise* end; pointdit's `(-0.8, 0.8)` is a different point and is the natural next cycle.
- **No ratio sweep yet.** 0.1 is pointdit's released default. Sweep 0.1 / 0.3 / 0.5 only if this arm moves.
- **No fresh run from scratch.** Fine-tune from iter 200000. A from-scratch arm costs ~65 h on BSC for the same
  question.

## 2 Analysis

Gradient mass under the shared `1/clamp(1-t,0.05)²` weight, 2M Monte-Carlo draws:

| schedule | mean weight | `t == 0` exactly | `t < 0.1` | `t > 0.95` |
|---|---|---|---|---|
| our compose arm — `uniform` | 39.0 | 0% | 0.286% | 51.2% |
| **this arm — `uniform` + 10% forced `t=0`** | 35.2 | **0.283%** | **0.566%** | 51.0% |
| `uniform` + 30% forced `t=0` | 27.4 | 1.09% | 1.37% | 50.8% |
| pointdit `logitnormal(-0.8, 0.8)` + 10% forced `t=0` | 2.96 | 3.60% | 5.14% | 0.04% |

The last row is the one the analysis file quoted, and it is 13× cheaper per sample than ours because of the
`t_dist`, not because of the forced zeros. Read the `mean weight` column: the forced-`t=0` samples always carry
weight 1, so their share is `r / (r + (1-r)·mean)`. On uniform that denominator is 39; on pointdit's schedule
it is 3. **The forced-`t=0` trick is a `t_dist`-relative knob, and we are importing it without its `t_dist`.**

Why run it anyway. The combined-recipe framing bundled forced `t=0` with the x-MSE term and would have
attributed any movement to "the low-`t` recipe". Splitting makes the attribution honest, and this half is the
cheap one: it is a 4-line patch, one 12 h BSC job, and a flat result is itself the argument for changing
`t_dist` rather than adding more low-`t` tricks on top of `uniform`.

Prior that eval-only knobs on this baseline do nothing: four sampler / `t`-schedule arms were all flat or worse
(`../results/2026-08-20_flow_sampler_ablation.html`). This one changes what the net is trained on, which
that sweep never did — but it changes it by 0.28% of the gradient.

## 3 Solution

**Prerequisite, shared with the x-MSE cycle: the wrapper does not forward an init checkpoint.**
`sh/train_deltatok_flow.sh:17` has `--ckpt` commented out and hardcoded to a dead `ehpc551` path.
`train_deltatok_flow.py:80` accepts it (weights-only pretrained init). Add the forward before either arm:

```bash
# sh/train_deltatok_flow.sh, replacing the commented-out line 17
[[ -n "${INIT_CKPT:-}" ]] && ARGS+=(--ckpt "$INIT_CKPT")   # weights-only fine-tune init
```

**Code — one new key, defaulting off.** In `flow_noising`, after the `t_dist` branch raises at
`occrae/deltatok_flow_trainer.py:482` and **before** `t = t.expand(b, t_dim).contiguous()` (line 485):

```python
# pointdit --force_zero_t: pin a fraction of the batch to exactly t=0 (denoiser.py:76-85).
zr = float(self.cfg.model.get("force_zero_t_ratio", 0.0))
if zr > 0 and self._eval_t_gen is None:                                    # train only; eval t must stay replayable
    m = torch.rand(b, n_draw, generator=g, device=device) < zr             # (b, 1|t)
    t = t.masked_fill(m, 0.0)                                              # (b, 1|t)
```

The `self._eval_t_gen is None` guard matters: eval seeds `t` on purpose (commit 4aa38dd) so `LossFlow` compares
weights and not draws. Zeroing a fraction of eval `t` would re-break that. Placing it inside the `else` also
keeps `train_fixed_t` unaffected.

Declare it in `configs/deltatok_flow/train_deltatok_flow.yaml` beside `train_fixed_t` (line 69):
`force_zero_t_ratio: 0.0  # pointdit --force_zero_t: fraction of the batch pinned to t=0`.

Log the mean `t` and the realised zero fraction through `metric_logger`, or a mis-wired mask reads as a null.

**Run.** Copy the baseline, do not rewrite:

```bash
cp slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_bsc.slurm \
   slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_forcet0_bsc.slurm
```

Edit, keeping every other flag byte-identical to the baseline:

- `RUN_NAME=df_ctx3fwd2_tc128mg9s005compose_forcet0_xxl` — a fresh name, or it resumes the baseline's `current.pth`.
- `model.force_zero_t_ratio=0.1`.
- `training.epoch=110`, `training.max_iter=220000` — 20k updates on top of the 200k init.
- `training.eval_num_steps=1` — the metric under test. Re-run the numsteps eval slurm on the finished ckpt for
  the full N curve.
- `export INIT_CKPT="$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit/ckpts/current.pth"`.
- `#SBATCH --time=12:00:00`, `--qos=acc_ehpc`. Walltime is absent from BSC's priority formula, so a short
  request only ever backfills sooner.

**The `ft20k` matched-compute control, submitted once and shared with the x-MSE cycle.** The same copy with
both knobs off, `RUN_NAME=df_ctx3fwd2_tc128mg9s005compose_ft20k_xxl`, same `INIT_CKPT`, same 220000 `max_iter`.
Without it every Δ is confounded with 20k extra updates. Three 12 h jobs total across the two cycles, not two.

**Pre-flight, then submit:**

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n \"force_zero_t_ratio\" occrae/deltatok_flow_trainer.py configs/deltatok_flow/train_deltatok_flow.yaml && grep -n INIT_CKPT sh/train_deltatok_flow.sh && grep -E \"RUN_NAME|max_iter|force_zero_t\" slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_forcet0_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_forcet0_bsc.slurm'"
```

The user syncs manually — if a grep comes back empty, the cluster copy is stale. Ask, do not `rsync`.

**Budget.** BSC steady state on this arm is **1.16 s/iter** at 4 GPU × bsize 16 (job 45111164, `[60/2000]`
line), so 2000 updates ≈ **39 min/epoch**. 20k updates = 10 epochs ≈ **6.5 h** plus evals. One 12 h job.

## 4 Results

_Pending._ Matched-epoch table, one block per eval set (Waymo val ×128, KITTI, nuScenes):

| arm | ep | N | MSEToken | LossDepth | LossPointmap | LossRaymap |
|---|---|---|---|---|---|---|
| baseline (iter 200000) | 100 | 1 | 0.7417 | 3.6563 | 8.1345 | 5.6041 |
| baseline | 100 | 20 | 0.8837 | 3.8796 | 9.4117 | 6.7993 |
| ft20k control | 110 | 1 | | | | |
| forcet0 | 110 | 1 | | | | |
| forcet0 | 110 | 20 | | | | |

Δ% against the **`ft20k` control** at the same `N`, not against the ep-100 baseline row. The `_tok` teacher
ceilings must be identical across rows; if they move, the eval pool changed.

Tracking: job | arm | state | epoch | note. Logs `slurm/output/`, TB mirror
`/mnt/d/tb_logs/deltatok_flow_log/df_ctx3fwd2_tc128mg9s005compose_forcet0_xxl/tb_logs/`.

## 5 Findings

_Pending._

## → Next hypothesis

`open`. Three branches, decided by §5:

- **Moves** — sweep `force_zero_t_ratio` 0.1 / 0.3 / 0.5, and combine with the x-MSE arm.
- **Flat** (expected) — the mass arithmetic was right and the lever is `t_dist`. Open a cycle on pointdit's
  `logitnormal(-0.8, 0.8)`, which is where its 5.1% sub-`t=0.1` mass actually comes from. Note this is *not*
  the `mu=-0.7, sigma=1.4` arm already refuted in `../results/2026-08-20_flow_sampler_ablation.html`.
- **Both this and the x-MSE arm flat** — the 1-step readout was never starved, and the wall is the frozen
  decoder or the conditional variance per `../analysis/2026-07-19_flow_wall.html`. Then the blocker is
  the realism probe: port pointdit's scale-invariant depth-boundary F1 (`engine.py:869-893`).
