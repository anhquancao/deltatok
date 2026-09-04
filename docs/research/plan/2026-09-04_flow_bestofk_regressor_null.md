# flow — can the eval see generation at all? best-of-K, K-spread, and a regressor null

Created 2026-09-04 · thread `flow` · prior cycle: [`../analysis/2026-09-01_flow_numsteps_why_more_steps_degrade.md`](../analysis/2026-09-01_flow_numsteps_why_more_steps_degrade.md)
· arms: `df_ctx3fwd2_tc128mg9s005compose_fixedt0_xxl` (regressor null) plus eval-only sweeps on existing checkpoints
· control: `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit` at `iter_100000` and `iter_200000`
· jobs: `_pending_` · deck: `_pending_`

Executes diagnostic to-do 1 of the prior cycle (seed spread at N=1 vs N=20), sharpened, and adds the null
model the thread has never run. Motivated by the pointditT read
(`../results/2026-09-04_pointdit_lowt_numsteps_ep50_slides.html`): 13× more gradient at `t=0` made the
`t=0` readout *worse*, which the current eval cannot explain.

## 1 Hypothesis

**The thread is stuck on a measurement problem, not a modelling problem.** Every metric it ranks on —
`MSEToken`, `LossDepth`, `LossPointmap`, `LossRaymap` — is a distortion against one GT, and the conditional
mean wins those by theorem (prior cycle, cause 1). So no run so far can tell a working sampler from a broken
one. Three readouts on the existing checkpoints, plus one new arm, decide which of two problems this is:

- **A — the sampler generates; the eval punishes it.** Predicted: at N=20 with K=8 draws per item, best-of-8
  `LossDepth` beats the 1-step 3.6325 (control `iter_100000`, 128 Waymo val seqs), and the K-spread of
  `z_hat` accounts for the whole N=20 penalty (drift ≈ 0, defined in §3).
- **B — the sampler is mean-seeking; extra steps only drift.** Predicted: K-spread ≪ the N=20 penalty, best-of-8
  ≈ single-sample, drift ≫ 0.

And independently: **the flow's 1-step readout is starved (prior cycle, cause 2).** Predicted: a from-scratch
regressor with the identical 1.2B net, `train_fixed_t=0`, beats the control's 1-step `MSEToken` 0.6677 at
matched `iter_100000`.

**Falsifiers.** Each bullet is the routing for §4.

- **Best-of-8 beats 1-step, drift ≈ 0 (A).** The "1 step is best" finding was an artifact. Best-of-K and
  spread become the headline flow metrics; the next arm is CFG (condition dropout in training, `cfg_w` in the
  sampler already exists at `generation_helper.py:24`). The four null sampler / `t`-schedule ablations get
  re-read on the new metrics before any new one is run.
- **Spread ≈ 0, best-of-8 ≈ single (B, collapse).** The model copies at high `t` and never learned to place a
  sample. The lever is the *weight*, not the placement of `t`: cap `1/clamp(1−t, 0.05)²` at ~5 (min-SNR-γ)
  instead of 400. pointditT already showed that moving mass without changing the cap loses.
- **Spread large but drift ≫ 0 (B, exposure bias).** Samples are diverse and systematically off. Prior cycle
  cause 3; the fix is noise-augmented training of the trajectory, not the loss weight.
- **Regressor ≪ 0.6677.** Cause 2 confirmed: the flow's `t=0` head was never trained to the regression floor.
  Whatever branch A/B says, the regressor is the honest baseline the paper's main table must carry.
- **Regressor ≈ 0.6677.** Cause 2 refuted. The `t=0` head is already at the floor and that floor is
  conditional variance plus SIGReg's dead dims (`../analysis/2026-07-19_flow_wall.html`). Nothing in the
  flow loss moves it; the tokenizer does.
- **Regressor ≫ 0.6677.** The high-`t` supervision is what makes the `t=0` map good. The pointditT paradox
  generalises, and the mechanism is representational, not a schedule.

**Not doing.**

- **No CFG arm, no weight-cap arm.** Each is gated on a branch above. Running either before the branch is
  known repeats the thread's pattern of ablating against a metric that cannot see the effect.
- **No depth-edge F1 realism probe.** No code exists for it. The K-spread and best-of-K are the
  distributional readouts here; F1 is a second cycle if branch A holds.
- **No 8-seed sweep via `training.seed`.** The prior to-do's design. Superseded: K draws in one pass keeps the
  data and the `LossFlow` `t` pinned and varies only the prior, and yields best-of-K for free. `training.seed`
  also reseeds `_eval_t_gen`, which the seed sweep would have confounded.
- **No pointditT ep100.** BSC:45344713 cancelled 2026-09-04 at epoch 58 after the ep50 read lost 19 of 20
  cells. Its `iter_100000.pth` stays in the t-bins job below because it is the cleanest probe of the paradox.
- **No FVD.** TODO 10, its own item.

## 2 Why it is worth the GPU hours

**The distortion trap is already in the ledger and has never been measured.** Prior cycle, cause 1: "more
steps = higher MSE is the expected direction for any working generative sampler". Its own to-do 1 asks for the
spread measurement and it was not run. Since then two more arms were ablated on the same metrics
(`../results/2026-08-20_flow_sampler_ablation.html`, pointditT) and both read null-or-worse, which is what the
trap predicts regardless of whether they helped.

**The `t=0` paradox from 2026-09-04** (`../results/2026-09-04_pointdit_lowt_numsteps_ep50_slides.html`, both
arms `iter_100000`, 128 Waymo val seqs, `ode` + `linear`):

| N | control MSEToken | pointditT | Δ | control LossRaymap | pointditT | Δ |
|---|---|---|---|---|---|---|
| 1 | **0.6677** | 0.7221 | +8.2% | 5.0347 | 6.0057 | +19.3% |
| 20 | 0.8850 | 0.8726 | −1.4% | 5.8777 | 7.6995 | +31.0% |

N=1 *is* `x̂(noise, t=0)` (`generation_helper.py:69-85` with `num_steps=1`, linear, alpha 0) and `MSEToken`
*is* its error on the forecast slots (`deltatok_flow_trainer.py:881`). pointditT sends ~13× the relative
gradient there (`../plan/2026-09-02_pointdit_lowt_schedule.md` §2, mean weight 2.77 vs 39.0) and is worse
at exactly that point. Two readings, and only the train-split t-bins separates them: the extra low-`t` gradient
overfit the conditional mean (train better, val worse), or the high-`t` supervision is what shapes the `t=0`
map (worse on both).

**The `t=0` floor is close.** SIGReg pins `z` to unit scale, so the predict-zero error is ≈1.0 and the control's
0.6677 explains ~33% of the variance. The prior note reads ~47% of the flow loss as dead SIGReg dims
(`../analysis/2026-07-19_flow_wall.html`). If most of the 0.6677 is irreducible, no `t`-schedule can move it
and the thread should stop trying. `x_var` from `eval_deltatok_flow_t_bins.py` measures the floor directly
instead of inferring it.

**The control regresses over training on the only metric that matters here.** Control N=1 `MSEToken` 0.6677
at `iter_100000` → 0.7417 at `iter_200000` (+11.1%); `LossRaymap` 5.0347 → 5.6041 (+11.3%). The training
objective drifts away from the eval. A regressor that trains *only* the eval point cannot drift this way, which
is the second reason it belongs in the table.

**The decoded damage sits in pose, not depth.** Every degradation measured this week moves `LossRaymap` 3–8×
more than `MSEToken` while `LossDepth` barely moves (+1.9% to +3.2%). The v-pred arm failed the same way
(`../results/2026-08-25_flow_vpred_vs_xpred_slides.html`). Not tested here, but branch A's follow-up should
weight the flow loss toward the pose-carrying dims or add a decode-through raymap term.

**Cost against the prize.** Five eval jobs at minutes each, and one training run, decide whether the next ten
arms are scored on a metric that can see them. The thread has spent four training runs on the wrong metric.

## 3 How it is run

### The patch — K draws per item in `eval_one_epoch`

`occrae/deltatok_flow_trainer.py`, eval loop at :795-885. Wrap the existing sample → rollout → decode → frame
losses in a loop over `K = training.eval_num_samples`; sample 0 keeps the existing keys and the viz path
byte-unchanged, so the training-time eval (K=1) does not move.

- **The draws.** Each `_sample_noise(x_spatial)` call advances `_eval_noise_gen`, so draw `k` at item `i` is
  the same on every arm. Nothing else in the loop is reseeded.
- **Per-item minimum needs `val_bsize=1`.** `_compute_frame_losses` (`deltatok_shared.py:415`) returns
  batch means. At batch 1 the batch mean is the item, and `min_k` over the K dicts is a true best-of-K. No
  change to the loss code.
- **New keys, appended to `_EVAL_KEYS` (:33-38)** so every rank still reduces the same vector:
  `MSEToken_best`, `LossPointmap_best`, `LossDepth_best`, `LossRaymap_best` (min over k, then mean over
  items); `MSEToken_meanK` (MSE of the K-sample mean against `z`); `ZHatSpreadK` (per-item unbiased variance
  of `z_hat` across k on the forecast slots, mean over dims and items). At K=1 the `_best` keys equal their
  plain twins and `ZHatSpreadK` is 0.
- **The yaml key.** `eval_num_samples: 1` beside `eval_num_steps` in
  `configs/deltatok_flow/train_deltatok_flow.yaml:130`. Without it `compose()` rejects the override.
- **The readout.** Print the resolved `eval_num_samples` next to `eval_num_steps` in the
  `[INFO] ===== sampler_step_mode=...` banner of `eval_deltatok_flow_sampler.py:208-211`, so a K that
  silently stayed 1 is not read as collapse.

**The decomposition the keys buy.** Write `M1 = MSEToken` at N=1, `M20 = MSEToken` at N=20 (single draw),
`S = ZHatSpreadK`, `Mbar = MSEToken_meanK` at N=20. For draws `x_k = μ̂ + ε_k` with `Var ε = S`:
`M20 − Mbar = S·(1 − 1/K)` is the variance share, and **`drift = Mbar − M1 − S/K`** is how far the 20-step
trajectory's centre sits from the 1-step readout. Branch A is `drift ≈ 0`; branch B is `drift` carrying most
of `M20 − M1` (0.217 on the control at `iter_100000`).

### The second patch — the regressor's startup line

`train_fixed_t` is read at `deltatok_flow_trainer.py:474` but absent from the `t schedule:` print at :94-97,
so a fixed-`t` arm prints `t_dist=uniform` and reads as the control. Add `train_fixed_t=` to that line.

### Eval sweeps — existing checkpoints, `acc_debug`, no sync beyond the patch

`slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm` already takes `CKPT` / `OUTPUT_DIR` / `NUM_STEPS`
from the environment. Add `training.eval_num_samples=8` and `training.val_bsize=1` through `EXTRA_ARGS`; the
script's `training.val_bsize=4` at :61 is overridden by a later `--cfg` of the same key.

```bash
# control, matched to the 2026-09-04 read
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  CKPT=/gpfs/scratch/ehpc1001/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit/ckpts/iter_100000.pth \
  OUTPUT_DIR=results/deltatok_flow_bestofk/control_ep50 NUM_STEPS=1,20 \
  EXTRA_ARGS=\"--cfg training.eval_num_samples=8 training.val_bsize=1\" \
  sbatch slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm'"
```

Same line for `.../iter_200000.pth` → `control_ep100` and for
`/gpfs/scratch/ehpc1001/deltatok_flow_log/df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl/ckpts/iter_100000.pth`
→ `pointditT_ep50`. `acc_debug` allows one job per user; run the second and third on `acc_ehpc` with
`--qos=acc_ehpc --time=01:00:00` on the `sbatch` line, as on 2026-09-04. Budget: 128 items × 8 draws × 20
steps at batch 1, ~25 min each.

### t-bins — train and val, both `iter_100000` checkpoints

`slurm/eval_deltatok_flow_t_bins_jz.slurm` is Jean Zay only and the checkpoints are on BSC. Copy it:

```bash
cp slurm/eval_deltatok_flow_t_bins_jz.slurm slurm/eval_deltatok_flow_t_bins_bsc.slurm
```

Change only: the header to `--account=ehpc880 --partition=acc --qos=acc_debug --cpus-per-task=20` and the
`_bsc_%j` log names; `cd /gpfs/projects/ehpc1001/code/deltatok`; `source env_bsc.sh`;
`--config-name train_deltatok_flow_waymo_bsc`; and the arch flags of
`slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm:42-60` passed as `--cfg` (the `.py` defaults are
the Jean Zay whitenpos run). `SPLITS=train,val`, defaults `NUM_BINS=50 NUM_BATCHES=8 NOISE_REPEATS=4`. The
train split draws from the config's `train_dataset`, 8 × `bsize` 16 = 128 items, the val count.

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && CKPT=<control iter_100000> OUTPUT_DIR=results/deltatok_flow_t_bins/control_ep50 sbatch slurm/eval_deltatok_flow_t_bins_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && CKPT=<pointditT iter_100000> OUTPUT_DIR=results/deltatok_flow_t_bins/pointditT_ep50 sbatch slurm/eval_deltatok_flow_t_bins_bsc.slurm'"
```

Read `mse_x` in the first bin and `x_var`, train against val, per arm. ~30 min each.

### The arm — `train_fixed_t=0` regressor

Copy the control, do not rewrite:

```bash
cp slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_bsc.slurm \
   slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_fixedt0_bsc.slurm
```

Change only these:

- `RUN_NAME=df_ctx3fwd2_tc128mg9s005compose_fixedt0_xxl`. A fresh name, or it resumes the control.
- `--job-name`, `--output`, `--error` together.
- `model.train_fixed_t=0`. Pins every forecast slot to `t=0`; context slots stay at 1 via the existing
  override (`flow_noising` :507-510). Under `loss_mode=v` the weight at `t=0` is exactly 1, so the loss is
  plain x-MSE with no other flag touched. `t_dist`, `mu`, `sigma`, `force_zero_t_ratio` become inert.
- `training.eval_num_steps=1`. The live curve must read the only point this arm trains. At 20 the sampler
  would query `t = 0.05 … 0.95`, none of which it has seen.

Budget: one 72 h `acc_ehpc` job, ~39 min/epoch (pointditT's rate at this shape). `iter_100000` lands at ~33 h;
that is the read. Let it run to `max_iter` for the ep100 row if the wall allows.

### Pre-flight, then submit

The user syncs manually. An empty grep means the cluster copy is stale: ask, never `rsync`.

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n eval_num_samples occrae/deltatok_flow_trainer.py configs/deltatok_flow/train_deltatok_flow.yaml && grep -n train_fixed_t= occrae/deltatok_flow_trainer.py && ls slurm/eval_deltatok_flow_t_bins_bsc.slurm && grep -E \"RUN_NAME|train_fixed_t|eval_num_steps\" slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_fixedt0_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_fixedt0_bsc.slurm'"
```

A job id is not a launch. Watch until the first loss line, and grep the `.out` for `train_fixed_t=0` on the
`t schedule:` line and `eval_num_samples=8` in each sweep's banner.

### The read

Eval sweeps and t-bins: as soon as they finish, into a `results/2026-09-xx_flow_bestofk_regressor_null_slides.html`.
The regressor: at `iter_100000` against the control's `iter_100000` row above, N=1 only.

| readout | control ep50 | control ep100 | pointditT ep50 | regressor ep50 |
|---|---|---|---|---|
| `MSEToken` N=1 (`M1`) | 0.6677 | 0.7417 | 0.7221 | |
| `MSEToken` N=20 (`M20`) | 0.8850 | 0.8837 | 0.8726 | — |
| `ZHatSpreadK` N=20 K=8 (`S`) | | | | — |
| `MSEToken_meanK` N=20 K=8 (`Mbar`) | | | | — |
| `drift = Mbar − M1 − S/8` | | | | — |
| `LossDepth` N=1 | 3.6325 | 3.6563 | 3.7485 | |
| `LossDepth_best` N=20 K=8 | | | | — |
| t-bins `mse_x` bin 1, train / val | | — | | — |
| t-bins `x_var` | | — | | — |

## 4 Outcome

`_pending_`. Routing is the falsifier list in §1.
