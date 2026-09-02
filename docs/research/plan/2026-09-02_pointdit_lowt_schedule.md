# pointdit — does the low-`t` training schedule beat uniform, trained from scratch?

Created 2026-09-02 · thread `pointdit` · prior cycle: [`../analysis/2026-09-02_pointdit_vs_deltatok.md`](../analysis/2026-09-02_pointdit_vs_deltatok.md)
· arm: `df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl`
· control: `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit` @ ep 100 (already measured)
· jobs: BSC:45344713, submitted 2026-09-02, `PENDING` · deck: `_pending_`

Merges `2026-09-02_pointdit_schedule_scratch.md` and `2026-09-02_pointdit_schedule.md`, both deleted 2026-09-02.

## 1 Hypothesis

Training the tc128 compose flow arm **from scratch** under pointdit's schedule, `t = sigmoid(-0.8 + 0.8·ε)` with
10% of each batch pinned to exactly `t=0`, cuts 1-step `MSEToken` below **0.7417** and `LossDepth` below
**3.6563** on the 128 held-out Waymo val sequences at matched **ep 100**. The control is the uniform-`t` arm read
with the same sampler (`../results/2026-09-01_flow_numsteps_tc128compose_slides.html`, `ode`, `linear`, N=1).
`N=20` is predicted to degrade.

The schedule is the only variable. Every other flag stays byte-identical to the control, including the frozen
tokenizer `epoch_100.pth`, `t_shared=true`, `sampler_alpha=0`, `loss_mode=v` and `pred_mode=x`.

**Why both knobs at once.** Forced `t=0` samples carry weight 1, so their gradient share is
`r / (r + (1-r)·E[w])`. On `uniform`, `E[w] = 39.0` and 10% zeros buy **0.28%**. Under `logitnormal(-0.8, 0.8)`,
`E[w] = 2.96` and the same 10% buy **3.62%**. Neither knob alone reaches pointdit's regime (§2).

**Falsifiers.** Each bullet is also the routing for §4.

- **N=1 improves, N=20 degrades.** Expected: the schedule spends the mass above `t=0.95` (51.2% → 0.03%) on
  1-step. Declare 1-step the operating point and port pointdit's boundary-F1 (`engine.py:355-367`) next.
- **N=1 improves, N=20 holds.** Uniform was wasting half its mass. Make the schedule the flow default and
  close the two `pointdit` fine-tune arms.
- **Everything degrades.** Reproduces `logitnormal(-0.7, 1.4)` (+2.9% LossPointmap, +3.3% LossRaymap at
  ep 60–99, `../results/2026-08-20_flow_sampler_ablation.html` §3). The collision is `loss_mode=v` itself.
  Hand off to `2026-09-02_pointdit_xloss_additive.md`.
- **N=1 flat.** Starvation was never the wall. `../analysis/2026-07-19_flow_wall.html` (frozen decoder,
  conditional variance) becomes the thread's blocker.
- **Train loss drops, every rollout metric worsens.** The refuted arm's signature (`sampler_ablation` §6:
  train 0.499 vs 0.593 at ep 90). `Train/Loss` and `Eval/LossFlow` weight `t` differently across `t_dist`
  arms, so read only the decoded metrics.

**Not doing.**

- **No single-knob arm.** `2026-09-02_pointdit_lowt_recipe_finetune.md` is the zeros-on-uniform arm, and
  `logitnormal` alone sits 0.04 pp from the refuted `(-0.7, 1.4)` arm on sub-`t=0.1` mass (§2).
- **No `mu` / `sigma` / `force_zero_t_ratio` sweep.** `(-0.8, 0.8)` and 0.1 are pointdit's released defaults
  (`main.py:82-83`, `denoiser.py:76-85`). Sweep only if this arm moves.
- **No fine-tune.** 20k updates off iter 200000 leaves a "too short to re-fit" escape hatch. From scratch is
  the true head-to-head against a control already measured at ep 100.
- **No x-MSE term, no zeros ODE init, no `t_shared=false`.** Separate arm, dropped, and a different question
  (`tindep` measured +0.8%).

## 2 Why it is worth the GPU hours

Gradient mass under the shared `1/clamp(1-t, 0.05)²` weight of `loss_mode=v`, 4M Monte-Carlo draws. Mass
fraction is `E[w(t)·1{t∈A}] / E[w(t)]`; `med t` is the unweighted median.

| schedule | mean `w` | `t == 0` | `t < 0.1` | `t < 0.5` | `t > 0.95` | med `t` |
|---|---|---|---|---|---|---|
| `uniform` — the control | 39.00 | 0% | 0.28% | 2.56% | 51.16% | 0.500 |
| `uniform` + 10% `t=0` — open `pointdit` arm | 35.22 | 0.28% | 0.57% | 2.84% | 51.21% | 0.444 |
| `logitnormal(-0.7, 1.4)` — **refuted** arm | 10.44 | 0% | 1.55% | 12.56% | 17.81% | 0.332 |
| `logitnormal(-0.8, 0.8)` — schedule alone | 2.96 | 0% | 1.59% | 59.32% | 0.02% | 0.310 |
| **`logitnormal(-0.8, 0.8)` + 10% `t=0` — this arm** | **2.77** | **3.62%** | **5.16%** | **60.81%** | **0.03%** | **0.287** |

- **The knobs multiply.** Rows 2 and 4 each move sub-`t=0.1` mass by under 1.4 pp. Row 5 moves it 18×. The
  forced zeros only bite once the mean weight has collapsed.
- **`sigma` kills the high-`t` tail; it does not buy low-`t` mass.** Rows 3 → 4 move sub-`t=0.1` by 0.04 pp
  while `t > 0.95` goes 17.81% → 0.02%. That is what this arm *stops* training, and the risk it carries.

**The control.** `../results/2026-09-01_flow_numsteps_tc128compose_slides.html`: same checkpoint, `ode` +
`linear`, 128 val sequences, seed 42. Teacher ceilings `LossPointmap_tok 4.0557 · LossDepth_tok 2.7923 ·
LossRaymap_tok 1.6763`.

| steps | MSEToken | LossPointmap | LossDepth | LossRaymap |
|---|---|---|---|---|
| **1** | **0.7417** | 8.1345 | **3.6563** | 5.6041 |
| 2 | 0.7699 | **7.9687** | 3.7226 | **5.2918** |
| 3 | 0.7986 | 8.1355 | 3.7721 | 5.4819 |
| 4 | 0.8166 | 8.3646 | 3.8001 | 5.7242 |
| 20 | 0.8837 | 9.4117 | 3.8796 | 6.7993 |

`MSEToken` is strictly monotone in step count, and the best row still sits 2.0× the pointmap ceiling. That
deck's `LossFlow` column is not readable: each pass drew its own `t`, fixed since by `_eval_t_gen` (4aa38dd).

**Why the refuted arm does not settle this.** `df_ctx3fwd2_tc128mg9s005compose_logitnorm_xxl` (BSC:44767716)
lost at ep 60–99 (LossPointmap 9.341 vs 9.076, LossRaymap 6.525 vs 6.317, LossDepth 3.860 vs 3.868). But it
was read at N=20 only, on the `epoch_40.pth` tokenizer at 1000 iters/epoch, with no forced zeros and
`sigma=1.4`, which leaves 17.8% of its mass above `t=0.95`.

**Free prior, before spending 65 h.** Sweep that arm and its base twin
(`deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_xxl_dit`, both still on BSC 2026-09-02) at
N=1 through `slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm` with a `CKPT` override. Both need
`model.deltatok_ckpt` repointed to `epoch_40.pth`; the script hardcodes `epoch_100.pth` at line 44. Two
`acc_debug` jobs, minutes each. A prior, not a gate.

## 3 How it is run

### The patch

Written 2026-09-02, uncommitted. Three pieces, shared with `2026-09-02_pointdit_lowt_recipe_finetune.md`:

- **The mask.** `flow_noising` takes `force_zero_t_ratio` (`occrae/deltatok_flow_trainer.py:467`) and masks
  `t` to 0 after the `t_dist` draw, inside the `else` so `train_fixed_t` is untouched (:490-494). Only the
  train call site passes it (:609-612); the eval call keeps the default 0.0, so the seeded eval `t` stays
  replayable (4aa38dd).
- **The yaml key.** `force_zero_t_ratio: 0.0` beside `train_fixed_t`
  (`configs/deltatok_flow/train_deltatok_flow.yaml:70`). Without it Hydra `compose()` rejects the override.
- **The readout.** `__init__` resolves the ratio once and prints
  `t schedule: t_dist=… mu=… sigma=… force_zero_t_ratio=…` (:91-97). One grep on the `.out` proves the
  override landed; a schedule that silently stayed `uniform` would otherwise read as a clean null.

`t_dist` / `mu` / `sigma` need no code (`train_deltatok_flow.yaml:65-67`, `flow_noising` :480-489).

### The arm

Copy the control, do not rewrite:

```bash
cp slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_bsc.slurm \
   slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_pointditT_bsc.slurm
```

Change only these:

- `RUN_NAME=df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl`. A fresh name, or it resumes the control's `current.pth`.
- `--job-name`, `--output`, `--error` (lines 2, 13, 14), or `slurm/output/` collides with the control's.
- `model.t_dist=logitnormal`, `model.mu=-0.8`, `model.sigma=0.8`. The config defaults are `-0.7 / 1.4`, the refuted pair.
- `model.force_zero_t_ratio=0.1`.

`training.eval_num_steps=20` stays. The live curve then tracks the metric this arm is predicted to *lose*, so a
worsening N=20 curve is falsifier 1, not a reason to kill the run. No live N=1 curve: the control has none, so
it could never carry the head-to-head.

### Reading N=1

`eval_deltatok_flow_sampler.py` loops `--num_steps` and re-sets `cfg.training.eval_num_steps` per pass
(:208-211). It loads `ema_state` when present (:197-199), and `eval_one_epoch` runs inside `ema_scope()`
(`deltatok_flow_trainer.py:734`) exactly as in training, so an offline sweep sees the in-training weights.
`CKPT=<run>/ckpts/current.pth NUM_STEPS=1` in `acc_debug`, minutes, whenever the number is wanted.

### Pre-flight, then submit

The user syncs manually. An empty grep means the cluster copy is stale: ask, never `rsync`.

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n force_zero_t_ratio occrae/deltatok_flow_trainer.py configs/deltatok_flow/train_deltatok_flow.yaml && grep -E \"RUN_NAME|t_dist|model.mu|model.sigma|force_zero_t|eval_num_steps|max_iter|deltatok_ckpt\" slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_pointditT_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_pointditT_bsc.slurm'"
```

A job id is not a launch. Watch until the first loss line, and grep the `.out` for
`t schedule: t_dist=logitnormal mu=-0.8 sigma=0.8 force_zero_t_ratio=0.1`.

**Budget.** 1.16 s/iter at 4 GPU × bsize 16 (job 45111164): 2000 updates ≈ 39 min/epoch, 100 epochs ≈ **65 h**
plus evals. One 72 h `acc_ehpc` job, the QoS cap. Evals may push it past the wall; resume is automatic, so a
plain relaunch finishes it.

### The read

At **ep 100**, never earlier. Sweep both ep-100 checkpoints through
`slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm` (`NUM_STEPS=1,2,3,4,20`, `STEP_MODES=ode`). The
control rows must reproduce the §2 table, which doubles as the checkpoint-load check. Δ% against the control at
the same `N`. The `_tok` ceilings must match within a block; if they move, the eval pool changed. The
free-prior rows (`epoch_40` tokenizer, 1000 iters/ep) go in their own table, not comparable to these.

| arm | ep | N | MSEToken | LossPointmap | LossDepth | LossRaymap |
|---|---|---|---|---|---|---|
| control `uniform` | 100 | 1 | 0.7417 | 8.1345 | 3.6563 | 5.6041 |
| control `uniform` | 100 | 20 | 0.8837 | 9.4117 | 3.8796 | 6.7993 |
| pointditT | 100 | 1 | | | | |
| pointditT | 100 | 20 | | | | |

Logs `slurm/output/` and `../monitor_jobs/data/logs/BSC/`; TB mirror
`/mnt/d/tb_logs/deltatok_flow_log/df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl/tb_logs/`.

## 4 Outcome

`_pending_`. Routing is the falsifier list in §1.

**Status 2026-09-02.** BSC:45344713 submitted and `PENDING`. Pre-flight passed: the cluster copy carries the
patch, the config key and all five slurm overrides, `$SCRATCH/deltatok_flow_log/df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl`
did not exist (fresh start, no accidental resume), and the tokenizer `epoch_100.pth` is on disk (8.2 GB,
2026-08-22). Still to confirm at first iteration: the `t schedule:` line reads
`t_dist=logitnormal mu=-0.8 sigma=0.8 force_zero_t_ratio=0.1`.
