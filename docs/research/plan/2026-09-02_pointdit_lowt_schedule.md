# pointdit — does the low-`t` training schedule beat uniform, trained from scratch?

Created 2026-09-02 · thread `pointdit` · prior cycle: [`../analysis/2026-09-02_pointdit_vs_deltatok.md`](../analysis/2026-09-02_pointdit_vs_deltatok.md)
· arm: `df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl`
· control: `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit` @ ep 100 (already measured)
· jobs: `_pending_` · deck: `_pending_`

Merges `2026-09-02_pointdit_schedule_scratch.md` (hypothesis) and `2026-09-02_pointdit_schedule.md`
(implementation), both deleted 2026-09-02 when the plan template went to one file per cycle.

## 1 Hypothesis

Training the tc128 compose flow arm **from scratch** under pointdit's schedule — `t = sigmoid(-0.8 + 0.8·ε)`
with 10% of each batch pinned to exactly `t=0` — cuts 1-step `MSEToken` below **0.7417** and `LossDepth` below
**3.6563** on the 128 held-out Waymo val sequences at matched **ep 100**, against the uniform-`t` control read
at the same epoch with the same sampler (`../results/2026-09-01_flow_numsteps_tc128compose_slides.html`,
`ode`, `linear`, N=1). `N=20` is predicted to degrade.

The schedule is the only variable. Every other flag stays byte-identical to the control, including the frozen
tokenizer `epoch_100.pth`, `t_shared=true`, `sampler_alpha=0`, `loss_mode=v` and `pred_mode=x`.

**Why both knobs at once.** They are not separable in effect — see the mass table in §2. Forced `t=0` samples
carry weight 1, so their share of the gradient is `r / (r + (1-r)·E[w])`. On `uniform`, `E[w] = 39.0` and 10%
zeros buy **0.28%**. Under `logitnormal(-0.8, 0.8)`, `E[w] = 2.96` and the same 10% buy **3.62%**. The
logit-normal alone buys **0.00%** at `t=0` and only 1.59% below `t=0.1`, which is where the already-refuted
`(-0.7, 1.4)` arm sits (1.55%). Neither single knob reaches pointdit's regime; the pair does.

**Falsifiers.**

- **N=1 improves, N=20 degrades.** The expected outcome. The schedule buys the 1-step readout by spending the
  mass that used to sit above `t=0.95` (51.2% → 0.03%). Then 1-step inference becomes the arm's declared
  operating point, and the blocker moves to realism: every metric we log is a distortion metric, so port
  pointdit's boundary-F1 next.
- **N=1 improves and N=20 holds.** The uniform schedule was simply wasting half its mass. Then
  `logitnormal(-0.8, 0.8) + 10% zeros` becomes the default training schedule for the flow arm, and the two
  open `pointdit` fine-tune arms are subsumed.
- **Everything degrades.** Reproduces `logitnormal(-0.7, 1.4)` (+2.9% LossPointmap, +3.3% LossRaymap at
  ep 60–99, `../results/2026-08-20_flow_sampler_ablation.html` §3). Then the collision is with
  `loss_mode=v` itself and not with `sigma`, low-`t` training is refuted for this objective, and the lever is
  the loss — the additive unweighted x-MSE of `2026-09-02_pointdit_xloss_additive.md`.
- **N=1 flat.** 5.16% of the mass below `t=0.1` is still not enough, or the 1-step wall is not starvation at
  all but the frozen decoder / conditional variance per `../analysis/2026-07-19_flow_wall.html`. That file
  then becomes the thread's blocker and this schedule is closed.
- **Train loss drops while every rollout metric worsens.** The signature of the refuted arm
  (`sampler_ablation` §6: train 0.499 vs 0.593 at ep 90, rollouts worse). **`Train/Loss` and `Eval/LossFlow`
  are not readable across `t_dist` arms** — they optimise a differently-weighted objective. Read only the
  decoded metrics.

**Not doing.**

- **No `t_dist`-alone arm and no `force_zero_t`-alone arm here.** Both singles already exist:
  `2026-09-02_pointdit_lowt_recipe_finetune.md` is the 10%-zeros-on-uniform arm, and `logitnormal` alone is
  0.04 pp from the refuted `(-0.7, 1.4)` arm on sub-`t=0.1` mass. Running them again would buy attribution we
  can already reconstruct from the mass table.
- **No `sigma` or `mu` sweep.** `(-0.8, 0.8)` is pointdit's released default (`main.py:82-83`). Sweep only if
  this arm moves.
- **No fine-tune.** A 20k fine-tune off iter 200000 leaves a "20k was too short to re-fit a schedule this
  different from the 200k of uniform behind it" escape hatch. From scratch is the true head-to-head against a
  control that is already measured at ep 100, and it matches the protocol of the refuted arm.
- **No `force_zero_t_ratio` sweep.** 0.1 is pointdit's default.
- **No x-MSE term, no zeros ODE init.** Separate arm and dropped, respectively — see the `pointdit` ledger.
- **No `t_shared=false`.** `tindep` was measured at +0.8% and is a different question.

## 2 Why it is worth the GPU hours

### The mass table

Gradient mass under the shared `1/clamp(1-t, 0.05)²` weight of `loss_mode=v`, 4M Monte-Carlo draws. Mass
fraction is `E[w(t)·1{t∈A}] / E[w(t)]`; `med t` is the unweighted median.

| schedule | mean `w` | `t == 0` | `t < 0.1` | `t < 0.5` | `t > 0.95` | med `t` |
|---|---|---|---|---|---|---|
| `uniform` — the control | 39.00 | 0% | 0.28% | 2.56% | 51.16% | 0.500 |
| `uniform` + 10% `t=0` — open `pointdit` arm | 35.22 | 0.28% | 0.57% | 2.84% | 51.21% | 0.444 |
| `logitnormal(-0.7, 1.4)` — **refuted** arm | 10.44 | 0% | 1.55% | 12.56% | 17.81% | 0.332 |
| `logitnormal(-0.8, 0.8)` — schedule alone | 2.96 | 0% | 1.59% | 59.32% | 0.02% | 0.310 |
| **`logitnormal(-0.8, 0.8)` + 10% `t=0` — this arm** | **2.77** | **3.62%** | **5.16%** | **60.81%** | **0.03%** | **0.287** |

Three readings decide the design.

- **The two knobs are multiplicative, not additive.** Row 2 and row 4 each move sub-`t=0.1` mass by under
  1.4 pp. Row 5 moves it 18×. The forced zeros only bite once the mean weight has collapsed.
- **`sigma` does not buy low-`t` mass; it kills the high-`t` tail.** Rows 3 → 4 move sub-`t=0.1` mass by
  0.04 pp while `t > 0.95` goes 17.81% → 0.02% and the mean weight goes 10.44 → 2.96. So "narrower
  logit-normal" is a statement about what the arm *stops* training, and that is the risk this cycle carries.
- **This refines, and does not contradict, the analysis file.**
  `../analysis/2026-09-02_pointdit_vs_deltatok.md` quotes 3.6% / 5.2% / 0.02% for the combined row; the MC
  reproduces it at 3.62% / 5.16% / 0.03%. The two new rows are the decomposition that file did not have.

### The control

`../results/2026-09-01_flow_numsteps_tc128compose_slides.html`, same checkpoint, `ode` + `linear`, 128 val
sequences, seed 42, teacher ceilings `LossPointmap_tok 4.0557 · LossDepth_tok 2.7923 · LossRaymap_tok 1.6763`:

| steps | MSEToken | LossPointmap | LossDepth | LossRaymap |
|---|---|---|---|---|
| **1** | **0.7417** | 8.1345 | **3.6563** | 5.6041 |
| 2 | 0.7699 | **7.9687** | 3.7226 | **5.2918** |
| 3 | 0.7986 | 8.1355 | 3.7721 | 5.4819 |
| 4 | 0.8166 | 8.3646 | 3.8001 | 5.7242 |
| 20 | 0.8837 | 9.4117 | 3.8796 | 6.7993 |

`MSEToken` is strictly monotone in step count. The best arm still sits 2.0× the pointmap ceiling. The
`LossFlow` column of that deck is not readable — each pass drew its own `t`, fixed since by `_eval_t_gen`
(commit 4aa38dd).

### Why the refuted arm does not settle this

`df_ctx3fwd2_tc128mg9s005compose_logitnorm_xxl` (BSC:44767716) lost at ep 60–99: LossPointmap 9.341 vs 9.076
(+2.9%), LossRaymap 6.525 vs 6.317 (+3.3%), LossDepth 3.860 vs 3.868 (−0.2%). Three reasons it is not this
arm's answer.

- **Read at N=20 only.** That was its training-time `eval_num_steps`. Nobody has read it at N=1, which is the
  metric this cycle is about, and N=20 is the step count the low-`t` schedule is predicted to lose.
- **Different tokenizer and different epoch length.** All five ablation arms used `epoch_40.pth` and 1000
  iters/epoch; the current control uses `epoch_100.pth` and 2000. Its ep 100 is not this control's ep 100.
- **No forced zeros, and `sigma=1.4`.** It keeps 17.8% of its mass above `t=0.95` and 0% at `t=0` — a third of
  the way to pointdit, not at it.

### The free prior, before spending 65 h

Both ablation checkpoints are still on BSC and `slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm`
already takes a `CKPT` override, so the N-sweep on the refuted arm and its own base twin costs two `acc_debug`
jobs, minutes each:

```
$SCRATCH/deltatok_flow_log/df_ctx3fwd2_tc128mg9s005compose_logitnorm_xxl/ckpts/current.pth        # verified 2026-09-02
$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_xxl_dit/ckpts/current.pth
$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0/ckpts/epoch_40.pth
```

Both need `model.deltatok_ckpt` repointed to **`epoch_40.pth`**; the checked-in eval script hardcodes
`epoch_100.pth` at line 44. `model.t_dist` at eval touches only the teacher-forced `LossFlow` term, never the
ODE rollout, so it can stay as-is for the decoded metrics.

This is a prior, not a gate. If the refuted arm already wins at N=1 against its own base twin, the from-scratch
arm is well motivated before it is submitted. If it loses at N=1 too, the third falsifier is live and the
`sigma` story has to carry the whole cycle — worth knowing for 30 minutes of `acc_debug`.

## 3 How it is run

### The patch — `force_zero_t_ratio` does not exist yet

`grep -rn force_zero_t occrae/ configs/ sh/ slurm/` is empty on 2026-09-02. The patch is shared with
`2026-09-02_pointdit_lowt_recipe_finetune.md` — land it once. In `flow_noising`, after the `t_dist` branch
raises at `occrae/deltatok_flow_trainer.py:482` and **before** `t = t.expand(b, t_dim).contiguous()` at
line 485:

```python
# pointdit --force_zero_t: pin a fraction of the batch to exactly t=0 (denoiser.py:76-85).
zr = float(self.cfg.model.get("force_zero_t_ratio", 0.0))
if zr > 0 and self._eval_t_gen is None:                                    # train only; eval t must stay replayable
    m = torch.rand(b, n_draw, generator=g, device=device) < zr             # (b, 1|t)
    t = t.masked_fill(m, 0.0)                                              # (b, 1|t)
```

The `self._eval_t_gen is None` guard matters: eval seeds `t` on purpose (commit 4aa38dd) so `LossFlow` compares
weights and not draws. Placing the block inside the `else` keeps `train_fixed_t` unaffected.

Declare it beside `train_fixed_t` at `configs/deltatok_flow/train_deltatok_flow.yaml:69`:
`force_zero_t_ratio: 0.0  # pointdit --force_zero_t: fraction of the batch pinned to t=0`.

Log the mean `t` and the realised zero fraction through `metric_logger`, or a mis-wired mask reads as a null.

**`t_dist` needs no code.** `model.mu` / `model.sigma` / `model.t_dist` already exist
(`configs/deltatok_flow/train_deltatok_flow.yaml:65-67`), `flow_noising` reads them
(`occrae/deltatok_flow_trainer.py:473-482`), and the eval path forwards them. That half is config-only.

### The arm

Copy the control, do not rewrite:

```bash
cp slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_bsc.slurm \
   slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_pointditT_bsc.slurm
```

Edit, keeping every other flag byte-identical to the control:

- `RUN_NAME=df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl` — a fresh name, or it resumes the control's `current.pth`.
- `model.t_dist=logitnormal` (control: `uniform`).
- `model.mu=-0.8` and `model.sigma=0.8` — new overrides; the config defaults are `-0.7 / 1.4`, the refuted pair.
- `model.force_zero_t_ratio=0.1`.
- `#SBATCH --time=40:00:00`, `--qos=acc_ehpc`, chained ×2 with the `chain-slurm-jobs` skill.
  `training.exit_before_time_limit=true` is already set and keeps the split checkpoint-safe.
- **`training.eval_num_steps=20` stays as the control has it.** See below — do not set it to 1.

Unchanged and load-bearing: `CONFIG_NAME=train_deltatok_flow_waymo_ep128k_bsc`,
`model.deltatok_ckpt=...compose1.0/ckpts/epoch_100.pth`, `training.epoch=100`, `training.max_iter=200000`,
`training.effective_bsize=64`, `model.t_shared=true`, `model.sampler_scheduler_mode=linear`,
`model.sampler_alpha=0`, `model.loss_mode=v`, `training.val_bsize=4`.

### Reading it at N=1 needs no new code

The comparison is an **offline** sweep on the two ep-100 checkpoints, not the live curve.
`eval_deltatok_flow_sampler.py:206-210` already loops over `--num_steps` and re-sets
`cfg.training.eval_num_steps` per pass, and `slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm` runs
`NUM_STEPS=1,2,3,4,20`, `STEP_MODES=ode` on any `CKPT` in one 2 h `acc_debug` job. That is exactly how the
control's §2 numbers were produced, so it is the only matched read.

This is why the arm keeps `eval_num_steps=20`: the control logged its live curve at 20, and dropping the arm
to 1 would make the two TB curves incomparable for the whole 65 h with nothing gained.

**Optional watch knob, ~6 lines, not part of the comparison.** In-training eval is single-valued —
`occrae/deltatok_flow_trainer.py:713` reads one scalar and `eval_one_epoch` runs one pass. To also log N=1
every epoch, give `eval_one_epoch` a `num_steps` / `tag` pair, suffix the key at line 1023
(`Eval/{test_name}{tag}/{key}`), and loop at the call site (line 1108) over a new
`training.eval_extra_num_steps: []`. Its only value is catching the schedule early instead of at ep 100; the
control has no such curve, so it can never carry the head-to-head. **Cost is unmeasured** — the extra pass
saves 19/20 of the DiT forwards but repeats the full DeltaTok rollout and OccRAE decode, which likely dominate.
Set `eval_num_visualizations=0` on the extra pass if it lands.

### Pre-flight, then submit

The user syncs manually. If a grep comes back empty the cluster copy is stale — ask, never `rsync`.

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n force_zero_t_ratio occrae/deltatok_flow_trainer.py configs/deltatok_flow/train_deltatok_flow.yaml && grep -E \"RUN_NAME|t_dist|model.mu|model.sigma|force_zero_t|eval_num_steps|max_iter|deltatok_ckpt\" slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_pointditT_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_pointditT_bsc.slurm'"
```

A job id is not a launch. Watch until the first loss line, then check the logged zero fraction is ≈0.1 and the
mean `t` ≈0.29 — a schedule that silently stayed `uniform` reads as a clean null.

### Budget

BSC steady state on this arm is **1.16 s/iter** at 4 GPU × bsize 16 (job 45111164), so 2000 updates ≈
**39 min/epoch** and 100 epochs ≈ **65 h** plus evals. Two chained 40 h `acc_ehpc` jobs. Walltime is absent from
BSC's priority formula, so the 40 h split only ever backfills sooner than one 72 h request.

### The read

At **ep 100**, never earlier. Sweep both ep-100 checkpoints through
`slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm` with `NUM_STEPS=1,2,3,4,20`, `STEP_MODES=ode`, so
the control row reproduces §2 and doubles as a checkpoint-load check.

Waymo val ×128, `ode` · `linear` · seed 42:

| arm | ep | N | MSEToken | LossPointmap | LossDepth | LossRaymap |
|---|---|---|---|---|---|---|
| control `uniform` | 100 | 1 | 0.7417 | 8.1345 | 3.6563 | 5.6041 |
| control `uniform` | 100 | 20 | 0.8837 | 9.4117 | 3.8796 | 6.7993 |
| pointditT | 100 | 1 | | | | |
| pointditT | 100 | 2 | | | | |
| pointditT | 100 | 20 | | | | |

Free-prior block, `epoch_40` tokenizer, **not comparable to the rows above** (different tokenizer, 1000
iters/ep):

| arm | ep | N | MSEToken | LossPointmap | LossDepth | LossRaymap |
|---|---|---|---|---|---|---|
| ablation base `uniform` | 100 | 1 | | | | |
| ablation base `uniform` | 100 | 20 | | | | |
| `logitnorm(-0.7,1.4)` | 100 | 1 | | | | |
| `logitnorm(-0.7,1.4)` | 100 | 20 | | | | |

Δ% against the control at the same `N`. The `_tok` teacher ceilings must be identical within each block; if
they move, the eval pool changed. Do not tabulate `Train/Loss` or `Eval/LossFlow` across `t_dist` arms.

Tracking: job | arm | state | epoch | note. Logs `slurm/output/` and `../monitor_jobs/data/logs/BSC/`, TB mirror
`/mnt/d/tb_logs/deltatok_flow_log/df_ctx3fwd2_tc128mg9s005compose_pointditT_xxl/tb_logs/`.

## 4 Outcome

`_pending_`. Four branches, decided by the §3 read, in falsifier order:

- **N=1 wins, N=20 loses** — declare 1-step the operating point and port pointdit's scale-invariant
  depth-boundary F1 (`engine.py:869-893`), since every metric we log is minimised by the conditional mean.
- **N=1 wins, N=20 holds** — make the schedule the flow default and close the two open `pointdit` fine-tune arms.
- **Everything degrades** — the collision is `loss_mode=v`, not `sigma`. Hand the thread to
  `2026-09-02_pointdit_xloss_additive.md`.
- **Flat** — starvation was never the wall. `../analysis/2026-07-19_flow_wall.html` becomes the blocker and
  this thread closes.
