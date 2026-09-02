# pointdit — does an additive, unweighted x-MSE lift the 1-step readout?

Created 2026-09-02 · thread `pointdit` · prior cycle: `../analysis/2026-09-02_pointdit_vs_deltatok.md`
· init ckpt: `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit` iter 200000
· arm: `df_ctx3fwd2_tc128mg9s005compose_xmse13_xxl` · control: `df_ctx3fwd2_tc128mg9s005compose_ft20k_xxl`
· jobs: _pending_ · deck: _pending_

Mitigation 2 of the three in the analysis, **alone**. Split from the combined recipe on 2026-09-02; mitigation 1
is now `2026-09-02_pointdit_lowt_recipe_finetune.md`.

## 1 Hypothesis

Fine-tuning the trained compose arm with an additive **unweighted** x-MSE — pointdit's `rel_point_loss`,
nothing else changed — cuts 1-step `MSEToken` below **0.7417** and `LossDepth` below **3.6563** on the 128
held-out Waymo val sequences (`../results/2026-09-01_flow_numsteps_tc128compose_slides.html`, same ckpt,
N=1, `ode`, `linear`), and below the matched-compute `ft20k` control read at the same epoch 110.

The weight is **1.3, not pointdit's 0.1.** The v-MSE is the x-MSE divided by `(1-t)²`, and under this arm's
`t_dist=uniform` that weight has mean **39.0**; under pointdit's `logitnormal(-0.8, 0.8)` it has mean **2.96**.
So pointdit's 0.1 buys its x-term **3.6%** of the gradient mass, while 0.1 here would buy **0.28%** — inert.
`x_loss_weight=1.3` reproduces pointdit's 3.6% share on our schedule. Porting the number instead of the
balance would test nothing.

**Falsifiers.**

- **1-step MSEToken drops.** The 1-step readout was starved and an unweighted x-term is the instrument that
  reaches it. The recipe becomes default for new flow arms, and the weight gets a proper sweep.
- **1-step MSEToken flat vs the `ft20k` control.** 3.6% of gradient mass spread uniformly over `t` is not
  enough to move the `t=0` point, or the point was never the bottleneck. Combined with a flat forced-`t=0` arm
  this hands the question back to the flow thread and the wall argument.
- **1-step improves but N=20 degrades further.** The two regimes trade off; the x-term pulls the net toward the
  conditional mean everywhere. That is *fine* for our distortion metrics but is exactly the
  perception/distortion trade-off, and it makes the missing realism probe (boundary F1) the blocker.
- **Training destabilises or `LossFlow` rises.** At 3.6% mass share the x-term should not be able to move
  `LossFlow` much. If it does, the two terms are fighting and the weight is above the useful range — drop to
  0.3 (0.85% share) rather than abandoning.

**Not doing.**

- **No forced `t=0`.** That is `2026-09-02_pointdit_lowt_recipe_finetune.md`, run as its own arm off the same init
  so the two knobs are separable. The combined recipe is a follow-up only if both move alone.
- **No `loss_mode=x` swap.** `x` *replaces* the flow objective rather than supplementing it, which forfeits
  comparability with every existing flow number. pointdit keeps both terms, and so should we.
- **No literal 0.1 arm.** It buys 0.28% of the gradient mass on this schedule — below the run-to-run noise
  floor of 2.6–8% on Pointmap/Raymap (`../results/2026-08-20_flow_sampler_ablation.html`). A null there
  would be uninformative.
- **No relative-L1 form.** pointdit normalises by distance from the origin because pointmaps carry metric
  scale; SIGReg already pins our latents to unit scale, so plain MSE is the right analogue.
- **No `t_dist` change.** Keeping `uniform` isolates this one knob.
- **No fresh run from scratch.** Fine-tune from iter 200000; from scratch is ~65 h on BSC for the same question.

## 2 Analysis

What the weight buys, mean v-weight and x-term mass share, 2M Monte-Carlo draws:

| `x_loss_weight` | share on ours (`uniform`, mean w 39.0) | share on pointdit (`ln(-0.8,0.8)`, mean w 2.96) |
|---|---|---|
| 0.1 — pointdit's released default | 0.28% | **3.61%** |
| 0.3 | 0.85% | 10.1% |
| 1.0 | 2.77% | 27.3% |
| **1.3 — this arm** | **3.57%** | 32.8% |
| 3.0 | 7.87% | 52.9% |

`rel_point_loss_weight` is a `t_dist`-relative knob. The combined-recipe framing carried 0.1 over unexamined;
splitting the cycle is what surfaced that 0.1 on a uniform-`t` arm is 13× too small to reproduce pointdit's
balance.

Why this half is the one with teeth. The x-term is uniform in `t`, so 10% of its mass sits below `t=0.1` — at
weight 1.3 that is 0.36% of total gradient mass added to a band that currently holds 0.286%. It roughly
**doubles** the sub-`t=0.1` gradient, and unlike forced `t=0` it does so without pinning any sample, so the
flow objective keeps its full support. Forced `t=0` at 10% moves the same band 0.286% → 0.566% but costs 10% of
the batch's flow supervision (`2026-09-02_pointdit_lowt_recipe_finetune.md` §2).

Why now: `../results/2026-08-25_flow_vpred_vs_xpred_slides.html` showed `pred_mode=v` loses on every
geometry metric *and* on `LossFlow` itself, so the `x`-side of the parameterisation is where this arm lives.
Adding an explicit x-term is continuous with that result.

Prior that this is not free: four sampler / `t`-schedule arms off this same baseline were all flat or worse
(`../results/2026-08-20_flow_sampler_ablation.html`). Those were *eval-only* knobs — this one changes what
the net is trained on, which that sweep never did.

## 3 Solution

**Prerequisite, shared with the forced-`t=0` cycle: the wrapper does not forward an init checkpoint.**
`sh/train_deltatok_flow.sh:17` has `--ckpt` commented out and hardcoded to a dead `ehpc551` path.
`train_deltatok_flow.py:80` accepts it (weights-only pretrained init). Add the forward before either arm:

```bash
# sh/train_deltatok_flow.sh, replacing the commented-out line 17
[[ -n "${INIT_CKPT:-}" ]] && ARGS+=(--ckpt "$INIT_CKPT")   # weights-only fine-tune init
```

**Code — one new key, defaulting off.** In `flow_loss`, after the `loss_mode` branch closes and before
`return loss` (`occrae/deltatok_flow_trainer.py:541-544`):

```python
# pointdit rel_point_loss (engine.py:121-140): plain x-MSE, every t weighted equally.
x_w = float(self.cfg.model.get("x_loss_weight", 0.0))
if x_w > 0:
    loss = loss + x_w * ((x_pred - x) ** 2)[mask].mean()
```

`x_pred` is already defined for all three `pred_mode` branches (lines 522-534), so this needs no extra
reconstruction, and `mask` already excludes the context slots.

Declare it in `configs/deltatok_flow/train_deltatok_flow.yaml` beside `loss_mode` (line 61):
`x_loss_weight: 0.0  # pointdit rel_point_loss: additive unweighted x-MSE`.

**Log the two terms separately** through `metric_logger` (`LossFlow` and a new `LossX`), or the run reports one
number and the read cannot tell a flow regression from an x-term gain.

**Run.** Copy the baseline, do not rewrite:

```bash
cp slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_bsc.slurm \
   slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_xmse_bsc.slurm
```

Edit, keeping every other flag byte-identical to the baseline:

- `RUN_NAME=df_ctx3fwd2_tc128mg9s005compose_xmse13_xxl` — a fresh name, or it resumes the baseline's `current.pth`.
- `model.x_loss_weight=1.3`.
- `training.epoch=110`, `training.max_iter=220000` — 20k updates on top of the 200k init.
- `training.eval_num_steps=1` — the metric under test. Re-run the numsteps eval slurm on the finished ckpt for
  the full N curve.
- `export INIT_CKPT="$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit/ckpts/current.pth"`.
- `#SBATCH --time=12:00:00`, `--qos=acc_ehpc`. Walltime is absent from BSC's priority formula, so a short
  request only ever backfills sooner.

**The `ft20k` matched-compute control is shared with the forced-`t=0` cycle — submit it once**, from whichever
cycle launches first, and read both arms against it. Definition in
`2026-09-02_pointdit_lowt_recipe_finetune.md` §3.

**Pre-flight, then submit:**

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n \"x_loss_weight\" occrae/deltatok_flow_trainer.py configs/deltatok_flow/train_deltatok_flow.yaml && grep -n INIT_CKPT sh/train_deltatok_flow.sh && grep -E \"RUN_NAME|max_iter|x_loss_weight\" slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_xmse_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_xmse_bsc.slurm'"
```

The user syncs manually — if a grep comes back empty, the cluster copy is stale. Ask, do not `rsync`.

**Budget.** BSC steady state on this arm is **1.16 s/iter** at 4 GPU × bsize 16 (job 45111164, `[60/2000]`
line), so 2000 updates ≈ **39 min/epoch**. 20k updates = 10 epochs ≈ **6.5 h** plus evals. One 12 h job. The
extra x-MSE term is one elementwise square over a tensor already in memory — no measurable cost.

## 4 Results

_Pending._ Matched-epoch table, one block per eval set (Waymo val ×128, KITTI, nuScenes):

| arm | ep | N | MSEToken | LossDepth | LossPointmap | LossRaymap |
|---|---|---|---|---|---|---|
| baseline (iter 200000) | 100 | 1 | 0.7417 | 3.6563 | 8.1345 | 5.6041 |
| baseline | 100 | 20 | 0.8837 | 3.8796 | 9.4117 | 6.7993 |
| ft20k control | 110 | 1 | | | | |
| xmse13 | 110 | 1 | | | | |
| xmse13 | 110 | 20 | | | | |

Δ% against the **`ft20k` control** at the same `N`, not against the ep-100 baseline row. Report `LossFlow` and
`LossX` separately — a total that improves while `LossFlow` regresses is the trade-off, not a win. The `_tok`
teacher ceilings must be identical across rows; if they move, the eval pool changed.

Tracking: job | arm | state | epoch | note. Logs `slurm/output/`, TB mirror
`/mnt/d/tb_logs/deltatok_flow_log/df_ctx3fwd2_tc128mg9s005compose_xmse13_xxl/tb_logs/`.

## 5 Findings

_Pending._

## → Next hypothesis

`open`. Three branches, decided by §5:

- **Moves** — sweep `x_loss_weight` at matched mass share (0.3 / 1.3 / 3.0 = 0.85% / 3.6% / 7.9%), then combine
  with forced `t=0` if that arm also moved.
- **Flat** — 3.6% of gradient mass spread over all `t` does not reach the `t=0` readout. Try the share that
  pointdit's *median* `t` implies rather than its mean, or accept that the schedule, not the loss, is the lever
  and open the `logitnormal(-0.8, 0.8)` cycle.
- **Both this and the forced-`t=0` arm flat** — the 1-step readout was never starved, and the wall is the
  frozen decoder or the conditional variance per `../analysis/2026-07-19_flow_wall.html`. Then the
  blocker is the realism probe: port pointdit's scale-invariant depth-boundary F1 (`engine.py:869-893`).
