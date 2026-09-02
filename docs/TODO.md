# DeltaTok TODO — the queue

The work queue for [`ROADMAP.md`](ROADMAP.md), the CVPR paper skeleton. `Paper` says which section or table each
item fills. A TODO becomes a `task` file in its thread when work starts, and its row here then points at that
file. Status is one of `not started`, `started: <task file>`, `running: <jobs>, read at ep N: <task file>`.
A row that reaches `done: <file>` or `dropped: <reason>` moves to [`DONE.md`](DONE.md) keeping its `#`; nothing is
deleted. Previous backlog: `research/plan/2026-08-18_cross_backlog.html`.

**Next big step (2026-09-02):** finalize the training and evaluation setting, then get baseline results. TODO 3 and 4;
they go before 2.

**This week (2026-09-02):** TODO 7 (pointdit choices), 8 (sum at 0.02 tc512), 10 (FVD on OccAny features).

| # | Item | Thread | Paper | Status |
|---|---|---|---|---|
| 2 | tc512 + sigreg 0.02 / 0.04 / 0.08 | tc_width | ablation: SIGReg weight | `running: BSC:45296347 (0.02) ep 33 · BSC:45297731 (0.04) ep 0 · BSC:45296349 (0.08) cancelled ep 33, read at ep 67: research/plan/2026-09-01_tc_width_sigreg_weight_tc512.md`. Sweep is broken up — see the note below |
| 3 | Finalize the training and evaluation setting | cross | main results: target setting | `not started`. Open per `ROADMAP.md` "Where the code is against it": OpenScene loader, training mixture, FVD or not, final eval sets |
| 4 | Baseline results: DeltaTok, VGGT-World, Gen3R | cross | main results: main table | `not started`. Depends on 3. Per `ROADMAP.md` "Baselines": none run on our setting, Gen3R not in `third_party/` |
| 5 | Re-read the width sweep at the calibrated weight | tc_width | ablation: channel width Cz | `not started`. Depends on 2. The sweep held weight at 0.005 across Cz 128–1536, so width and regulariser pressure are not separated |
| 7 | Implement the pointdit choices in the flow trainer | pointdit | ablation: flow recipe | `started`, three plans, none submitted. Mitigation 1 alone: `research/plan/2026-09-02_pointdit_lowt_recipe_finetune.md` (10% forced `t=0`, ~6.5 h BSC fine-tune off iter 200000). Mitigation 2 alone: `research/plan/2026-09-02_pointdit_xloss_additive.md` (unweighted x-MSE at weight 1.3). Both knobs from scratch: `research/plan/2026-09-02_pointdit_lowt_schedule.md` (`logitnormal(-0.8,0.8)` + 10% `t=0`, 100 ep, ~65 h BSC, 2 chained 40 h jobs). All three share one `force_zero_t_ratio` patch — land it once. The zeros ODE init is `dropped: not important for us` — eval-time sampler knob, no plan file was written. Mechanism: `research/analysis/2026-09-02_pointdit_vs_deltatok.md`. The fourth choice, the boundary-F1 realism probe, has no plan file yet. Inherited from [TODO 1](DONE.md): re-run the `1,2,3,4,20` step sweep once a low-`t` arm exists, where N=1 is no longer starved |
| 6 | Attack the rank ceiling directly instead of via the weight | sigreg | contribution: SIGReg | `not started`. A sliced-Gaussian statistic is a weak lever on spectrum *shape*; candidates are whitening or a per-channel rescale before the isotropy term, or a direct `‖Σ̂ − I‖²_F` covariance penalty as a third arm. Never a swap — it forfeits comparability to the no-SIGReg twin |
| 8 | Sum is actually better → run again with 0.02 tc512 (0.01 undecided); compare with 512 plain | sigreg | ablation: SIGReg on the composed sum | `not started`. The one sum arm measured is BSC:45122721 at weight **0.005** (`research/plan/2026-08-27_sigreg_compose_z.md`, ep 99; re-read in `research/results/2026-09-01_tc_width_tc512_sigreg_weight_slides.html`). Its effective pressure is ~1/3 the twin's — 3× live rows drops `scale` 17.0 → 6.33 — so 0.005 with the sum is not 0.005 without it |
| 10 | Add FVD on OccAny features for eval: average patch tokens | flow | main results: main table | `not started`. `occrae/util/training_summary.py:57` already consumes a `"FVD"` metric key and nothing emits it. Settles the FVD line in `ROADMAP.md` "Target setting" |

**TODO 2 status, 2026-09-02.** The three arms are no longer comparable at a matched epoch, and none has reached the ep-67 read point.

- **0.02 · BSC:45296347 · RUNNING, ep 33.** The healthy arm. Train 0.0896 (Recon 0.0430, raw `SIGReg:` 0.0039), Eval 0.0507, nuScenes `ZPartRank` 86.1. ~28 h of walltime left, enough to pass ep 67.
- **0.04 · BSC:45296348 FAILED, resubmitted as BSC:45297731 · RUNNING, ep 0.** The failure was infrastructural, not the objective: `torch.cuda.set_device` raised `CUDA error: out of memory` at startup on node `as06r3b02`, 1 m 17 s in. The resubmit queued ~19 h and started 2026-09-02 ~13:10, so it is 33 epochs behind the others in wall-clock.
- **0.08 · BSC:45296349 · CANCELLED 2026-09-02 13:40 at ep 33.** A `current.pth` exists, so it is resumable.

**Progress note on 0.08, not the read.** At matched ep 32 it sits at roughly twice the 0.02 arm's loss on every term — Train 0.1777 vs 0.0900, Recon 0.0813 vs 0.0432, Eval 0.0932 vs 0.0511 — and its rank is *lower*, not higher (`ZPartRank` 38.7 KITTI / 54.3 nuScenes vs 86.1 nuScenes at 0.02). That is the shape of over-regularisation, and it points at a turnover below 0.08. It is **not** the pool-32768 warmup blow-up: the raw `SIGReg:` term goes 0.0116 → 0.0141 → 0.0107 across the end of the 2000-iter warmup, a bump, not the 39× spike from `research/analysis/2026-07-31_sigreg_pool_not_weight_neutral.html`. One seed, ep 33 of a read fixed at 67; do not conclude the turnover from this.
