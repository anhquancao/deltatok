# DeltaTok DONE — closed items

Items closed out of [`TODO.md`](TODO.md), newest first. A row keeps its original `#`, so a `TODO 7`
reference stays valid after the item moves here. `Closed` is the date the status was set, not the date the
measurement ran. `dropped` rows live here too — a dropped item is closed, and its reason is the record.

| # | Item | Thread | Paper | Closed | Status |
|---|---|---|---|---|---|
| 1 | Try 1, 2, 3, 4 steps like PointDiT | flow | ablation: ODE steps | 2026-09-02 | `done: research/results/2026-09-01_flow_numsteps_tc128compose_slides.html`. `1,2,3,4,20` steps on `tc128compose` ckpt iter 200000, BSC:45296762. Fewer steps win monotonically — MSEToken 0.7417 at N=1 vs 0.8837 at N=20, LossDepth 3.6563 vs 3.8796. It does **not** measure step count: the sampler does not integrate, so N steps returns x̂ at `t = 1 − 1/N`, and N=1 reads out the arm's least-trained regime (0.28% of gradient mass below `t=0.1`). Mechanism: `research/analysis/2026-09-01_flow_numsteps_why_more_steps_degrade.md`. Left open: the seed spread at N=1 vs N=20 (8 seeds, plus a decoded depth-edge F1, one `acc_debug` job) was never submitted, so "the rise is sample variance" stays unmeasured; the re-read on a low-`t` arm moved to [TODO 7](TODO.md) |
| 9 | How is the sky handled when computing the eval loss for the flow matching? | flow | main results: eval protocol | 2026-09-02 | `done: research/analysis/2026-09-02_flow_sky_in_eval_loss.md`. Dropped implicitly — `valid_mask = depth>0 & depth<z_far(50)` over sparse LiDAR, so sky has no return and never enters `LossDepth` / `LossPointmap`. The flow loss is latent-space and keeps sky at full weight. Left open: the valid-pixel fraction per eval set is not logged |
