# DeltaTok roadmap — the CVPR paper

**Goal:** a CVPR paper. Deadline: _fill in_. This file is the paper skeleton. Each section lists what the paper needs,
what is measured (with the doc that holds the number), and what is missing. A TODO becomes a `task` file in its thread
when work starts, and its row here then points at that file. Status is one of `not started`, `started: <task file>`,
`running: <jobs>, read at ep N: <task file>`, `done: <file>`, `dropped: <reason>`; rows are never deleted.
Previous backlog: `research/2026-08-18_task_backlog.html`.

**Next big step (2026-09-02):** finalize the training and evaluation setting, then get baseline results. TODO 3 and 4;
they go before 1 and 2.

## Contributions

For a geometry-driven driving world model: a delta-token latent that is compact, composable, and spread.

- **SIGReg on the delta code** (`research/sigreg/`, `research/tc_width/`). A sliced-Gaussian regulariser that keeps the code spread and sets its usable rank. ~10× more of the channel budget used than without it (participation ratio 173 vs 17.6 at matched ep 40), and at tc512 doubling its weight cut eval recon 29% and lifted rank 2.6×. Recon tracks rank at r = −0.99 across arms; rank, not width, is the state variable.
- **Additive composition** (`research/compose/`). Two hop deltas add to the span delta, `d13 + d35 ≈ d15`, so a long transition is one decode instead of a chained rollout. At equal budget (ep 99) composed reconstruction is 3.4× better than the plain tokenizer for +3% on single-step recon, and the plain arm regresses on composability after ep 11.
- **Delta-token compression** (`research/tc_width/`). How small the geometry code can be: cutting the token count beats squeezing channels at equal floats (channel squeeze converges ~7× worse), and tc128 is the best width at matched compute once SIGReg sets the rank.

## Method

What each component is in the code and where its design lives.

| Component | Code | Design doc | State today |
|---|---|---|---|
| Delta tokenizer over frozen DA3 layer-12 tokens | `occrae/deltatok_trainer.py`, `occrae/deltatok_shared.py` | `occrae/occrae.md` | production recipe dtok64 · nozn · maxgap9 · vpt1to2 |
| SIGReg on z | `occrae/sigreg.py` | `research/sigreg/2026-07-25_analysis_sigreg_debug.md`, `research/sigreg/2026-07-29_impl_bsc_migration_rolling_pool.html` | rolling pool 8192, ns = 2·Cz, warmup 2000, weight 0.005 (0.01 at tc512) |
| Additive composition | `_compose_forward` in `occrae/deltatok_trainer.py` | `research/compose/2026-08-12_impl_additive_composition.md` | `compose_weight` 1.0 |
| Compression knobs K, Cz | `num_delta_tokens`, `target_channels` | `research/tc_width/2026-07-03_task_num_delta_tokens_sweep.html` | best: K=64, tc128 |
| Flow world model, DiT over delta tokens | `occrae/deltatok_flow_trainer.py`, sampler in `occrae/generation_helper.py` | `research/flow/2026-05-31_impl_flow_matching_trainer.md`, `research/flow/2026-07-07_impl_dtok32.html` | XXL DiT, `pred_mode=x`, `loss_mode=v`, 5 consecutive frames, cam 0, Waymo-only |

## Main results

**Target setting.**

- **Input.** Monocular, metric-scale, 1 view.
- **Train.** Waymo, DDAD, Pandaset, ONCE, OpenScene.
- **Eval.** Depth forecast + pointmap forecast. In-domain: OpenScene, Waymo. OOD: KITTI, nuScenes. Open: FVD on the features?

**Where the code is against it.**

- Loaders exist for Waymo, DDAD, Pandaset, ONCE, VKITTI, KITTI, nuScenes (`occany/datasets/`). **OpenScene has no loader and no preprocessing.**
- The one multi-dataset flow arm trained on Waymo + VKITTI + DDAD + Pandaset + ONCE and **trails Waymo-only at matched epochs** on all three eval sets (`research/flow/2026-08-24_report_alldata_xxl_eval.html`). The multi-dataset recipe is unsolved.
- Flow eval logs `LossDepth`, `LossPointmap`, `LossRaymap`, their `_tok` teacher ceilings, and `MSEToken`. **FVD is not implemented.**
- Current flow eval sets: KITTI, nuScenes, 32 held-out Waymo scenes.

**Baselines.**

| Baseline | In checkout | Run on our setting |
|---|---|---|
| DeltaTok (upstream) | `third_party/deltatok` | no |
| VGGT-World | `third_party/VGGT-World` | no; used only as a recipe reference in `research/flow/2026-07-18_findings_whitenpos.html` |
| Gen3R | **not in `third_party/`** | no |

**Main table.** Rows: ours and the three baselines. Columns: depth and pointmap forecast on OpenScene, Waymo, KITTI, nuScenes. **No cell is measured yet.** The closest numbers are Waymo-only flow arms evaluated on Waymo / KITTI / nuScenes, e.g. `research/flow/2026-09-01_report_numsteps_tc128compose_slides.html`.

## Ablations

What is measured, on what, and where. Every row is one seed. Tokenizer rows are evaluated on KITTI + nuScenes
round-trip reconstruction, not on forecasting; none is re-run at the target setting yet.

| Ablation | Knob | Measured on | Result | Doc |
|---|---|---|---|---|
| SIGReg on / off | `sigreg_weight` 0 vs 0.05 | tokenizer, ep 40 | PR 173 vs 17.6 of 768; baseline collapses to ~50 dims | `research/sigreg/2026-07-28_analysis_z_spread.html` |
| SIGReg weight | 0.002 … 0.08 at tc512 | tokenizer, ep 42 / 67 | monotone gain to 0.01 (−29% recon); 0.02–0.08 running | `research/tc_width/2026-09-01_task_sigreg_weight_tc512.md` |
| SIGReg pool size | 8192 vs 32768 | tokenizer | not weight-neutral; 32768 kills training | `research/sigreg/2026-07-31_findings_pool_not_weight_neutral.html` |
| Gap scaling | `sigreg_gap_sigma` | tokenizer, ep 67 | 6.5–13.9% worse on all geometry losses | `research/sigreg/2026-08-24_report_gapsig_vs_plain_slides.html` |
| Composition on / off | `compose_weight` 0 vs 1 | tokenizer, ep 99 | Comp 3.4× better, recon +3%; plain regresses on Comp after ep 11 | `research/compose/2026-08-25_report_compose_vs_plain_slides.html` |
| SIGReg on the composed sum | `sigreg_compose_z` | tokenizer, ep 67 | half the gain of doubling the weight | `research/sigreg/2026-08-27_impl_sigreg_compose_z.md` |
| Token count K | 1 … 64 | tokenizer, pre-1ecb17c eval | cutting K beats squeezing Cz at equal floats | `research/tc_width/2026-07-07_report_num_delta_tokens_ablation.html` |
| Channel width Cz | 64 … 1536 | tokenizer, ep 40 / 67 | tc128 best; rank, not width, predicts recon | `research/tc_width/2026-08-16_report_tc_sweep_sigreg_slides.html`, `2026-08-27_report_tc_plain_vs_compose_slides.html` |
| Pair sampling | random-interval vs consecutive; `max_gap` | tokenizer | random-interval 3× faster; maxgap9 adopted | `research/pair_sampling/2026-08-05_report_randint_vs_consec_convergence_slides.html`, `2026-08-10_report_maxgap_sweep.html` |
| Tokenizer width into the flow | tc128 / 768 / 1536 | flow, Waymo, ep 100 | depth improves with tc; pointmap and raymap do not | `research/flow/2026-08-18_report_tc_sweep_slides.html` |
| Latent whitening / SIGReg for the flow | raw vs whitened vs SIGReg z | flow, Waymo | whitening was never the bottleneck | `research/flow/2026-08-16_report_sigreg_vs_whiten_slides.html`, `2026-07-18_findings_whitenpos.html` |
| Prediction target | v vs x | flow, Waymo | x wins on every geometry metric and on LossFlow | `research/flow/2026-08-25_report_vpred_vs_xpred_slides.html` |
| Sampler and t-schedule | 4 arms | flow, Waymo, ep 101 | all flat or worse | `research/flow/2026-08-20_report_sampler_ablation.html` |
| ODE steps at eval | 1 … 20 | flow, one checkpoint | 1 step best; error rises monotonically | `research/flow/2026-09-01_findings_numsteps_why_more_steps_degrade.md` |
| Frames and cameras | consecutive vs random; front vs random cam | flow, Waymo | consecutive required; front-cam wins | `research/flow/2026-08-05_report_frontcam_vs_randcam_slides.html` |
| Training mixture | Waymo vs 5 sources | flow, ep 77 | multi-dataset trails Waymo-only | `research/flow/2026-08-24_report_alldata_xxl_eval.html` |

**Gaps.** No seeds. No ablation at the target setting. The K sweep predates 1ecb17c, so its eval is not comparable to
later arms. The compose and SIGReg ablations exist only as tokenizer reconstruction, not as forecast quality.

## TODO

The queue. `Paper` says which section or table the item fills.

| # | Item | Thread | Paper | Status |
|---|---|---|---|---|
| 1 | Try 1, 2, 3, 4 steps like PointDiT | flow | ablation: ODE steps | `not started`. Context: eval-time step sweep in `research/flow/2026-09-01_findings_numsteps_why_more_steps_degrade.md` |
| 2 | tc512 + sigreg 0.02 / 0.04 / 0.08 | tc_width | ablation: SIGReg weight | `running: BSC:45296347–49, read at ep 67: research/tc_width/2026-09-01_task_sigreg_weight_tc512.md` |
| 3 | Finalize the training and evaluation setting | cross | main results: target setting | `not started`. Open per "Where the code is against it": OpenScene loader, training mixture, FVD or not, final eval sets |
| 4 | Baseline results: DeltaTok, VGGT-World, Gen3R | cross | main results: main table | `not started`. Depends on 3. Per "Baselines": none run on our setting, Gen3R not in `third_party/` |
