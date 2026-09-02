# DeltaTok roadmap — the CVPR paper

**Goal:** a CVPR paper. Deadline: _fill in_. This file is the paper skeleton. Each section lists what the paper needs,
what is measured (with the doc that holds the number), and what is missing. The work queue — what to run next and
its status — is in [`todo/02-09-2026.md`](todo/02-09-2026.md).

## Contributions

For a geometry-driven driving world model: a delta-token latent that is compact, composable, and spread.

- **SIGReg on the delta code** (threads `sigreg`, `tc_width`). A sliced-Gaussian regulariser that keeps the code spread and sets its usable rank. ~10× more of the channel budget used than without it (participation ratio 173 vs 17.6 at matched ep 40), and at tc512 doubling its weight cut eval recon 29% and lifted rank 2.6×. Recon tracks rank at r = −0.99 across arms; rank, not width, is the state variable.
  - **The weight is not width-transferable; it must be tuned jointly with `Cz`.** At a fixed 0.005 the same weight pins the code's *scale* at every width — `TotalVar/Cz` stays 0.94–1.10 from Cz 128 to 1536 — while rank grows only 2.3× (78.9 → 183.1, KITTI) for a 12× budget, so the fraction of the budget used collapses 62% → 12% and tc512 and tc768 land on the *identical* nuScenes rank of 133.9 (`research/results/2026-08-16_tc_width_tc_sweep_sigreg_slides.html`, ep 47–48). Across the ep-67 compose sweep, Cz vs rank is r = −0.10 (`research/results/2026-08-27_tc_width_tc_sigreg_ab_slides.html`).
  - **Rank follows the weight, not the width.** tc512 needs 0.01 to reach the rank tc128 has at 0.005, and there the two tie on eval recon at matched rank (0.0416 / 0.0414; rank 97.2 / 90.3). So SIGReg does buy rank — at fixed width, 0.005 → 0.01 lifted it 36.7 → 97.2 — but it does not deliver rank in proportion to `Cz`, which is what makes extra channels dead. The `weight ∝ Cz` rule is the calibration under test in [TODO 2](todo/02-09-2026.md).
- **Additive composition** (thread `compose`). Two hop deltas add to the span delta, `d13 + d35 ≈ d15`, so a long transition is one decode instead of a chained rollout. At equal budget (ep 99) composed reconstruction is 3.4× better than the plain tokenizer for +3% on single-step recon, and the plain arm regresses on composability after ep 11.
- **Delta-token compression** (thread `tc_width`). How small the geometry code can be: cutting the token count beats squeezing channels at equal floats (channel squeeze converges ~7× worse), and tc128 is the best width at matched compute once SIGReg sets the rank.

## Method

What each component is in the code and where its design lives.

| Component | Code | Design doc | State today |
|---|---|---|---|
| Delta tokenizer over frozen DA3 layer-12 tokens | `occrae/deltatok_trainer.py`, `occrae/deltatok_shared.py` | `occrae/occrae.md` | production recipe dtok64 · nozn · maxgap9 · vpt1to2 |
| SIGReg on z | `occrae/sigreg.py` | `research/analysis/2026-07-25_sigreg_debug.md`, `research/plan/2026-07-29_sigreg_bsc_migration_rolling_pool.html` | rolling pool 8192, ns = 2·Cz, warmup 2000, weight 0.005 (0.01 at tc512). **Weight is width-dependent, not a constant** — see the contribution bullet |
| Additive composition | `_compose_forward` in `occrae/deltatok_trainer.py` | `research/plan/2026-08-12_compose_additive_composition.md` | `compose_weight` 1.0 |
| Compression knobs K, Cz | `num_delta_tokens`, `target_channels` | `research/plan/2026-07-03_tc_width_num_delta_tokens_sweep.html` | best: K=64, tc128 |
| Flow world model, DiT over delta tokens | `occrae/deltatok_flow_trainer.py`, sampler in `occrae/generation_helper.py` | `research/plan/2026-05-31_flow_matching_trainer.md`, `research/plan/2026-07-07_flow_dtok32.html` | XXL DiT, `pred_mode=x`, `loss_mode=v`, 5 consecutive frames, cam 0, Waymo-only |

## Main results

**Target setting.**

- **Input.** Monocular, metric-scale, 1 view.
- **Train.** Waymo, DDAD, Pandaset, ONCE, OpenScene.
- **Eval.** Depth forecast + pointmap forecast. In-domain: OpenScene, Waymo. OOD: KITTI, nuScenes. Plus FVD on
  OccAny features, averaging the patch tokens ([TODO 10](todo/02-09-2026.md)).

**Where the code is against it.**

- Loaders exist for Waymo, DDAD, Pandaset, ONCE, VKITTI, KITTI, nuScenes (`occany/datasets/`). **OpenScene has no loader and no preprocessing.**
- The one multi-dataset flow arm trained on Waymo + VKITTI + DDAD + Pandaset + ONCE and **trails Waymo-only at matched epochs** on all three eval sets (`research/results/2026-08-24_flow_alldata_xxl_eval.html`). The multi-dataset recipe is unsolved.
- Flow eval logs `LossDepth`, `LossPointmap`, `LossRaymap`, their `_tok` teacher ceilings, and `MSEToken`. **FVD is not implemented.**
- Current flow eval sets: KITTI, nuScenes, 32 held-out Waymo scenes.

**Baselines.**

| Baseline | In checkout | Run on our setting |
|---|---|---|
| DeltaTok (upstream) | `third_party/deltatok` | no |
| VGGT-World | `third_party/VGGT-World` | no; used only as a recipe reference in `research/analysis/2026-07-18_flow_whitenpos.html` |
| Gen3R | **not in `third_party/`** | no |

**Main table.** Rows: ours and the three baselines. Columns: depth and pointmap forecast on OpenScene, Waymo, KITTI, nuScenes. **No cell is measured yet.** The closest numbers are Waymo-only flow arms evaluated on Waymo / KITTI / nuScenes, e.g. `research/results/2026-09-01_flow_numsteps_tc128compose_slides.html`.

## Ablations

What is measured, on what, and where. Every row is one seed. Tokenizer rows are evaluated on KITTI + nuScenes
round-trip reconstruction, not on forecasting; none is re-run at the target setting yet.

| Ablation | Knob | Measured on | Result | Doc |
|---|---|---|---|---|
| SIGReg on / off | `sigreg_weight` 0 vs 0.05 | tokenizer, ep 40 | PR 173 vs 17.6 of 768; baseline collapses to ~50 dims | `research/analysis/2026-07-28_sigreg_z_spread.html` |
| SIGReg weight | 0.002 … 0.08 at tc512 | tokenizer, ep 42 / 67 | monotone gain to 0.01 (−29% recon); 0.02–0.08 running | `research/plan/2026-09-01_tc_width_sigreg_weight_tc512.md` |
| SIGReg weight × width | 0.005 held over Cz 128 … 1536 | tokenizer, ep 47 / 67 | at one weight SIGReg pins scale (`TotalVar/Cz` 0.94–1.10) but not rank: 12× budget buys 2.3× rank, 62% → 12% used | `research/results/2026-08-16_tc_width_tc_sweep_sigreg_slides.html`, `research/results/2026-08-27_tc_width_tc_sigreg_ab_slides.html` |
| SIGReg pool size | 8192 vs 32768 | tokenizer | not weight-neutral; 32768 kills training | `research/analysis/2026-07-31_sigreg_pool_not_weight_neutral.html` |
| Gap scaling | `sigreg_gap_sigma` | tokenizer, ep 67 | 6.5–13.9% worse on all geometry losses | `research/results/2026-08-24_sigreg_gapsig_vs_plain_slides.html` |
| Composition on / off | `compose_weight` 0 vs 1 | tokenizer, ep 99 | Comp 3.4× better, recon +3%; plain regresses on Comp after ep 11 | `research/results/2026-08-25_compose_vs_plain_slides.html` |
| SIGReg on the composed sum | `sigreg_compose_z` | tokenizer, ep 67 | half the gain of doubling the weight — verdict disputed by a later read, re-tested at 0.02 in [TODO 8](todo/02-09-2026.md) | `research/plan/2026-08-27_sigreg_compose_z.md`, `research/results/2026-09-01_tc_width_tc512_sigreg_weight_slides.html` |
| Token count K | 1 … 64 | tokenizer, pre-1ecb17c eval | cutting K beats squeezing Cz at equal floats | `research/results/2026-07-07_tc_width_num_delta_tokens_ablation.html` |
| Channel width Cz | 64 … 1536 | tokenizer, ep 40 / 67 | tc128 best; rank, not width, predicts recon | `research/results/2026-08-16_tc_width_tc_sweep_sigreg_slides.html`, `research/results/2026-08-27_tc_width_tc_plain_vs_compose_slides.html` |
| Pair sampling | random-interval vs consecutive; `max_gap` | tokenizer | random-interval 3× faster; maxgap9 adopted | `research/results/2026-08-05_pair_sampling_randint_vs_consec_convergence_slides.html`, `research/results/2026-08-10_pair_sampling_maxgap_sweep.html` |
| Tokenizer width into the flow | tc128 / 768 / 1536 | flow, Waymo, ep 100 | depth improves with tc; pointmap and raymap do not | `research/results/2026-08-18_flow_tc_sweep_slides.html` |
| Latent whitening / SIGReg for the flow | raw vs whitened vs SIGReg z | flow, Waymo | whitening was never the bottleneck | `research/results/2026-08-16_flow_sigreg_vs_whiten_slides.html`, `research/analysis/2026-07-18_flow_whitenpos.html` |
| Prediction target | v vs x | flow, Waymo | x wins on every geometry metric and on LossFlow | `research/results/2026-08-25_flow_vpred_vs_xpred_slides.html` |
| Sampler and t-schedule | 4 arms | flow, Waymo, ep 101 | all flat or worse | `research/results/2026-08-20_flow_sampler_ablation.html` |
| ODE steps at eval | 1 … 20 | flow, one checkpoint | 1 step best; error rises monotonically | `research/analysis/2026-09-01_flow_numsteps_why_more_steps_degrade.md` |
| Frames and cameras | consecutive vs random; front vs random cam | flow, Waymo | consecutive required; front-cam wins | `research/results/2026-08-05_flow_frontcam_vs_randcam_slides.html` |
| Training mixture | Waymo vs 5 sources | flow, ep 77 | multi-dataset trails Waymo-only | `research/results/2026-08-24_flow_alldata_xxl_eval.html` |

**Gaps.** No seeds. No ablation at the target setting. The K sweep predates 1ecb17c, so its eval is not comparable to
later arms. The compose and SIGReg ablations exist only as tokenizer reconstruction, not as forecast quality.

**The width sweep is confounded with regulariser pressure.** Every width in `research/results/2026-08-16_tc_width_tc_sweep_sigreg_slides.html`
and `research/results/2026-08-27_tc_width_tc_plain_vs_compose_slides.html` ran at `sigreg_weight=0.005`, a value inherited from tc128 and never
itself optimised. Since the weight that a width needs scales with `Cz`, the wide arms were under-regularised, and "width does
not help" is measured only at one weight. The tc512 point already moved: 0.005 → 0.01 cut eval recon 29%. [TODO 2](todo/02-09-2026.md) calibrates
the rule; until it lands, no width conclusion in the paper is safe.
