# sigreg — making the delta code spread

The SIGReg regulariser on z: estimator, pool, weight, and the geometry it needs. Its weight is what sets usable rank
(see `../tc_width/`), so the live weight question is tracked there.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-07-25 | [sigreg_debug](2026-07-25_analysis_sigreg_debug.md) | analysis + solution | Why the SIGReg arm failed to learn | Estimator fix `e95de40`, validated on a 2-epoch dev run |
| 2026-07-26 | [layernorm_geometry](2026-07-26_findings_layernorm_geometry.html) | findings | Does SIGReg work after the fix? | Yes, but the z LayerNorm is the wrong geometry and the weight is 20× too high → nozn arms |
| 2026-07-28 | [param_search_plan](2026-07-28_task_param_search.md) | hypothesis | Which SIGReg knobs to search | Plan only, never submitted as written |
| 2026-07-28 | [z_spread_analysis](2026-07-28_analysis_z_spread.html) | analysis | How much of the 768-dim budget SIGReg buys | ~10×: participation ratio 173 vs 17.6 at matched ep 40 |
| 2026-07-29 | [bsc_migration_rolling_pool](2026-07-29_impl_bsc_migration_rolling_pool.html) | solution | SIGReg as a rolling sample pool; BSC on ehpc1001 | Landed |
| 2026-07-31 | [pool_not_weight_neutral](2026-07-31_findings_pool_not_weight_neutral.html) | findings | Is pool size weight-neutral? | No: pool 32768 at the same weight kills training, spike at warmup end |
| 2026-08-24 | [gapsig_vs_plain_slides](2026-08-24_report_gapsig_vs_plain_slides.html) | results + findings | Does Brownian gap-scaling (`sigreg_gap_sigma`) help? | No: 6.5–13.9% worse on all geometry losses, both eval sets |
| 2026-08-27 | [sigreg_compose_z_plan](2026-08-27_impl_sigreg_compose_z.md) | solution | SIGReg on all three compose streams | Code `1140de1`; half the gain of doubling the weight (tc_width Q4) |
