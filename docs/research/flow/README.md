# flow — flow matching over frozen delta tokens

The world model: a DiT that denoises DeltaTok codes conditioned on context frames. Twenty-two docs, from the June
collapse through the September step-count analysis.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-05-31 | [flow_matching_trainer_plan](2026-05-31_impl_flow_matching_trainer.md) | solution | Simplify the trainer for token flow matching | Landed as `occrae/deltatok_flow_trainer.py` |
| 2026-06-23 | [concat_cond_plan](2026-06-23_impl_concat_cond.md) | solution | Pooled frame-0 context token instead of cross-attention | Plan |
| 2026-06-23 | [camera_swap](2026-06-23_analysis_camera_swap.md) | analysis | Cameras swap in generated views | Per-camera embedding fix, unconfirmed on a run |
| 2026-06-25 | [mse_collapse](2026-06-25_analysis_mse_collapse.md) | analysis | Tiny loss but mean-collapsed samples | Confounded, see next row |
| 2026-06-26 | [overfit_data_confound](2026-06-26_findings_overfit_data_confound.md) | findings | Was the single-scene overfit fitting one sample? | No: `ray_map_prob=0.0` randomised ~9 view orders per batch; supersedes the collapse read |
| 2026-07-07 | [dtok32_plan](2026-07-07_impl_dtok32.html) | solution | Native-K flow, run K=32 | Landed; K=1 squeeze removed |
| 2026-07-08 | [dtok32_smoketest](2026-07-08_report_dtok32_smoketest.html) | results | Smoke test | Pose-estimation crash fixed; the dtok32 checkpoint is NaN |
| 2026-07-08 | [dtok16_deltactx_global](2026-07-08_report_dtok16_deltactx_global.html) | results | Convergence on the dtok16 tokenizer | Teacher-forced loss vs sampled metrics over 40 epochs |
| 2026-07-17 | [latent_whitening_plan](2026-07-17_impl_latent_whitening.html) | solution | RAE-style per-channel whitening | Landed behind a flag |
| 2026-07-18 | [whitenpos_failure](2026-07-18_findings_whitenpos.html) | findings | Why the whitened run does not work | Whitening was never the bottleneck |
| 2026-07-19 | [wall](2026-07-19_findings_wall.html) | findings | What the flow wall is | Decoder + conditional variance, not architecture or sampler; in-context fix proposed |
| 2026-07-22 | [waymo_dtok64_experiments_slides](2026-07-22_report_waymo_dtok64_experiments_slides.html) | results | Eight Waymo dtok64 runs | One wall: MSEToken ≈ 0.29, LossDepth ≈ 5.8 |
| 2026-08-05 | [frontcam_vs_randcam_slides](2026-08-05_report_frontcam_vs_randcam_slides.html) | results + findings | Consecutive frames; front-cam vs rand-cam | Non-consecutive never generates; consecutive works; front-cam wins every cam-0 decode |
| 2026-08-10 | [tc1536_maxgap9_plan](2026-08-10_impl_tc1536_maxgap9.md) | solution | Swap the frozen tokenizer to tc1536 maxgap9 nosigreg | Landed on BSC |
| 2026-08-13 | [whiten_vs_raw_nosigreg_slides](2026-08-13_report_whiten_vs_raw_nosigreg_slides.html) | results | Whiten vs raw z on the nosigreg tokenizer | Deck |
| 2026-08-16 | [sigreg_vs_whiten_slides](2026-08-16_report_sigreg_vs_whiten_slides.html) | results | Latent choices: SIGReg vs whiten vs raw | Deck |
| 2026-08-18 | [tc_sweep_slides](2026-08-18_report_tc_sweep_slides.html) | results | Tokenizer width into the same XXL DiT | Depth improves with tc; pointmap and raymap do not, tc1536 worst |
| 2026-08-20 | [sampler_ablation](2026-08-20_report_sampler_ablation.html) | results + findings | Four sampler / t-schedule arms off the compose tc128 baseline | All flat or worse; logit-normal loses |
| 2026-08-24 | [alldata_xxl_eval](2026-08-24_report_alldata_xxl_eval.html) | results + findings | Multi-dataset training | Degrades geometric losses vs Waymo-only at matched epochs |
| 2026-08-25 | [vpred_vs_xpred_slides](2026-08-25_report_vpred_vs_xpred_slides.html) | results + findings | Does v-prediction beat x-prediction? | No: loses on every geometry metric and on LossFlow |
| 2026-09-01 | [numsteps_tc128compose_slides](2026-09-01_report_numsteps_tc128compose_slides.html) | results | 1…20 ODE steps on one checkpoint | Error rises monotonically with steps |
| 2026-09-01 | [numsteps_why_more_steps_degrade](2026-09-01_findings_numsteps_why_more_steps_degrade.md) | findings, **open** | Why more steps degrade | The sampler does not integrate: N steps returns x̂ at t = 1−1/N. t≈0 starvation under `loss_mode=v` is untested |
