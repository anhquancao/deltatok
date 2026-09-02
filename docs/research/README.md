# DeltaTok research

Three folders, one per stage. A doc's **thread** — the question it belongs to — is the tag in its filename, and
the ledger for each thread is a section below.

| Folder | Holds | Name |
|---|---|---|
| [`plan/`](plan/) | The hypothesis, why it is worth GPU hours, and how it is run: the patch, the arm scripts, the `sbatch` line, the job ids | `<date>_<thread>_<slug>.<ext>` |
| [`results/`](results/) | Matched-epoch numbers and plots against the control. Decks end `_slides.html` | `<date>_<thread>_<slug>.<ext>` |
| [`analysis/`](analysis/) | What it means: a mechanism, a diagnosis, or the verdict on a plan | `<date>_<thread>_<slug>.<ext>` |

[`TEMPLATE.md`](TEMPLATE.md) is the plan template. Copy it; never start from a blank file.

## Open questions (2026-09-02)

| Thread | Open |
|---|---|
| [`tc_width`](#tc_width--channel-and-token-budget-of-the-delta-token) | Where `sigreg_weight` turns over at tc512 and whether the optimum scales with Cz. Arms BSC:45296347–49 pending, read at ep 67. |
| [`sigreg`](#sigreg--making-the-delta-code-spread) | Does a direct `‖E[zzᵀ]−I‖²_F` penalty break the ~90/512 rank ceiling? Plan only, no code, not submitted. The `weight ∝ Cz` question is tracked in `tc_width`. |
| [`compose`](#compose--additive-composition-of-delta-tokens) | Composed ≠ autoregressive. Whether a short-schedule plain arm buys composability for free. |
| [`pair_sampling`](#pair_sampling--which-frame-pairs-and-gaps-the-tokenizer-trains-on) | None. |
| [`flow`](#flow--flow-matching-over-frozen-delta-tokens) | t≈0 starvation under `loss_mode=v`, and the compose-arm diagnostic in the 2026-09-01 numsteps analysis. |
| [`pointdit`](#pointdit--which-of-pointdits-recipe-choices-transfer-to-our-flow-arm) | Three arms planned, none submitted. The zeros ODE init is dropped. |

## Cross-thread

| Date | File | Stage | Holds |
|---|---|---|---|
| 2026-08-18 | [backlog](plan/2026-08-18_cross_backlog.html) | plan | The previous work queue, superseded by [`../todo/02-09-2026.md`](../todo/02-09-2026.md) |
| 2026-09-02 | [4week_review_slides](results/2026-09-02_cross_4week_review_slides.html) | results | Four-week review deck across every thread |

## tc_width — channel and token budget of the delta token

How many channels (`target_channels`, Cz) and tokens (`num_delta_tokens`, K) the delta code needs, and what sets its
usable rank. Rank turned out to be set by SIGReg weight, not width, so this thread and `sigreg` share their open
question.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-07-03 | [num_delta_tokens_sweep](plan/2026-07-03_tc_width_num_delta_tokens_sweep.html) | plan | Make K configurable and sweep it | K=32 best of the sweep; cutting K beats squeezing channels |
| 2026-07-07 | [num_delta_tokens_ablation](results/2026-07-07_tc_width_num_delta_tokens_ablation.html) | results | Eval recon vs K | Companion tables; per-step CSV `results/2026-07-07_tc_width_pointmap_predvsorig_losses.csv` |
| 2026-07-24 | [bottleneck_diagnosis](analysis/2026-07-24_tc_width_bottleneck_diagnosis.html) | analysis | Why the 768-channel bottleneck fails | No good linear subspace found, and the code collapses; motivates SIGReg and stage-2 |
| 2026-07-29 | [channel_compression_slides](results/2026-07-29_tc_width_channel_compression_slides.html) | results | Squeeze 1536→768 seven ways vs the dtok32 control | Channel squeeze converges ~7× worse than uncompressed, ~4× worse than dtok32 at equal floats |
| 2026-08-05 | [stage2_bottleneck_spread_slides](results/2026-08-05_tc_width_stage2_bottleneck_spread_slides.html) | results | SIGReg at 1536, then freeze and bottleneck to 768 | Fixed stage-2 keeps spread (ZPartRank 364/768 KITTI); the ~150 read was the broken twin |
| 2026-08-16 | [tc_sweep_sigreg_slides](results/2026-08-16_tc_width_tc_sweep_sigreg_slides.html) | results | tc 128…1536 at a fixed recipe | tc128 best at matched ep 40; usable rank stalls at 30–96 dims whatever Cz |
| 2026-08-25 | [tc_compose_losses_slides](results/2026-08-25_tc_width_tc_compose_losses_slides.html) | results | Same sweep at compose 1.0, matched 40 h | tc128 ≈ tc256 lead on eval recon |
| 2026-08-26 | [compose_convergence_sigreg_tc](plan/2026-08-26_tc_width_compose_convergence_sigreg_tc.md) | plan | Q1 plain reaches the compose floor? Q2 wide-tc regression a SIGReg tax? Q3 raise sigreg at tc512? Q4 SIGReg on the composed sum? | Q1 no. Q2 no, 95% of the penalty survives. Q3 yes: −29% recon, rank 2.6×. Q4 half the gain of doubling the weight. Q5 → 2026-09-01 |
| 2026-08-27 | [tc_plain_vs_compose_slides](results/2026-08-27_tc_width_tc_plain_vs_compose_slides.html) | results | Width under plain vs compose | Plain: every doubling helps, tc512 plain best. Compose: narrowest best |
| 2026-08-27 | [tc_sigreg_ab_slides](results/2026-08-27_tc_width_tc_sigreg_ab_slides.html) | results | Deck for Q2 | Halving the weight makes recon worse at tc256 and tc512 |
| 2026-09-01 | [sigreg_weight_tc512](plan/2026-09-01_tc_width_sigreg_weight_tc512.md) | plan, **open** | Where `sigreg_weight` turns over at tc512; is it `∝ Cz`? | Arms 0.02 / 0.04 / 0.08 = BSC:45296347–49, read at ep 67 |
| 2026-09-01 | [tc512_sigreg_weight_slides](results/2026-09-01_tc_width_tc512_sigreg_weight_slides.html) | results | Deck for Q3/Q4 | 0.005 vs 0.01 vs sigregsum at ep 67 |

## sigreg — making the delta code spread

The SIGReg regulariser on z: estimator, pool, weight, and the geometry it needs. Its weight is what sets usable rank
(see `tc_width`), so the live weight question is tracked there.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-07-25 | [sigreg_debug](analysis/2026-07-25_sigreg_debug.md) | analysis | Why the SIGReg arm failed to learn | Estimator fix `e95de40`, validated on a 2-epoch dev run |
| 2026-07-26 | [layernorm_geometry](analysis/2026-07-26_sigreg_layernorm_geometry.html) | analysis | Does SIGReg work after the fix? | Yes, but the z LayerNorm is the wrong geometry and the weight is 20× too high → nozn arms |
| 2026-07-28 | [param_search_plan](plan/2026-07-28_sigreg_param_search.md) | plan | Which SIGReg knobs to search | Plan only, never submitted as written |
| 2026-07-28 | [z_spread_analysis](analysis/2026-07-28_sigreg_z_spread.html) | analysis | How much of the 768-dim budget SIGReg buys | ~10×: participation ratio 173 vs 17.6 at matched ep 40 |
| 2026-07-29 | [bsc_migration_rolling_pool](plan/2026-07-29_sigreg_bsc_migration_rolling_pool.html) | plan | SIGReg as a rolling sample pool; BSC on ehpc1001 | Landed |
| 2026-07-31 | [pool_not_weight_neutral](analysis/2026-07-31_sigreg_pool_not_weight_neutral.html) | analysis | Is pool size weight-neutral? | No: pool 32768 at the same weight kills training, spike at warmup end |
| 2026-08-24 | [gapsig_vs_plain_slides](results/2026-08-24_sigreg_gapsig_vs_plain_slides.html) | results | Does Brownian gap-scaling (`sigreg_gap_sigma`) help? | No: 6.5–13.9% worse on all geometry losses, both eval sets |
| 2026-08-27 | [sigreg_compose_z_plan](plan/2026-08-27_sigreg_compose_z.md) | plan | SIGReg on all three compose streams | Code `1140de1`; half the gain of doubling the weight (tc_width Q4) |
| 2026-09-02 | [sum_at_weight_0.02](plan/2026-09-02_sigreg_sum_at_weight_0.02.md) | plan, **open** | Does the sum win survive at weight 0.02, or was it buying effective weight? | Arm BSC:45345063 vs plain twin BSC:45296347, read at ep 67 |
| 2026-09-02 | [cov_penalty](plan/2026-09-02_sigreg_cov_penalty.md) | plan, **open** | Does a direct `‖E[zzᵀ]−I‖²_F/Cz` penalty break the ~90/512 rank ceiling the weight axis cannot? | Not submitted; one arm at `cov_weight=3e-5` on top of sigreg 0.02, twin BSC:45296347 |

**Open.** Whether `sigreg_compose_z` still beats its plain twin at weight 0.02 (`2026-09-02`), and whether a
direct covariance penalty moves the rank ceiling the weight axis has turned over on (`2026-09-02`, TODO 6). The live
`sigreg_weight ∝ Cz` question is tracked in [`tc_width`](#tc_width--channel-and-token-budget-of-the-delta-token).

## compose — additive composition of delta tokens

Train `d13 + d35 ≈ d15` so a span is composed in latent space instead of chained through the decoder. The Q1 cycle
(plain vs compose at equal budget) lives in `plan/2026-08-26_tc_width_compose_convergence_sigreg_tc.md`.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-08-12 | [additive_composition_plan](plan/2026-08-12_compose_additive_composition.md) | plan | Compose a span as plain addition of hop deltas | Landed 2026-08-13 as the compose arm |
| 2026-08-18 | [token_composition_impact](results/2026-08-18_compose_token_composition_impact.html) | results | What composition does to the code | Deck |
| 2026-08-25 | [compose_vs_plain_slides](results/2026-08-25_compose_vs_plain_slides.html) | results | Compose vs plain at equal budget, ep 99 | Compose loses single-step recon +3% but wins LossRecon_Comp 3.4×; plain regresses on composability after ep 11 |

**Open.** Composed ≠ autoregressive: compose is worse on LossRecon_AR while owning LossRecon_Comp. A short-schedule
plain tokenizer may buy most of the composability for free; untested.

## pair_sampling — which frame pairs and gaps the tokenizer trains on

Timestep selection, gap range (`max_gap`), and the 1ecb17c switch to consecutive windows.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-05-31 | [augmentation_parity_plan](plan/2026-05-31_pair_sampling_augmentation_parity.md) | plan | Random per-pair stride + horizontal flip, as upstream DeltaTok | `reverse_seq` arm ran 2026-08-04 |
| 2026-08-02 | [timestep_sampling_1ecb17c](analysis/2026-08-02_pair_sampling_timestep_sampling_1ecb17c.html) | analysis | What an item contains before and after 1ecb17c | Post-change eval is easier (all gaps one slot); eval loss is not comparable across the boundary |
| 2026-08-05 | [randint_vs_consec_convergence_slides](results/2026-08-05_pair_sampling_randint_vs_consec_convergence_slides.html) | results | Random-interval vs consecutive frames | Random-interval converges ~3× faster |
| 2026-08-10 | [maxgap_sweep_report](results/2026-08-10_pair_sampling_maxgap_sweep.html) | results | Loss, pointmap AR and z-spread vs `max_gap` | maxgap9 adopted for all later arms |

**Open.** None.

## flow — flow matching over frozen delta tokens

The world model: a DiT that denoises DeltaTok codes conditioned on context frames. Twenty-two docs, from the June
collapse through the September step-count analysis. The pointdit comparison spun out into its own thread on
2026-09-02.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-05-31 | [flow_matching_trainer_plan](plan/2026-05-31_flow_matching_trainer.md) | plan | Simplify the trainer for token flow matching | Landed as `occrae/deltatok_flow_trainer.py` |
| 2026-06-23 | [concat_cond_plan](plan/2026-06-23_flow_concat_cond.md) | plan | Pooled frame-0 context token instead of cross-attention | Plan |
| 2026-06-23 | [camera_swap](analysis/2026-06-23_flow_camera_swap.md) | analysis | Cameras swap in generated views | Per-camera embedding fix, unconfirmed on a run |
| 2026-06-25 | [mse_collapse](analysis/2026-06-25_flow_mse_collapse.md) | analysis | Tiny loss but mean-collapsed samples | Confounded, see next row |
| 2026-06-26 | [overfit_data_confound](analysis/2026-06-26_flow_overfit_data_confound.md) | analysis | Was the single-scene overfit fitting one sample? | No: `ray_map_prob=0.0` randomised ~9 view orders per batch; supersedes the collapse read |
| 2026-07-07 | [dtok32_plan](plan/2026-07-07_flow_dtok32.html) | plan | Native-K flow, run K=32 | Landed; K=1 squeeze removed |
| 2026-07-08 | [dtok32_smoketest](results/2026-07-08_flow_dtok32_smoketest.html) | results | Smoke test | Pose-estimation crash fixed; the dtok32 checkpoint is NaN |
| 2026-07-08 | [dtok16_deltactx_global](results/2026-07-08_flow_dtok16_deltactx_global.html) | results | Convergence on the dtok16 tokenizer | Teacher-forced loss vs sampled metrics over 40 epochs |
| 2026-07-17 | [latent_whitening_plan](plan/2026-07-17_flow_latent_whitening.html) | plan | RAE-style per-channel whitening | Landed behind a flag |
| 2026-07-18 | [whitenpos_failure](analysis/2026-07-18_flow_whitenpos.html) | analysis | Why the whitened run does not work | Whitening was never the bottleneck |
| 2026-07-19 | [wall](analysis/2026-07-19_flow_wall.html) | analysis | What the flow wall is | Decoder + conditional variance, not architecture or sampler; in-context fix proposed |
| 2026-07-22 | [waymo_dtok64_experiments_slides](results/2026-07-22_flow_waymo_dtok64_experiments_slides.html) | results | Eight Waymo dtok64 runs | One wall: MSEToken ≈ 0.29, LossDepth ≈ 5.8 |
| 2026-08-05 | [frontcam_vs_randcam_slides](results/2026-08-05_flow_frontcam_vs_randcam_slides.html) | results | Consecutive frames; front-cam vs rand-cam | Non-consecutive never generates; consecutive works; front-cam wins every cam-0 decode |
| 2026-08-10 | [tc1536_maxgap9_plan](plan/2026-08-10_flow_tc1536_maxgap9.md) | plan | Swap the frozen tokenizer to tc1536 maxgap9 nosigreg | Landed on BSC |
| 2026-08-13 | [whiten_vs_raw_nosigreg_slides](results/2026-08-13_flow_whiten_vs_raw_nosigreg_slides.html) | results | Whiten vs raw z on the nosigreg tokenizer | Deck |
| 2026-08-16 | [sigreg_vs_whiten_slides](results/2026-08-16_flow_sigreg_vs_whiten_slides.html) | results | Latent choices: SIGReg vs whiten vs raw | Deck |
| 2026-08-18 | [tc_sweep_slides](results/2026-08-18_flow_tc_sweep_slides.html) | results | Tokenizer width into the same XXL DiT | Depth improves with tc; pointmap and raymap do not, tc1536 worst |
| 2026-08-20 | [sampler_ablation](results/2026-08-20_flow_sampler_ablation.html) | results | Four sampler / t-schedule arms off the compose tc128 baseline | All flat or worse; logit-normal loses |
| 2026-08-24 | [alldata_xxl_eval](results/2026-08-24_flow_alldata_xxl_eval.html) | results | Multi-dataset training | Degrades geometric losses vs Waymo-only at matched epochs |
| 2026-08-25 | [vpred_vs_xpred_slides](results/2026-08-25_flow_vpred_vs_xpred_slides.html) | results | Does v-prediction beat x-prediction? | No: loses on every geometry metric and on LossFlow |
| 2026-09-01 | [numsteps_tc128compose_slides](results/2026-09-01_flow_numsteps_tc128compose_slides.html) | results | 1…20 ODE steps on one checkpoint | Error rises monotonically with steps |
| 2026-09-01 | [numsteps_why_more_steps_degrade](analysis/2026-09-01_flow_numsteps_why_more_steps_degrade.md) | analysis, **open** | Why more steps degrade | The sampler does not integrate: N steps returns x̂ at t = 1−1/N. t≈0 starvation under `loss_mode=v` is untested |
| 2026-09-02 | [sky_in_eval_loss](analysis/2026-09-02_flow_sky_in_eval_loss.md) | analysis | How is the sky handled in the flow eval loss? | Dropped implicitly: the geometry losses are masked by `valid_mask = depth>0 & depth<50`, and sky has no LiDAR return. The flow loss keeps it at full weight |

## pointdit — which of pointdit's recipe choices transfer to our flow arm

pointdit (`/mnt/d/code/pointdit`) is a single-image pointmap flow model with the same Euler update and the same
`pred_mode=x` + `loss_mode=v` algebra as our arm. Split out of `flow` on 2026-09-02: that thread owns *why more
ODE steps degrade*, this one owns *what pointdit does differently and which of it is worth porting*.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-09-02 | [pointdit_vs_deltatok](analysis/2026-09-02_pointdit_vs_deltatok.md) | analysis | What does pointdit do differently? | Same algebra; three mitigations missing, two worth porting. Code reading only, nothing measured on our data |
| 2026-09-02 | [lowt_recipe_finetune](plan/2026-09-02_pointdit_lowt_recipe_finetune.md) | plan, **open** | Mitigation 1 alone: does 10% forced `t=0` lift the 1-step readout? | Not submitted. Predicted near-flat — pinning 10% at weight 1 against a mean weight of 39.0 |
| 2026-09-02 | [xloss_additive](plan/2026-09-02_pointdit_xloss_additive.md) | plan, **open** | Mitigation 2 alone: does an additive unweighted x-MSE lift it? | Not submitted. Weight 1.3, not pointdit's 0.1 — 0.1 would buy 0.28% of the gradient mass here |
| 2026-09-02 | [lowt_schedule](plan/2026-09-02_pointdit_lowt_schedule.md) | plan, **open** | Does pointdit's low-`t` schedule beat uniform, trained from scratch? | Not submitted. `t = sigmoid(-0.8 + 0.8·ε)` + 10% at `t=0`, 100 ep from scratch (~65 h), read at matched ep 100. Merges the two 2026-09-02 `pointdit_schedule*` files |


**Open.** Three arms planned, none submitted. All read 1-step `MSEToken` against 0.7417 and `LossDepth` against
3.6563 (`results/2026-09-01_flow_numsteps_tc128compose_slides.html`). The zeros ODE init is **dropped** — an
eval-time sampler knob, not important for us; its plan file was never written.
