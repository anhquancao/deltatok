# tc_width — channel and token budget of the delta token

How many channels (`target_channels`, Cz) and tokens (`num_delta_tokens`, K) the delta code needs, and what sets its
usable rank. Rank turned out to be set by SIGReg weight, not width, so this thread and `../sigreg/` share their open
question.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-07-03 | [num_delta_tokens_sweep](2026-07-03_task_num_delta_tokens_sweep.html) | results | Make K configurable and sweep it | K=32 best of the sweep; cutting K beats squeezing channels |
| 2026-07-07 | [num_delta_tokens_ablation](2026-07-07_report_num_delta_tokens_ablation.html) | results | Eval recon vs K | Companion tables; per-step CSV `2026-07-07_report_pointmap_predvsorig_losses.csv` |
| 2026-07-24 | [bottleneck_diagnosis](2026-07-24_analysis_bottleneck_diagnosis.html) | analysis | Why the 768-channel bottleneck fails | No good linear subspace found, and the code collapses; motivates SIGReg and stage-2 |
| 2026-07-29 | [channel_compression_slides](2026-07-29_report_channel_compression_slides.html) | results | Squeeze 1536→768 seven ways vs the dtok32 control | Channel squeeze converges ~7× worse than uncompressed, ~4× worse than dtok32 at equal floats |
| 2026-08-05 | [stage2_bottleneck_spread_slides](2026-08-05_report_stage2_bottleneck_spread_slides.html) | results | SIGReg at 1536, then freeze and bottleneck to 768 | Fixed stage-2 keeps spread (ZPartRank 364/768 KITTI); the ~150 read was the broken twin |
| 2026-08-16 | [tc_sweep_sigreg_slides](2026-08-16_report_tc_sweep_sigreg_slides.html) | results | tc 128…1536 at a fixed recipe | tc128 best at matched ep 40; usable rank stalls at 30–96 dims whatever Cz |
| 2026-08-25 | [tc_compose_losses_slides](2026-08-25_report_tc_compose_losses_slides.html) | results | Same sweep at compose 1.0, matched 40 h | tc128 ≈ tc256 lead on eval recon |
| 2026-08-26 | [compose_convergence_sigreg_tc](2026-08-26_task_compose_convergence_sigreg_tc.md) | cycle ×5 | Q1 plain reaches the compose floor? Q2 wide-tc regression a SIGReg tax? Q3 raise sigreg at tc512? Q4 SIGReg on the composed sum? | Q1 no. Q2 no, 95% of the penalty survives. Q3 yes: −29% recon, rank 2.6×. Q4 half the gain of doubling the weight. Q5 → 2026-09-01 |
| 2026-08-27 | [tc_plain_vs_compose_slides](2026-08-27_report_tc_plain_vs_compose_slides.html) | results | Width under plain vs compose | Plain: every doubling helps, tc512 plain best. Compose: narrowest best |
| 2026-08-27 | [tc_sigreg_ab_slides](2026-08-27_report_tc_sigreg_ab_slides.html) | results | Deck for Q2 | Halving the weight makes recon worse at tc256 and tc512 |
| 2026-09-01 | [sigreg_weight_tc512](2026-09-01_task_sigreg_weight_tc512.md) | cycle, **open** | Where `sigreg_weight` turns over at tc512; is it `∝ Cz`? | Arms 0.02 / 0.04 / 0.08 = BSC:45296347–49, read at ep 67 |
| 2026-09-01 | [tc512_sigreg_weight_slides](2026-09-01_report_tc512_sigreg_weight_slides.html) | results | Deck for Q3/Q4 | 0.005 vs 0.01 vs sigregsum at ep 67 |
