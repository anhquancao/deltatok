# Why the tc1536 forced-bottleneck arms stall at the copy plateau

**Date:** 2026-09-14 · **Arms:** `deltatok_tc1536_bneck_sigreg0.02_ns3072_pool24576_rawrecon_sw0` (`BSC:45831175`), its
whitened twin (`BSC:45831174`), `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.06_ns3072_pool8192_compose1.0_decnoise0.8_detach_bneck_sw0`
(`BSC:45678654`) · **Controls:** `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.02_ns3072_pool24576_compose1.0_decnoise0.8_detach_sw0`
(`BSC:45727710`), `BSC:45725881`, `BSC:45610897`, tc512 `BSC:45578805` · **Diagnostics:** AdamW-state and weight probe
`BSC:45855319` · **prior cycle:** `analysis/2026-09-13_tc_width_tc1536_rank.md` · **Code:** `795e971`, `d851761`

## Answer

- **The decoder stops reading z.** Noise on z changes nothing: σ = 0.82 costs +0.1% KITTI `LossRecon` against +32% in
  the no-bottleneck arm.
- **The encoder then gets almost no gradient.** Encoder ÷ decoder gradient RMS is 0.006, against 0.131 without the
  bottleneck and 0.109 for the tc512 bottleneck.
- **z stays content-poor.** Eval `ZPartRank` is 15–19 of 1536, close to the rank of the learned per-slot query (19.5).
- **The loop locks itself.** No reader means no gradient, which means no content, which means no reader.
- **The common factor is the forced bottleneck.** The noised decode, SIGReg weight, SIGReg pool and recon whitening
  all vary across the three stalled arms.
- **Why the no-bottleneck arms escape is untested.** Each leaves the plateau at ep 3–6 during a z-scale blow-up, and
  the LayerNorm caps that scale.

## 1 The two arms differ in two settings

| setting | `BSC:45727710` | `BSC:45831175` |
|---|---|---|
| `model.deltatok.force_bottleneck` | false: no norm, no projection | true: `LayerNorm(1536)` + square `Linear(1536,1536)` down and up |
| `model.deltatok.decode_noise_tau` | 0.8, detached second decode | 0.0 |

Both run SIGReg 0.02, 3072 slices, pool 24576, warmup 0, compose 1.0, lr 1e-3, clip 0.1, `max_gap` 9, bsize 2 /
effective 16, nt10, and the raw log-cosh recon (`recon_whiten_alpha` 0).

## 2 Every forced-bottleneck arm stalls

KITTI eval `LossRecon`. At init the decoder copies `x_prev` (`layer_scale_init` 1e-5), so ~0.13–0.14 is the copy level.

| arm | job | bottleneck | noised decode | SIGReg weight / pool / warmup | recon | ep 3 | ep 6 | ep 13 |
|---|---|---|---|---|---|---|---|---|
| base | `45727710` | none | 0.8 | 0.02 / 24576 / 0 | raw | 0.1219 | 0.0659 | 0.0389 |
| pool 8192 | `45725881` | none | 0.8 | 0.02 / 8192 / 0 | raw | 0.1033 | 0.0459 | 0.0333 |
| 0.06 control | `45610897` | none | 0.8 | 0.06 / 8192 / 2000 | raw | 0.1184 | 0.0697 | 0.0400 |
| tc512 | `45578805` | 1536 → 512 | 0.8 | 0.02 / 8192 / 2000 | raw | 0.1208 | 0.1033 | 0.0820 |
| forced + noised decode | `45678654` | square | 0.8 | 0.06 / 8192 / 0 | raw | 0.1339 | 0.1328 | 0.1210 |
| forced, raw recon | `45831175` | square | off | 0.02 / 24576 / 0 | raw | 0.1330 | 0.1311 | 0.1293 |
| forced, whitened | `45831174` | square | off | 0.02 / 24576 / 0 | α 0.5 | 0.1334 | 0.1333 | 0.1304 |

- **The noised decode is not the separator.** `45678654` has it and stalls. The plain tc1536 `BSC:44590128` lacks it
  and escapes at ep 4 (`results/2026-09-11_tc_width_bneck_sw0_tc1536_read_slides.html`).
- **The base vs raw-recon pair has no warmup confound.** Both use `sigreg_warmup` 0.

## 3 The decoder ignores z

Eval `LossRecon` at σ = 0 / 0.32 / 0.55 / 0.82 on z, ep 13. z sits near unit scale in every arm (`ZRowMeanSquare` ≈ 1).

| arm | KITTI | nuScenes | σ = 0.82 cost, KITTI / nuScenes |
|---|---|---|---|
| base `45727710` | 0.0389 / 0.0412 / 0.0452 / 0.0514 | 0.0274 / 0.0294 / 0.0328 / 0.0377 | +32% / +38% |
| raw recon `45831175` | 0.1293 / 0.1293 / 0.1293 / 0.1294 | 0.1003 / 0.1003 / 0.1003 / 0.1003 | +0.1% / 0.0% |
| whitened `45831174` | 0.1304 / 0.1304 / 0.1304 / 0.1304 | 0.1006 / 0.1006 / 0.1006 / 0.1007 | 0.0% / +0.1% |

`45678654` read the same at ep 22: +0.3% / +0.3%, against +55% / +59% for its control.

## 4 The encoder gets almost no gradient

`sqrt(exp_avg_sq)` from AdamW, mean over 2-D weights, at `epoch_10.pth` (iter 11250). Gradient clipping scales every
parameter equally, so compare ratios within a row, not raw values across rows. The optimizer state was mapped to names
by reproducing `_build_optimizer` (sorted Linear weights, then the sorted rest) and shape-checked.

| arm | encoder ÷ decoder | `z_proj_down` ÷ decoder | `z_proj_up` ÷ decoder | `pre_bottleneck_norm` γ ÷ decoder |
|---|---|---|---|---|
| base `45727710` | 0.131 | none | none | none |
| tc512 `45578805` | 0.109 | 1.92 | 3.12 | 12.5 |
| forced + noised decode `45678654` | 0.051 | 0.45 | 0.67 | 9.9 |
| forced, raw recon `45831175` | 0.006 | 0.28 | 0.14 | 13.3 |

- **The raw-recon encoder gets 22× less relative gradient** than the base arm's.
- **Its up-projection gets 22× less** than tc512's, the bottleneck arm that trains.
- **The LN gain is not the separator.** It gets 10–13× the decoder's gradient in every bottleneck arm, tc512 included.

## 5 Where the rank goes

Weights at `epoch_10.pth`. PR is the participation ratio of the squared singular values. Eval `ZPartRank` is KITTI at the
same checkpoint.

| arm | `z_proj_up` PR | `z_proj_up` s0 / s10 | `z_proj_up` bias norm | `z_proj_down` PR | `z_embed` PR | eval `ZPartRank` |
|---|---|---|---|---|---|---|
| tc512 `45578805` | 41.7 / 512 | 13.0 / 8.8 | 4.70 | 45.9 | 47.9 | 27.8 / 512 |
| forced + noised decode `45678654` | 22.2 / 1536 | 44.3 / 15.6 | 8.96 | 512.4 | 46.8 | 37.2 / 1536 |
| forced, raw recon `45831175` | 8.9 / 1536 | 58.4 / 12.2 | 9.33 | 165.7 | 19.5 | 17.6 / 1536 |
| base `45727710` | none | none | none | none | 27.3 | 197.6 / 1536 |

- **The up-projection collapses onto a few directions** with a large bias, most in the arm that stalls hardest.
- **The down-projection is not collapsed.** So z's low rank comes from the normalised encoder output.
- **That rank sits near the learned query's.** This suggests z is mostly the per-slot query `z_embed` with little frame
  content. It is inferred from weights, not measured on activations.
- **The LN does not zero channels.** Median |γ| is 0.33 in the raw-recon arm and under 0.1% of channels fall below 0.1.

## 6 Why the no-bottleneck arms escape (untested)

Every no-bottleneck tc1536 arm leaves the plateau at ep 3–6 while its z scale blows up. Eval `ZMeanAbsMax` reaches 516
at ep 1 in `45727710`, 6098 at ep 0 in `45610897`, and 76 at ep 4 in `45725881`. With the LN in front of the projection
and SIGReg on its output, the raw-recon arm's `ZMeanAbsMax` stays at 0.14–3.8 from step 0. That closes this route.
The link between blow-up and escape is a correlation.

tc512 escapes slowly despite the same LN. Its up-projection still gets 3.12× the decoder's gradient. What makes the
square case lock is not measured. The 2026-09-11 read's candidate is that a square projection applies no compression.

## Implications for a norm on z (untested)

- **Gate on gradient, not rank.** Read the encoder ÷ decoder gradient ratio and the σ ladder in the first 5 epochs.
  `ZPartRank` did not flag this stall.
- **Drop the square projections at tc1536.** They compress nothing, and the signal collapses there.
- **Warm-start the norm arm from `BSC:45727710`'s encoder and decoder.** The decoder then already reads z.

## Caveats and not done

- **No forward pass.** "z is mostly the learned query" and the blow-up link are inferred from weights and logs.
- **The whitened arm `45831174` was not probed** for gradients or weights, only for eval.
- **The Adam state is a ~20-step average** (β2 = 0.95) at one checkpoint.
- **Single seed.** The trajectory table is KITTI only.
- **Both forced arms are still `RUNNING`** at ep 14 of their 48 h walls. Neither was cancelled.
- **Probe artifacts are on BSC `$SCRATCH`, not cleaned:** `tmp_bneck_adam.py`, `tmp_bneck_adam.sh`,
  `tmp_bneck_adam_45855287.out` (failed in 2 s: `sbatch --wrap` runs `sh`), `tmp_bneck_adam_45855319.out`.
