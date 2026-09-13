# Why the tc1536 delta code has low ZPartRank

**Date:** 2026-09-13 · **Arms:** `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.02_ns3072_pool8192_compose1.0_decnoise0.8_detach_sw0`
(`BSC:45725881`) and its pool-24576 twin (`BSC:45727710`), both at ep 30 · **Diagnostics:** z spectrum `BSC:45822565`,
row/token attribution `BSC:45822767`, DA3 input rank `BSC:45825393`, SIGReg force probe (CPU, `occrae/sigreg.py` verbatim)
· **prior cycle:** `results/2026-09-13_tc_width_sigreg_pool_pair_tc1536_read_slides.html` · **Code:** `795e971`

## Answer

- **pool 8192, `ZPartRank` 1.0:** one register token is a massive activation. Delta slot `k=44` carries +2000 to +7000
  on channel 1109 (later 1105) in every sequence. Nothing normalises the raw residual that becomes `z` at tc1536, and
  SIGReg cannot see a 1.6% tail. Without that slot the same code has rank 244–278.
- **pool 24576, `ZPartRank` 246:** no register slot. Rank is set by reconstruction. The DA3 input has the dimensions,
  the encoder drops the low-variance tail, and forcing it back with `cov_weight` did not improve recon at tc512.

## Part 1 — the register slot (pool 8192)

From `BSC:45822767`, 10240 val rows (160 sequences × 64 slots):

| ckpt | tail rows (\|z\|>50) | slot | channel, value p50 | all-row rank | non-tail rank |
|---|---|---|---|---|---|
| ep 10 | 1.562% | 44 only | ch1109, +4394 | 1.0 | 244.4 |
| ep 20 | 1.562% | 44 only | ch1109, +2644 | 1.0 | 254.8 |
| ep 30 | 1.562% | 44 only | ch1105, +5930 | 1.0 | 278.0 |
| pool 24576, ep 30 | 0% | — | max \|z\| 5.6 | 246.1 | 246.1 |

- **Every sequence, one slot, always positive.** 160 of 160 sequences have exactly one tail token, and it is `k=44`.
  It holds 99.5–99.8% of the total variance. Its other channels are also large (RMS 57–155 vs 1.05 for bulk rows).
- **Born in encoder block 0.** The per-block trace shows the value already at ~4200 after block 0 and roughly constant
  through block 11. It is an attention-sink register, the same phenomenon as ViT "massive activations".
- **Why the norms do not stop it.** `DINOv3ViTLayer` is pre-norm: `h = h + ls1·Attn(LN1(h))`, `h = h + ls2·MLP(LN2(h))`.
  LN only sees the branch input. The residual `h` is never normalised, and `z = hidden[:, :, :K]` of the last block
  (`occrae/deltatok_trainer.py:359`). At `target_channels == hidden_size` with `z_norm=false`, no bottleneck is built, so
  nothing sits between that residual and SIGReg. Reference copy: `third_party/hf_dinov3_vit_ref/modeling_dinov3_vit.py:406`.
- **Why SIGReg does not pull it back.** A tail of fraction `f` costs about `k·f²·1.2597` (k = 1–2 by sign structure).
  At `f = 1.56%` that is 3e-4 to 6e-4. The restoring force on a tail row peaks at amplitude ~100, turns negative by ~1000
  and is pure noise at 1e4. The live tail sits at 2000–7000, past the point where the gradient points home. Pool size
  changes the floor, not the mean force.
- **Recovery in siblings was luck of the blow-up.** Siblings recovered only when most rows blew up early (SIGReg stat
  ≥ 0.2), which gives the estimator a whole-batch signal. The pool-24576 twin did so at ep 6. The pool-8192 twin kept
  one slot out and stayed at 1.0.

**Fix landed in `795e971`:** `model.deltatok.force_bottleneck` now defaults to `true`, putting `LayerNorm → Linear` on `z`
even at tc1536. The seven raw-z tc1536 slurm arms pin it to `false` so their checkpoints still resume. The flow config
key defaults to `false` and must match the frozen tokenizer.

## Part 2 — why the healthy code is still ~250/1536

### The code

pool 24576, ep 30, `BSC:45822565`:

| ZPartRank | n50 / n90 / n99 | eig[500] | eig[1000] | top-290 share |
|---|---|---|---|---|
| 246.1 | 98 / 297 / 435 | 0.04 | 0 | 0.893 |

The code lives in a ~450-dim subspace. The same plateau shows at every width: tc512 ≈ 100, tc768 ≈ 290, tc1536 250–313.

### SIGReg cannot see it at this width

A random unit slice of a 1536-d code has variance ≈ tr(Σ)/1536 whichever directions are dead. Measured slice variance is
1.23 ± 0.10, so every slice looks like the same Gaussian.

| SIGReg statistic (Epps–Pulley, 3072 slices) | value |
|---|---|
| actual code | 0.0043 |
| Gaussian with the same covariance | 0.0060 |
| isotropic, full rank, slice scale 1.23 | 0.0053 |
| unit scale, rank 246 spread only | 0.0008 |
| N(0, I) floor | 0.00016 |

Most of the statistic is the 1.23 slice scale. The part due to low rank is ~0.0006 above the floor, so at weight 0.02
almost none of the gradient asks for more dimensions.

### The input has the dimensions

`BSC:45825393`: participation ratio of DA3 layer-12 patch tokens `f` and of `f[t+g] − f[t]` over 40 nt10 train batches.

| rows | ZPartRank /1536 | n50 / n90 / n99 | trace |
|---|---|---|---|
| raw f | 98 | 53 / 521 / 1209 | 587 |
| gap 1 | 184 | 89 / 616 / 1263 | 563 |
| gap 3 | 166 | 81 / 597 / 1253 | 662 |
| gap 5 | 159 | 78 / 591 / 1249 | 700 |
| gap 9 | 153 | 76 / 579 / 1239 | 745 |
| gaps 1–9 pooled | 167 | 82 / 599 / 1254 | 657 |
| **z (pool 24576)** | **246** | **98 / 297 / 435** | 1886 |

- **Gap barely changes the shape.** Gap 1 to 9 moves the ratio 184 → 153 while the trace grows 563 → 745. Longer gaps
  add variance along the same directions.
- **The input spectrum has a long tail; z cuts it.** The input needs ~650 more dims to go from 90% to 99% of variance.
  z needs ~140 and is exactly zero past ~500. Each of those tail directions holds ~1e-4 of the input variance, so recon
  pays almost nothing to drop them, and weight decay removes them.
- The participation ratio of z is higher than the input's because z is flatter at the head, not because it spans more.

### Forcing the rank up does not buy recon

`training.cov_weight` adds `‖E[zzᵀ] − I‖²_F / Cz` on the SIGReg pool, which sees the spectrum directly. At tc512,
`sigreg 0.02` (`plan/2026-09-02_sigreg_cov_penalty.md`):

| arm | job | eval ZPartRank /512 |
|---|---|---|
| cov 0 | `BSC:45296347` | ~85–101 |
| cov 3e-5 | `BSC:45416718` | 105 |
| cov 1e-4 | `BSC:45498520` | 95 |
| cov 3e-4 | `BSC:45498521` | 125 (train 163 at ep 29) |

Rank rose 1.9× with no ceiling. Eval `LossRecon` moved a few percent out of dose order, and only 3e-5 beat the control.

## Options if rank is the goal

- **Tokenizer side:** one tc1536 arm with `force_bottleneck=true` and `cov_weight`. The tc512 doses do not transfer,
  because the Frobenius term grows with Cz. Expect rank up and recon flat.
- **Flow side (preferred):** keep the code and let the flow work in its used subspace. Top-k PCA of Σ_z, or
  `whiten_stats` from `compute_deltatok_latent_stats.py`. The Σ_z-shaped noise probe
  (`plan/2026-09-13_tc_width_zshaped_ladder_err_spectrum.md`) already measures this.

## Caveats and not done

- **Input rank covers one shape.** All 40 sampled batches were single-camera at 168×518 (444 patches). Multi-camera
  and taller resolutions are not in the table.
- **Pooled gaps are not uniform.** The pooled row weights gap `g` by `T − g` windows, so gap 1 has 9× the rows of gap 9.
  The per-gap spectra are close enough that this does not change the conclusion.
- **Row rank is not a strict bound.** z summarises 2 × 444 patch tokens into 64 slots, so token-level input rank does
  not bound z's rank in either direction.
- **Not measured:** how much of z's 246 is between-slot mean (at most 63 dims from 64 learned queries) vs within-slot
  variance. No tc1536 `cov_weight` calibration.
- **Pre-existing bug, not fixed:** `compute_deltatok_z_spread.py:160` passes `per_dataset_sampling` to
  `get_data_loader`, which rejects it, so its `--split train` path crashes.
- **Artifacts on BSC `$SCRATCH`, not cleaned:** `tmp_zdump_out/zdump_{pool8192_ep10,ep20,ep30,pool24576_ep30}.pt`,
  `tmp_featrank_spectra.pt`, the probe scripts and their `tmp_*_<jobid>.out` logs.
