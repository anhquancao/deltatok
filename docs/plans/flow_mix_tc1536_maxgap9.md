# Plan: Waymo Flow with tc1536 maxgap9 Tokenizer

Swap the frozen tokenizer in the Waymo flow arm from tc768 stage-2 bottleneck to the
maxgap9 tc1536 nosigreg arm, keeping front-cam-only and the rest of the recipe fixed.
Runs on **BSC**, **Waymo-only**, **raw z (no whitening)** — a deliberate test: if the
mis-scaled latent breaks flow training, we want to see it break.

**Baseline script:** `slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_sigreg_bsc.slurm`
**Baseline config:** `configs/deltatok_flow/train_deltatok_flow_waymo_bsc.yaml`
**New script:** `slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_bsc.slurm`
**New config:** `configs/deltatok_flow/train_deltatok_flow_waymo_tc1536mg9_bsc.yaml`
**Tokenizer source:** `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_nosigreg` (BSC `$SCRATCH`)
**Pinned ckpt:** `ckpts/epoch_49.pth` (copied from `current.pth`, global_epoch 49 = last)

## 1. Tokenizer state

The maxgap9 nosigreg arm stopped at **epoch 48 of 100** (cosine 1e-3 → 1e-5, mid-schedule).
Eval recon 0.0128 — best of the maxgap sweep; the tc768 stage-2 bottleneck is 0.0283.

A SIGReg-weight sweep on the same maxgap9 backbone is queued (BSC 44465439–41, PENDING).
If it lands in time, the sigreg variant is the cleaner swap — its pinned scale lets the
flow run with raw z and no whitening, exactly like the current recipe.

**Decision gate:** wait for sigreg sweep, or proceed with nosigreg + whitening now.

### z-spread stats (maxgap9 nosigreg, ep48)

| Metric | KITTI | nuScenes | Concern |
|---|---|---|---|
| ZPartRank | 66.5 | 68.5 | 67/1536 dims active — thin manifold |
| ZTotalVar | 66 579 | 71 309 | — |
| ZRowMeanSquare | 189 | 195 | per-element RMS ~14 |
| ZMeanAbsMax | 408 | 410 | worst channel offset |

With raw z, the flow noising `z_t = t·x + (1−t)·ε` has ε ~ N(0,1) negligible against
signal RMS ~14 for almost all t. The effective schedule collapses near t = 0, v-targets
are 14–400× unit scale, and MSE is dominated by a few giant-offset channels.

## 2. Whitening (required for nosigreg)

If proceeding without sigreg, **per-channel whitening is mandatory**.

### Steps

1. **Pin an immutable checkpoint.** Copy `current.pth` to `epoch_48.pth` (or use whichever
   `epoch_N.pth` exists) so the tokenizer arm can resume without affecting the flow.
   The ckpt lives on BSC `$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_nosigreg/ckpts/`.
   For JZ, the user must sync it manually.

2. **Regenerate whitening stats** from the pinned ckpt via
   `STATS_MODE=channel slurm/compute_deltatok_latent_stats_jz.slurm`. **Do not** reuse the
   existing `uniformt_cross` stats file — it is from a different ckpt. At Cz = 1536 it now
   **passes** the shape assert (unlike the tc768 arm where it was rejected), so a stale file
   would silently apply the wrong moments. Check the ckpt-provenance line in stdout.

3. **Inspect the stats.** With rank 67/1536, many channels are near-dead. Per-channel
   whitening amplifies them into noise the flow must model (the `_WHITEN_STD_EPS=1e-3`
   floor caps amplification but doesn't remove the problem). Watch the `[GATE]` ratio and
   `n_floored` count. A std-ratio in the hundreds with a large floored-channel count pushes
   strongly toward waiting for the sigreg arm.

4. **Set `model.whiten_stats`** in the SLURM script to the regenerated stats path.

## 3. Architecture — keep xxlarge_d20

The flow ViT stays `xxlarge_d20` (width 1536, 20 depth, ~1.2B params).

- Width = 1× latent Cz (was 2× at tc768). This is the VGGT-World regime — validated.
  The repo's own failure analysis (`deltatok_flow_whitenpos_failure_analysis.html`) showed
  width < Cz is the root cause of "loss falls, samples stay at mean+noise".
  Width = Cz satisfies the rule with zero margin.
- Changing architecture alongside the tokenizer makes results unattributable. The in/out
  projection delta (768 → 1536) is ~2.4M on 1.2B — negligible.
- **Failure signature to watch:** teacher-forced LossFlow falling while MSEToken is
  flat/degrading from epoch 0 = width starvation. Escalation: `2b/giant` (2048w × 32d,
  1.33× Cz) or an RAE-style wide DDT head near the output.

### Latent dimensionality note

Per-sample scalars double to ~393k (4 deltas × 64 tokens × 1536 channels at T=5, N=1).
The current recipe has no t-distribution shift for high dims (uniform t, linear sampler).
Don't change preemptively — but if samples look under-noised/over-committed, a
logit-normal shift or SD3-style `t' = s·t / (1 + (s−1)t)` on both train and sampler
is the first knob.

## 4. Datasets — canonical 5-mix, front-cam-only

Use the existing `train_deltatok_flow_mix_jeanzay.yaml` dataset block verbatim:
Waymo 7200 + VKitti 3200 + DDAD 7200 + Pandaset 7200 + ONCE 7200 = 32k items/epoch =
1000 steps/epoch at eff 32.

- **In-distribution:** the maxgap9 tokenizer trained on exactly these 5 datasets
  (`configs/deltatok/train_deltatok_nt10_bsc.yaml`).
- **Front-cam-only (`fixed_cams=[0]`)** is the repo's mixed-dataset convention.
  The front-cam vs rand-cam experiment showed front-cam wins on all decode metrics
  (depth 4.40 vs 4.76, pointmap 15.96 vs 17.13). The tokenizer's `vpt1to2` means
  N=1 encoding is in-distribution.
- **Frame rates are harmonized by pkl choice:** 10 Hz datasets use `sub5` (0.5 s slots),
  nuScenes 2 Hz keyframes use `sub1` = 0.5 s. A 5-frame window = 2 s everywhere.
- **Val:** the same Waymo 32-scene val as both existing arms — numbers are comparable.
- **nuScenes/KITTI stay held out** as zero-shot eval domains.

### Pre-launch check

Eyeball one decoded frame per dataset from the frozen tokenizer to confirm all five
cam-0 streams are forward-facing. If one isn't, it's a silent domain skew.

## 5. Recipe — keep everything else frozen

The tokenizer is the only variable. Copy these from the mix arm verbatim:

| Param | Value | Note |
|---|---|---|
| `cond_mode` | `delta_ctx_cross` | clean context + frame-0 cross-attn |
| `num_ctx_deltas` | 2 | 3 given, 2 forecast (mix arm convention) |
| `attn_mode` | `global` | one attention over all delta tokens |
| `vit_use_camera_embed` | true | degenerates to constant bias at N=1 |
| `loss_mode` | `v` | v-MSE |
| `t_dist` | `uniform` | uniform t |
| `t_shared` | true | one t per sample, all slots |
| `sampler_scheduler_mode` | `linear` | uniform-in-t grid |
| `sampler_alpha` | 0 | no per-frame stagger |
| `lr_schedule` | `constant` | 1e-4 after warmup |
| `weight_decay` | 0.05 | default |
| `epoch` | 100 | 100k steps, chained across SLURM walls |
| `eval_num_steps` | 20 | no residual noise |

## 6. Tokenizer flags — mirror exactly

Parameterless flags silently yield the wrong z if mismatched. `strict=True` cannot catch them.

```
model.deltatok.num_delta_tokens=64
model.deltatok.target_channels=1536        # no bottleneck (== hidden_size → no proj built)
model.deltatok.z_norm=false
model.deltatok.norm_affine=false
model.deltatok.use_camera_rope=false
model.deltatok.bottleneck_mlp=false        # inert at tc1536
```

**Changed from the tc768 mix arm:**
- `target_channels`: 768 → 1536
- `deltatok_ckpt`: stage-2 bottleneck → maxgap9 nosigreg pinned ckpt
- `whiten_stats`: null → regenerated stats path (if nosigreg; null if sigreg)

## 7. New files to create

### Config: `configs/deltatok_flow/train_deltatok_flow_mix_tc1536mg9_jeanzay.yaml`

Overlay on `train_deltatok_flow_jeanzay.yaml`. Identical dataset block to
`train_deltatok_flow_mix_jeanzay.yaml` — copy verbatim (same 5-mix, same val).

### SLURM: `slurm/deltatok_flow/train_deltatok_flow_mix_xxl_tc1536mg9_jz.slurm`

Copy `train_deltatok_flow_mix_xxl_sigreg_jz.slurm` and apply:
- `CONFIG_NAME=train_deltatok_flow_mix_tc1536mg9_jeanzay`
- `RUN_NAME=deltatok_flow_mix5mono_ctx3fwd2_tc1536mg9nosigreg_xxl_dit`
- `model.deltatok_ckpt=<pinned maxgap9 ckpt path on JZ>`
- `model.deltatok.target_channels=1536`
- Remove `model.deltatok.bottleneck_mlp=false` comment about "as this arm trained"
- Add `model.deltatok.bottleneck_mlp=false` explicitly
- Add `model.whiten_stats=<path>` if nosigreg (omit if sigreg)
- Update header comment to describe the tokenizer swap
- Update `--output`/`--error` filenames

## 8. Evaluation plan

Token-space metrics (LossFlow, MSEToken) are meaningless across tokenizers — different
z-spaces. Compare in **decode space**:

- LossDepth, LossPointmap, LossRaymap — absolute values on the shared Waymo val
- GT-z teacher ceiling (`*_tok` keys) — record the new tokenizer's ceiling early.
  The 0.0128 vs 0.0283 recon gap is the whole reason for this swap; ceiling-normalized
  decode loss is the honest readout of whether the flow closes it.

## 9. Sequencing (Fable recommendation)

Cleanest ladder:

1. Let the current tc768 mix arm (`deltatok_flow_mix5mono_ctx3fwd2_tc768s2bneck_xxl_dit`)
   establish the mix baseline.
2. **If sigreg sweep lands:** run mix + maxgap9-sigreg as the pure tokenizer swap (no
   whitening confound).
3. **If forced by timing:** run mix + maxgap9-nosigreg + whitening now, accepting the
   normalization confound. Check `[GATE]` output first.

## 10. Pre-launch checklist

- [ ] Decide: wait for sigreg sweep or proceed with nosigreg + whitening
- [ ] Pin immutable tokenizer checkpoint (copy `current.pth` → `epoch_48.pth`)
- [ ] If nosigreg: regenerate whitening stats, inspect `[GATE]` / `n_floored`
- [ ] Sync pinned ckpt (and stats if applicable) from BSC to JZ
- [ ] Create config yaml (copy mix config, rename)
- [ ] Create slurm script (copy mix slurm, apply changes from §7)
- [ ] Sync new files to JZ
- [ ] Verify cluster files match local (`grep` key fields)
- [ ] `sbatch` and watch until first iteration
