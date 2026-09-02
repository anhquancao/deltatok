# Plan: OccRAE Image Decoder Training

## Context

GLD trains an MAE decoder to reconstruct RGB from frozen DA3 multi-level features. We want to adapt this to use OccRAE (layer-18 token caching) as the frozen encoder, extracting 3-level features from OccRAE's decode path. For DA3-Giant-1.1 (the 1B variant), `out_layers=[19, 27, 33, 39]` — all 4 levels are >= layer 18, but we use only the first 3 (`[19, 27, 33]`) to reduce decoder size and keep the deepest features closer to the cached layer-18 representation.

**Goal:** Train an `OccRAEImageDecoder` (MAE-based) to reconstruct RGB images from frozen OccRAE 3-level features (layers 19, 27, 33), following GLD's training recipe (L1 + LPIPS + GAN losses).

**Precondition:** DA3-Giant-1.1 only. DA3-Large has `out_layers=[11,15,19,23]` — only 2 of 4 levels are >= 19, so multi-level feature extraction from decode is not viable.

---

## GLD Reference Training Settings

For comparison, GLD's original image decoder training configuration:

| Parameter | GLD Value |
|-----------|-----------|
| GPUs | 4 |
| Batch size (per GPU) | 1 |
| Views per sample | 4 |
| Epochs | 100 |
| Image resolution | 504x504 (multi-res: 504x504, 504x378, 504x336, 504x280) |
| Precision | bf16 |
| Encoder | RAE_DA3 (DA3-Base, frozen) |
| Decoder hidden_size | 6144 (4 x 1536, 4 DA3-Base levels) |
| Decoder config | ViTXL, patch_size=14 |
| Dataset | 50k scenes (25k DL3DV + 25k RE10K) |
| Optimizer | AdamW (lr=2e-4 → 2e-5, betas=0.9/0.95, no weight decay) |
| LR schedule | Cosine, warmup 1 epoch, decay until epoch 50 |
| EMA decay | 0.9978 |
| Checkpoint interval | 5,000 steps |
| Steps per epoch | ~12.5k (50k scenes / 4 GPUs) |
| Total steps | ~1.25M |

**Loss schedule:**
- L1 (recon_weight=1.0) + LPIPS (perceptual_weight=1.0) from epoch 0
- Discriminator updates start epoch 4
- GAN loss (disc_weight=0.75) starts epoch 6
- Disc LR decays separately (cosine, decay end epoch 16)
- Discriminator: DINO ViT-S/8, hinge loss, DiffAug (prob=1.0)

---

## Architecture Overview

```
Images (ImageNet-normed)
  |
  v
OccRAE.encode() [frozen, layers 0-18]
  -> layer-18 tokens: (B, S, N, 1536)
  |
  v
OccRAE backbone tail [frozen, layers 19-33]
  -> 3-level features at out_layers [19, 27, 33]
  -> each: (B*V, N+1, feature_dim)  where feature_dim = 3072 (1536*2 with cat_token)
  |
  v
Strip CLS, concatenate 3 levels -> (B*V, N, 9216)
  |
  v
OccRAEImageDecoder [TRAINABLE]
  -> predict 14x14 RGB patches -> unpatchify -> (B*V, 3, H, W) in ImageNet space
  |
  v
Denormalize to [0,1] for loss computation
  |
  v
L1 + LPIPS + GAN(DINO discriminator) losses
```

**Key dimension:** DA3-Giant with cat_token=True gives 3072 per level, so decoder `hidden_size = 3 * 3072 = 9216`.

---

## Files to Create / Modify

### 1. Modify: `occany/model/occ_rae.py`

Add a `decode_to_features()` method that:
- Takes cached layer-18 tokens from `encode()`
- Runs the backbone from layer 19 using `backbone.forward_from_layer()`
- Collects features at the first 3 `out_layers` [19, 27, 33]
- Returns a list of 3 feature tensors (CLS already stripped by `forward_from_layer`)

Key reference: `inference_batch_from_layer()` at `model_da3.py:828` already calls `backbone.forward_from_layer()`. The new method is similar but returns intermediate features instead of passing them to the DPT head.

**Important:** `forward_from_layer()` already strips CLS+register tokens (see `vision_transformer.py:593`). The return format is `tuple(zip(outputs, camera_tokens)), aux_outputs` where each `output_i` is `(B*V, N_patches, 3072)`. No manual CLS stripping needed.

```python
@torch.no_grad()
def decode_to_features(self, latents: dict, num_levels: int = 3) -> list[torch.Tensor]:
    """Run decode backbone and return multi-level features (CLS already stripped).

    Returns list of `num_levels` tensors, each (B*V, N_patches, 3072).
    Default num_levels=3 returns features at out_layers [19, 27, 33].
    """
    x = latents["tokens"]
    h, w = latents["H"], latents["W"]
    start_layer = self.encode_layer + 1

    feats, _aux = self.model.model.backbone.forward_from_layer(
        x, x,  # local_x = x (layer 18 is local attention)
        start_layer, h, w,
        ref_view_strategy="first",
    )
    # feats is tuple of (features, camera_token) pairs; features already CLS-stripped
    return [feat for feat, _cam in feats[:num_levels]]
```

Also expose `encoder_mean` / `encoder_std` (ImageNet normalization constants) as properties for the training script's loss denormalization.

### 2. Create: `train_occrae_img_decoder.py` (root)

Adapted from GLD's `third_party/GLD/src/train_stage1_mae.py`. Key changes:

| Aspect | GLD | OccRAE Image Decoder |
|--------|-----|-----------------|
| Encoder | `RAE_DA3` (DA3-Base, 4 levels at [5,7,9,11]) | `OccRAE` (DA3-Giant, 3 levels at [19,27,33] via decode) |
| hidden_size | 6144 (4 * 1536) | 9216 (3 * 3072 with cat_token) |
| Data | `cut3r_data` (50k scenes: DL3DV + RE10K, 504x504) | 45k scenes: Waymo/VKitti/DDAD/Pandaset/Once (front + surround) |
| Resolution | Fixed 504x504 (multi-res: 504x504/378/336/280) | Variable: 518x{168,210,266,280,294} |
| Imports | GLD's `stage1`, `disc`, `cut3r_data` | OccAny's `occ_rae`, GLD's `disc` + `GeneralDecoder_Variable` |

**Reuse from GLD (import directly from `third_party/GLD/src/`):**
- `disc/` — DINO discriminator, DiffAug, LPIPS, GAN losses
- `stage1/decoders/decoder.py` — `GeneralDecoder_Variable` (wrapped as `OccRAEImageDecoder`)
- `utils/optim_utils.py` — `build_optimizer`, `build_scheduler`

**Reuse from OccAny:**
- `occany/model/occ_rae.py` — OccRAE (frozen encoder + feature extraction)
- `occany/datasets/` — data loading (via the existing `get_data_loader()` infrastructure)

**Training loop structure** (same as GLD):
1. Load batch → ImageNet-normalized images `(B, V, C, H, W)`
2. Frozen encode: `OccRAE.encode()` → layer-18 tokens
3. Frozen decode features: `OccRAE.decode_to_features()` → list of 3 tensors, each `(B*V, N, 3072)`, CLS already stripped
4. Concat 3 levels along feature dim → `(B*V, N, 9216)`
5. Trainable decoder forward → RGB logits → unpatchify
6. Denormalize to [0,1]
7. Losses: L1 + LPIPS + GAN (with adaptive weighting)
8. Discriminator update step
9. EMA update on decoder

**Data loading:** Use OccAny's existing `occany.datasets.get_data_loader()` via `eval()` dataset strings (same as `train_occany_plus_recon_1B.sh`). Each batch item provides `img` tensors. Image normalization is ImageNet-norm, matching OccRAE's expected input.

**Dataset composition (45k scenes total):**

| Split | Dataset | Scenes | Type | Views |
|-------|---------|--------|------|-------|
| Front-facing | WaymoSeqMultiView | 6,000 | seq_exact_len_sub5_stride9 | 2–20 |
| Front-facing | VKittiSeqMultiView | 5,000 | seq_exact_len_sub5_stride9 | 2–20 |
| Front-facing | DDADSeqMultiView | 6,000 | seq_exact_len_sub5_stride9 | 2–20 |
| Front-facing | PandasetSeqMultiView | 6,000 | seq_exact_len_sub5_stride9 | 2–20 |
| Front-facing | OnceSeqMultiView | 6,000 | seq_exact_len_sub5_stride9 | 2–20 |
| Surround | WaymoSeqMultiView | 4,000 | seq_surround_temporal_sub5_stride9 | 5–20 |
| Surround | DDADSeqMultiView | 4,000 | seq_surround_temporal_sub5_stride9 | 6–20 |
| Surround | PandasetSeqMultiView | 4,000 | seq_surround_temporal_sub5_stride9 | 6–20 |
| Surround | OnceSeqMultiView | 4,000 | seq_surround_temporal_sub5_stride9 | 5–20 |

**Validation:** 206 KITTI scenes (518x168, 5 views) + 206 Occ3d-nuScenes scenes (518x294, 24 views / 6 per timestep).

All datasets use: `resolution=[(518,294),(518,280),(518,266),(518,210),(518,168)]`, `aug_crop=128`, `z_far=50`, `transform=SeqColorJitter`, `aug_focal=0.9`, `distill_model_name='SAM3'`, `base_model='da3'`.

### 3. Create: `configs/train_occrae_img_decoder.yaml`

```yaml
# OccRAE Image Decoder: MAE decoder on 3-level decode features

occrae:
  weights_path: checkpoints/occany_plus_recon_1B.pth
  output_resolution: [518, 518]
  encode_layer: 18

decoder:
  config_path: third_party/GLD/src/configs/decoder/ViTXL
  patch_size: 14
  hidden_size: 9216    # 3072 * 3 (3 DA3-Giant levels with cat_token)
  dropout: 0.0

training:
  epochs: 100
  ema_decay: 0.9978
  batch_size: 2
  effective_batch_size: 64
  num_workers: 12
  clip_grad: 0.0
  log_interval: 100
  checkpoint_interval: 5000
  optimizer:
    lr: 2.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.0
  scheduler:
    type: cosine
    warmup_epochs: 1
    decay_end_epoch: 50
    base_lr: 2.0e-4
    final_lr: 2.0e-5

gan:
  disc:
    arch:
      dino_ckpt_path: models/discs/dino_vit_small_patch8_224.pth
      ks: 9
      norm_type: bn
      using_spec_norm: true
      recipe: S_8
    optimizer:
      lr: 2.0e-4
      betas: [0.9, 0.95]
      weight_decay: 0.0
    scheduler:
      type: cosine
      warmup_epochs: 1
      decay_end_epoch: 16
      base_lr: 2.0e-4
      final_lr: 2.0e-5
    augment:
      prob: 1.0
      cutout: 0.0
  loss:
    disc_loss: hinge
    gen_loss: vanilla
    disc_weight: 0.75
    perceptual_weight: 1.0
    recon_weight: 1.0
    disc_start: 6
    disc_upd_start: 4
    lpips_start: 0
    max_d_weight: 10000.0
    disc_updates: 1
```

### 4. Create: `sh/train_occrae_img_decoder.sh`

Launch script using `torchrun` for DDP. Uses the same dataset strings as `train_occany_plus_recon_1B.sh` (45k training scenes, front-facing + surround). Sources `env_bsc.sh` and prepends vendored paths.

Key training args carried over from `train_occany_plus_recon_1B.sh`:
- `--batch_size=2`, `--effective_batch_size=64` (gradient accumulation)
- `--epochs=100`, `--warmup_epochs=3`
- `--amp bf16`
- `--num_workers=12`
- `--save_freq=3`, `--eval_freq=1`
- Same `--train_dataset` and `--test_dataset` strings (Waymo/VKitti/DDAD/Pandaset/Once + KITTI/nuScenes val)

---

## Key Implementation Details

### Variable resolution handling

GLD uses fixed 504x504. OccAny uses variable resolutions (518x168 to 518x294). `OccRAEImageDecoder` already supports variable input sizes via `interpolate_pos_encoding()` — pass `input_size=(H, W)` to `forward()`.

### Discriminator crop for small resolutions

GLD uses `RandomCrop(224, 224)`, but OccAny's smallest resolution is 518x168, where H=168 < 224. This would crash.

**Fix:** Adaptive crop size per batch: `crop_h, crop_w = min(H, 224), min(W, 224)`. Both must be divisible by 8 (DINO ViT-S/8 patch size). OccAny resolutions are all multiples of 14 (DA3 patch size), so divisibility by 8 is guaranteed (lcm(14,8)=56 and all heights 168/210/266/280/294 are multiples of 14). The DINO ViT-S/8 discriminator handles variable input sizes since ViTs process variable-length sequences.

### ImageNet normalization flow

OccAny's data pipeline provides ImageNet-normalized images. OccRAE.encode() expects this format. For loss computation:
- Denormalize predictions and GT to [0,1]: `x * std + mean`
- Scale to [-1,1] for LPIPS: `x * 2 - 1`
- Same flow as GLD's training script

### CLS token handling

`forward_from_layer` already strips CLS + register tokens at `vision_transformer.py:593`. The returned features are `(B*V, N_patches, 3072)` — no manual stripping needed. (DA3-Giant has `num_register_tokens=0`, so only CLS is stripped, but the code is robust to register tokens.)

---

## Verification

1. **Smoke test (single GPU, overfit on 1 sample):**
   ```bash
   ssh karolina "conda activate occany && source env_bsc.sh && \
     python train_occrae_img_decoder.py --config configs/train_occrae_img_decoder.yaml \
       --precision bf16 --image-size 518"
   ```
   Verify: loss decreases, reconstruction images look reasonable at step ~1000.

2. **Check feature dimensions:** Print shapes of OccRAE.decode_to_features() output to confirm 3 levels each with expected dim.

3. **Full training (4 GPUs):**
   ```bash
   ssh karolina "sbatch sh/train_occrae_img_decoder.sh"
   ```
   Monitor wandb for loss/recon, loss/lpips, loss/gan curves matching GLD's training profile.
