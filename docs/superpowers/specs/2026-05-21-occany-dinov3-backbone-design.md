# OccAny+ recon with DINOv3 backbone — design

Date: 2026-05-21
Status: draft (pre-implementation)

## 1. Goal & two-stage approach

Train an OccAny+ reconstruction model with the same loss stack as
`sh/train_occany_plus_recon_1B_infinite_depth.sh`, but with the backbone ViT
replaced by **DINOv3 ViT-H+/16 distilled (LVD-1689M)** instead of the
DA3-GIANT-1.1 DinoV2 backbone.

We split this into two stages and **only Stage 1 is committed by this spec**:

- **Stage 1 (this spec): vanilla DINOv3, per-view.** Each view is processed
  independently through an unmodified DINOv3 ViT — no `alt_start` alternating
  attention, no `camera_token`, no `cat_token`. The DPT head sees per-view
  features and produces per-view depth / ray / pointmap. We verify the simplest
  possible swap works end-to-end and produces reasonable metrics before adding
  any cross-view machinery.
- **Stage 2 (deferred — separate spec when Stage 1 lands).** Port DA3's
  cross-view design (alt_start alternating local/global attention, camera_token
  per view, cat_token, RoPE handling for cross-view blocks) on top of the same
  DINOv3 ViT. This is the harder, more invasive change; doing it only after
  Stage 1 validates lets us isolate "did DINOv3 help?" from "did the cross-view
  port preserve DA3 behavior?".

### Stage 1 in scope

- Thin wrapper that runs DINOv3 ViT-H+/16 per-view on a `(B, T, 3, H, W)` batch.
- New wrapper class (recon-only) sibling to `DA3Wrapper`.
- New training entrypoint, launcher, shell script, and SLURM file wired for
  Karolina (`qgpu`, account `eu-25-92`).
- DPT head producing depth / depth_conf / ray / ray_conf per view, with the
  same output schema as `DA3Wrapper.inference_batch`.
- Patch size 16 with resolutions rounded to nearest multiples of 16
  (`528×288`, `528×272`, `528×256`, `528×208`, `528×176`).
- YAML config for backbone architecture under `occany/configs/dinov3/`.

### Stage 1 out of scope (defer to Stage 2 or later)

- Cross-view attention of any kind. Each view is independent in Stage 1.
- `alt_start`, `camera_token`, `cat_token` — none of these exist in Stage 1.
- Global-mode RoPE rearrangement (irrelevant when we never do global passes).
- Generation mode (no `gen_input_encoder`, `forward_gen`, `init_gen_encoders`).
- SAM3 head and SAM3 distillation (`head_sam`, `forward_sam_features`,
  `--distill_model`, `--sam3_*`, `distill_model_name='SAM3'` in datasets).
- `--loss_enc_feat` intermediate-feature loss.
- `--aux_branch_layers` / `init_aux_branch` aux teacher path.
- Touching `DA3Wrapper`, `training_da3.py`, `launch_da3.py`, the
  `depth_anything_3` vendored tree, or the `dinov3` vendored tree.
- Evaluation shell wrappers (`sh/eval_*.sh`) and metric pipelines — follow-up.
- OccRAE / DeltaTok paths.

## 2. New files (Stage 1)

```
occany/configs/dinov3/vith16plus.yaml             # backbone arch (no multi-view kwargs)
occany/model/dinov3_backbone.py                   # thin per-view wrapper over DinoVisionTransformer
occany/model/model_dinov3.py                      # Dinov3Wrapper (recon-only, per-view)
occany/training_dinov3.py                         # recon-only training loop (fork of training_da3.py)
launch_dinov3.py                                  # launcher (parallel to launch_da3.py)
sh/train_occany_plus_recon_dinov3_vith16plus.sh   # shell wrapper
slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm   # SLURM wrapper
test_dinov3_pca.py                                # standalone PCA-feature visualization script
```

### Edited in place (small, additive)

- `occany/datasets/base_seq_dataset.py` (lines ~179 and ~499): add a
  `'dinov3'` branch alongside the existing `'da3'` branch. Both routes call
  `InputProcessor.NORMALIZE(to_tensor(img))` — same ImageNet normalization.
  Simplest form is to widen the check to `if self.base_model in ('da3', 'dinov3'):`.
- `occany/datasets/kitti.py` (line ~493) and `occany/datasets/nuscenes.py`
  (line ~835): same widening of the `'da3'` check to include `'dinov3'`.

No other edits to existing files.

### Reused unchanged

- Dataset constructors (`WaymoSeqMultiView`, `VKittiSeqMultiView`,
  `DDADSeqMultiView`, `PandasetSeqMultiView`, `OnceSeqMultiView`,
  `KittiSeqMultiView`, `Occ3dNuscenesSeqMultiView`). We pass
  `base_model='dinov3'` (new value) and add a one-line branch in the dataset
  code that maps it to the same normalization path as `'da3'`: ImageNet stats
  via `InputProcessor.NORMALIZE` (`mean=(0.485, 0.456, 0.406)`,
  `std=(0.229, 0.224, 0.225)` — see
  `third_party/Depth-Anything-3/src/depth_anything_3/utils/io/input_processor.py:56`).
  This matches DINOv3's pretraining normalization
  (`third_party/dinov3/dinov3/data/transforms.py:32-33`) and avoids reusing the
  `'da3'` string for a different backbone family. See §6 for the exact dataset
  edits.
- All loss modules: pointmap (lidar + pseudo), depth (lidar + pseudo), raymap,
  InfiniDepth pseudo-supervision pipeline (precomputed sibling `.infinidepth.png`
  files; no aux teacher at train time).
- `occany.model.checkpoint_utils`, `occany.utils.helpers`,
  `occany.utils.runtime_paths`, `occany.utils.image_util.ImgNorm`.
- `sh/train_common.sh` helpers (`occany_prepare_train_env`,
  `occany_compute_accum_iter`, `occany_log_train_config`,
  `occany_select_train_cmd`, `occany_log_start_cmd`).

### Not used

- `occany.model.raymap_encoder_da3.RaymapEncoderDA3` (gen-only).
- `occany.model.must3r_blocks.head.SAM3Head` (SAM3 disabled).

## 3. Per-view DINOv3 backbone — `dinov3_backbone.py`

No subclass of `DinoVisionTransformer` is needed in Stage 1. Instead, a small
helper module that instantiates the published DINOv3 ViT, loads its checkpoint,
and exposes a per-view forward that taps multiple intermediate layers.

### Class: `DinoV3PerViewBackbone(nn.Module)`

Attributes:
- `self.vit: DinoVisionTransformer` — constructed directly from
  `dinov3.models.vision_transformer.DinoVisionTransformer(**arch_kwargs)`.
  Pretrained weights loaded via `torch.load` (see below).
- `self.out_layers: tuple[int, ...]` — block indices whose outputs are tapped.

### Method: `forward_multiview(images: (B, T, 3, H, W)) -> list[(feat, cam_token)]`

1. Reshape to `(B*T, 3, H, W)`.
2. Run a single per-image forward through the DINOv3 blocks that taps every
   index in `self.out_layers` and returns the **normed patch tokens** at each
   tap (DINOv3's `DinoVisionTransformer.get_intermediate_layers` already does
   this — call it with `n=self.out_layers`, `reshape=False`, `return_class_token=True`,
   `norm=True`). Result is a list of `(patch_tokens, cls_token)` pairs where:
   - `patch_tokens` shape `(B*T, N, embed_dim)` with `N = (H // 16) * (W // 16)`.
   - `cls_token` shape `(B*T, embed_dim)`.
3. Reshape each pair to `(B, T, N, embed_dim)` and `(B, T, embed_dim)`,
   returning a list of `(feat, cam_token)` tuples that **structurally matches
   what `DA3Wrapper`'s downstream code expects**. The `cam_token` value here is
   just DINOv3's CLS token per view (good enough for any downstream code that
   indexes `feats[i][1]`; the recon-only path in Stage 1 doesn't actually use
   cam_token, but matching the signature keeps the wrapper drop-in shaped).
4. Storage tokens (`n_storage_tokens=4` in DINOv3 ViT-H+) are not returned —
   they're internal to DINOv3 and the DA3-side code never asks for them.

### Method: `get_backbone_metadata() -> dict`

Returns the same dict shape `DA3Wrapper.get_backbone_metadata()` returns, so
the DPT head and any log lines do not branch on backbone family:
```python
{
    "name": "dinov3_vith16plus",
    "token_dim": embed_dim,                # 1280 for ViT-H+/16
    "feature_dim": embed_dim,              # cat_token disabled in Stage 1 → no doubling
    "out_layers": tuple(out_layers),
    "alt_start": -1,                       # sentinel: no cross-view in Stage 1
    "total_layers": self.vit.n_blocks,     # 32
    "num_heads": self.vit.num_heads,       # 20
    "cat_token": False,                    # always False in Stage 1
}
```

### Construction & weight loading

YAML-driven, via a `build_dinov3_per_view(yaml_path, weights_path=None)`
factory:

```python
def build_dinov3_per_view(yaml_path, weights_path=None):
    cfg = yaml.safe_load(open(yaml_path))
    vit = DinoVisionTransformer(**cfg["arch"])
    if weights_path is not None:
        # weights_only=True matches the official DINOv3 load path
        # (torch.hub.load_state_dict_from_url in _make_dinov3_vit).
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = vit.load_state_dict(state_dict, strict=True)
        # strict=True is fine in Stage 1: we add no parameters of our own.
        assert not missing and not unexpected, (missing, unexpected)
    backbone = DinoV3PerViewBackbone(vit, out_layers=tuple(cfg["out_layers"]))
    return backbone
```

No `from_pretrained`, no HuggingFace Hub, no `dinov3.hub.backbones.*` hub
functions. The `arch` block in the YAML mirrors the kwargs that
`dinov3_vith16plus` passes to `_make_dinov3_vit`, so the parameter shapes align
bit-for-bit with the published checkpoint.

### YAML config

```yaml
# occany/configs/dinov3/vith16plus.yaml
arch:
  img_size: 224                       # required by PatchEmbed.__init__; forward still accepts dynamic H/W via RoPE
  patch_size: 16
  in_chans: 3
  embed_dim: 1280
  depth: 32
  num_heads: 20
  ffn_ratio: 6.0
  ffn_layer: swiglu
  layerscale_init: 1.0e-05
  norm_layer: layernormbf16
  n_storage_tokens: 4
  mask_k_bias: true
  qkv_bias: true
  proj_bias: true
  ffn_bias: true
  drop_path_rate: 0.0
  pos_embed_rope_base: 100.0
  pos_embed_rope_normalize_coords: separate
  pos_embed_rope_rescale_coords: 2
  pos_embed_rope_dtype: fp32

# Layers tapped for the DPT head — picked by analogy to DA3-GIANT's
# out_layers quartered across depth. Tunable.
out_layers: [7, 15, 23, 31]
```

`img_size: 224` is the standard DINOv3 default and only affects PatchEmbed's
init-time bookkeeping; per-block RoPE is computed from the *actual* patch grid
at forward time, so the same weights load and run unchanged at 528×176.

## 4. `Dinov3Wrapper` (recon-only) — `model_dinov3.py`

`Dinov3Wrapper(nn.Module)`. Does **not** extend `DepthAnything3`. No
`gen_input_encoder`, no `head_sam`, no `aux_blocks` / `aux_head`.

### Attributes

- `self.backbone: DinoV3PerViewBackbone` — from §3.
- `self.head` — DPT-style head producing depth / depth_conf / ray / ray_conf.

### Public surface

- `__init__(self, config_path: str, weights_path: str | None)`:
  - Builds backbone via `build_dinov3_per_view(config_path, weights_path)`.
  - Builds head from `backbone.get_backbone_metadata()` (`feature_dim`,
    `len(out_layers)`).
- `forward(self, images, **kwargs)` → `self.inference_batch(images, **kwargs)`.
- `inference_batch(self, images, **_)` returns a dict with the same keys as
  `DA3Wrapper.inference_batch`:
  ```python
  {
      "pointmap": (B, T, H, W, 3),
      "depth": (B, T, H, W),
      "depth_conf": (B, T, H, W),
      "ray": (B, T, H, W, 6),
      "ray_conf": (B, T, H, W),
      "c2w": None,
      "intrinsics": None,
      "sam_feats": None,
  }
  ```
- `get_backbone_metadata()` — delegates to the backbone.

### `_process_depth_output`

Direct port of `DA3Wrapper._process_depth_output` with:
- `head_sam` branch removed (always `sam_feats=None`).
- `pose_from_cam_dec` branch removed.
- `pose_from_depth_ray` branch removed (we keep `c2w=None`, `intrinsics=None`;
  pointmap is computed as `depth * ray_dirs + ray_origins`, the DA3 default).
- `save_outputs` debug block removed.

Computed:
```python
depth        = head_output["depth"] * default_scale      # default_scale = 20
depth_conf   = head_output["depth_conf"]
ray          = head_output["ray"]                        # (..., 6)
ray[..., 3:] = ray[..., 3:] * default_scale              # scale ray origins
ray_conf     = head_output["ray_conf"]
pointmap     = depth.unsqueeze(-1) * ray[..., :3] + ray[..., 3:]
```

### DPT head choice

Plan A — import DA3's existing DPT head class
(`depth_anything_3.model.dualdpt.DualDPT`, per the DA3 config at
`third_party/Depth-Anything-3/src/depth_anything_3/configs/da3-giant.yaml`)
and instantiate a fresh one sized from `backbone.get_backbone_metadata()`. The
head needs no DA3 model state, only its construction kwargs (`dim_in`,
`output_dim`, `features`, `out_channels`).

Plan B (fallback) — if the head class isn't cleanly importable in isolation,
write a minimal ~100-line DPT head producing the same dict
(`depth`, `depth_conf`, `ray`, `ray_conf`).

Decision: try Plan A first during implementation; fall back to Plan B if it
can't be imported without dragging in the rest of the DA3 model.

## 5. Training entrypoint, shell script, SLURM file

### `launch_dinov3.py`

Parallel to `launch_da3.py`. Same DDP / torchrun boilerplate; imports
`occany.training_dinov3` and invokes its `main(args)`.

### `occany/training_dinov3.py`

Fork of `training_da3.py` with the following changes.

**Deletions:**
- All `gen_*` / `model_gen` / `init_gen_encoders` / `forward_gen` paths.
- All `head_sam` / SAM3 distillation paths (`--distill_model`,
  `--distill_criterion`, `--sam3_proj_lr_mult`, `--sam3_use_dpt_proj`).
- `--loss_enc_feat` path.
- `--aux_branch_layers` / `init_aux_branch` path.

**Argument renames:**
- `--da3_model_name` removed.
- New flags: `--dinov3_config <path-to-yaml>`, `--dinov3_weights <path-to-pth>`.
- Model construction:
  ```python
  model = Dinov3Wrapper(args.dinov3_config, args.dinov3_weights)
  ```

**Model-attribute rewiring (important).** In `training_da3.py` the trainer
touches `model.model.backbone.parameters()`,
`model.model.backbone.pretrained.blocks[layer_idx]`,
`model.model.head.parameters()`, and the unused `model.model.cam_enc/cam_dec/
gs_head/gs_adapter`. `Dinov3Wrapper` exposes `model.backbone` and `model.head`
directly (no extra `.model.` nesting). The fork must rewire every
`model.model.X` access to `model.X`. Specifically:
- `model.model.backbone.parameters()` → `model.backbone.vit.parameters()`.
- `model.model.backbone.pretrained.blocks[layer_idx]` →
  `model.backbone.vit.blocks[layer_idx]` (fine-tune-layer filtering).
- `model.model.head.parameters()` → `model.head.parameters()`.
- `model.model.cam_enc / cam_dec / gs_head / gs_adapter` references → deleted
  (these modules don't exist in `Dinov3Wrapper`).

The recon loop body (forward, loss assembly, backward, eval) is otherwise
identical. Logging keys (`loss_pointmap_lidar`, `loss_pointmap_pseudo`,
`depth_loss_depth_lidar`, `depth_loss_depth_pseudo`, etc.) carry over unchanged.

**DDP wrapping.** Confirm `find_unused_parameters=True` is set (or that every
learnable parameter in `Dinov3Wrapper` is touched by the loss every step). DA3
already needs this, so the fork should inherit the same setting.

### `sh/train_occany_plus_recon_dinov3_vith16plus.sh`

Clone of `sh/train_occany_plus_recon_1B_infinite_depth.sh` (the non-SAM3
variant) with these edits:

- `EXP_NAME="occany_plus_recon_dinov3_vith16plus"`.
- `WIDTH=528`, `HEIGHT=176` (was 518 / 168).
- All `resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)]`
  → `resolution=[(528, 288), (528, 272), (528, 256), (528, 208), (528, 176)]`
  in every train dataset constructor.
- Test datasets:
  - KITTI `(518, 168)` → `(528, 176)`.
  - Occ3dNuscenes `(518, 266)` → `(528, 256)`.
- Remove `distill_model_name='SAM3'` from every dataset constructor.
- Change `base_model='da3'` → `base_model='dinov3'` on every dataset constructor
  (train and test). The dataset edits in §2 route this to the same ImageNet
  normalization path.
- Replace `--da3_model_name depth-anything/DA3-GIANT-1.1` with:
  ```
  --dinov3_config occany/configs/dinov3/vith16plus.yaml \
  --dinov3_weights $PROJECT/checkpoints/dinov3/dinov3_vith16plus_pretrain_lvd1689m.pth
  ```
- Remove `--sam3_proj_lr_mult 10.0`, `--sam3_use_dpt_proj`, `--loss_enc_feat`.
- `--fine_tune_layers 34,35,36,37,38,39` → `--fine_tune_layers 26,27,28,29,30,31`
  (last 6 of the 32 DINOv3 blocks).
- Remove the "SAM3 distillation disabled for now" comment block (no longer
  relevant).
- Launcher: `CMD="$(occany_select_train_cmd 'launch_dinov3.py')"`.

Unchanged: lr (`5e-5`), `min_lr`, `warmup_epochs`, `epochs`, batch sizing
helpers, save / eval frequencies, `--amp bf16`, `--fixed_eval_set`,
`--training_objective pointmap_depth_ray`, `--loss_type L1`,
`--pointmap_lambda_c 1.0`, `--depth_lambda_c 0.0`, `--raymap_lambda_c 1.0`,
`--infinidepth_pseudo_supervision`, all `--lambda_*` flags.

### `slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm`

Clone of `slurm/karolina_train_occany_plus_recon_1B_infinite_depth_sam3_distill.slurm`
(only Karolina SLURM template available in repo) with:

- `--job-name=train_occany_plus_recon_dinov3_vith16plus`.
- `--output=slurm/output/train_occany_plus_recon_dinov3_vith16plus_%j.out`.
- `--error=slurm/output/train_occany_plus_recon_dinov3_vith16plus_%j.err`.
- Final line: `bash sh/train_occany_plus_recon_dinov3_vith16plus.sh`
  (the template currently calls `..._sam3_distill.sh` — change this line to
  point at the new non-SAM3 wrapper).
- Drop the SAM3 memory-budget comment table (won't apply to this model).
- Keep `--nodes=4`, `--ntasks-per-node=8`, `--gres=gpu:8`, `--cpus-per-task=16`,
  `--time=48:00:00`, `--partition=qgpu`, `-A eu-25-92`, env activation
  (`conda activate occany`), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 5b. Standalone PCA visualization — `test_dinov3_pca.py`

A small standalone script to confirm the backbone is loading and producing
reasonable dense features. Modeled on
`third_party/dinov3/notebooks/pca.ipynb` (the official DINOv3 PCA
visualization), but as a CLI script so it runs on Karolina from SSH.

**Inputs:**
- `--config` (default `occany/configs/dinov3/vith16plus.yaml`) — backbone arch.
- `--weights` — path to the DINOv3 `.pth` checkpoint.
- `--input_dir` — directory of images (or a single image path).
- `--output_dir` — where to write `pca_<basename>.png` for each input.
- `--layer` (default last value of `out_layers`) — which tap to PCA.
- `--resolution` (default `528 528`) — resize input to this (H, W) before
  forward. Both must be multiples of 16.
- `--device` (default `cuda`).

**Pipeline:**
1. Build the backbone via `build_dinov3_per_view(args.config, args.weights)`;
   set `eval()` mode and move to device.
2. For each image:
   1. Load PIL → tensor via the **same path datasets use**:
      `InputProcessor.NORMALIZE(to_tensor(pil_resized))` — verifies the chosen
      normalization end-to-end.
   2. Add fake T dim → shape `(1, 1, 3, H, W)`.
   3. `feats_list = backbone.forward_multiview(img)` — pick the tap matching
      `--layer`. Take the patch features `(1, 1, N, embed_dim)` and reshape to
      `(N, embed_dim)`.
   4. Fit PCA with 3 components on the patch features (torch SVD on the
      centered matrix is fine — no sklearn dependency needed). Optionally
      apply the standard "foreground PCA" trick from `pca.ipynb`: PCA-1 is
      used as a foreground mask via threshold + sign-flip, then PCAs 2/3/4 are
      computed on the foreground patches only for cleaner RGB.
   5. Min-max normalize the 3 components to `[0, 1]`, reshape to
      `(H_p, W_p, 3)`, bilinear-upsample to `(H, W, 3)`, multiply by 255 and
      save as PNG.
3. Print a one-line summary per image: input path, output path, tap layer,
   `(H_p, W_p)`, feature L2-norm mean.

**Success criterion:** the output PNG shows spatially coherent
object-vs-background structure (matching the DINOv3 paper's example
visualizations). If it looks like noise, the normalization or checkpoint load
is wrong.

This is a CPU-fast / GPU-fast script; intended to run as a first sanity check
before launching any training.

## 6. Verification steps (do these at the top of the implementation plan)

1. **DINOv3 ViT-H+/16 checkpoint availability.** Confirm
   `$PROJECT/checkpoints/dinov3/dinov3_vith16plus_pretrain_lvd1689m.pth` exists
   on Karolina, or downloaded with `wget` from the Meta link. (The user must
   download manually — Meta's license requires accepting the access form.)

2. **Bit-for-bit checkpoint compatibility.** Build the ViT from the YAML and
   confirm `vit.load_state_dict(state_dict, strict=True)` succeeds with no
   missing/unexpected keys. This validates the YAML arch kwargs against the
   checkpoint shipped by Meta.

3. **Standalone forward smoke test.** Build the backbone, push a
   `(2, 4, 3, 528, 176)` tensor through `forward_multiview`, and confirm:
   - 4 exported `(feat, cam_token)` tuples for `out_layers=[7,15,23,31]`.
   - `feat.shape == (2, 4, n_patches, 1280)` where
     `n_patches = (528 // 16) * (176 // 16) = 33 * 11 = 363`.
   - `cam_token.shape == (2, 4, 1280)`.

4. **DPT head importability.** Try
   `from depth_anything_3.model.dualdpt import DualDPT` from inside `occany/`.
   If it imports cleanly without instantiating the full DA3 model, go with Plan
   A in §4; otherwise switch to Plan B (minimal DPT).

5. **Dataset `base_model` semantics — already understood, but verify edits.**
   Confirm the three dataset edits in §2 (`base_seq_dataset.py` ~lines 179 /
   499, `kitti.py` ~line 493, `nuscenes.py` ~line 835) widen the `'da3'`
   branch to also accept `'dinov3'`. Verify no other site reads `base_model`
   in a way that would diverge between the two values (search:
   `grep -rn "base_model" occany/`).

6. **Normalization parity — `InputProcessor.NORMALIZE` vs. DINOv3 pretrain.**
   Already confirmed at design time:
   `third_party/Depth-Anything-3/.../input_processor.py:56` uses ImageNet
   stats, and so does `third_party/dinov3/dinov3/data/transforms.py:32-33`.
   At implementation time, sanity-check both files still match.

7. **DDP unused-parameter check.** After a first single-GPU forward+backward,
   list any backbone params whose `.grad` is None. If empty, default DDP works;
   if not, the SLURM/training script must wrap with `find_unused_parameters=True`.

8. **Aspect-ratio / metrics sanity.** Patch-16 changes presets from
   `(518, 168)` (ratio 3.08) to `(528, 176)` (ratio 3.00); the longest preset
   loses ~3% aspect. Run a small fixed-set eval after the first few epochs to
   confirm depth/IoU metrics are in the expected ballpark before scaling up.

## 7. Open questions & risks

- **No cross-view information in Stage 1.** Stage 1 is intentionally a "weaker"
  model than DA3 because each view sees no other view. Depth/pointmap losses
  remain per-view, so training is well-defined, but final-metric parity with
  DA3-GIANT is not the Stage 1 goal — usefulness as a baseline for Stage 2 is.
  If Stage 1 metrics are catastrophically worse than DA3, that's a signal that
  cross-view attention is doing important work and Stage 2 must follow.

- **`out_layers=[7, 15, 23, 31]` for a 32-block ViT.** Picked by analogy to
  DA3-GIANT (out_layers quartered across the depth). May be retuned after a
  first training run.

- **Patch-size resolution change.** Multi-aspect-ratio training presets
  changed; could subtly affect data augmentation balance. Sanity check on a
  few batches before the long run.

- **InfiniDepth pseudo-supervision files.** Continue to use the precomputed
  `<stem>.infinidepth.png` sibling files; verified by
  `extract_infinidepth_pseudo.py` / `verify_infinidepth_pseudo.py`. No change.

## 8. Stage 2 — deferred (separate spec when Stage 1 lands)

Add DA3-style cross-view machinery on top of the same DINOv3 ViT. Touchpoints,
recorded here so we don't re-discover them in three months:

- Subclass `DinoVisionTransformer` to add `alt_start`, per-view `camera_token`
  parameter `(1, T_max, embed_dim)`, optional `cat_token` doubling of exported
  feature dim, and a multi-view forward with alternating local/global blocks.
- The hard part is RoPE under global attention: DINOv3's `apply_rope`
  (`third_party/dinov3/dinov3/layers/attention.py:66-85`) skips a *contiguous
  prefix* of `N - sin.shape[-2]` tokens. Naive `(B, T, P+N, C) → (B, T*(P+N),
  C)` reshape interleaves prefix tokens per view and breaks this prefix math.
  The fix is to rearrange tokens to `[all_prefix_across_views,
  all_patches_across_views]` order before each global block, tile the patch
  RoPE T times, and invert the rearrangement after. Documented in detail when
  Stage 2 is written.
- `Dinov3Wrapper.get_backbone_metadata()` would then return
  `alt_start=<configured>`, `cat_token=<configured>`, `feature_dim = embed_dim
  * (2 if cat_token else 1)` instead of the Stage 1 sentinels.

## 9. Follow-ups (after Stage 1 lands)

- Evaluation wrappers (`sh/eval_*.sh`) for the DINOv3 model.
- Stage 2 (cross-view machinery) — separate spec.
- Smaller-arch YAMLs (`vitl16.yaml`, `vit7b16.yaml`) for ablations.
- A SAM3 distillation path on top of DINOv3, if the recon path validates.
