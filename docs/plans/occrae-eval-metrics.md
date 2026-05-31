# Objective
Extend `OccRAETrainer.eval_one_epoch` to log downstream geometric metrics
(`Pointmap`, `Depth`, `Raymap`) by sampling spatial tokens with the flow model,
decoding them through `OccRAE`, and comparing the decoded outputs against
ground-truth depth / intrinsics / cam2world stored in the preprocessed `.npz`
samples.

# Key Files & Context
- `occrae/dataset/preprocessed_sequence.py` — dataset; needs an opt-in GT loader.
- `occrae/occrae_trainer.py` — collate, `fit`, `eval_one_epoch`.
- `occany/loss/losses_da3.py` — `PointmapLoss`, `DepthLosses`, `RaymapLoss`.
- `occany/utils/helpers.py` — joint crop helper (`crop_resize_if_necessary`),
  `intrinsics_c2w_to_raymap`, `convert_depth_to_point_cloud`.
- `occrae/generation_helper.py` — `flow_euler_sample`.
- `occany/model/occ_rae.py` — `OccRAE.encode/decode`. `decode` returns
  `pointmap (B,V,H,W,3)`, `depth (B,V,H,W)`, `depth_conf`, `ray (B,V,H,W,6)`,
  `ray_conf`, `c2w`, `intrinsics`.

# Resolution Standardization
- Standardize all **internal** resolution handling on `(H, W)`.
- Rationale:
  - model APIs, decoded tensors, and loss shapes are already height-first;
  - tensor shapes in the trainer are `(B, C, H, W)` / `(B, V, H, W)`;
  - the current mix of `(H, W)` and `(W, H)` is an avoidable source of silent
    decode and crop bugs.
- Keep `(W, H)` only at external boundaries where a library explicitly expects
  image-size order, and convert there locally.
- Concretely:
  - `cfg.model.output_resolution` stays `(H, W)`.
  - `SequenceRecord.output_resolution` and `_normalize_resolutions()` should be
    migrated to emit `(H, W)` as well.
  - In the trainer, rename ambiguous variables to `output_resolution_hw`.
  - At helper boundaries that still expect PIL-style ordering, convert with an
    explicit local variable like `resolution_wh = (width, height)`.

# Conventions to keep straight
- Canonical internal resolution order is `(H, W)`.
- Preprocessed `.npz` keys are `image`, `depthmap`, `intrinsics`, `cam2world`
  (verified across waymo / vkitti / ddad preprocessors).
- DA3 outputs are bf16 when the trainer runs in bf16; cast inputs to fp32
  before metric reduction to avoid bf16 underflow on `(diff)**2`.
- The model now predicts metric scale, so evaluation should use raw metric
  scale rather than per-scene normalization.

# Implementation Steps

## 0. Resolution cleanup first
Files: `occrae/dataset/preprocessed_sequence.py`, `occrae/occrae_trainer.py`

- Change `_normalize_resolutions()` to normalize into `(H, W)` tuples.
- Update `ProcessedDatasetConfig.output_resolution` call sites to treat stored
  resolutions as `(H, W)`.
- In `collate_preprocessed`, rename the chosen resolution variable to
  `output_resolution_hw` and unpack it as `height, width`.
- Only convert to `(W, H)` at helper boundaries that require it.
- Add a small assertion after preprocessing to catch swapped resolutions early,
  e.g. `assert imgs.shape[-2:] == (height, width)`.

## 1. `PreprocessedSequenceDataset` — opt-in GT loading
File: `occrae/dataset/preprocessed_sequence.py`

- Add `load_gt: bool = False` to `__init__` and store on `self`.
- In `__getitem__`, when `load_gt`:
  - For every frame stem load `depthmap`, `intrinsics`, `cam2world` from the
    same `.npz`.
  - Append to per-item lists `depthmaps`, `intrinsics`, `cam2worlds` as numpy
    arrays at native resolution.
  - Add the three lists to the returned dict only when `load_gt` is true.
- Do not stack here; stacking happens after the joint crop in collate.

## 2. Collate — joint crop and GT tensors
File: `occrae/occrae_trainer.py`

The current `collate_preprocessed` calls the image-only
`occany.utils.image_util.crop_resize_if_necessary`. Switch to the joint helper
`occany.utils.helpers.crop_resize_if_necessary(image, depthmap, intrinsics,
resolution)` for both train and val so images and GT share an identical crop.

- Use `output_resolution_hw` as the canonical variable.
- If the helper expects PIL order, convert locally with
  `resolution_wh = (width, height)`.
- Branch on whether the first item has `depthmaps`; if absent (train), pass a
  zero placeholder depth map and identity intrinsics into the helper and
  discard the GT outputs.

For a val batch (GT present):
- Apply the joint helper per `(image, depthmap, intrinsics)` triple.
- Normalize the cropped image with `InputProcessor.NORMALIZE(to_tensor(...))`
  exactly as today.
- Stack into:
  - `imgs`         : `(B, V, C, H, W)`
  - `gt_depth`     : `(B, V, H, W)`         float32
  - `gt_intrinsics`: `(B, V, 3, 3)`         float32
  - `gt_c2w`       : `(B, V, 4, 4)`         float32
- Compute `gt_raymap = intrinsics_c2w_to_raymap(gt_intrinsics, gt_c2w, H, W)`
  → `(B, V, H, W, 6)`.
- Compute `gt_pointmap = gt_depth.unsqueeze(-1) * gt_raymap[..., :3] +
  gt_raymap[..., 3:]` → `(B, V, H, W, 3)`.
- Compute `gt_mask = (gt_depth > 0)` boolean and add it to the batch.

## 3. `OccRAETrainer` — criteria and val dataset wiring
File: `occrae/occrae_trainer.py`

In `__init__` (after super init):
```python
self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
self.raymap_criterion = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
```
- Use `gt_scale=True` because the model predicts metric scale and we want raw
  metric-scale evaluation, not per-scene normalization.
- `alpha=0.0` disables gradient loss; the eval metric should reflect depth
  accuracy, not the auxiliary regularizer.
- The criteria have no parameters; do not register them as submodules and do
  not move them to device.

In `fit()`:
- Construct `self.test_data` with `load_gt=True`.
- Replace the batch-budget idea with an item-budget config:
  - `eval_num_items = cfg.training.get("eval_num_items", 200)`
  - `eval_num_steps = cfg.training.get("eval_num_steps", 25)`
- During eval, stop after exactly 200 items have been consumed globally for the
  epoch, or as close as possible without splitting a batch.

## 4. `eval_one_epoch` — sample, decode, compute metrics
File: `occrae/occrae_trainer.py`

Wrap the existing logic; reuse `with self.ema_scope():` and keep
`self.vit.eval()`.

Per eval batch:
1. Encode images to teacher tokens and keep computing the existing flow loss.
2. Sample spatial tokens:
   - `z = torch.randn_like(x_spatial)`
   - `gen_spatial = flow_euler_sample(self._ema_model(), z,
        pred_mode=cfg.model.pred_mode, context=0,
        num_steps=eval_num_steps, autocast_ctx=self.autocast)`
3. Convert sampled spatial tokens back to DA3 token layout:
   - `gen = self._spatial_to_tokens(gen_spatial)`
   - `gen = gen.squeeze(-1).transpose(-1, -2).contiguous()`  # `(B, V, S, C)`
4. Decode using canonical `(H, W)`:
   ```python
   height, width = batch["output_resolution_hw"]
   decoded = self.occ_rae.decode({"tokens": gen, "H": height, "W": width})
   ```
5. Compute metrics when GT is present, all in fp32:
   - `mask = batch["gt_mask"]`
   - Pointmap:
     ```python
     loss_pm, _ = self.pointmap_criterion(
         decoded["pointmap"].float(),
         batch["gt_pointmap"].float(),
         mask=mask)
     ```
   - Depth:
     ```python
     pred_d = decoded["depth"].float().reshape(B * V, 1, height, width)
     gt_d = batch["gt_depth"].float().reshape(B * V, 1, height, width)
     d_mask = mask.reshape(B * V, 1, height, width).float()
     loss_d, _ = self.depth_criterion(pred_d, gt_d, confidence=None, mask=d_mask)
     ```
   - Raymap:
     ```python
     loss_ray, _ = self.raymap_criterion(
         decoded["ray"].float(),
         decoded.get("ray_conf"),
         batch["gt_c2w"].float(),
         batch["gt_intrinsics"].float(),
         batch["gt_raymap"].float())
     ```
6. Accumulate weighted sums using the exact number of evaluated items so the
   final reduction is a true mean over the 200-item budget.
7. Track `items_seen += B` and stop once `items_seen >= eval_num_items`.

After the loop:
- All-gather across ranks if `self.distributed`.
- On master, log `Eval/loss_pointmap`, `Eval/loss_depth`, `Eval/loss_raymap`,
  and the existing `Eval/loss_flow`.

## 5. Config additions
Files: `configs/train_occrae*.yaml`

Add:
- `training.eval_num_items` (int, default 200)
- `training.eval_num_steps` (int, default 25)

# Verification & Testing
- Sanity: print one collated val batch and assert shapes
  `gt_depth (B,V,H,W)`, `gt_intrinsics (B,V,3,3)`, `gt_c2w (B,V,4,4)`,
  `gt_raymap (B,V,H,W,6)`, `gt_pointmap (B,V,H,W,3)`, `gt_mask (B,V,H,W)`.
- Add one targeted assertion that
  `batch["output_resolution_hw"] == imgs.shape[-2:]`.
- Run a short eval-only loop and confirm:
  - exactly 200 items are evaluated per eval epoch;
  - the three new scalars appear under `Eval/` in TensorBoard;
  - no OOM occurs at the configured batch size.
- Confirm the resolution migration does not introduce swapped-size decode bugs.