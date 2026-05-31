# Plan: Per-Image Pseudo Depth with a Frozen Teacher Tail

Keep the current idea of a duplicated frozen aux branch, but change how it is executed.
The goal is to use two passes per training step:

1. A **per-image teacher pass** that runs through the frozen shared backbone prefix,
   then through a **frozen copied tail** (`aux_blocks`) and **frozen copied head** (`aux_head`)
   to produce pseudo depth.
2. A **multi-view student pass** that runs through the same frozen shared prefix,
   then through the **trainable tail** (the model's last few backbone blocks) and the
   **trainable main head** to produce the actual reconstruction outputs.

This preserves the key property we want: the pseudo target stays frozen with respect to the
trainable tail, while avoiding the current multi-view token-splicing logic in `inference_batch_aux`.

## Target Architecture

Let the backbone be split at a boundary layer:

- **Frozen prefix**: layers `0..boundary`
- **Tail**: layers `boundary+1..last`

Then each training step does:

1. **Teacher / pseudo-depth pass** (`torch.no_grad()`)
   - Reshape images from `(B, T, 3, H, W)` to `(B*T, 1, 3, H, W)`
   - Run the backbone **only up to the boundary** using the frozen shared prefix
   - Feed the boundary tokens into the **frozen copied tail** (`aux_blocks`)
   - Decode with the **frozen copied head** (`aux_head`)
   - Reshape depth back to `(B, T, H, W)` as `aux_depth`

2. **Student / reconstruction pass** (normal training forward)
   - Run the multi-view images `(B, T, 3, H, W)` through the same frozen prefix
   - Continue through the model's **trainable tail blocks**
   - Decode with the **trainable main head**
   - Use these outputs for depth / pointmap / raymap supervision as today

After that, `aux_depth` feeds into the existing `build_aux_pseudo_depth` scale-alignment,
sky-masking, and pseudo-supervision logic.

## Why This Design

- The pseudo branch is a true frozen teacher with respect to the trainable tail.
- The student branch can still adapt by tuning only the last few backbone blocks plus the head.
- The teacher pass is per-image, which removes the fragile view-reordering and token-splicing
  assumptions in the current aux inference path.
- No second full DA3 model is needed; the existing duplicated frozen tail and head are sufficient.

Compute cost increases because the frozen prefix is run twice per step, but that is acceptable for
this design.

## Proposed Changes

### 1. `occany/model/model_da3.py`

- **Keep** `init_aux_branch`, `aux_blocks`, `aux_head`, and `aux_input_layer_idx`.
- **Change** the aux execution path from the current multi-view token-splicing approach to a
  per-image boundary-state approach.
- **Add** a new method on `DA3Wrapper`, for example `inference_batch_individual(images)`, that:
  - takes `images` with shape `(B, T, 3, H, W)`
  - flattens to `(B*T, 1, 3, H, W)`
  - runs the frozen shared backbone prefix up to `aux_input_layer_idx`
  - runs the frozen copied tail (`aux_blocks`)
  - decodes with `head=self.aux_head`
  - returns depth with shape `(B, T, H, W)`

- **Refactor or replace** `inference_batch_aux`:
  - the current version assumes multi-view raw intermediate tokens from the backbone
  - it should either be removed entirely or rewritten as a lower-level helper that operates on
    single-view boundary states

- **Refactor or replace** `_process_attention_aux` only as needed:
  - if the new per-image teacher path still uses the same tail-block stepping logic, this helper may
    still be useful
  - if the new implementation routes through an existing lower-level backbone helper, it may become
    unnecessary

- **Keep** `_process_depth_output(..., head=...)` as the decoding entry point.
  No extra `_process_depth_output_with_head` method is needed because the existing method already
  accepts an explicit `head` argument.

### 2. `occany/da3_inference.py` (`compute_da3_loss`)

- In the `aux_metric_pseudo_supervision` branch, stop reading pseudo depth from
  `output['aux_outputs']`.
- Instead, compute pseudo depth explicitly from the teacher pass:

  ```python
  metric_imgs = torch.stack([b['img'] for b in img_views], dim=1).to(device)
  aux_depth = model.inference_batch_individual(metric_imgs)
  ```

- Everything below that remains the same in spirit:
  - scale-align `aux_depth` to lidar depth with `build_aux_pseudo_depth`
  - compute sky mask with `da3_metric_model`
  - build `pseudo_pointmap`
  - apply pseudo supervision to the student outputs

- Remove the dependency on `output['aux_outputs']` for the pseudo-supervision path.

- The legacy `scale_inv_depth_loss` path should be treated separately:
  - either remove it entirely if this repo is consolidating on pseudo supervision
  - or keep it temporarily and document that it still uses the old aux-branch wiring until removed

### 3. `occany/training_da3.py`

- **Keep** `--aux_branch_layers` because it still defines the teacher-tail depth.
- **Keep** the call to `model.init_aux_branch(n_layers=args.aux_branch_layers)` when aux pseudo
  supervision is enabled.
- **Strengthen validation** for selective fine-tuning:
  - `fine_tune_layers` must match the final contiguous `aux_branch_layers` of the backbone
  - not just have the same length

  For example:
  - if `aux_branch_layers=6` on a 24-layer model, `fine_tune_layers` must be `18,19,20,21,22,23`
  - if `aux_branch_layers=6` on a 40-layer model, `fine_tune_layers` must be `34,35,36,37,38,39`

- **Keep** the checkpoint key sanitizer for `aux_head` / `aux_blocks` so old checkpoints still load.

- `da3_metric_model` stays as-is for sky masking.

### 4. Training shell wrappers

- Keep `--aux_branch_layers N` because the frozen teacher tail still exists.
- Keep `--fine_tune_layers ...` but ensure those values exactly match the final `N` backbone layers.
- Update wrapper comments or documentation to clarify that training now uses:
  - a per-image frozen teacher pass for pseudo depth
  - a multi-view student pass for reconstruction

At minimum this applies to:

- `sh/train_occany_plus_recon.sh`
- `sh/train_occany_plus_recon_1B.sh`
- `sh/train_occany_plus_recon_1B_bsc.sh`

## What is kept

- `aux_blocks` as the frozen copied teacher tail
- `aux_head` as the frozen copied teacher decoder
- `aux_input_layer_idx` as the boundary between frozen prefix and tail
- `build_aux_pseudo_depth`, scale alignment, pseudo pointmap construction
- `da3_metric_model` for sky masking
- checkpoint compatibility for old aux-branch checkpoints

## What changes

| Changed | New behavior |
|---|---|
| Aux execution path | per-image teacher pass instead of multi-view token-splicing |
| Pseudo depth source | explicit teacher forward, not `output['aux_outputs']` from the student pass |
| Fine-tune validation | enforce exact match between trainable tail and frozen teacher tail |

## What may be removed

| Candidate removal | Condition |
|---|---|
| `inference_batch_aux` | if fully replaced by the new per-image teacher path |
| `_process_attention_aux` | if no longer needed by the new teacher implementation |
| `scale_inv_depth_loss` and related flags | if this repo standardizes on aux pseudo supervision only |
