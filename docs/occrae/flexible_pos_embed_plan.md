# Plan: Resolution-Flexible Positional Embeddings for OccRAE Transformer

## Root Cause

The OccRAE transformer in `occrae/network/efficient_transformer.py` uses `nn.Embedding` lookup tables for both temporal and spatial positional embeddings. These are sized at init time based on `input_size` (defaults: `num_views=8`, `seq_len=16`), but actual data has 10 temporal frames and 778 or 445 spatial tokens. This causes an index-out-of-bounds CUDA crash on the first forward pass when indices exceed the embedding table size.

### Data dimension mismatch

| Parameter | Config default | Actual data (nuScenes) | Actual data (KITTI) |
|-----------|---------------|----------------------|-------------------|
| `num_views` (temporal) | **8** | **10** | **10** |
| `seq_len` (spatial) | **16** | **778** | **445** |
| `temporal_pos` entries | **8** | needs **10** | needs **10** |
| `spatial_pos` entries | **16** | needs **778** | needs **445** |

Spatial token counts are derived from `(width // 14) × (height // 14) + 1` (the +1 is the CLS token):
- 518×294 → 37×21 + 1 = **778**
- 518×168 → 37×12 + 1 = **445**

---

## DA3 Reference: Two Mechanisms

Depth Anything 3 handles variable spatial resolution with two complementary mechanisms:

1. **Learned absolute `pos_embed` + bicubic interpolation** — A learnable `(1, N+1, D)` parameter initialized for a reference square patch grid, reshaped to 2D and bicubic-interpolated to the actual patch grid `(h_patches, w_patches)` at runtime via `interpolate_pos_encoding()`.
   - Reference: `third_party/Depth-Anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py` L218–243

2. **2D Rotary Position Embeddings (RoPE)** — Applied per-layer in attention from a configurable start layer. `PositionGetter` generates integer `(y, x)` coordinates dynamically for any grid size. The rotation is inherently resolution-independent — no fixed embedding table.
   - Reference: `third_party/Depth-Anything-3/src/depth_anything_3/model/dinov2/layers/rope.py`

---

## Recommended Approach: Two-Part Fix Following DA3

### Part 1 — Spatial: Learnable 2D grid + bicubic interpolation (DA3 style)

- Replace `self.spatial_pos = nn.Embedding(...)` with `self.spatial_pos = nn.Parameter(torch.zeros(1, ref_h * ref_w + 1, hidden_dim))`.
- The +1 slot stores the CLS token's positional embedding (separated, same as DA3).
- Store the reference grid shape as `self.ref_spatial_size = (ref_h, ref_w)`, added as a new `__init__` parameter.
- Add `interpolate_pos_encoding(self, h_patches, w_patches)` method:
  - Separate the CLS embedding (index 0) from patch embeddings (indices 1:).
  - Reshape patch embeddings to `(1, ref_h, ref_w, D)`, permute to `(1, D, ref_h, ref_w)`.
  - Bicubic-interpolate to `(1, D, h_patches, w_patches)`.
  - Permute back and flatten to `(1, h_patches * w_patches, D)`.
  - Concatenate CLS embedding back at index 0.
  - Fast-path: skip interpolation when `(h_patches, w_patches) == (ref_h, ref_w)`.

The current spatial layout in `efficient_transformer.py` flattens tokens as `(C, T, S, 1)` with `w=1`. To do 2D bicubic interpolation, the 2D patch grid `(patch_h, patch_w)` must be known at init time. This comes from `output_resolution` in the dataset config divided by `patch_size=14`.

### Part 2 — Temporal: Sinusoidal encoding at runtime

- Remove `self.temporal_pos = nn.Embedding(self.t, hidden_dim)`.
- In `forward()`, compute temporal position embeddings as:
  ```python
  t_embed = gem_timestep_embedding(torch.arange(t, device=x.device), self.hidden_dim)
  ```
  — this function is already implemented in the same file. No learnable parameter needed.
- Handles any number of views with zero added parameters.

### Alternative (future work): RoPE in Attention Layers

A more radical but cleaner long-term approach following DA3's primary mechanism:
- Add 1D/2D RoPE to `Attention.forward()` in `transformer_block.py`.
- Pass spatial `(y, x)` positions to spatial attention, temporal position indices to temporal attention.
- Remove additive positional embeddings entirely.
- Not recommended as the first fix — it's a larger architectural change.

---

## Implementation Steps

### Phase 1: Core positional embedding changes in `occrae/network/efficient_transformer.py`

1. **Add `ref_spatial_size` parameter to `Transformer.__init__`** — e.g. `ref_spatial_size=(37, 21)` for 518×294. Store as `self.ref_spatial_size`.

2. **Replace `self.spatial_pos` from `nn.Embedding` to `nn.Parameter`** — shape `(1, ref_h * ref_w + 1, hidden_dim)`. Initialize with `nn.init.trunc_normal_(..., std=0.02)`.

3. **Remove `self.temporal_pos = nn.Embedding(...)`** entirely.

4. **Add `interpolate_pos_encoding(self, h_patches, w_patches)` method** — follows DA3's `vision_transformer.py` L218–243. CLS slot separated, patch grid reshaped to 2D, bicubic-interpolated, reassembled.

5. **Update `forward()` positional embedding block** (currently L228–231):
   - Replace `self.temporal_pos(t_pos)` with sinusoidal: `gem_timestep_embedding(torch.arange(t), hidden_dim)` expanded per spatial token.
   - Replace `self.spatial_pos(s_pos)` with `self.interpolate_pos_encoding(h, w)` tiled across `t` frames.
   - Note: the CLS token from `interpolate_pos_encoding` is included in the spatial sequence because `seq_len = patch_h * patch_w + 1`.

6. **Update `initialize_weights()`** — remove `nn.init.normal_` calls for the old `nn.Embedding` weights; add `nn.init.trunc_normal_(self.spatial_pos, std=0.02)`.

7. **Update `self.num_spatial`** — must be computed from `ref_h * ref_w + 1` (matching the actual token count including CLS).

### Phase 2: Propagate 2D grid info through the trainer

8. **Update `occrae/occrae_trainer.py` `__init__`** — read `output_resolution` from config, derive `patch_h = height // 14`, `patch_w = width // 14`. Change `self.input_size` from `(1536, num_views, seq_len, 1)` to `(1536, num_views, patch_h, patch_w)`. Pass `ref_spatial_size=(patch_h, patch_w)` to `Transformer(...)`.

9. **Update configs** — add `output_resolution: [518, 294]` (nuScenes) or `[518, 168]` (KITTI) to the `dataset:` section of each training config.

### Phase 3: Checkpoint compatibility

10. **Add key remapping in checkpoint loading** — if an old checkpoint contains `temporal_pos.weight` or `spatial_pos.weight`, remap/drop as needed. `nn.Embedding` stores `.weight`; `nn.Parameter` stores directly under the parameter name. Handle in the trainer's checkpoint loading logic.

### Phase 4: Verification

11. Run the roundtrip smoke test — no crash:
    ```bash
    source env_bsc.sh && python test_occ_rae.py \
      --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
      --input_dir ./demo_data/input \
      --output_dir ./demo_data/output_occ_rae
    ```

12. Test with 518×294 token files (seq_len=778) and 518×168 token files (seq_len=445) — confirm `interpolate_pos_encoding` produces the correct output shape in both cases.

13. Verify gradient flow — run one training step and confirm `model.spatial_pos.grad` is non-None.

---

## Relevant Files

| File | What changes |
|------|-------------|
| `occrae/network/efficient_transformer.py` | Main changes: `Transformer.__init__`, `Transformer.forward`, new `interpolate_pos_encoding` |
| `occrae/occrae_trainer.py` | Update `input_size` construction (~L36–38, ~L104); pass `ref_spatial_size` |
| `configs/train_occrae.yaml` | Add `output_resolution` under `dataset:` |
| `configs/train_occrae_overfit.yaml` | Add `output_resolution` under `dataset:` |
| `configs/train_occrae_fm_overfit.yaml` | Add `output_resolution` under `dataset:` |
| `occrae/network/transformer_block.py` | Only if going the RoPE route |

**Reference implementations:**
- `third_party/Depth-Anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py` L218–243 — `interpolate_pos_encoding`
- `third_party/Depth-Anything-3/src/depth_anything_3/model/dinov2/layers/rope.py` — `PositionGetter`, `RotaryPositionEmbedding2D`

---

## Decisions

- **Spatial pos**: learnable 2D grid `nn.Parameter` + bicubic interpolation at runtime — preserves learned spatial priors while supporting arbitrary resolutions, directly following DA3.
- **Temporal pos**: sinusoidal via `gem_timestep_embedding()` (already in codebase) — small 1D dimension, no benefit from learned embeddings, handles any view count with zero extra parameters.
- **Not RoPE initially** — bigger architectural change requiring `Attention` modifications; can be layered on later.
- **CLS token**: separate learnable slot at index 0 of `spatial_pos`, not interpolated, matching DA3's design.

## Further Considerations

1. **2D grid knowledge**: The current architecture encodes spatial as 1D with `w=1` in `input_size`. Proper 2D bicubic interpolation requires `(patch_h, patch_w)`. Recommended: derive from `output_resolution` in the dataset config (`width // 14`, `height // 14`). Alternative: treat as 1D and use `F.interpolate(..., mode='linear')` — simpler but loses 2D spatial structure.

2. **Max temporal length**: Sinusoidal encoding handles arbitrarily long sequences. If learned temporal embeddings are needed later, a generous max (e.g., 64) with interpolation is straightforward to add.

3. **Mixed-resolution batches**: If a single batch contains sequences from different resolutions (e.g., 518×294 and 518×168 mixed), standard collation will fail due to shape mismatch. The positional embedding fix here is a prerequisite, not the complete solution — dataset-level resolution grouping or padding would also be needed.
