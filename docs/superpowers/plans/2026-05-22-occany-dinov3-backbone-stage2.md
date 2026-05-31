# OccAny+ DINOv3 Backbone — Cross-View Extension (Stage 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. All GPU work (forward smokes, training launches) must use the `karolina-job` skill — there is no local GPU.

**Depends on:** Stage 1 (`docs/superpowers/plans/2026-05-22-occany-dinov3-backbone-stage1.md`) is implemented and merged. The following Stage 1 artifacts are reused: `occany/model/dinov3_backbone.py` (extended in-place), `occany/model/model_dinov3.py` (extended in-place), `occany/training_dinov3.py` (extended in-place), `occany/configs/dinov3/occany_dinov3_vith16plus.yaml`, `launch_dinov3.py`, `test_dinov3_pca.py`, dataset edits in `occany/datasets/*`, the training wrapper `sh/train_occany_plus_recon_dinov3_vith16plus.sh` (extended in-place), and the SLURM file `slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm`.

**Not a separate training stage. Not a parallel class hierarchy.** Stage 2 is the next code increment over Stage 1 — implemented as **modifications to Stage 1's existing files**, not as new parallel modules. The cross-view feature is exposed via an `alt_start` kwarg on the existing `DinoV3PerViewBackbone` class (default `-1` = vanilla per-view, Stage 1 behavior). There is **no** new backbone class, **no** new `dinov3_cross_view.py` module, **no** parallel `_stage2` wrapper or SLURM. Cross-view mode is enabled by setting a handful of env vars on Stage 1's existing wrapper:

```bash
EXP_NAME=occany_plus_recon_dinov3_vith16plus_cross_view \
CROSS_VIEW_ALT_START=11 \
FINE_TUNE_LAYERS=11-31 \
DINOV3_CONFIG=occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml \
bash sh/train_occany_plus_recon_dinov3_vith16plus.sh
```

With all four vars unset (defaults), the wrapper produces bit-identical Stage 1 behavior.

**Goal:** Add cross-view attention to the DINOv3 ViT-H+/16 backbone by introducing local↔global parity alternation at `alt_start = 11`, with gated QK-norm initialized so the model starts at pure DINOv3 features. Train end-to-end from the pretrained DINOv3 checkpoint plus random head (no Stage 1 checkpoint loading) so cross-view is a clean ablation against Stage 1.

**Architecture:**
- The existing `DinoV3PerViewBackbone` (Stage 1) learns an optional `alt_start: int = -1` kwarg.
  - When `alt_start == -1` (default), the forward path is identical to Stage 1 — calls `vit.get_intermediate_layers(...)`, no cross-view machinery, no QK-norm installed.
  - When `alt_start >= 0`, gated QK-norm is installed on blocks `[alt_start:]` in `__init__`, and the forward path switches to a custom loop with local↔global parity alternation.
- Layers `0..alt_start-1` are pure pretrained DINOv3 (frame-local). From `alt_start = 11` onward, blocks alternate by index parity: even-index → frame-local (same as before), odd-index → cross-view global pass.
- Global pass follows DA3's pattern: flatten `(B, T, N, D) → (B, T*N, D)` keeping all tokens (cls + storage + patches), with a custom RoPE sin/cos that has identity rotation `(sin=0, cos=1)` for cls/storage positions and the standard intra-frame 2D RoPE *tiled identically* across all T frames for patch positions. Patches at the same `(x, y)` across frames get identical rotations → relative position 0 → pure content-based attention. DA3's `pos_special=zeros` trick, transposed to DINOv3's contiguous-grid RoPE.
- Gated QK-norm wraps DINOv3's `SelfAttention`. Scalar gates `q_gate, k_gate` are zero-initialized so Q/K pass through unchanged at step 0 → bit-identical to pretrained DINOv3. Training learns to ramp the gates as long-sequence global passes start producing distribution shift.
- All new helper code (`GatedQKNormSelfAttention`, `install_gated_qk_norm_on_blocks`, `build_global_rope_sincos`) lives in **`occany/model/dinov3_backbone.py`** — no new file is created.
- `Dinov3Wrapper.__init__` learns a `cross_view_alt_start: int = -1` kwarg and forwards it to `build_dinov3_per_view`. When `-1`, behaves exactly as Stage 1.
- `training_dinov3.py` learns a `--cross_view_alt_start` flag plumbed through to `Dinov3Wrapper`.
- Stage 1's training wrapper is made env-overridable so the cross-view config can be selected at launch time without forking the wrapper. The four overridable knobs are `EXP_NAME`, `DINOV3_CONFIG`, `FINE_TUNE_LAYERS`, and `CROSS_VIEW_ALT_START`. All default to their Stage 1 values.

**Tech stack:** Same as Stage 1 — PyTorch + bf16 autocast + DDP, vendored `dinov3`, DA3's `DualDPT` head, `karolina-job` skill for cluster execution, SLURM `qgpu` / account `eu-25-92` / 4×8 A100.

---

## File map

**Create:**

| Path | Responsibility |
|---|---|
| `occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml` | DINOv3 ViT-H+/16 arch + `alt_start: 11` + tap layers re-tuned for the 32-block depth |
| `test_dinov3_cross_view.py` | Standalone forward + grad-flow smoke for cross-view mode |

**Modify (all in-place, additive — Stage 1 default behavior preserved bit-for-bit):**

| Path | What changes |
|---|---|
| `occany/model/dinov3_backbone.py` | Add `GatedQKNormSelfAttention` (subclass of `dinov3.layers.attention.SelfAttention`), `install_gated_qk_norm_on_blocks`, and `build_global_rope_sincos` helpers. Extend `DinoV3PerViewBackbone.__init__` to accept `alt_start: int = -1`; install QK-norm when `alt_start >= 0`. Split `forward_multiview` into `_forward_per_view` (current Stage 1 path) and `_forward_cross_view` (new cross-view path); dispatch based on `alt_start`. Update `get_backbone_metadata` to reflect `alt_start`. Extend `build_dinov3_per_view` factory to read optional `alt_start` from YAML and accept `alt_start_override`. |
| `occany/model/model_dinov3.py` | Add `cross_view_alt_start: int = -1` kwarg to `Dinov3Wrapper.__init__`; forward to `build_dinov3_per_view(..., alt_start_override=cross_view_alt_start if cross_view_alt_start >= 0 else None)`. |
| `occany/training_dinov3.py` | Add `--cross_view_alt_start` argparse flag (default `-1`). Pass to `Dinov3Wrapper`. Extend `--fine_tune_layers` parsing to accept range syntax (`11-31`) in addition to the existing comma-separated form. |
| `sh/train_occany_plus_recon_dinov3_vith16plus.sh` | Make `EXP_NAME`, `DINOV3_CONFIG`, `FINE_TUNE_LAYERS`, `CROSS_VIEW_ALT_START` env-overridable. Pass `--cross_view_alt_start` through. Default values reproduce Stage 1 behavior bit-for-bit. |

**Not created (cross-view is an in-place extension, not a parallel implementation):**

- ~~`occany/model/dinov3_cross_view.py`~~ — all cross-view code is added to `occany/model/dinov3_backbone.py` and folded into the existing `DinoV3PerViewBackbone` class.
- ~~`DinoV3CrossViewBackbone` class~~ — `DinoV3PerViewBackbone` itself supports both modes via `alt_start`.
- ~~`build_dinov3_cross_view` factory~~ — `build_dinov3_per_view` itself supports both modes.
- ~~`sh/train_occany_plus_recon_dinov3_vith16plus_stage2.sh`~~ — Stage 1's wrapper is env-parametric.
- ~~`slurm/karolina_train_occany_plus_recon_dinov3_vith16plus_stage2.slurm`~~ — submit Stage 1's SLURM with sbatch CLI overrides (`-J`, `-o`, `-e`) and env vars.

**Out of scope (left untouched):** dataset files (already accept `base_model='dinov3'`); `launch_dinov3.py` (reused as-is); the `dinov3` vendored tree; eval pipeline; Stage 1's YAML.

**Naming note:** The class name `DinoV3PerViewBackbone` becomes slightly misleading when `alt_start >= 0` (cross-view is not per-view). This is an acceptable trade-off vs. a rename: the class name is stable across Stage 1's existing call sites (`occany/model/model_dinov3.py`, `test_dinov3_pca.py`, `occany/training_dinov3.py`), and the module docstring + `__init__` docstring document the dual-mode behavior. A future cleanup could rename the class once cross-view becomes the only mode in use.

---

## Task 1: Preflight verification

**Files:** none modified.

- [ ] **Step 1.1: Confirm Stage 1 code is present on local + Karolina**

```bash
cd /home/acao/code/OccAny && ls \
  occany/model/dinov3_backbone.py \
  occany/model/model_dinov3.py \
  occany/training_dinov3.py \
  occany/configs/dinov3/occany_dinov3_vith16plus.yaml \
  launch_dinov3.py \
  sh/train_occany_plus_recon_dinov3_vith16plus.sh \
  slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm 2>&1
```

Expected: every file listed, no `cannot access`. If any are missing, Stage 1 isn't merged yet — stop and execute Stage 1 first.

- [ ] **Step 1.2: Confirm `dinov3.layers.attention.SelfAttention` is the upstream class to subclass**

```bash
cd /home/acao/code/OccAny && grep -n "class SelfAttention\b" third_party/dinov3/dinov3/layers/attention.py
```

Expected: one match at `third_party/dinov3/dinov3/layers/attention.py:43`. If the file path changed, update the import in Task 3 accordingly.

- [ ] **Step 1.3: Confirm `Block.attn` is `SelfAttention` (not a sibling class) in our YAML config**

```bash
cd /home/acao/code/OccAny && PYTHONPATH=third_party/dinov3 python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.model.dinov3_backbone import build_dinov3_per_view
bb = build_dinov3_per_view('occany/configs/dinov3/occany_dinov3_vith16plus.yaml', weights_path=None)
blk0 = bb.vit.blocks[0]
print('block class:', type(blk0).__name__)
print('attn class:', type(blk0.attn).__name__)
print('attn module:', type(blk0.attn).__module__)
print('has q_norm?', hasattr(blk0.attn, 'q_norm'))
"
```

Expected:
- `block class: SelfAttentionBlock`
- `attn class: SelfAttention`
- `attn module: dinov3.layers.attention`
- `has q_norm? False`

If `has q_norm? True` — DINOv3 already exposes QK-norm and Task 3's subclass approach is unnecessary; stop and re-design.

- [ ] **Step 1.4: Confirm RoPE shape contract**

```bash
cd /home/acao/code/OccAny && PYTHONPATH=third_party/dinov3 python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.model.dinov3_backbone import build_dinov3_per_view
bb = build_dinov3_per_view('occany/configs/dinov3/occany_dinov3_vith16plus.yaml', weights_path=None)
# Training uses the (W, H) resolution set [(512, 288), (512, 272), (512, 256), (512, 208), (512, 176), (512, 160)]
# Smokes pick the (W=512, H=176) shape; patch grid = (H//16, W//16) = (11, 32) -> 11*32 = 352 patch tokens.
sin, cos = bb.vit.rope_embed(H=11, W=32)
print('sin shape:', sin.shape, 'dtype:', sin.dtype)
print('cos shape:', cos.shape, 'dtype:', cos.dtype)
assert sin.shape == (11 * 32, 1280 // 20), f'expected (352, 64), got {sin.shape}'
print('OK')
"
```

Expected: `sin shape: torch.Size([352, 64]) dtype: torch.float32`, `cos shape: torch.Size([352, 64]) dtype: torch.float32`, `OK`. Head dim = `embed_dim / num_heads = 1280 / 20 = 64`.

---

## Task 2: Cross-view YAML config

**Files:**
- Create: `occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml`

Same arch as Stage 1's YAML plus `alt_start: 11` and re-tuned tap layers. The four taps must straddle the cross-view region so the DPT head sees pre-cross-view, mid-cross-view, and post-cross-view features. With `alt_start=11` and `depth=32`, a reasonable choice is to put one tap before `alt_start` and three after.

- [ ] **Step 2.1: Create the YAML**

Write `occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml`:

```yaml
arch:
  img_size: 224
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

# Cross-view alternation starts here.
# Layers 0..10 are pure pretrained DINOv3 (frame-local).
# Layers 11..31: even index -> frame-local, odd index -> global cross-view.
# Global pass count = |{11,13,15,17,19,21,23,25,27,29,31}| = 11.
alt_start: 11

# DPT tap layers — one pre-cross-view, three post.
# 9 = last pre-alt_start tap; 17/24/31 spread across the alternating region.
out_layers: [9, 17, 24, 31]
```

- [ ] **Step 2.2: Commit**

```bash
git add occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml
git commit -m "feat(occany): add DINOv3 cross-view YAML config (alt_start=11)"
```

---

## Task 3: Extend `dinov3_backbone.py` with cross-view machinery

**Files:**
- Modify: `occany/model/dinov3_backbone.py`

All cross-view code is added **to the existing Stage 1 file** — no new module is created. The work is structured so the file remains a clean linear read: imports → helpers → `GatedQKNormSelfAttention` → `install_gated_qk_norm_on_blocks` → `build_global_rope_sincos` → `DinoV3PerViewBackbone` (extended) → `build_dinov3_per_view` (extended). All additions are gated on `alt_start >= 0`; default-mode behavior is bit-identical to Stage 1.

The full structure after this task is given below; smokes verify the result step-by-step.

- [ ] **Step 3.1: Replace `occany/model/dinov3_backbone.py` with the extended file**

Use the Read tool to inspect the current file first, then rewrite. The complete intended content:

```python
"""DINOv3 ViT-H+/16 backbone wrapper for OccAny+.

Default mode (alt_start = -1, Stage 1 behavior):
    Per-view backbone. Each view is processed independently through an
    unmodified `dinov3.models.vision_transformer.DinoVisionTransformer`;
    the forward calls `vit.get_intermediate_layers(...)` to tap multiple
    blocks. No cross-view machinery, no QK-norm.

Cross-view mode (alt_start >= 0, Stage 2 extension):
    From layer `alt_start` onward, blocks alternate by index parity --
    even-index blocks remain frame-local, odd-index blocks flatten across
    frames for cross-view attention. Gated QK-norm is installed on all
    blocks at and after `alt_start` (gates zero-initialized, so bit-identical
    to default mode at step 0). Global passes use a custom RoPE: identity
    rotation for cls/storage tokens, intra-frame 2D RoPE tiled across
    frames for patch tokens (DA3-style).

The class name `DinoV3PerViewBackbone` is preserved across modes for
backward compatibility with Stage 1 call sites; it is slightly inaccurate
when alt_start >= 0 and should be revisited once cross-view becomes the
default mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from dinov3.layers.attention import SelfAttention
from dinov3.models.vision_transformer import DinoVisionTransformer


# --------------------------------------------------------------------------- #
# Gated QK-norm and installer (used only when alt_start >= 0).                #
# --------------------------------------------------------------------------- #


class GatedQKNormSelfAttention(SelfAttention):
    """SelfAttention + per-head-dim LayerNorm on Q and K, scalar-gated.

    At init (q_gate = k_gate = 0), the attention is bit-identical to the
    parent class because:
        q_out = q + 0 * (LN(q) - q) = q
        k_out = k + 0 * (LN(k) - k) = k

    Training learns to ramp the gates toward 1 (full QK-norm).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        proj_bias: bool,
        mask_k_bias: bool,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        device=None,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        head_dim = dim // num_heads
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-5, device=device)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-5, device=device)
        # Scalar-per-head gates. Start at zero so the norm is a no-op at init.
        self.q_gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1, device=device))
        self.k_gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1, device=device))

    def compute_attention(self, qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
        """Override of SelfAttention.compute_attention with gated QK-norm.

        Original (third_party/dinov3/dinov3/layers/attention.py:106-118):
            q, k, v = unbind(qkv); transpose; [optional rope]; SDPA
        This override inserts the gated norm between the transpose and the
        optional RoPE / SDPA call.
        """
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]   # [B, head, N, head_dim]

        # Gated QK-norm — bit-identical to upstream when q_gate == k_gate == 0.
        q = q + self.q_gate * (self.q_norm(q) - q)
        k = k + self.k_gate * (self.k_norm(k) - k)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


def install_gated_qk_norm_on_blocks(
    vit: DinoVisionTransformer,
    alt_start: int,
) -> None:
    """Replace `block.attn` on `vit.blocks[alt_start:]` with GatedQKNormSelfAttention.

    Weights of `qkv` and `proj` are copied from the existing module so the
    model is bit-identical to pretrained DINOv3 at step 0. The new parameters
    (q_norm, k_norm, q_gate, k_gate) are freshly initialized — gates at 0,
    LayerNorm at standard reset.
    """
    if alt_start < 0:
        return
    if alt_start >= len(vit.blocks):
        raise ValueError(f"alt_start={alt_start} >= n_blocks={len(vit.blocks)}")

    for blk in vit.blocks[alt_start:]:
        old = blk.attn
        if not isinstance(old, SelfAttention):
            raise TypeError(
                f"Expected block.attn to be SelfAttention, got {type(old).__name__}"
            )
        if isinstance(old, GatedQKNormSelfAttention):
            # Idempotent: already installed.
            continue

        dim = old.qkv.in_features
        num_heads = old.num_heads
        qkv_bias = old.qkv.bias is not None
        proj_bias = old.proj.bias is not None
        mask_k_bias = hasattr(old.qkv, "bias_mask")
        device = old.qkv.weight.device

        new = GatedQKNormSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        new.qkv.load_state_dict(old.qkv.state_dict())
        new.proj.load_state_dict(old.proj.state_dict())
        new = new.to(dtype=old.qkv.weight.dtype)
        blk.attn = new


# --------------------------------------------------------------------------- #
# Global-pass RoPE construction (used only on cross-view forward).            #
# --------------------------------------------------------------------------- #


def build_global_rope_sincos(
    rope_embed,
    H_patch: int,
    W_patch: int,
    T: int,
    n_special_per_frame: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (sin, cos) for the flattened cross-view sequence.

    Per-frame layout: [cls(1), storage(n_special_per_frame-1), patches(H_p*W_p)].
    Special-token positions get identity rotation (sin=0, cos=1) so their
    Q/K pass through unchanged. Patch positions get the standard intra-frame
    2D RoPE, tiled identically across all T frames so same-(x, y) patches in
    different frames have identical rotations -> relative-position 0 ->
    content-based cross-view attention.

    DINOv3's RoPE prefix-skip mechanism only handles a contiguous front prefix,
    which doesn't match our interleaved layout — hence the hand-built sin/cos.
    """
    sin_p, cos_p = rope_embed(H=H_patch, W=W_patch)
    sin_p = sin_p.to(device)
    cos_p = cos_p.to(device)

    sin_special = torch.zeros(
        n_special_per_frame, head_dim, device=device, dtype=sin_p.dtype
    )
    cos_special = torch.ones(
        n_special_per_frame, head_dim, device=device, dtype=cos_p.dtype
    )

    sin_frame = torch.cat([sin_special, sin_p], dim=0)
    cos_frame = torch.cat([cos_special, cos_p], dim=0)

    sin_global = sin_frame.repeat(T, 1)
    cos_global = cos_frame.repeat(T, 1)
    return sin_global, cos_global


# --------------------------------------------------------------------------- #
# Backbone wrapper — single class, dual mode.                                 #
# --------------------------------------------------------------------------- #


class DinoV3PerViewBackbone(nn.Module):
    """DINOv3 ViT-H+/16 backbone, optionally with cross-view attention.

    See module docstring for the default-mode vs. cross-view-mode contract.
    """

    def __init__(
        self,
        vit: DinoVisionTransformer,
        out_layers: Iterable[int],
        alt_start: int = -1,
    ):
        super().__init__()
        self.vit = vit
        self.out_layers = tuple(int(idx) for idx in out_layers)
        if len(self.out_layers) == 0:
            raise ValueError("DinoV3PerViewBackbone requires at least one tap layer")
        if max(self.out_layers) >= self.vit.n_blocks:
            raise ValueError(
                f"out_layers={self.out_layers} exceeds n_blocks={self.vit.n_blocks}"
            )
        if alt_start >= self.vit.n_blocks:
            raise ValueError(
                f"alt_start={alt_start} >= n_blocks={self.vit.n_blocks}"
            )
        self.alt_start = int(alt_start)

        # Cross-view mode: install gated QK-norm on the alternating block range.
        # Default mode: no-op (`install_gated_qk_norm_on_blocks` returns early).
        install_gated_qk_norm_on_blocks(self.vit, alt_start=self.alt_start)

    @property
    def n_special_tokens_per_frame(self) -> int:
        # cls (1) + storage (n_storage_tokens). Only used in cross-view mode.
        return 1 + int(self.vit.n_storage_tokens)

    def forward_multiview(
        self, images: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run DINOv3 ViT and return tapped features.

        Args:
            images: (B, T, 3, H, W). H, W must be multiples of patch_size (16).

        Returns:
            List of length len(out_layers). Each entry is a tuple
            `(patch_feat, cls_token)` where:
              - patch_feat: (B, T, N_patch, embed_dim) with N_patch = (H//16)*(W//16)
              - cls_token: (B, T, embed_dim)
        """
        if images.ndim != 5:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(images.shape)}")
        if self.alt_start < 0:
            return self._forward_per_view(images)
        return self._forward_cross_view(images)

    def _forward_per_view(
        self, images: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Vanilla per-view forward (Stage 1 path)."""
        B, T, C, H, W = images.shape
        flat = images.reshape(B * T, C, H, W)

        taps = self.vit.get_intermediate_layers(
            flat,
            n=self.out_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for patch_tokens, cls_token in taps:
            N_patch = patch_tokens.shape[-2]
            E = patch_tokens.shape[-1]
            results.append(
                (
                    patch_tokens.reshape(B, T, N_patch, E),
                    cls_token.reshape(B, T, E),
                )
            )
        return results

    def _forward_cross_view(
        self, images: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Cross-view forward with local<->global parity alternation."""
        B, T, C, H, W = images.shape
        if H % self.vit.patch_size != 0 or W % self.vit.patch_size != 0:
            raise ValueError(
                f"H={H}, W={W} not divisible by patch_size={self.vit.patch_size}"
            )

        flat = images.reshape(B * T, C, H, W)
        x, (H_p, W_p) = self.vit.prepare_tokens_with_masks(flat)
        # x: (B*T, N, D) where N = 1 + n_storage + H_p*W_p
        N = x.shape[1]
        D = x.shape[2]
        n_special = self.n_special_tokens_per_frame
        N_patch = H_p * W_p
        assert N == n_special + N_patch, (N, n_special, N_patch)

        head_dim = D // self.vit.num_heads
        device = x.device

        # Per-frame RoPE for local passes (both pre-alt_start and even-index post).
        # block.attn.apply_rope uses prefix = N - sin.shape[-2] = n_special, so cls/storage
        # skip rotation and patches are rotated. Matches upstream DINOv3.
        sin_local, cos_local = self.vit.rope_embed(H=H_p, W=W_p)
        rope_local = (sin_local.to(device), cos_local.to(device))

        # Global RoPE for cross-view passes — identity for cls/storage, tiled
        # intra-frame 2D RoPE for patches. Built once per forward (T may vary).
        rope_global = build_global_rope_sincos(
            rope_embed=self.vit.rope_embed,
            H_patch=H_p,
            W_patch=W_p,
            T=T,
            n_special_per_frame=n_special,
            head_dim=head_dim,
            device=device,
        )

        out_layer_set = set(self.out_layers)
        tapped: dict[int, torch.Tensor] = {}

        for i, blk in enumerate(self.vit.blocks):
            if i < self.alt_start or (i % 2 == 0):
                # Frame-local pass: (B*T, N, D).
                x = blk(x, rope_local)
            else:
                # Global cross-view pass: (B, T*N, D).
                x_g = x.reshape(B, T * N, D)
                x_g = blk(x_g, rope_global)
                x = x_g.reshape(B * T, N, D)

            if i in out_layer_set:
                tapped[i] = x

        # Apply final norm (matches `get_intermediate_layers(norm=True)`).
        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for tap_idx in self.out_layers:
            normed = self.vit.norm(tapped[tap_idx])               # (B*T, N, D)
            cls_tok = normed[:, 0]                                  # (B*T, D)
            patch = normed[:, n_special:]                           # (B*T, N_patch, D)
            results.append(
                (
                    patch.reshape(B, T, N_patch, D),
                    cls_tok.reshape(B, T, D),
                )
            )
        return results

    def get_backbone_metadata(self) -> dict:
        embed_dim = int(self.vit.embed_dim)
        name = (
            "dinov3_vith16plus_cross_view" if self.alt_start >= 0 else "dinov3_vith16plus"
        )
        return {
            "name": name,
            "token_dim": embed_dim,
            "feature_dim": embed_dim,
            "out_layers": tuple(self.out_layers),
            "alt_start": int(self.alt_start),
            "total_layers": int(self.vit.n_blocks),
            "num_heads": int(self.vit.num_heads),
            "cat_token": False,
        }


# --------------------------------------------------------------------------- #
# Factory.                                                                    #
# --------------------------------------------------------------------------- #


def build_dinov3_per_view(
    yaml_path: str | Path,
    weights_path: str | Path | None = None,
    alt_start_override: Optional[int] = None,
) -> DinoV3PerViewBackbone:
    """Build a `DinoV3PerViewBackbone` from a YAML config + optional checkpoint.

    The `alt_start` field is read from the YAML if present (default mode if absent).
    `alt_start_override`, when provided, takes precedence over the YAML value.
    The checkpoint is loaded with strict=True against the pristine DINOv3 state
    dict (QK-norm params are added by the class `__init__`, AFTER load).
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    arch_kwargs = dict(cfg["arch"])
    out_layers = tuple(int(idx) for idx in cfg["out_layers"])

    if alt_start_override is not None:
        alt_start = int(alt_start_override)
    else:
        alt_start = int(cfg.get("alt_start", -1))

    vit = DinoVisionTransformer(**arch_kwargs)

    if weights_path is not None:
        state_dict = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )
        # strict=True against the pristine DINOv3 state dict — QK-norm params are
        # added AFTER load (inside DinoV3PerViewBackbone.__init__).
        missing, unexpected = vit.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"DINOv3 strict load failed: "
                f"missing={missing[:5]}... unexpected={unexpected[:5]}..."
            )

    return DinoV3PerViewBackbone(vit, out_layers=out_layers, alt_start=alt_start)
```

- [ ] **Step 3.2: Default-mode regression smoke — Stage 1 behavior must be bit-identical**

Critical sanity check: the in-place edit must not perturb Stage 1 behavior when `alt_start=-1` (the default).

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
import torch
from occany.model.dinov3_backbone import build_dinov3_per_view

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
bb = build_dinov3_per_view(\"occany/configs/dinov3/occany_dinov3_vith16plus.yaml\", weights_path=ckpt).eval().cuda()
md = bb.get_backbone_metadata()
assert md[\"alt_start\"] == -1, md
assert md[\"name\"] == \"dinov3_vith16plus\", md[\"name\"]
print(\"metadata default mode:\", md)

# Verify NO block has GatedQKNormSelfAttention in default mode.
from occany.model.dinov3_backbone import GatedQKNormSelfAttention
for i, blk in enumerate(bb.vit.blocks):
    assert not isinstance(blk.attn, GatedQKNormSelfAttention), f\"block {i} unexpectedly patched\"

x = torch.randn(2, 4, 3, 176, 512, device=\"cuda\")
with torch.no_grad(), torch.autocast(\"cuda\", dtype=torch.bfloat16):
    feats = bb.forward_multiview(x)
print(\"num taps:\", len(feats))
for i, (f, cls) in enumerate(feats):
    print(f\"tap{i}: feat={tuple(f.shape)} cls={tuple(cls.shape)}\")
assert feats[-1][0].shape == (2, 4, 352, 1280)
print(\"default-mode regression OK\")
"'
```

Expected: `alt_start=-1`, no block uses `GatedQKNormSelfAttention`, all four tap shapes correct, `default-mode regression OK`. If any assert fires, Task 3's in-place edit broke Stage 1 — revert and reconcile.

- [ ] **Step 3.3: CPU bit-identical smoke — gates=0 produces identical features to default mode (on the same VIT instance)**

```bash
cd /home/acao/code/OccAny && PYTHONPATH=third_party/dinov3 python -c "
import torch
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.model.dinov3_backbone import (
    build_dinov3_per_view,
    install_gated_qk_norm_on_blocks,
    GatedQKNormSelfAttention,
)

# Build default-mode VIT, forward, then install QK-norm in place and re-forward.
# This isolates the QK-norm install from any other init differences.
bb = build_dinov3_per_view('occany/configs/dinov3/occany_dinov3_vith16plus.yaml', weights_path=None)
bb.eval()

x = torch.randn(1, 3, 32, 32)  # tiny CPU grid
with torch.no_grad():
    f_before = bb.vit.get_intermediate_layers(x, n=[31], reshape=False, return_class_token=False, norm=True)[0]

install_gated_qk_norm_on_blocks(bb.vit, alt_start=11)
for i in range(11, 32):
    assert isinstance(bb.vit.blocks[i].attn, GatedQKNormSelfAttention)

with torch.no_grad():
    f_after = bb.vit.get_intermediate_layers(x, n=[31], reshape=False, return_class_token=False, norm=True)[0]

diff = (f_before - f_after).abs().max().item()
print(f'max abs diff after QK-norm install (gates=0): {diff:.2e}')
assert diff < 1e-6, diff
print('QK-norm install bit-identical at init')
"
```

Expected: `diff < 1e-6`, `QK-norm install bit-identical at init`.

- [ ] **Step 3.4: GPU cross-view forward + strict load smoke**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
import torch
from occany.model.dinov3_backbone import build_dinov3_per_view

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
# alt_start read from the cross-view YAML (= 11).
bb = build_dinov3_per_view(\"occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml\", weights_path=ckpt)
bb = bb.eval().cuda()
md = bb.get_backbone_metadata()
assert md[\"alt_start\"] == 11, md
assert md[\"name\"] == \"dinov3_vith16plus_cross_view\"
print(\"strict load + metadata OK:\", md)

x = torch.randn(2, 4, 3, 176, 512, device=\"cuda\")
with torch.no_grad(), torch.autocast(\"cuda\", dtype=torch.bfloat16):
    feats = bb.forward_multiview(x)
print(\"num taps:\", len(feats))
for i, (f, cls) in enumerate(feats):
    print(f\"tap{i}: feat={tuple(f.shape)} cls={tuple(cls.shape)}\")
assert len(feats) == 4
assert feats[-1][0].shape == (2, 4, 352, 1280), feats[-1][0].shape

assert feats[-1][1].shape == (2, 4, 1280), feats[-1][1].shape
print(\"GPU cross-view forward OK\")
"'
```

Expected: metadata showing `alt_start: 11`, tap shapes `(2, 4, 352, 1280)` for the last tap, `GPU cross-view forward OK`.

- [ ] **Step 3.5: GPU init-equivalence smoke — cross-view (gates=0, T=1) ≈ default mode**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
import torch
from occany.model.dinov3_backbone import build_dinov3_per_view

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
bb_def = build_dinov3_per_view(\"occany/configs/dinov3/occany_dinov3_vith16plus.yaml\", weights_path=ckpt).eval().cuda()
bb_cv  = build_dinov3_per_view(\"occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml\", weights_path=ckpt).eval().cuda()

# T=1: the global block is invoked but T*N == N. Gates=0 -> identity QK-norm.
# Last tap is layer 31 for both configs.
torch.manual_seed(0)
x = torch.randn(2, 1, 3, 176, 512, device=\"cuda\")

with torch.no_grad(), torch.autocast(\"cuda\", dtype=torch.bfloat16):
    f_def = bb_def.forward_multiview(x)[-1][0]
    f_cv  = bb_cv.forward_multiview(x)[-1][0]

diff = (f_def - f_cv).abs().max().item()
print(f\"max abs diff (default vs cross-view, T=1, gates=0): {diff:.2e}\")
assert diff < 5e-2, diff
print(\"init-equivalence OK\")
"'
```

Expected: `diff ~ 1e-2` (bf16 nondeterminism), `init-equivalence OK`. If diff is large (> 0.1), the cross-view forward path is broken.

- [ ] **Step 3.6: GPU grad-flow smoke — every QK-norm gate on blocks 11..31 receives a grad**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
import torch
from occany.model.dinov3_backbone import build_dinov3_per_view

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
bb = build_dinov3_per_view(\"occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml\", weights_path=ckpt).cuda()
bb.train()

x = torch.randn(1, 3, 3, 176, 512, device=\"cuda\")
with torch.autocast(\"cuda\", dtype=torch.bfloat16):
    feats = bb.forward_multiview(x)
loss = sum(f.float().mean() + cls.float().mean() for f, cls in feats)
loss.backward()

missing = []
for i in range(11, 32):
    attn = bb.vit.blocks[i].attn
    for name in (\"q_gate\", \"k_gate\", \"q_norm.weight\", \"k_norm.weight\"):
        mod, *rest = name.split(\".\")
        param = getattr(attn, mod)
        for r in rest:
            param = getattr(param, r)
        if param.grad is None:
            missing.append(f\"blocks[{i}].attn.{name}\")
if missing:
    print(\"MISSING GRADS:\", missing[:10])
    raise SystemExit(1)
print(\"grad-flow OK (all QK-norm params on blocks 11..31 received grads)\")
"'
```

Expected: `grad-flow OK (all QK-norm params on blocks 11..31 received grads)`.

- [ ] **Step 3.7: Commit**

```bash
git add occany/model/dinov3_backbone.py
git commit -m "feat(occany): extend DinoV3PerViewBackbone with optional cross-view mode (alt_start>=0)"
```

---

## Task 4: Plumb `cross_view_alt_start` through `Dinov3Wrapper`

**Files:**
- Modify: `occany/model/model_dinov3.py`

Additive change. When `cross_view_alt_start == -1` (default), Stage 1 behavior is preserved bit-for-bit. When `>= 0`, the value is forwarded to `build_dinov3_per_view` via `alt_start_override`.

- [ ] **Step 4.1: Edit `occany/model/model_dinov3.py`**

Extend the docstring and `__init__` signature; the rest of the wrapper is unchanged.

```python
"""Recon-only DINOv3 wrapper for OccAny+.

Default mode (cross_view_alt_start = -1, Stage 1 behavior):
    Backbone runs in vanilla per-view mode (no cross-view, no QK-norm).

Cross-view mode (cross_view_alt_start >= 0, Stage 2 extension):
    Backbone runs in cross-view mode -- local<->global parity alternation
    starting at `cross_view_alt_start`, with gated QK-norm on blocks at and
    after that layer (gates zero-initialized, so bit-identical to default
    mode at step 0).

Both modes go through the same `DinoV3PerViewBackbone` class and the same
`build_dinov3_per_view` factory; the mode is selected by the `alt_start`
kwarg / YAML field. See `occany/model/dinov3_backbone.py` for details.

No SAM3 head, no gen mode, no aux teacher path in either mode.

Public surface mirrors a subset of `occany.model.model_da3.DA3Wrapper`:
- `forward(images, **kwargs)` -> `inference_batch(images, **kwargs)`
- `inference_batch` returns the same dict keys as DA3Wrapper's recon path
- `get_backbone_metadata()` returns a dict with the same shape

Differences from DA3Wrapper:
- `self.backbone` / `self.head` live directly on the wrapper (no `self.model.` nesting)
- `c2w`, `intrinsics`, `sam_feats` are always None
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from depth_anything_3.model.dualdpt import DualDPT

from occany.model.dinov3_backbone import build_dinov3_per_view


class Dinov3Wrapper(nn.Module):
    def __init__(
        self,
        config_path: str | Path,
        weights_path: Optional[str | Path] = None,
        cross_view_alt_start: int = -1,
    ):
        super().__init__()
        # Forward cross_view_alt_start to the factory only when explicitly enabled.
        # When -1, alt_start is read from the YAML if present, else defaults to -1
        # (= Stage 1 behavior, bit-identical to the pre-Stage-2 codepath).
        alt_start_override = (
            cross_view_alt_start if cross_view_alt_start >= 0 else None
        )
        self.backbone = build_dinov3_per_view(
            config_path,
            weights_path=weights_path,
            alt_start_override=alt_start_override,
        )
        meta = self.backbone.get_backbone_metadata()
        self.head = DualDPT(
            dim_in=meta["feature_dim"],
            patch_size=16,
            output_dim=2,
            features=256,
            out_channels=(256, 512, 1024, 1024),
            head_names=("depth", "ray"),
        )

    def get_backbone_metadata(self) -> dict:
        return self.backbone.get_backbone_metadata()

    def forward(self, images: torch.Tensor, **kwargs):
        return self.inference_batch(images, **kwargs)

    def inference_batch(self, images: torch.Tensor, **_unused):
        b, t, c, h, w = images.shape
        feats = self.backbone.forward_multiview(images)
        return self._process_depth_output(feats=feats, h=h, w=w)

    def _process_depth_output(self, feats, h, w):
        output = self.head(feats, h, w, patch_start_idx=0)

        default_scale = 20

        depth = output["depth"] * default_scale
        depth_conf = output["depth_conf"]

        ray = output["ray"]
        ray_conf = output["ray_conf"]
        ray[..., 3:] = ray[..., 3:] * default_scale

        pointmap = depth.unsqueeze(-1) * ray[..., :3] + ray[..., 3:]

        return {
            "pointmap": pointmap,
            "depth": depth,
            "depth_conf": depth_conf,
            "ray": ray,
            "ray_conf": ray_conf,
            "c2w": None,
            "intrinsics": None,
            "sam_feats": None,
        }
```

- [ ] **Step 4.2: Smoke — default behavior preserved**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.model.model_dinov3 import Dinov3Wrapper

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
# Default kwarg: cross_view_alt_start = -1 -> Stage 1 mode.
m = Dinov3Wrapper(\"occany/configs/dinov3/occany_dinov3_vith16plus.yaml\", weights_path=ckpt)
md = m.get_backbone_metadata()
assert md[\"name\"] == \"dinov3_vith16plus\", md[\"name\"]
assert md[\"alt_start\"] == -1, md[\"alt_start\"]
print(\"default (Stage 1) mode OK\")
"'
```

Expected: `default (Stage 1) mode OK`.

- [ ] **Step 4.3: Smoke — cross-view mode forward + backward**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
import torch
from occany.model.model_dinov3 import Dinov3Wrapper

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
m = Dinov3Wrapper(
    \"occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml\",
    weights_path=ckpt,
    cross_view_alt_start=11,
).cuda()
m.train()

md = m.get_backbone_metadata()
assert md[\"name\"] == \"dinov3_vith16plus_cross_view\"
assert md[\"alt_start\"] == 11
print(\"backbone metadata:\", md)

x = torch.randn(1, 3, 3, 176, 512, device=\"cuda\")
with torch.autocast(\"cuda\", dtype=torch.bfloat16):
    out = m(x)
print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in out.items()})

loss = sum(v.float().mean() for k, v in out.items() if torch.is_tensor(v))
loss.backward()
print(\"backward OK; loss =\", float(loss))
"'
```

Expected: backbone metadata showing `alt_start=11`, every recon key present, `backward OK`.

- [ ] **Step 4.4: Commit**

```bash
git add occany/model/model_dinov3.py
git commit -m "feat(occany): plumb cross_view_alt_start through Dinov3Wrapper"
```

---

## Task 5: Add `--cross_view_alt_start` and range-syntax `--fine_tune_layers` to `training_dinov3.py`

**Files:**
- Modify: `occany/training_dinov3.py`

Three small edits: an argparse flag, a model-construction kwarg, and a tweak to the existing `--fine_tune_layers` parser so it accepts range syntax (`11-31`) in addition to comma lists. Default `-1` preserves Stage 1 behavior bit-for-bit; the parser tweak is fully backwards-compatible with the existing comma form.

- [ ] **Step 5.1: Add the argparse flag**

Find the block in `occany/training_dinov3.py` where `--dinov3_config` and `--dinov3_weights` are defined. Add immediately after them:

```python
    parser.add_argument(
        "--cross_view_alt_start", type=int, default=-1,
        help=(
            "DINOv3 layer index at which to start local<->global cross-view "
            "alternation. -1 (default) disables cross-view (Stage 1 behavior). "
            ">= 0 enables it (cross-view extension); typically 11 for ViT-H+/16."
        ),
    )
```

- [ ] **Step 5.2: Extend `--fine_tune_layers` to accept range syntax**

Find the existing block at `occany/training_dinov3.py` that parses `args.fine_tune_layers`:

```python
    # Before:
    fine_tune_layers = None
    if args.fine_tune_layers is not None:
        fine_tune_layers = [int(x.strip()) for x in args.fine_tune_layers.split(',')]
```

Replace with a small helper that accepts either form — `"11,12,13"`, `"11-31"`, or a mix like `"0,5,11-13"` — and returns a sorted, deduplicated `list[int]`:

```python
    # After:
    def _parse_layer_spec(spec: str) -> list[int]:
        out: set[int] = set()
        for tok in spec.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if '-' in tok:
                lo, hi = tok.split('-', 1)
                lo_i, hi_i = int(lo), int(hi)
                if lo_i > hi_i:
                    raise ValueError(f"invalid range {tok!r}: lo > hi")
                out.update(range(lo_i, hi_i + 1))
            else:
                out.add(int(tok))
        return sorted(out)

    fine_tune_layers = None
    if args.fine_tune_layers is not None:
        fine_tune_layers = _parse_layer_spec(args.fine_tune_layers)
```

Also update the `--fine_tune_layers` `help=` string to mention the range form (e.g., "Comma-separated indices and/or `lo-hi` ranges, e.g. `26,27,28,29,30,31` or `11-31`.").

Sanity check from the shell:

```bash
cd /home/acao/code/OccAny && python -c "
def _parse_layer_spec(spec):
    out = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok: continue
        if '-' in tok:
            lo, hi = tok.split('-', 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(tok))
    return sorted(out)
assert _parse_layer_spec('11-31') == list(range(11, 32))
assert _parse_layer_spec('26,27,28,29,30,31') == [26, 27, 28, 29, 30, 31]
assert _parse_layer_spec('0,5,11-13') == [0, 5, 11, 12, 13]
print('range-spec parser OK')
"
```

Expected: `range-spec parser OK`.

- [ ] **Step 5.3: Plumb the flag into `Dinov3Wrapper`**

Find the `Dinov3Wrapper(...)` call (look for `args.dinov3_config`). Replace:

```python
    # Before:
    model = Dinov3Wrapper(args.dinov3_config, weights_path=args.dinov3_weights)

    # After:
    model = Dinov3Wrapper(
        args.dinov3_config,
        weights_path=args.dinov3_weights,
        cross_view_alt_start=args.cross_view_alt_start,
    )
```

- [ ] **Step 5.4: Argparse smoke**

```bash
cd /home/acao/code/OccAny && PYTHONPATH=third_party python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.training_dinov3 import get_args_parser
ap = get_args_parser()
flag_set = {a.option_strings[0] for a in ap._actions if a.option_strings}
assert '--cross_view_alt_start' in flag_set
ns = ap.parse_args([
    '--dinov3_config', 'x', '--dinov3_weights', 'y',
    '--train_dataset', '1 @ X()', '--test_dataset', '1 @ Y()',
    '--output_dir', '/tmp/x', '--training_objective', 'pointmap_depth_ray',
])
assert ns.cross_view_alt_start == -1, ns.cross_view_alt_start
print('argparse OK')
"
```

Expected: `argparse OK`.

- [ ] **Step 5.5: Commit**

```bash
git add occany/training_dinov3.py
git commit -m "feat(occany): add --cross_view_alt_start + range-syntax --fine_tune_layers"
```

---

## Task 6: Standalone cross-view smoke script

**Files:**
- Create: `test_dinov3_cross_view.py`

CLI sanity-check that the cross-view path produces sensible features and that training-mode forward+backward runs end-to-end with the real DINOv3 weights. Complements `test_dinov3_pca.py`. Uses the same `build_dinov3_per_view` factory in both modes.

- [ ] **Step 6.1: Write `test_dinov3_cross_view.py`**

```python
"""Standalone cross-view smoke for DinoV3PerViewBackbone in cross-view mode.

Validates:
1. Loads DINOv3 ViT-H+/16 from a pristine pretrained checkpoint in BOTH default
   mode (alt_start=-1) and cross-view mode (alt_start=11).
2. Cross-view forward runs at T = 1, 2, 4 views.
3. With gates = 0, the T=1 output of cross-view mode matches default mode at
   machine precision (modulo bf16 nondeterminism).
4. Backward runs without exception and grads land on QK-norm gates.

Usage:
    python test_dinov3_cross_view.py \\
        --weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
"""
from __future__ import annotations

import argparse

import torch

from occany.utils.runtime_paths import prepend_vendored_import_paths

prepend_vendored_import_paths()

from occany.model.dinov3_backbone import build_dinov3_per_view  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--default_config", type=str, default="occany/configs/dinov3/occany_dinov3_vith16plus.yaml")
    p.add_argument("--cross_view_config", type=str, default="occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--T_values", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--height", type=int, default=176)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = get_args_parser().parse_args()
    torch.manual_seed(args.seed)

    bb_def = build_dinov3_per_view(args.default_config, weights_path=args.weights).eval().to(args.device)
    bb_cv  = build_dinov3_per_view(args.cross_view_config, weights_path=args.weights).eval().to(args.device)

    md_def = bb_def.get_backbone_metadata()
    md_cv = bb_cv.get_backbone_metadata()
    print("default-mode metadata:", md_def)
    print("cross-view metadata:", md_cv)
    assert md_def["alt_start"] == -1, md_def
    assert md_cv["alt_start"] >= 0, md_cv

    # --- Forward smoke at multiple T values (cross-view mode only) ---
    for T in args.T_values:
        x = torch.randn(1, T, 3, args.height, args.width, device=args.device)
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
            feats = bb_cv.forward_multiview(x)
        for tap_idx, (f, cls) in enumerate(feats):
            assert f.shape[1] == T, (f.shape, T)
            assert cls.shape[1] == T, (cls.shape, T)
        print(f"forward T={T} OK; last-tap feat shape {tuple(feats[-1][0].shape)}")

    # --- Init equivalence (T = 1, gates = 0) ---
    x1 = torch.randn(2, 1, 3, args.height, args.width, device=args.device)
    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
        f_def = bb_def.forward_multiview(x1)[-1][0]
        f_cv  = bb_cv.forward_multiview(x1)[-1][0]
    diff = (f_def - f_cv).abs().max().item()
    print(f"init-equivalence max abs diff (T=1, gates=0): {diff:.2e}")
    assert diff < 5e-2, f"expected near-identical features at init; got {diff}"

    # --- Backward smoke + QK-norm gate grad check ---
    bb_cv.train()
    x = torch.randn(1, 3, 3, args.height, args.width, device=args.device)
    with torch.autocast(args.device, dtype=torch.bfloat16):
        feats = bb_cv.forward_multiview(x)
    loss = sum(f.float().mean() + cls.float().mean() for f, cls in feats)
    loss.backward()

    alt_start = md_cv["alt_start"]
    n_blocks = md_cv["total_layers"]
    missing = []
    for i in range(alt_start, n_blocks):
        attn = bb_cv.vit.blocks[i].attn
        for name, param in [
            ("q_gate", attn.q_gate),
            ("k_gate", attn.k_gate),
            ("q_norm.weight", attn.q_norm.weight),
            ("k_norm.weight", attn.k_norm.weight),
        ]:
            if param.grad is None:
                missing.append(f"blocks[{i}].attn.{name}")

    if missing:
        print("MISSING GRADS:", missing[:10])
        raise SystemExit(1)
    print(f"grad-flow OK (all QK-norm params on blocks {alt_start}..{n_blocks - 1} received grads)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run on Karolina**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python test_dinov3_cross_view.py \
    --weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
```

Expected, in order:
- `default-mode metadata: {...alt_start: -1, ...}`
- `cross-view metadata: {...alt_start: 11, ...}`
- `forward T=1 OK; ...`
- `forward T=2 OK; ...`
- `forward T=4 OK; ...`
- `init-equivalence max abs diff ...: < 5e-2`
- `grad-flow OK (all QK-norm params on blocks 11..31 received grads)`
- `ALL CHECKS PASSED`

If any check fails, stop and fix the underlying issue in `dinov3_backbone.py` before proceeding.

- [ ] **Step 6.3: Commit**

```bash
git add test_dinov3_cross_view.py
git commit -m "feat(occany): add DINOv3 cross-view standalone smoke"
```

---

## Task 7: Make Stage 1's training wrapper env-overridable

**Files:**
- Modify: `sh/train_occany_plus_recon_dinov3_vith16plus.sh`

The four cross-view-relevant knobs need to become env-overridable so the **same** wrapper produces Stage 1 behavior (no env set) or cross-view behavior (env set). All defaults match the current Stage 1 values exactly — running the wrapper with no env overrides must produce a bit-identical training launch to the current state.

Reasoning for the widened fine-tune default applied via env (not by changing the wrapper's default): every block from `alt_start` onward is modified (QK-norm installed, even-index blocks see new local state distribution, odd-index blocks do cross-view). All 21 such blocks need to learn the new task. The 11 pretrained blocks `0..10` stay frozen. Hence the cross-view launch must set `FINE_TUNE_LAYERS=11-31`.

- [ ] **Step 7.1: Edit `sh/train_occany_plus_recon_dinov3_vith16plus.sh`**

Make the following four edits.

**Edit 1: `EXP_NAME` env-overridable.** Change:

```bash
export EXP_NAME="occany_plus_recon_dinov3_vith16plus"
```

to:

```bash
: ${EXP_NAME:=occany_plus_recon_dinov3_vith16plus}
export EXP_NAME
```

**Edit 2: `DINOV3_CONFIG` env-overridable.** Find the line:

```bash
    --dinov3_config occany/configs/dinov3/occany_dinov3_vith16plus.yaml \
```

Replace with:

```bash
    --dinov3_config "${DINOV3_CONFIG:-occany/configs/dinov3/occany_dinov3_vith16plus.yaml}" \
```

**Edit 3: `FINE_TUNE_LAYERS` env-overridable.** Find the line:

```bash
    --training_objective pointmap_depth_ray --fine_tune_layers 26,27,28,29,30,31 \
```

Replace with:

```bash
    --training_objective pointmap_depth_ray --fine_tune_layers "${FINE_TUNE_LAYERS:-26,27,28,29,30,31}" \
```

**Edit 4: Add `--cross_view_alt_start` to the python invocation.** Find the block of `--lambda_*` flags at the end of the `$CMD` line. After `--lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0`, append:

```bash
     --cross_view_alt_start "${CROSS_VIEW_ALT_START:--1}"
```

(Mind the trailing-backslash chain — the new flag must be on the last line, with the previous line keeping its `\` continuation.)

- [ ] **Step 7.2: Verify default values reproduce Stage 1 behavior bit-for-bit**

```bash
cd /home/acao/code/OccAny && bash -n sh/train_occany_plus_recon_dinov3_vith16plus.sh && echo "SYNTAX_OK"
cd /home/acao/code/OccAny && grep -n -E '^\s*:\s*\$\{EXP_NAME|DINOV3_CONFIG:-|FINE_TUNE_LAYERS:-|CROSS_VIEW_ALT_START:-' sh/train_occany_plus_recon_dinov3_vith16plus.sh
```

Expected: `SYNTAX_OK`, plus four grep matches showing the new env-overridable patterns.

- [ ] **Step 7.3: Dry-run on Karolina to confirm no broken arg construction**

This is a no-launch syntax sanity — we replace `$CMD` with `echo` (via a tmp file) and inspect the assembled python CLI in both modes.

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  WRAPPER=sh/train_occany_plus_recon_dinov3_vith16plus.sh && \
  TMP=$(mktemp /tmp/dryrun_XXXXXX.sh) && \
  sed "s|\$CMD |echo CMD |g" "$WRAPPER" > "$TMP" && chmod +x "$TMP" && \
  echo === DEFAULT MODE === && \
    NUM_NODE=1 NUM_GPU_PER_NODE=1 BATCH_SIZE=1 EFFECTIVE_BATCH_SIZE=1 N_WORKERS=2 EPOCHS=1 \
    bash "$TMP" 2>&1 | head -60 && \
  echo === CROSS-VIEW MODE === && \
    NUM_NODE=1 NUM_GPU_PER_NODE=1 BATCH_SIZE=1 EFFECTIVE_BATCH_SIZE=1 N_WORKERS=2 EPOCHS=1 \
    EXP_NAME=occany_plus_recon_dinov3_vith16plus_cross_view \
    CROSS_VIEW_ALT_START=11 \
    FINE_TUNE_LAYERS=11-31 \
    DINOV3_CONFIG=occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml \
    bash "$TMP" 2>&1 | head -60 && \
  rm "$TMP"'
```

In the DEFAULT MODE block look for: `--dinov3_config occany/configs/dinov3/occany_dinov3_vith16plus.yaml`, `--fine_tune_layers 26,27,28,29,30,31`, `--cross_view_alt_start -1`.

In the CROSS-VIEW MODE block look for: `--dinov3_config occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml`, `--fine_tune_layers 11-31`, `--cross_view_alt_start 11`.

- [ ] **Step 7.4: Commit**

```bash
git add sh/train_occany_plus_recon_dinov3_vith16plus.sh
git commit -m "feat(sh): make DINOv3 training wrapper env-overridable for cross-view"
```

---

## Task 8: End-to-end single-GPU cross-view smoke on Karolina

**Files:** none.

Submit a tiny 1-GPU job for 1–2 iterations to confirm:

1. Forward + backward run end-to-end with the real DINOv3 weights + cross-view path.
2. DDP wraps the model without `RuntimeError: marked ready twice` or `parameter not used in producing loss` failures.
3. Loss values are finite for the first step.
4. The per-step log lines include the expected recon keys: `loss_pointmap_lidar`, `loss_pointmap_pseudo`, `depth_loss_depth_lidar`, `depth_loss_depth_pseudo`, `loss_raymap`.
5. VRAM is in budget. Cross-view adds ~1.4% sequence length on global passes but trains 21/32 of the backbone instead of 6/32 — significant grad memory increase.

Use the `karolina-job` skill for submission. The launch reuses Stage 1's wrapper with env overrides — **no new shell script needed**.

- [ ] **Step 8.1: Submit a 1-node 1-GPU smoke (qgpu_exp, ~30 min)**

SLURM's `--export=ALL,KEY=VAL` has comma-parsing trouble when `VAL` itself contains commas. The range form `FINE_TUNE_LAYERS=11-31` avoids that entirely, but several other env vars may still contain commas. Use an inline `export` block inside `--wrap` to keep all overrides comma-safe regardless of form:

```bash
ssh karolina 'cd ~/OccAny && \
  sbatch --partition=qgpu_exp --time=00:30:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
    --job-name=dinov3_cross_view_smoke \
    --output=slurm/output/dinov3_cross_view_smoke_%j.out \
    --error=slurm/output/dinov3_cross_view_smoke_%j.err \
    --wrap="eval \"\$(conda shell.bash hook)\" && conda activate occany && \
            export NUM_NODE=1 NUM_GPU_PER_NODE=1 BATCH_SIZE=1 EFFECTIVE_BATCH_SIZE=1 N_WORKERS=2 EPOCHS=1 && \
            export EXP_NAME=occany_plus_recon_dinov3_vith16plus_cross_view_smoke && \
            export CROSS_VIEW_ALT_START=11 && \
            export FINE_TUNE_LAYERS=11-31 && \
            export DINOV3_CONFIG=occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml && \
            bash sh/train_occany_plus_recon_dinov3_vith16plus.sh"'
```

- [ ] **Step 8.2: Wait + tail the smoke output**

Use the `karolina-job` skill to tail. Look for, in order:

1. Successful argparse + dataset construction.
2. `Dinov3Wrapper` constructed with `cross_view_alt_start=11` (backbone metadata log shows `name: dinov3_vith16plus_cross_view`, `alt_start: 11`).
3. First training-step log line with finite `loss_*` values and the expected key names.
4. No `marked ready twice`, `parameter not used in producing loss`, or `Found dtype Float but expected ...` errors.
5. Memory utilization < 80% of single-GPU VRAM (40 GiB on A100, 80 GiB on H100).

- [ ] **Step 8.3: DDP `find_unused_parameters` audit**

The QK-norm gates on **odd-index blocks past alt_start that fall on global passes** receive grads through the global-pass forward; **even-index blocks past alt_start** receive them through the local-pass forward. Both fire every step, so all 21 blocks' QK-norm params should always receive grads.

Decision matrix:
- All unused params are QK-norm on blocks `alt_start..n_blocks-1` → forward bug (a block isn't being visited); investigate `_forward_cross_view` routing in `dinov3_backbone.py`.
- Unused params are inside the frozen pre-`alt_start` blocks (0..10) and outside `--fine_tune_layers` — expected (they're frozen). Keep `find_unused_parameters=True`.
- Unused params are in `model.head` — wiring bug in `_process_depth_output`; the cross-view path might be returning a tap that's not consumed.

- [ ] **Step 8.4: Stage 1 regression smoke (recommended)**

Resubmit the same wrapper with **no env overrides** for a single step to confirm it still produces Stage 1 behavior bit-for-bit:

```bash
ssh karolina 'cd ~/OccAny && \
  sbatch --partition=qgpu_exp --time=00:30:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
    --job-name=dinov3_stage1_regression \
    --output=slurm/output/dinov3_stage1_regression_%j.out \
    --error=slurm/output/dinov3_stage1_regression_%j.err \
    --wrap="eval \"\$(conda shell.bash hook)\" && conda activate occany && \
            export NUM_NODE=1 NUM_GPU_PER_NODE=1 BATCH_SIZE=1 EFFECTIVE_BATCH_SIZE=1 N_WORKERS=2 EPOCHS=1 && \
            bash sh/train_occany_plus_recon_dinov3_vith16plus.sh"'
```

Check the log for: `--cross_view_alt_start -1`, backbone name `dinov3_vith16plus` (no `_cross_view`), `--fine_tune_layers 26,27,28,29,30,31`, default `--dinov3_config occany/configs/dinov3/occany_dinov3_vith16plus.yaml`. If anything differs, Task 7 broke Stage 1 defaults — revert and reconcile.

- [ ] **Step 8.5: Quick eval-time sanity check**

After the smoke job completes, glance at the `--fixed_eval_set` eval-time metrics in the log. They will be much worse than Stage 1's first-eval values (cross-view starts from random head + untrained gates; it needs more steps to catch up). The check is **not NaN/inf** and not absurd (`depth_loss_depth_lidar > 1e3`).

- [ ] **Step 8.6: Commit any fixes from 8.2/8.3, then proceed**

```bash
git add <fixed files>
git commit -m "fix(occany): <specific issue from cross-view smoke>"
```

---

## Task 9: Full multi-node Karolina launch (cross-view)

**Files:** none.

Once the single-GPU smoke is clean, submit a production 4-node × 8-GPU job using Stage 1's SLURM file with sbatch CLI overrides for job-name / output paths and env vars for the cross-view knobs. **No new SLURM file is created.**

- [ ] **Step 9.1: Submit**

```bash
ssh karolina 'cd ~/OccAny && \
  EXP_NAME=occany_plus_recon_dinov3_vith16plus_cross_view \
  CROSS_VIEW_ALT_START=11 \
  FINE_TUNE_LAYERS=11-31 \
  DINOV3_CONFIG=occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml \
  sbatch \
    --job-name=train_occany_plus_recon_dinov3_vith16plus_cross_view \
    --output=slurm/output/train_occany_plus_recon_dinov3_vith16plus_cross_view_%j.out \
    --error=slurm/output/train_occany_plus_recon_dinov3_vith16plus_cross_view_%j.err \
    --export=ALL \
    slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm'
```

Notes:
- `--export=ALL` forwards all env vars (including the four cross-view knobs) into the job's environment. This avoids the comma-parsing issue in `--export=ALL,X=...` style.
- `--job-name`, `--output`, `--error` CLI flags override the `#SBATCH` directives in Stage 1's SLURM file.
- All other resources (nodes, GPUs, time, partition, account) come from Stage 1's SLURM and apply unchanged.

Record the job ID.

- [ ] **Step 9.2: Tail the first ~30 minutes via `karolina-job` skill**

Watch for:
- All 32 ranks initialize without `NCCL` / `Address already in use` errors.
- The wrapper log shows the cross-view values: `EXP_NAME=...cross_view`, `--fine_tune_layers 11-31`, `--cross_view_alt_start 11`, `--dinov3_config .../occany_dinov3_vith16plus_cross_view.yaml`. If any default-Stage-1 value bleeds in, Task 7's env wiring is wrong — abort and fix.
- First epoch's per-step loss is in a sane range. Cross-view starts worse than Stage 1 (random head + untrained QK-norm + untrained cross-view blocks) and should converge over the first few epochs as the gates ramp.
- Memory utilization is well within budget per GPU. Cross-view has ~3.5× the trainable-block count of Stage 1; if VRAM is tight, drop `BATCH_SIZE` or raise `EFFECTIVE_BATCH_SIZE`.
- First eval pass at end of epoch 1 produces finite metrics on the KITTI + Occ3d-NuScenes test sets.

If anything fails at scale (NCCL hang, OOM at multi-node), pause and investigate before iterating.

- [ ] **Step 9.3: Once the run is stable, hand off to the user**

Report: job ID, output directory (`$PROJECT/tb_log_occany/occany_plus_recon_dinov3_vith16plus_cross_view`), expected first-eval timestamp. Compare cross-view vs Stage 1 eval metrics after both have run ≥ 5 epochs — this is the headline ablation answering "does cross-view alternation help."

Do **not** auto-iterate beyond Task 9. Post-cross-view follow-ups (alt_start sensitivity, QK-norm gate analysis, tap-layer retuning, longer training schedules) are separate work.

---

## Self-review notes

**Framing of Stage 2 as an in-place code extension, not a parallel implementation:**

The user's directive was that Stage 2 should not be a "dedicated training stage" and should "directly modify the code" rather than create a new `DinoV3CrossViewBackbone` class. This plan reflects that:

- All cross-view code lives in `occany/model/dinov3_backbone.py` (Stage 1's existing file). No new `dinov3_cross_view.py` module is created.
- `DinoV3PerViewBackbone` is extended in-place with an optional `alt_start` kwarg (default `-1` = Stage 1 behavior). No `DinoV3CrossViewBackbone` class is introduced.
- `build_dinov3_per_view` factory is extended in-place to handle both modes via an optional `alt_start_override` kwarg.
- No `_stage2.sh` wrapper or `_stage2.slurm` file is created — Stage 1's existing wrapper is made env-overridable; cross-view launches set four env vars to override the defaults. Stage 1's SLURM is invoked with sbatch CLI overrides for the job-name / output paths.
- Default-mode behavior is verified bit-identical to pre-Stage-2 Stage 1: a dedicated smoke (Task 3 Step 3.2) asserts that `alt_start=-1` builds the same VIT without QK-norm and produces the same forward outputs as the Stage 1 baseline. A second regression smoke (Task 8 Step 8.4) verifies the same at the full training-wrapper level.

**Design coverage:**

- The user agreed cross-view = "flip local↔global parity at `alt_start`, gated QK-norm so we start at pure DINOv3, DA3-style global RoPE." Mapped to:
  - alt_start=11, parity routing → Task 3 `_forward_cross_view`.
  - Gated QK-norm → Task 3 `GatedQKNormSelfAttention` + zero-init gates.
  - DA3-style global RoPE → Task 3 `build_global_rope_sincos` (identity for cls/storage, tiled intra-frame 2D for patches).
- The user explicitly chose alt_start=11, init from fresh DINOv3 pretrain (not Stage 1 ckpt), DA3-style include-all-tokens-in-global (with identity rotation for special tokens). All locked in.

**Stage 1 isolation:** Stage 1's behavior is preserved exactly. `DinoV3PerViewBackbone(... , alt_start=-1)` is bit-identical to the Stage 1 class (same forward path, no QK-norm installed); `Dinov3Wrapper(cross_view_alt_start=-1)` is bit-identical to Stage 1's wrapper; the training script's `--cross_view_alt_start` defaults to `-1`; the shell wrapper's env-overridable defaults all match the current Stage 1 hardcoded values.

**Filename reconciliation with on-disk state:** This plan references the actual Stage 1 filenames as committed:
- Stage 1 YAML: `occany/configs/dinov3/occany_dinov3_vith16plus.yaml`.
- Cross-view YAML follows the same naming convention: `occany/configs/dinov3/occany_dinov3_vith16plus_cross_view.yaml`.

**Discrepancies between design and code reality, surfaced during planning:**

1. The user originally proposed "skip QK-norm" but I noted that without it long-sequence attention can saturate. The user agreed to gated QK-norm. Implemented as a zero-init scalar gate per Q and per K (one gate vector per head-dim, not per-channel) so the optimizer ramps it as needed.

2. The user proposed "include cls/storage in global the same way DA3 does." DA3 uses per-token position tensors with zeros for cls; DINOv3 uses a contiguous-prefix sin/cos. Bridged via `build_global_rope_sincos` — hand-built sin/cos with identity rotation `(sin=0, cos=1)` for the 5 special-token slots per frame and standard intra-frame 2D RoPE tiled across T frames for patch positions. Mathematically equivalent to DA3's approach.

3. The user chose alt_start=11. The Stage 1 fine-tune set (last 6 blocks, `26..31`) doesn't cover that — cross-view launches must override `FINE_TUNE_LAYERS=11-31` (all 21 blocks past alt_start). Without this widening, ~15 blocks worth of QK-norm + cross-view-relevant block weights would be frozen and the cross-view machinery couldn't learn. This is exposed via the new env override.

4. The original Stage 2 plan introduced a parallel `DinoV3CrossViewBackbone` class in a new `occany/model/dinov3_cross_view.py` module, plus a separate `_stage2.sh` wrapper and `_stage2.slurm`. Per the user's directive that Stage 2 should "directly modify the code," all those parallel files are eliminated:
   - `DinoV3PerViewBackbone` is extended in-place to support both modes.
   - All helpers (`GatedQKNormSelfAttention`, `install_gated_qk_norm_on_blocks`, `build_global_rope_sincos`) live in the existing `dinov3_backbone.py`.
   - The training wrapper is made env-parametric; no `_stage2` variant is created.
   - The SLURM file is reused via sbatch CLI overrides.

5. The class name `DinoV3PerViewBackbone` is slightly misleading when `alt_start >= 0` (cross-view is not per-view). This is documented in the module docstring; a rename is left as a follow-up if cross-view becomes the default mode.

**Type consistency:** `DinoV3PerViewBackbone.forward_multiview` returns `list[(patch_feat: (B,T,N,E), cls_token: (B,T,E))]` in both modes — same contract Stage 1 already provides. `Dinov3Wrapper.inference_batch` and `_process_depth_output` consume the same contract in both modes. `get_backbone_metadata()` returns the same dict shape in both, with only `name` and `alt_start` differing in value.

**No placeholders / TBDs / `# implement later` strings.** Every step contains the actual command or code to run.
