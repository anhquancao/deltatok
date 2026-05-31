"""DINOv3 ViT-H+/16 backbone wrapper for OccAny+.

Cross-view backbone built on `dinov3.models.vision_transformer.DinoVisionTransformer`.
Blocks 0..alt_start-1 are pure pretrained DINOv3 (frame-local). From `alt_start`
onward, blocks alternate by index parity -- even-index blocks remain frame-local,
odd-index blocks flatten across frames for cross-view attention. Gated QK-norm is
installed on all blocks at and after `alt_start` (gates zero-initialized, so the
backbone is bit-identical to pretrained DINOv3 at step 0). Global passes use a
custom RoPE: identity rotation for cls/storage tokens, intra-frame 2D RoPE tiled
across frames for patch tokens (DA3-style).

`alt_start` is read from the YAML config (required field). DPT taps via `out_layers`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import yaml

from dinov3.layers.attention import SelfAttention
from dinov3.models.vision_transformer import DinoVisionTransformer


# --------------------------------------------------------------------------- #
# Gated QK-norm and installer.                                                #
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
    if alt_start >= len(vit.blocks):
        raise ValueError(f"alt_start={alt_start} >= n_blocks={len(vit.blocks)}")

    for blk in vit.blocks[alt_start:]:
        old = blk.attn
        if not isinstance(old, SelfAttention):
            raise TypeError(
                f"Expected block.attn to be SelfAttention, got {type(old).__name__}"
            )
        if isinstance(old, GatedQKNormSelfAttention):
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
# Global-pass RoPE construction.                                              #
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
# Backbone wrapper.                                                           #
# --------------------------------------------------------------------------- #


class DinoV3Backbone(nn.Module):
    """DINOv3 ViT-H+/16 with cross-view attention from layer `alt_start` onward."""

    def __init__(
        self,
        vit: DinoVisionTransformer,
        out_layers: Iterable[int],
        alt_start: int,
    ):
        super().__init__()
        self.vit = vit
        self.out_layers = tuple(int(idx) for idx in out_layers)
        if len(self.out_layers) == 0:
            raise ValueError("DinoV3Backbone requires at least one tap layer")
        if max(self.out_layers) >= self.vit.n_blocks:
            raise ValueError(
                f"out_layers={self.out_layers} exceeds n_blocks={self.vit.n_blocks}"
            )
        if alt_start < 0 or alt_start >= self.vit.n_blocks:
            raise ValueError(
                f"alt_start={alt_start} out of range [0, {self.vit.n_blocks})"
            )
        self.alt_start = int(alt_start)

        install_gated_qk_norm_on_blocks(self.vit, alt_start=self.alt_start)

    @property
    def n_special_tokens_per_frame(self) -> int:
        # cls (1) + storage (n_storage_tokens).
        return 1 + int(self.vit.n_storage_tokens)

    def forward_multiview(
        self, images: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run DINOv3 ViT with cross-view alternation and return tapped features.

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
        return {
            "name": "dinov3_vith16plus",
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


def build_dinov3_backbone(
    yaml_path: str | Path,
    weights_path: str | Path | None = None,
) -> DinoV3Backbone:
    """Build a `DinoV3Backbone` from a YAML config + optional checkpoint.

    YAML must contain `arch` (DINOv3 arch kwargs), `out_layers` (DPT tap indices),
    and `alt_start` (layer index at which cross-view alternation begins).
    The checkpoint is loaded with strict=True against the pristine DINOv3 state
    dict (QK-norm params are added by the class `__init__`, AFTER load).
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    arch_kwargs = dict(cfg["arch"])
    out_layers = tuple(int(idx) for idx in cfg["out_layers"])
    alt_start = int(cfg["alt_start"])

    vit = DinoVisionTransformer(**arch_kwargs)

    # DINOv3's __init__ does not run init_weights; mirror dinov3/hub/backbones.py
    # which either loads pretrained weights or calls init_weights() explicitly.
    # Without either, LinearKMaskedBias.bias_mask stays NaN and forward poisons.
    if weights_path is not None:
        state_dict = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )
        missing, unexpected = vit.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"DINOv3 strict load failed: "
                f"missing={missing[:5]}... unexpected={unexpected[:5]}..."
            )
    else:
        vit.init_weights()

    return DinoV3Backbone(vit, out_layers=out_layers, alt_start=alt_start)
