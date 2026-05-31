"""Standalone smoke for DinoV3Backbone (cross-view).

Validates:
1. Loads DINOv3 ViT-H+/16 with cross-view alternation from a pristine pretrained
   checkpoint via build_dinov3_backbone.
2. Forward runs at T = 1, 2, 4 views.
3. Backward runs without exception and grads land on QK-norm gates and norms
   on every block at and after alt_start.

Usage:
    python test_dinov3_cross_view.py \\
        --weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
"""
from __future__ import annotations

import argparse

import torch

from occany.utils.runtime_paths import prepend_vendored_import_paths

prepend_vendored_import_paths()

from occany.model.dinov3_backbone import build_dinov3_backbone  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="occany/configs/dinov3/occany_dinov3_vith16plus.yaml")
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

    bb = build_dinov3_backbone(args.config, weights_path=args.weights).eval().to(args.device)
    md = bb.get_backbone_metadata()
    print("backbone metadata:", md)
    assert md["alt_start"] >= 0, md

    # --- Forward smoke at multiple T values ---
    for T in args.T_values:
        x = torch.randn(1, T, 3, args.height, args.width, device=args.device)
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
            feats = bb.forward_multiview(x)
        for tap_idx, (f, cls) in enumerate(feats):
            assert f.shape[1] == T, (f.shape, T)
            assert cls.shape[1] == T, (cls.shape, T)
        print(f"forward T={T} OK; last-tap feat shape {tuple(feats[-1][0].shape)}")

    # --- Backward smoke + QK-norm gate grad check ---
    bb.train()
    x = torch.randn(1, 3, 3, args.height, args.width, device=args.device)
    with torch.autocast(args.device, dtype=torch.bfloat16):
        feats = bb.forward_multiview(x)
    loss = sum(f.float().mean() + cls.float().mean() for f, cls in feats)
    loss.backward()

    alt_start = md["alt_start"]
    n_blocks = md["total_layers"]
    missing = []
    for i in range(alt_start, n_blocks):
        attn = bb.vit.blocks[i].attn
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
