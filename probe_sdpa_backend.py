#!/usr/bin/env python3
"""Which SDPA kernel does the DeltaTok tokenizer's attention run on, with and without cuDNN SDPA?

Builds a DeltaTokModule with the training run's shape (DA3-giant: 1536 hidden, 24 heads,
patch 14, 12+12 layers, K=64), runs one bf16-autocast forward on a training-sized pair
batch, and lists the autograd nodes of every scaled_dot_product_attention call.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.amp import autocast

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(
    Path(__file__).resolve().parent,
    extra=["third_party/GLD/src", "third_party/deltatok"],
)

from occrae.deltatok_trainer import DeltaTokModule  # noqa: E402


def sdpa_nodes(out: torch.Tensor) -> Counter:
    """Count ScaledDotProduct* autograd nodes reachable from `out`."""
    seen, stack, names = set(), [out.grad_fn], Counter()
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        if "ScaledDotProduct" in fn.name():
            names[fn.name()] += 1
        stack.extend(nxt for nxt, _ in fn.next_functions)
    return names


def main() -> None:
    torch.manual_seed(0)
    net = DeltaTokModule(hidden_size=1536, num_heads=24, patch_size=14, num_hidden_layers=12,
                         num_delta_tokens=64, z_norm=False, target_channels=1536,
                         force_bottleneck=False, layer_scale_init=0.1).cuda().train()
    H, W = 294, 518                                                   # training res -> P = 21 * 37 = 777
    x_prev = torch.randn(6, 1, (H // 14) * (W // 14), 1536, device="cuda")   # (B*3, N, P, C) bsize 2 triplet
    x = torch.randn_like(x_prev)                                      # (B*3, N, P, C)
    for label, cudnn in (("default", True), ("cudnn_sdp off", False)):
        torch.backends.cuda.enable_cudnn_sdp(cudnn)
        with autocast("cuda", dtype=torch.bfloat16):
            x_hat = net(x_prev, x, H, W, num_cameras=1)
        print(f"{torch.__version__} | {label}: {dict(sdpa_nodes(x_hat))}", flush=True)


if __name__ == "__main__":
    main()
