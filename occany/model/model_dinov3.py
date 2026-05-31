"""Recon-only DINOv3 wrapper for OccAny+.

DINOv3 ViT-H+/16 with cross-view alternation (configured via the YAML's
`alt_start` field) feeds DA3's DualDPT head to produce per-view
depth / depth_conf / ray / ray_conf.

Public surface deliberately mirrors a subset of `occany.model.model_da3.DA3Wrapper`:
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

from occany.model.dinov3_backbone import build_dinov3_backbone


class Dinov3Wrapper(nn.Module):
    def __init__(
        self,
        config_path: str | Path,
        weights_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.backbone = build_dinov3_backbone(config_path, weights_path=weights_path)
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
