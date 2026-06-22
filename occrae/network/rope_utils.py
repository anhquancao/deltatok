"""Shared rope helpers for DeltaTok / flow-ViT camera rope.

Kept dependency-light (torch only) so both the tokenizer (occrae.deltatok_trainer)
and the flow ViT (occrae.network.efficient_transformer) can build their camera rope
from the SAME construction without importing each other.
"""
import torch


def compute_camera_rope(rope_embeddings, max_cameras, device, dtype):
    """Resolution-independent DINOv3 axial rope over camera positions 0..max_cameras-1.

    Lays the camera slots along a synthetic ``1 x max_cameras`` patch grid so the coord
    normalization (DINOv3 divides patch coords by the grid) uses a FIXED ``max_cameras``
    instead of the image's ``Hp x Wp``; a camera idx therefore maps to the same rotation
    at any image resolution. Only the width (x) coord varies — the row-0 y coord is
    constant, so the y-half of head_dim is identity here.

    Args:
        rope_embeddings: a ``DINOv3ViTRopePositionEmbedding`` (its ``.config.patch_size``
            sizes the synthetic grid; augmentation only fires when it is in train mode).
    Returns:
        (cos, sin), each of shape ``(max_cameras, head_dim)``.
    """
    ps = rope_embeddings.config.patch_size
    # 1 x max_cameras grid: Hp=1, Wp=max_cameras -> max_cameras grid positions.
    dummy = torch.zeros(1, 3, ps, ps * int(max_cameras), device=device, dtype=dtype)  # (1,3,ps,ps*MAX)
    return rope_embeddings(dummy)  # (cos, sin), each (max_cameras, head_dim)
