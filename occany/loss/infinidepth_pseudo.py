"""InfiniDepth pseudo-GT depth helpers.

Used by the offline precompute (``extract_infinidepth_pseudo.py``) and by the
depth-metrics eval script. Training consumes the precomputed PNGs from disk
and does not import these helpers.

Provides:
- ``denormalize_imagenet``: undo ImageNet normalization to recover [0,1] RGB.
- ``sample_sparse_prompt``: build the InfiniDepth prompt from a LiDAR depth
  map. ``num_samples`` controls the cap; passing ``None`` (or 0/negative)
  uses every valid LiDAR pixel as the prompt (dense mode).
- ``predict_infinidepth_view``: run InfiniDepth_DepthSensor on one view,
  conditioned on LiDAR as both ``gt_depth`` and prompt.
- ``predict_infinidepth_batched``: same, but accepts a (N, 3, H, W) batch and
  runs a single InfiniDepth forward.
- ``load_infinidepth_model``: load InfiniDepth_DepthSensor frozen on a device.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# Filesystem encoding for InfiniDepth pseudo-depth, shared by the
# extract / verify scripts and the training dataset loader so all three
# agree on the sibling filename and the uint16 fixed-point scale
# (depth_m = uint16 / scale, 0 = invalid).
INFINIDEPTH_PSEUDO_SUFFIX = ".infinidepth.png"
INFINIDEPTH_PSEUDO_SCALE = 256.0


def denormalize_imagenet(imgs: torch.Tensor) -> torch.Tensor:
    """imgs: (..., 3, H, W) ImageNet-normalized -> same shape in [0, 1]."""
    mean = _IMAGENET_MEAN.to(imgs.device, imgs.dtype)
    std = _IMAGENET_STD.to(imgs.device, imgs.dtype)
    flat = imgs.reshape(-1, *imgs.shape[-3:])
    flat = flat * std + mean
    return flat.view_as(imgs).clamp(0.0, 1.0)


def sample_sparse_prompt(
    gt_depth: torch.Tensor,
    num_samples: Optional[int],
    depth_min: float,
    depth_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a (sparse or dense) prompt from gt_depth (1, 1, H, W).

    Returns ``(prompt_depth, prompt_mask)`` of the same shape as ``gt_depth``.
    ``prompt_depth`` is zero outside the prompt pixels (matching InfiniDepth's
    ``load_depth()`` behavior).

    - When ``num_samples`` is a positive int and there are more valid LiDAR
      points than that, a random subset of ``num_samples`` points is used.
    - When ``num_samples`` is ``None`` or ``<= 0``, all valid LiDAR points
      are used as the prompt (dense mode).
    """
    valid = (gt_depth > depth_min) & (gt_depth < depth_max) & torch.isfinite(gt_depth)
    valid_idx = torch.nonzero(valid.view(-1), as_tuple=False).squeeze(-1)
    prompt = torch.zeros_like(gt_depth)
    if valid_idx.numel() == 0:
        return prompt, prompt
    use_dense = num_samples is None or num_samples <= 0
    if not use_dense and valid_idx.numel() > num_samples:
        perm = torch.randperm(valid_idx.numel(), device=gt_depth.device)[:num_samples]
        sel = valid_idx[perm]
    else:
        sel = valid_idx
    prompt.view(-1)[sel] = gt_depth.view(-1)[sel]
    prompt_mask = (prompt > 0).to(gt_depth.dtype)
    return prompt, prompt_mask


def predict_infinidepth_view(
    model: torch.nn.Module,
    img_chw_raw: torch.Tensor,
    gt_depth_hw: torch.Tensor,
    num_prompt_samples: Optional[int],
    depth_min: float,
    depth_max: float,
) -> torch.Tensor:
    """Run InfiniDepth_DepthSensor on one view.

    ``img_chw_raw``: (3, H, W) image in [0, 1].
    ``gt_depth_hw``: (H, W) metric depth from the LiDAR projection (used both
        as the full sensor depth and as the source for the prompt).
    Returns (H, W) metric depth.

    Pass ``num_prompt_samples=None`` to use every valid LiDAR pixel as the
    prompt (dense mode). Otherwise a random subset of at most
    ``num_prompt_samples`` points is used.
    """
    from InfiniDepth.utils.io_utils import depth_to_disparity
    from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS

    device = img_chw_raw.device
    h_orig, w_orig = img_chw_raw.shape[-2], img_chw_raw.shape[-1]
    # InfiniDepth's DINOv3 backbone requires H,W divisible by 16; pad here,
    # crop the prediction back to (h_orig, w_orig) before returning.
    pad_h = (-h_orig) % 16
    pad_w = (-w_orig) % 16

    image = img_chw_raw.unsqueeze(0).float()
    gt_depth = gt_depth_hw.unsqueeze(0).unsqueeze(0).float()
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        gt_depth = F.pad(gt_depth, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    gt_mask = (
        (gt_depth > depth_min)
        & (gt_depth < depth_max)
        & torch.isfinite(gt_depth)
    ).float()
    prompt_depth, prompt_mask = sample_sparse_prompt(
        gt_depth, num_prompt_samples, depth_min, depth_max,
    )

    gt_disp = depth_to_disparity(gt_depth)
    prompt_disp = depth_to_disparity(prompt_depth)

    h_sample, w_sample = image.shape[-2], image.shape[-1]
    query = SAMPLING_METHODS["2d_uniform"]((h_sample, w_sample)).unsqueeze(0).to(device)

    pred_2d, _ = model.inference(
        image=image,
        query_coord=query,
        gt_depth=gt_disp,
        gt_depth_mask=gt_mask,
        prompt_depth=prompt_disp,
        prompt_mask=prompt_mask.bool(),
    )
    pred_depthmap = pred_2d.permute(0, 2, 1).view(1, 1, h_sample, w_sample)
    pred_depthmap = pred_depthmap[..., :h_orig, :w_orig]
    return pred_depthmap.squeeze(0).squeeze(0)


def predict_infinidepth_batched(
    model: torch.nn.Module,
    imgs_nchw_raw: torch.Tensor,
    gt_depth_nhw: torch.Tensor,
    num_prompt_samples: Optional[int],
    depth_min: float,
    depth_max: float,
    chunk_size: int = 10,
) -> torch.Tensor:
    """Batched counterpart to ``predict_infinidepth_view``.

    ``imgs_nchw_raw``: (N, 3, H, W) images in [0, 1].
    ``gt_depth_nhw``: (N, H, W) metric LiDAR depth.
    Returns (N, H, W) metric depth predicted by InfiniDepth_DepthSensor.

    Pass ``num_prompt_samples=None`` to use every valid LiDAR pixel as the
    prompt for each view independently (dense mode).

    ``chunk_size`` caps how many views are pushed through InfiniDepth in a
    single forward pass — keeps GPU memory bounded when B*V is large (e.g.
    24 surround views). Set <= 0 to disable chunking.
    """
    from InfiniDepth.utils.io_utils import depth_to_disparity
    from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS

    device = imgs_nchw_raw.device
    n = imgs_nchw_raw.shape[0]
    h_orig, w_orig = imgs_nchw_raw.shape[-2], imgs_nchw_raw.shape[-1]
    pad_h = (-h_orig) % 16
    pad_w = (-w_orig) % 16

    image = imgs_nchw_raw.float()
    gt_depth = gt_depth_nhw.unsqueeze(1).float()
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        gt_depth = F.pad(gt_depth, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    gt_mask = (
        (gt_depth > depth_min)
        & (gt_depth < depth_max)
        & torch.isfinite(gt_depth)
    ).float()
    # sample_sparse_prompt's sparse path would sample a single global subset across the
    # whole batch — only the dense path is per-view-correct on a batched tensor.
    use_dense = num_prompt_samples is None or num_prompt_samples <= 0
    assert use_dense, (
        'predict_infinidepth_batched currently supports dense prompts only '
        '(num_prompt_samples=None); per-view sparse sampling is not implemented.'
    )
    prompt_depth, prompt_mask = sample_sparse_prompt(
        gt_depth, num_prompt_samples, depth_min, depth_max,
    )

    gt_disp = depth_to_disparity(gt_depth)
    prompt_disp = depth_to_disparity(prompt_depth)
    prompt_mask_bool = prompt_mask.bool()

    h_sample, w_sample = image.shape[-2], image.shape[-1]
    query_one = SAMPLING_METHODS["2d_uniform"]((h_sample, w_sample)).unsqueeze(0).to(device)

    # InfiniDepth's WarpMedian calls torch.quantile over LiDAR-valid pixels per
    # view and raises on an empty tensor. Run only views with at least one
    # valid LiDAR pixel; leave empty-mask views as zero so the caller's
    # depth_range / isfinite mask drops them downstream.
    view_has_valid = gt_mask.flatten(1).any(dim=1)
    valid_idx = view_has_valid.nonzero(as_tuple=False).squeeze(-1)
    n_valid = int(valid_idx.numel())

    pred_full = torch.zeros(n, 1, h_sample, w_sample, device=device, dtype=gt_disp.dtype)
    if n_valid > 0:
        image_v = image.index_select(0, valid_idx)
        gt_disp_v = gt_disp.index_select(0, valid_idx)
        gt_mask_v = gt_mask.index_select(0, valid_idx)
        prompt_disp_v = prompt_disp.index_select(0, valid_idx)
        prompt_mask_bool_v = prompt_mask_bool.index_select(0, valid_idx)

        step = n_valid if chunk_size is None or chunk_size <= 0 else min(int(chunk_size), n_valid)
        chunks = []
        for start in range(0, n_valid, step):
            end = min(start + step, n_valid)
            m = end - start
            query = query_one.expand(m, -1, -1).contiguous()
            pred_2d, _ = model.inference(
                image=image_v[start:end],
                query_coord=query,
                gt_depth=gt_disp_v[start:end],
                gt_depth_mask=gt_mask_v[start:end],
                prompt_depth=prompt_disp_v[start:end],
                prompt_mask=prompt_mask_bool_v[start:end],
            )
            chunks.append(pred_2d)
        pred_2d = torch.cat(chunks, dim=0)
        pred_valid = pred_2d.permute(0, 2, 1).view(n_valid, 1, h_sample, w_sample)
        pred_full.index_copy_(0, valid_idx, pred_valid.to(pred_full.dtype))

    pred_depthmap = pred_full[..., :h_orig, :w_orig]
    return pred_depthmap.squeeze(1)


def load_infinidepth_model(device: str, ckpt_path: str) -> torch.nn.Module:
    """Build and load InfiniDepth_DepthSensor in eval / frozen mode."""
    from InfiniDepth.utils.model_utils import build_model

    print(f"Loading InfiniDepth_DepthSensor: {ckpt_path}")
    model = build_model("InfiniDepth_DepthSensor", model_path=ckpt_path)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model
