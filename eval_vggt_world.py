#!/usr/bin/env python3
"""Dump pretrained VGGT-World forecasts (depth, points, c2w at DA3 scale, GT-mask
pixels only) on the DeltaTok flow-eval windows; score with eval_forecast_metrics.py.

Frames 0-1 are context, 2..T-1 are forecast. VGGT-World's flow model rolls the
part1 tokens forward, part2 + heads decode all T frames in frame-0 coordinates,
and DA3METRIC-LARGE on the context frames (predicted K) sets the metric scale.
Writes <output_dir>/<run>/<set>/<n:05d>.npz, n = window index in loader order.
Plan: docs/research/plan/2026-10-05_flow_forecast_dump_then_score.md

Usage (BSC GPU node):
  source env_bsc.sh && python eval_vggt_world.py \
      --config-name eval_deltatok_flow_alldata_ctx2fwd8_kitti_bsc \
      --ckpt /gpfs/scratch/ehpc1001/quan/vggt_world/kitti.pt
"""

import argparse
import os
import re
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(
    Path(__file__).resolve().parent,
    extra=[
        "third_party/VGGT-World",
    ],
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# VGGT-World imports tensorboardX only for a training SummaryWriter; not in the BSC venv.
try:
    import tensorboardX  # noqa: F401
except ModuleNotFoundError:
    _tbx = types.ModuleType("tensorboardX")
    _tbx.SummaryWriter = object
    sys.modules["tensorboardX"] = _tbx

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.alignment import (  # noqa: E402
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
)

from occany.datasets import get_data_loader  # noqa: E402
from occany.utils.helpers import convert_depth_to_point_cloud  # noqa: E402
from occrae.deltatok_shared import DeltaTokSharedMixin  # noqa: E402
from occrae.forecast_dump import save_windows  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
N_CTX = 2  # frames 0-1 given


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("VGGT-World on the DeltaTok flow eval", add_help=True)
    parser.add_argument("--config-dir", type=str, default="configs/deltatok_flow")
    parser.add_argument("--config-name", type=str, default="eval_deltatok_flow_alldata_ctx2fwd8_kitti_bsc")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_items", type=int, default=None,
                        help="Windows per set; above the split takes it whole (window_stride sets the size).")
    parser.add_argument("--test_filter", type=str, default=None)
    parser.add_argument("--bsize", type=int, default=4)
    parser.add_argument("--fm_steps", type=int, default=50)
    parser.add_argument("--rollout", choices=["stride2", "stride1"], default="stride1")  # authors' default (paper B.3)
    parser.add_argument("--resolution", choices=["native", "448"], default="native")
    parser.add_argument("--da3_metric_model", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--verbose_batches", type=int, default=1,
                        help="Print per-window scales for the first N batches of each set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/vggt_world_alldata_ctx2fwd8")
    return parser


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")


def _build_test_loaders(cfg, num_items=None, test_filter=None):
    """{test_name: loader} for cfg.dataset.test_dataset — mirrors how
    `DeltaTokFlowMatchingTrainer.fit` builds eval loaders, so `eval_one_epoch`
    sees the exact samples the training logs evaluated."""
    expr = cfg.dataset.get("test_dataset", None)
    if not expr:
        raise RuntimeError("Config has no dataset.test_dataset.")
    kw = dict(
        # val_bsize caps eval decode memory (0 = train bsize) — as the trainer does
        batch_size=int(cfg.training.get("val_bsize", 0)) or int(cfg.training.bsize),
        num_workers=int(cfg.training.get("val_num_workers", 2)),
        shuffle=False,
        drop_last=False,
    )
    loaders = {}
    for sub in str(expr).split("+"):
        sub = sub.strip()
        if not sub:
            continue
        if test_filter is not None and test_filter not in sub:
            continue
        if num_items is not None:
            ds_expr = sub.split("@", 1)[-1].strip()           # drop the config's "<n> @"
            loader = get_data_loader(f"{num_items} @ {ds_expr}", **kw)
            pool = len(loader.dataset.dataset)                # windows in the full split
            if pool < num_items:
                # ResizedDataset would repeat windows; take the split whole instead
                loader = get_data_loader(f"{pool} @ {ds_expr}", **kw)
            sub = f"{min(num_items, pool)} @ {ds_expr}"
        else:
            loader = get_data_loader(sub, **kw)
        test_name = sub.split("(")[0].strip()                 # tag carries the real count
        m = re.search(r"window_stride=(\d+)", sub)
        if m:
            test_name += f" stride{m.group(1)}"   # an unstrided set and its strided twin log under separate names
        loaders[test_name] = loader
    return loaders


def _load_vggt_world(ckpt_path, device):
    model = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        print(f"[INFO] ckpt top-level keys: {list(data.keys())[:10]}", flush=True)
    state = data["model"] if isinstance(data, dict) and "model" in data else data
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] VGGT-World load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"[INFO]   missing (first 10): {missing[:10]}", flush=True)
    print(f"[INFO]   unexpected prefixes: {sorted({k.split('.')[0] for k in unexpected})}", flush=True)
    del data, state
    model.to(device).eval()
    return model


@torch.no_grad()
def _forecast_tokens(model, xv, n_frames, fm_steps, rollout, patch_hw):
    """Context part1 tokens of frames 0-1, rolled forward by the flow model to n_frames."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = model.aggregator.part1(xv[:, :N_CTX])[0][0]           # (B, 2, Ntot, C)
    B, _, Ntot, C = ctx.shape
    fm_dtype = next(model.fm.parameters()).dtype
    shape_like = torch.zeros((B, 2, Ntot, C), device=ctx.device, dtype=fm_dtype)  # (B, 2, Ntot, C)
    frames = [ctx[:, 0:1], ctx[:, 1:2]]                             # T x (B, 1, Ntot, C)
    while len(frames) < n_frames:
        # stride2: condition on the 2 newest frames, keep both predictions.
        # stride1 (eval/kitti_val_mid.py): shift by one, keep only the new frame.
        first = len(frames) == N_CTX
        if rollout == "stride2" or first:
            cond = torch.cat(frames[-2:], dim=1)                    # (B, 2, Ntot, C)
        else:
            cond = torch.cat(frames[-3:-1], dim=1)                  # (B, 2, Ntot, C)
        gen = model.fm.sample_euler(
            cond_layers=[cond.to(fm_dtype)], shape_like=shape_like, steps=fm_steps, patch_hw=patch_hw,
        )                                                           # 2 x (B, 1, Ntot, C)
        if rollout == "stride2" or first:
            frames += gen
        else:
            frames.append(gen[1])
    return torch.cat(frames[:n_frames], dim=1)                      # (B, T, Ntot, C)


@torch.no_grad()
def _decode(model, tokens, xv, patch_hw, out_hw):
    """part2 + depth / camera heads over all T frames; depth, conf and K returned at out_hw."""
    B, T = tokens.shape[:2]
    h, w = xv.shape[-2:]
    H, W = out_hw
    with torch.autocast("cuda", dtype=torch.bfloat16):
        agg, psi = model.aggregator.part2([tokens.to(torch.bfloat16)], patch_hw=patch_hw)
    agg = [a.float() for a in agg]
    with torch.autocast("cuda", enabled=False):
        depth, conf = model.depth_head(agg, images=xv.float(), patch_start_idx=psi)  # (B, T, h, w, 1), (B, T, h, w)
        pose_enc = model.camera_head(agg)[-1]                                          # (B, T, 9)
        w2c, K = pose_encoding_to_extri_intri(pose_enc, (h, w))                       # (B, T, 3, 4), (B, T, 3, 3)
    depth = depth[..., 0]                                                              # (B, T, h, w)
    if (h, w) != (H, W):
        depth = F.interpolate(depth.flatten(0, 1)[:, None], size=(H, W), mode="bilinear",
                              align_corners=False).view(B, T, H, W)                    # (B, T, H, W)
        conf = F.interpolate(conf.flatten(0, 1)[:, None], size=(H, W), mode="bilinear",
                             align_corners=False).view(B, T, H, W)                     # (B, T, H, W)
        K = K.clone()
        K[..., 0, :] *= W / w
        K[..., 1, :] *= H / h
    w2c44 = torch.eye(4, device=w2c.device, dtype=torch.float64).repeat(B, T, 1, 1)   # (B, T, 4, 4)
    w2c44[..., :3, :] = w2c.double()
    c2w = torch.linalg.inv(w2c44).float()                                              # (B, T, 4, 4)
    return depth.float(), conf.float(), K.float(), c2w


def _da3_scale(m_depth, m_sky, depth, conf, K):
    """Per-window LSQ scale of VGGT depth to DA3-metric depth on the context frames
    (extract_recon.py:404-434). Returns (B,) scale and (B,) fallback flag."""
    md = apply_metric_scaling(m_depth, K[:, :N_CTX])                                  # (B, 2, H, W)
    non_sky = compute_sky_mask(m_sky, threshold=0.3)                                   # (B, 2, H, W)
    B = depth.shape[0]
    scale = torch.ones(B, device=depth.device)
    fallback = torch.ones(B, dtype=torch.bool, device=depth.device)
    for b in range(B):
        ns = non_sky[b]
        if ns.sum() <= 10:
            continue
        conf_ns = conf[b, :N_CTX][ns]
        if conf_ns.numel() == 0:
            continue
        median_conf = torch.quantile(sample_tensor_for_quantile(conf_ns, max_samples=100000), 0.5)
        align = compute_alignment_mask(conf[b, :N_CTX], ns, depth[b, :N_CTX], md[b], median_conf)
        if align.sum() == 0:
            continue
        s = least_squares_scale_scalar(md[b][align], depth[b, :N_CTX][align])
        if torch.isfinite(s):
            scale[b] = s
            fallback[b] = False
    return scale, fallback


def main() -> None:
    args = get_args_parser().parse_args()
    device = torch.device("cuda")

    ckpt_stem = _sanitize(Path(args.ckpt).stem)
    output_dir = os.path.join(os.path.abspath(args.output_dir), f"{ckpt_stem}_{args.rollout}_{args.resolution}")
    os.makedirs(output_dir, exist_ok=True)

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)
    with open_dict(cfg):
        cfg.training.val_bsize = args.bsize

    model = _load_vggt_world(args.ckpt, device)
    da3_metric = DepthAnything3.from_pretrained(args.da3_metric_model).to(device).eval()
    da3_metric.requires_grad_(False)
    batcher = DeltaTokSharedMixin()  # only _normalize_batch
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    loaders = _build_test_loaders(cfg, args.num_items, args.test_filter)
    for test_name, loader in loaders.items():
        # Pin data and noise per set, as eval_one_epoch does (rank 0).
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        ds = getattr(loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(0)
        torch.manual_seed(args.seed)

        set_dir = os.path.join(output_dir, _sanitize(test_name))
        os.makedirs(set_dir, exist_ok=True)
        n_fallback, n_items = 0, 0
        t0 = time.time()
        for it, raw in enumerate(loader):
            batch = batcher._normalize_batch(raw)
            imgs = batch["imgs"].to(device, non_blocking=True)                          # (B, T, 3, H, W) DA3-normed
            B, T, _, H, W = imgs.shape
            assert batch["num_cameras"] == 1, "cam-0 windows only"
            x01 = (imgs * std + mean).clamp(0, 1)                                       # (B, T, 3, H, W)
            if args.resolution == "448":
                xv = F.interpolate(x01.flatten(0, 1), size=(224, 448), mode="bilinear", align_corners=False,
                                   antialias=True).view(B, T, 3, 224, 448)              # (B, T, 3, 224, 448)
            else:
                assert H % 14 == 0 and W % 14 == 0, (H, W)
                xv = x01
            h, w = xv.shape[-2:]
            patch_hw = (h // 14, w // 14)

            tokens = _forecast_tokens(model, xv, T, args.fm_steps, args.rollout, patch_hw)  # (B, T, Ntot, C)
            depth, conf, K, c2w = _decode(model, tokens, xv, patch_hw, (H, W))          # (B, T, H, W) x 2, (B, T, 3, 3), (B, T, 4, 4)

            with torch.no_grad():
                m = da3_metric(imgs[:, :N_CTX], export_feat_layers=[])
            m_depth, m_sky = m["depth"].float(), m["sky"].float()                        # (B, 2, H, W)

            scale, fallback = _da3_scale(m_depth, m_sky, depth, conf, K)                 # (B,), (B,)
            depth = depth * scale[:, None, None, None]                                     # (B, T, H, W) metres
            c2w[..., :3, 3] *= scale[:, None, None]                                        # (B, T, 4, 4)
            point = convert_depth_to_point_cloud(depth, K, c2w)                            # (B, T, H, W, 3) frame-0 coords
            save_windows(set_dir, n_items, depth, point, c2w, batch)
            n_items += B
            n_fallback += int(fallback.sum())

            if it < args.verbose_batches:
                print(f"[DBG/{test_name}] imgs [{imgs.min():.3f}, {imgs.max():.3f}] -> x01 [{x01.min():.3f}, "
                      f"{x01.max():.3f}]  vggt in {tuple(xv.shape[-2:])}  patch_hw {patch_hw}", flush=True)
                print(f"[DBG/{test_name}] s_da3 {scale.tolist()}  fallback {fallback.tolist()}", flush=True)
            if it % 50 == 0:
                print(f"[INFO] {test_name}: {n_items} windows, {(time.time() - t0) / n_items:.2f} s/window",
                      flush=True)

        print(f"[INFO] {test_name}: wrote {n_items} windows, {n_fallback} DA3 fallbacks, to {set_dir}", flush=True)


if __name__ == "__main__":
    main()
