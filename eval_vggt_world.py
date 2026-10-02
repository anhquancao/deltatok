#!/usr/bin/env python3
"""Score pretrained VGGT-World forecasts with the DeltaTok flow-eval metrics.

Frames 0-1 are context, 2..T-1 are forecast. VGGT-World's flow model rolls the
part1 tokens forward, part2 + heads decode all T frames in frame-0 coordinates,
and DA3METRIC-LARGE on the context frames (predicted K) sets the metric scale.
Depth / Pointmap / Raymap losses and Chamfer are the trainer's own code
(``_compute_frame_losses``, ``compute_chamfer_metrics``), on the windows
``_build_test_loaders`` gives the flow eval.

Rows: main (DA3 scale), ``_oracle`` (median GT scale on forecast frames),
``_tok`` (part2 on GT part1 tokens of all T frames, DA3 scale).
Plan: docs/research/plan/2026-10-01_flow_vggt_world_baseline_eval.md

Usage (BSC GPU node):
  source env_bsc.sh && python eval_vggt_world.py \
      --config-name eval_deltatok_flow_alldata_ctx2fwd8_kitti_bsc \
      --ckpt /gpfs/scratch/ehpc1001/quan/vggt_world/kitti.pt
"""

import argparse
import json
import os
import re
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(
    Path(__file__).resolve().parent,
    extra=[
        "third_party/pyTorchChamferDistance",
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
from occany.loss import PointmapLoss, DepthLosses, RaymapLoss  # noqa: E402
from occany.utils.helpers import convert_depth_to_point_cloud, intrinsics_c2w_to_raymap  # noqa: E402
from occrae.chamfer_metrics import compute_chamfer_metrics  # noqa: E402
from occrae.deltatok_shared import DeltaTokSharedMixin  # noqa: E402

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
    parser.add_argument("--rollout", choices=["stride2", "stride1"], default="stride2")
    parser.add_argument("--resolution", choices=["native", "448"], default="native")
    parser.add_argument("--da3_metric_model", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--no_chamfer", action="store_true")
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


class _Scorer(DeltaTokSharedMixin):
    """The trainer's batch conversion and frame losses; criteria as deltatok_flow_trainer.py:181-183."""

    def __init__(self, device):
        self.device = device
        self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
        self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.raymap_criterion = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")


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


def _oracle_scale(depth, gt_depth, gt_mask, fslice):
    """Per-window median(GT) / median(pred) over valid forecast pixels. (B,)"""
    B = depth.shape[0]
    scale = torch.ones(B, device=depth.device)
    for b in range(B):
        m = gt_mask[b, fslice]
        if m.sum() > 0:
            scale[b] = gt_depth[b, fslice][m].median() / depth[b, fslice][m].median().clamp_min(1e-6)
    return scale


def _scaled_outputs(depth, K, c2w, scale):
    """Apply a per-window scale; build the dict _compute_frame_losses reads."""
    H, W = depth.shape[-2:]
    depth_s = depth * scale[:, None, None, None]                                       # (B, T, H, W)
    c2w_s = c2w.clone()
    c2w_s[..., :3, 3] = c2w_s[..., :3, 3] * scale[:, None, None]                       # (B, T, 4, 4)
    return {
        "depth": depth_s,
        "pointmap": convert_depth_to_point_cloud(depth_s, K, c2w_s),                   # (B, T, H, W, 3)
        "ray": intrinsics_c2w_to_raymap(K, c2w_s, H, W),                                # (B, T, H, W, 6)
    }


def _chamfer_rows(pointmap, gt_pointmap, mask):
    """Gen3R metrics per window: (B, 4) acc, comp, chamfer, relative %."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # Gen3R's float32 Umeyama must not run in TF32
    rows = []
    for b in range(pointmap.shape[0]):
        p = pointmap[b].float().contiguous()                                           # (F, H, W, 3); Umeyama .view()s it
        rows.append([v.item() for v in compute_chamfer_metrics(p, gt_pointmap[b], mask[b])[:4]])
    torch.backends.cuda.matmul.allow_tf32 = tf32
    return torch.tensor(rows)                                                           # (B, 4)


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
    scorer = _Scorer(device)
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

        sums = defaultdict(float)
        ratios, n_fallback, n_items = [], 0, 0
        t0 = time.time()
        for it, raw in enumerate(loader):
            batch = scorer._normalize_batch(raw)
            imgs = batch["imgs"].to(device, non_blocking=True)                          # (B, T, 3, H, W) DA3-normed
            B, T, _, H, W = imgs.shape
            assert batch["num_cameras"] == 1, "cam-0 windows only"
            fslice = slice(N_CTX, T)
            x01 = (imgs * std + mean).clamp(0, 1)                                       # (B, T, 3, H, W)
            if args.resolution == "448":
                xv = F.interpolate(x01.flatten(0, 1), size=(224, 448), mode="bilinear", align_corners=False,
                                   antialias=True).view(B, T, 3, 224, 448)              # (B, T, 3, 224, 448)
            else:
                assert H % 14 == 0 and W % 14 == 0, (H, W)
                xv = x01
            h, w = xv.shape[-2:]
            patch_hw = (h // 14, w // 14)

            tok_fc = _forecast_tokens(model, xv, T, args.fm_steps, args.rollout, patch_hw)  # (B, T, Ntot, C)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tok_gt = model.aggregator.part1(xv)[0][0]                               # (B, T, Ntot, C)
            dec_fc = _decode(model, tok_fc, xv, patch_hw, (H, W))                       # depth, conf, K, c2w
            dec_gt = _decode(model, tok_gt, xv, patch_hw, (H, W))

            with torch.no_grad():
                m = da3_metric(imgs[:, :N_CTX], export_feat_layers=[])
            m_depth, m_sky = m["depth"].float(), m["sky"].float()                        # (B, 2, H, W)

            gt_depth = batch["gt_depth"].to(device).float()                              # (B, T, H, W)
            gt_mask = batch["gt_mask"].to(device).bool().reshape(gt_depth.shape)         # (B, T, H, W)
            s_fc, fb_fc = _da3_scale(m_depth, m_sky, dec_fc[0], dec_fc[1], dec_fc[2])
            s_gt, _ = _da3_scale(m_depth, m_sky, dec_gt[0], dec_gt[1], dec_gt[2])
            s_or = _oracle_scale(dec_fc[0], gt_depth, gt_mask, fslice)

            rows = {
                "": _scaled_outputs(dec_fc[0], dec_fc[2], dec_fc[3], s_fc),
                "_oracle": _scaled_outputs(dec_fc[0], dec_fc[2], dec_fc[3], s_or),
                "_tok": _scaled_outputs(dec_gt[0], dec_gt[2], dec_gt[3], s_gt),
            }
            batch_losses = {}
            for suffix, dec in rows.items():
                l_pm, l_d, l_ray = scorer._compute_frame_losses(dec, batch, fslice, None, B, H, W)
                batch_losses[f"LossPointmap{suffix}"] = l_pm.item()
                batch_losses[f"LossDepth{suffix}"] = l_d.item()
                batch_losses[f"LossRaymap{suffix}"] = l_ray.item()
            if not args.no_chamfer:
                gt_pm = batch["gt_pointmap"][:, fslice].to(device).float()               # (B, F, H, W, 3)
                f_mask = gt_mask[:, fslice]                                              # (B, F, H, W)
                for suffix in ("", "_tok"):  # Sim(3)-aligned, so _oracle would equal the main row
                    cr = _chamfer_rows(rows[suffix]["pointmap"][:, fslice], gt_pm, f_mask)
                    for j, k in enumerate(("ChamferAcc", "ChamferComp", "Chamfer", "ChamferRel")):
                        batch_losses[f"{k}{suffix}"] = cr[:, j].mean().item()

            for k, v in batch_losses.items():
                sums[k] += v * B
            n_items += B
            keep = ~fb_fc.cpu()
            ratios.append((s_fc / s_or).cpu()[keep])
            n_fallback += int(fb_fc.sum())

            if it < args.verbose_batches:
                print(f"[DBG/{test_name}] imgs [{imgs.min():.3f}, {imgs.max():.3f}] -> x01 [{x01.min():.3f}, "
                      f"{x01.max():.3f}]  vggt in {tuple(xv.shape[-2:])}  patch_hw {patch_hw}", flush=True)
                print(f"[DBG/{test_name}] s_da3 {s_fc.tolist()}  s_oracle {s_or.tolist()}  "
                      f"s_da3_tok {s_gt.tolist()}  fallback {fb_fc.tolist()}", flush=True)
                for f in range(N_CTX, T):
                    mk = gt_mask[:, f]
                    e_fc = (rows[""]["depth"][:, f] - gt_depth[:, f]).abs()[mk].mean().item()
                    e_tok = (rows["_tok"]["depth"][:, f] - gt_depth[:, f]).abs()[mk].mean().item()
                    print(f"[DBG/{test_name}]   frame {f}: depth L1 forecast {e_fc:.3f} m  tok {e_tok:.3f} m",
                          flush=True)
            if it % 50 == 0:
                print(f"[INFO] {test_name}: {n_items} windows, {(time.time() - t0) / n_items:.2f} s/window",
                      flush=True)

        results = {k: v / n_items for k, v in sums.items()}
        r = torch.cat(ratios) if ratios else torch.zeros(0)
        if r.numel():
            q = torch.quantile(r.float(), torch.tensor([0.1, 0.5, 0.9]))
            results.update({"ScaleRatio_p10": q[0].item(), "ScaleRatio_p50": q[1].item(),
                            "ScaleRatio_p90": q[2].item()})
        results["ScaleFallback"] = n_fallback
        print(f"[Eval/{test_name}] n={n_items}  " + "  ".join(f"{k}: {v:.4f}" for k, v in results.items()),
              flush=True)
        out = os.path.join(output_dir, f"{_sanitize(test_name)}.json")
        with open(out, "w") as f:
            json.dump({"test_name": test_name, "n": n_items, "metrics": results, "args": vars(args),
                       "sec_per_window": (time.time() - t0) / max(n_items, 1)}, f, indent=2)
        print(f"[INFO] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
