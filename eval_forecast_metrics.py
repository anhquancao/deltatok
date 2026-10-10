#!/usr/bin/env python3
"""Score dumped forecasts with the DeltaTok flow-eval metrics.

Reads <pred_dir>/<set>/<n:05d>.npz for loader window n (format: occrae/forecast_dump.py).
Frames 0-1 are context, 2..T-1 are forecast. No fit or alignment: depth (LossDepth, AbsRel, delta1), points
(LossPointmap, Chamfer without Umeyama) and pose (ATE, RTE, RRE) as predicted.

Rows: main (saved scale), ``_oracle`` (median GT scale on forecast frames).
Plan: docs/research/plan/2026-10-05_flow_forecast_dump_then_score.md

Usage (BSC GPU node):
  source env_bsc.sh && python eval_forecast_metrics.py \
      --config-name eval_deltatok_flow_alldata_ctx2fwd8_kitti_bsc \
      --pred_dir /gpfs/scratch/ehpc1001/quan/forecast_preds/vggt_world_alldata_ctx2fwd8/cityscapes_stride1_native
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(Path(__file__).resolve().parent)

torch.backends.cuda.matmul.allow_tf32 = False  # float32 NN distances must not run in TF32

from depth_anything_3.utils.geometry import affine_inverse  # noqa: E402
from occany.datasets import get_data_loader  # noqa: E402
from occany.loss import PointmapLoss, DepthLosses  # noqa: E402
from occrae.chamfer_metrics import compute_chamfer_metrics  # noqa: E402
from occrae.deltatok_shared import DeltaTokSharedMixin  # noqa: E402
from occrae.forecast_dump import load_windows  # noqa: E402
from occrae.visualization_helper import _build_bev_panel, _log_viz_sample  # noqa: E402

# dust3r's dataset import sets file_system; a sibling job's /dev/shm cleanup then kills the loader.
torch.multiprocessing.set_sharing_strategy("file_descriptor")

N_CTX = 2  # frames 0-1 given


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Score dumped forecasts on the DeltaTok flow eval", add_help=True)
    parser.add_argument("--config-dir", type=str, default="configs/deltatok_flow")
    parser.add_argument("--config-name", type=str, default="eval_deltatok_flow_alldata_ctx2fwd8_kitti_bsc")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--num_items", type=int, default=None,
                        help="Windows per set; above the split takes it whole (window_stride sets the size).")
    parser.add_argument("--test_filter", type=str, default=None)
    parser.add_argument("--bsize", type=int, default=4)
    parser.add_argument("--no_chamfer", action="store_true")
    parser.add_argument("--verbose_batches", type=int, default=1,
                        help="Print per-frame depth L1 and step length for the first N batches of each set.")
    parser.add_argument("--viz", type=int, default=0, help="Save the trainer's eval panel for the first N windows of each set.")
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
    """The trainer's batch conversion and frame losses; criteria as deltatok_flow_trainer.py:181-182."""

    def __init__(self, device):
        self.device = device
        self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
        self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)

    def _frame_losses(self, dec, batch, fslice):
        """_compute_frame_losses (deltatok_shared.py:415) without the raymap term."""
        mask = batch["gt_mask"][:, fslice].to(self.device)                              # (B, F, H, W)
        B, F, H, W = mask.shape
        loss_pm, _ = self.pointmap_criterion(
            dec["pointmap"][:, fslice].float(),
            batch["gt_pointmap"][:, fslice].to(self.device).float(),
            mask=mask
        )
        pred_d = dec["depth"][:, fslice].float().reshape(B * F, 1, H, W)
        gt_d = batch["gt_depth"][:, fslice].to(self.device).float().reshape(B * F, 1, H, W)
        loss_d, _ = self.depth_criterion(pred_d, gt_d, confidence=None, mask=mask.reshape(B * F, 1, H, W).float())
        return loss_pm, loss_d


def _oracle_scale(depth, gt_depth, gt_mask, fslice):
    """Per-window median(GT) / median(pred) over valid forecast pixels. (B,)"""
    B = depth.shape[0]
    scale = torch.ones(B, device=depth.device)
    for b in range(B):
        m = gt_mask[b, fslice]
        if m.sum() > 0:
            scale[b] = gt_depth[b, fslice][m].median() / depth[b, fslice][m].median().clamp_min(1e-6)
    return scale


def _scaled_outputs(depth, point, c2w, scale):
    """Apply a per-window scale about the frame-0 origin."""
    s = scale[:, None, None, None]                                                     # (B, 1, 1, 1)
    c2w_s = c2w.clone()
    c2w_s[..., :3, 3] = c2w_s[..., :3, 3] * scale[:, None, None]                       # (B, T, 4, 4)
    return {"depth": depth * s, "pointmap": point * s[..., None], "c2w": c2w_s}


def _depth_metrics(depth, gt_depth, gt_mask, fslice):
    """Per-window AbsRel and delta1 over valid forecast pixels. (B,), (B,)"""
    d, g = depth[:, fslice], gt_depth[:, fslice]                                       # (B, F, H, W)
    m = gt_mask[:, fslice].float()                                                     # (B, F, H, W)
    n = m.sum((1, 2, 3)).clamp_min(1)                                                  # (B,)
    abs_rel = ((d - g).abs() / g.clamp_min(1e-6) * m).sum((1, 2, 3)) / n              # (B,)
    ratio = torch.maximum(d / g.clamp_min(1e-6), g / d.clamp_min(1e-6))                # (B, F, H, W)
    delta1 = ((ratio < 1.25).float() * m).sum((1, 2, 3)) / n                           # (B,)
    return abs_rel, delta1


def _pose_errors(c2w, gt_c2w, fslice):
    """Per-window ATE (m), RTE (m), RRE (deg): no fit or alignment, frame-0 coordinates.
    ATE: RMSE of camera positions c2w[:3, 3] over forecast frames. RTE / RRE: RMSE of the
    error in each step f-1 -> f into a forecast frame, in the previous camera's axes. (B,) x 3"""
    P, P_gt = c2w.double(), gt_c2w.double()                                            # (B, T, 4, 4)
    ate = (P[:, fslice, :3, 3] - P_gt[:, fslice, :3, 3]).norm(dim=-1).pow(2).mean(1).sqrt()  # (B,)
    cur = torch.arange(fslice.start, P.shape[1], device=P.device)                      # (F,) forecast frames
    step = affine_inverse(P[:, cur - 1]) @ P[:, cur]                                   # (B, F, 4, 4) predicted motion f-1 -> f
    step_gt = affine_inverse(P_gt[:, cur - 1]) @ P_gt[:, cur]                          # (B, F, 4, 4)
    E = affine_inverse(step_gt) @ step                                                 # (B, F, 4, 4) per-step error
    rte = E[..., :3, 3].norm(dim=-1).pow(2).mean(1).sqrt()                             # (B,)
    R = E[..., :3, :3]                                                                 # (B, F, 3, 3) per-step rotation error
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)                             # (B, F, 3) = 2 sin(θ) · axis
    ang = torch.atan2(v.norm(dim=-1), R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)       # (B, F) rad; tr - 1 = 2 cos(θ)
    rre = torch.rad2deg(ang).pow(2).mean(1).sqrt()                                     # (B,)
    return ate, rte, rre


def _chamfer_rows(pointmap, gt_pointmap, mask):
    """Gen3R metrics per window: (B, 4) acc, comp, chamfer, relative %."""
    rows = []
    for b in range(pointmap.shape[0]):
        rows.append([v.item() for v in compute_chamfer_metrics(pointmap[b], gt_pointmap[b], mask[b], align=False)[:4]])
    return torch.tensor(rows)                                                           # (B, 4)


def main() -> None:
    args = get_args_parser().parse_args()
    device = torch.device("cuda")

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)
    with open_dict(cfg):
        cfg.training.val_bsize = args.bsize

    scorer = _Scorer(device)

    loaders = _build_test_loaders(cfg, args.num_items, args.test_filter)
    for test_name, loader in loaders.items():
        # Same window order as the generator.
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        ds = getattr(loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(0)

        set_dir = os.path.join(args.pred_dir, _sanitize(test_name))
        sums = defaultdict(float)
        ratios, n_items = [], 0
        t0 = time.time()
        for it, raw in enumerate(loader):
            batch = scorer._normalize_batch(raw)
            gt_depth = batch["gt_depth"].to(device).float()                              # (B, T, H, W)
            B, T, H, W = gt_depth.shape
            assert batch["num_cameras"] == 1, "cam-0 windows only"
            fslice = slice(N_CTX, T)
            gt_mask = batch["gt_mask"].to(device).bool().reshape(gt_depth.shape)         # (B, T, H, W)

            depth, point, c2w, dense = load_windows(set_dir, n_items, batch, device)    # (B, T, H, W), (B, T, H, W, 3), (B, T, 4, 4)
            s_or = _oracle_scale(depth, gt_depth, gt_mask, fslice)                       # (B,) on top of the saved scale

            rows = {
                "": {"depth": depth, "pointmap": point, "c2w": c2w},
                "_oracle": _scaled_outputs(depth, point, c2w, s_or),
            }
            for b in range(min(B, args.viz - n_items)):  # deltatok_flow_trainer.py eval panel layout
                if dense[b] is None:  # only the dumps' first N_DENSE (16) windows per set
                    continue
                order = sorted(range(T), key=lambda v: int(batch["timesteps"][b][v]))
                bev = [_build_bev_panel(c, order, H, H) for c in (c2w[b], batch["gt_c2w"][b])]
                vis = depth.clone()
                vis[b] = dense[b]                                                        # (T, H, W)
                _log_viz_sample(batch, {"depth": vis}, b, 0, 0, os.path.join(args.pred_dir, "eval_viz"),
                                None, f"eval_depth/{test_name}", extra_panels=bev, view_order=order,
                                context_mask=[int(batch["timesteps"][b][v]) < N_CTX for v in order],
                                col_titles=["RGB", "Pred Depth", "BEV (pred)", "BEV (GT)"])
            gt_c2w = batch["gt_c2w"].to(device).float()                                  # (B, T, 4, 4) frame-0 coords
            gt_pm = batch["gt_pointmap"][:, fslice].to(device).float()                   # (B, F, H, W, 3)
            batch_losses = {}
            for suffix, dec in rows.items():
                l_pm, l_d = scorer._frame_losses(dec, batch, fslice)
                abs_rel, delta1 = _depth_metrics(dec["depth"], gt_depth, gt_mask, fslice)
                ate, rte, rre = _pose_errors(dec["c2w"], gt_c2w, fslice)
                batch_losses.update({f"LossDepth{suffix}": l_d.item(), f"AbsRel{suffix}": abs_rel.mean().item(),
                                     f"Delta1{suffix}": delta1.mean().item(), f"LossPointmap{suffix}": l_pm.item(),
                                     f"ATE{suffix}": ate.mean().item(), f"RTE{suffix}": rte.mean().item()})
                if not suffix:  # rotation does not depend on scale
                    batch_losses["RRE"] = rre.mean().item()
                if not args.no_chamfer:  # unaligned, so _oracle differs
                    cr = _chamfer_rows(dec["pointmap"][:, fslice], gt_pm, gt_mask[:, fslice])
                    for j, k in enumerate(("ChamferAcc", "ChamferComp", "Chamfer", "ChamferRel")):
                        batch_losses[f"{k}{suffix}"] = cr[:, j].mean().item()

            for k, v in batch_losses.items():
                sums[k] += v * B
            n_items += B
            ratios.append((1 / s_or).cpu())                                              # saved scale / GT-median scale

            if it < args.verbose_batches:
                print(f"[DBG/{test_name}] s_oracle {s_or.tolist()}  frame-0 position error "
                      f"{(c2w[:, 0, :3, 3] - gt_c2w[:, 0, :3, 3]).norm(dim=-1).tolist()} m", flush=True)
                gt_t = gt_c2w[0, :, :3, 3].double()                                      # (T, 3) frame-0 coords
                pr_t = c2w[0, :, :3, 3].double()                                         # (T, 3) saved scale
                for f in range(1, T):
                    e = (depth[:, f] - gt_depth[:, f]).abs()[gt_mask[:, f]].mean().item()
                    print(f"[DBG/{test_name}]   frame {f}: depth L1 {e:.3f} m  step "
                          f"{(pr_t[f] - pr_t[f - 1]).norm().item():.2f} m (GT {(gt_t[f] - gt_t[f - 1]).norm().item():.2f} m)",
                          flush=True)
            if it % 50 == 0:
                print(f"[INFO] {test_name}: {n_items} windows, {(time.time() - t0) / n_items:.2f} s/window",
                      flush=True)

        results = {k: v / n_items for k, v in sums.items()}
        q = torch.quantile(torch.cat(ratios).float(), torch.tensor([0.1, 0.5, 0.9]))
        results.update({"ScaleRatio_p10": q[0].item(), "ScaleRatio_p50": q[1].item(), "ScaleRatio_p90": q[2].item()})
        print(f"[Eval/{test_name}] n={n_items}  " + "  ".join(f"{k}: {v:.4f}" for k, v in results.items()),
              flush=True)
        out = os.path.join(args.pred_dir, f"{_sanitize(test_name)}.json")
        with open(out, "w") as f:
            json.dump({"test_name": test_name, "n": n_items, "metrics": results, "args": vars(args),
                       "sec_per_window": (time.time() - t0) / max(n_items, 1)}, f, indent=2)
        print(f"[INFO] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
