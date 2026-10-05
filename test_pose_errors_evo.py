#!/usr/bin/env python3
"""Check eval_forecast_metrics._pose_errors against evo (the metric classes behind evo_ape / evo_rpe).

Synthetic windows of T c2w poses in frame-0 coords, OpenCV axes (y down, z forward).
GT drives ~1 m per frame with a small yaw; the prediction is GT with per-frame SE(3)
noise and a 1.1x scale drift, so every term is non-zero. Three window groups:
  small   3 deg rotation noise
  large   60 deg rotation noise (error angles up to 180 deg, away from arccos' flat end)
  exact   pred == GT (error angles near 0, where arccos is worst conditioned)
Inputs are cast to float32 first, as the scorer feeds them.

  ATE  vs evo APE translation_part, no alignment, frames N_CTX..T-1
  RTE  vs evo RPE translation_part, delta 1 frame, frames N_CTX-1..T-1 (steps f-1 -> f, f >= N_CTX)
  RRE  vs evo RPE rotation_angle_deg, same pairs

Needs evo (pip install evo). Usage
-----
python test_pose_errors_evo.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from evo.core import metrics
from evo.core.trajectory import PosePath3D

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(Path(__file__).resolve().parent)

from eval_forecast_metrics import N_CTX, _pose_errors  # noqa: E402


def _exp_so3(w):
    """Rotation matrices from rotation vectors: (..., 3) -> (..., 3, 3)."""
    K = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype)                       # (..., 3, 3) skew(w)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    return torch.linalg.matrix_exp(K - K.transpose(-1, -2))


def _se3(R, t):
    """(..., 3, 3), (..., 3) -> (..., 4, 4)."""
    P = torch.eye(4, dtype=R.dtype).repeat(*R.shape[:-2], 1, 1)
    P[..., :3, :3] = R
    P[..., :3, 3] = t
    return P


def _windows(B, T, rot_noise_deg, gen):
    """GT and predicted c2w, (B, T, 4, 4) float64, frame 0 of GT = identity."""
    yaw = torch.zeros(B, T - 1, 3, dtype=torch.float64)                       # (B, T-1, 3)
    yaw[..., 1] = torch.randn(B, T - 1, generator=gen, dtype=torch.float64) * np.deg2rad(2.0)
    fwd = torch.zeros(B, T - 1, 3, dtype=torch.float64)                       # (B, T-1, 3)
    fwd[..., 2] = 1.0 + 0.3 * torch.rand(B, T - 1, generator=gen, dtype=torch.float64)
    step = _se3(_exp_so3(yaw), fwd)                                           # (B, T-1, 4, 4) GT motion f-1 -> f
    gt = [torch.eye(4, dtype=torch.float64).repeat(B, 1, 1)]
    for f in range(T - 1):
        gt.append(gt[-1] @ step[:, f])
    gt = torch.stack(gt, 1)                                                   # (B, T, 4, 4)
    rot = torch.randn(B, T, 3, generator=gen, dtype=torch.float64) * np.deg2rad(rot_noise_deg)
    trans = torch.randn(B, T, 3, generator=gen, dtype=torch.float64) * 0.2
    pred = gt @ _se3(_exp_so3(rot), trans)                                    # (B, T, 4, 4) noise in each camera's axes
    pred[..., :3, 3] *= 1.1                                                   # scale drift about the frame-0 origin
    return gt, pred


def _evo(gt, pred):
    """evo's ATE, RTE, RRE for one window: gt, pred (T, 4, 4) float64 numpy."""
    ape = metrics.APE(metrics.PoseRelation.translation_part)                  # evo_ape without -a / -s
    ape.process_data((PosePath3D(poses_se3=list(gt[N_CTX:])), PosePath3D(poses_se3=list(pred[N_CTX:]))))
    out = [ape.get_statistic(metrics.StatisticsType.rmse)]
    for rel in (metrics.PoseRelation.translation_part, metrics.PoseRelation.rotation_angle_deg):
        rpe = metrics.RPE(rel, delta=1, delta_unit=metrics.Unit.frames, all_pairs=False)  # evo_rpe --delta 1 --delta_unit f
        rpe.process_data((PosePath3D(poses_se3=list(gt[N_CTX - 1:])), PosePath3D(poses_se3=list(pred[N_CTX - 1:]))))
        out.append(rpe.get_statistic(metrics.StatisticsType.rmse))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--frames", type=int, default=10)                     # ctx2fwd8
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(args.seed)

    groups = {}
    for name, noise in (("small", 3.0), ("large", 60.0)):
        groups[name] = _windows(args.windows, args.frames, noise, gen)
    gt, _ = _windows(4, args.frames, 0.0, gen)
    groups["exact"] = (gt, gt.clone())

    tol = {"ATE": 1e-5, "RTE": 1e-5, "RRE": 1e-4}                              # m, m, deg
    worst = {k: 0.0 for k in tol}
    for name, (gt, pred) in groups.items():
        gt32, pred32 = gt.float(), pred.float()                               # as the scorer loads them
        ours = torch.stack(_pose_errors(pred32, gt32, slice(N_CTX, args.frames)), 1).numpy()  # (B, 3)
        ref = np.array([_evo(g.double().numpy(), p.double().numpy()) for g, p in zip(gt32, pred32)])  # (B, 3)
        diff = np.abs(ours - ref)                                             # (B, 3)
        for j, k in enumerate(tol):
            worst[k] = max(worst[k], diff[:, j].max())
            print(f"[{name:5s}] {k}: ours mean {ours[:, j].mean():.6f}  evo mean {ref[:, j].mean():.6f}  "
                  f"max |diff| {diff[:, j].max():.2e}", flush=True)
    ok = all(worst[k] <= tol[k] for k in tol)
    print("[result] " + "  ".join(f"{k} max |diff| {worst[k]:.2e} (tol {tol[k]:.0e})" for k in tol)
          + f"  -> {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
