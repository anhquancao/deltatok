#!/usr/bin/env python3
"""
Evaluate monocular/multi-view depth on the nuScenes val set against GT LiDAR.

Two model paths are supported:

* ``--model da3`` (default): run DA3 and fit a per-view least-squares scalar
  ``s`` so that ``s * pred ≈ gt`` on valid LiDAR pixels before computing
  Eigen depth metrics (Abs Rel, Sq Rel, RMSE, Log RMSE, δ < 1.25^k).
* ``--model infinidepth_depth_sensor``: run InfiniDepth_DepthSensor with the
  nuScenes LiDAR projection as both the full sensor depth and the source
  for a sparse prompt (~1500 random valid points). Predictions are metric,
  so no LS scaling is applied.

Usage:
    conda activate occany && source env_bsc.sh && python eval_depth_metrics.py \
        --dataset nuscenes --setting surround --model da3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
VENDORED_IMPORT_PATHS = [
    REPO_ROOT / "third_party",
    REPO_ROOT / "third_party" / "dust3r",
    REPO_ROOT / "third_party" / "croco" / "models" / "curope",
    REPO_ROOT / "third_party" / "Grounded-SAM-2",
    REPO_ROOT / "third_party" / "Grounded-SAM-2" / "grounding_dino",
    REPO_ROOT / "third_party" / "sam3",
    REPO_ROOT / "third_party" / "Depth-Anything-3" / "src",
    REPO_ROOT / "third_party" / "pyTorchChamferDistance",
    REPO_ROOT / "third_party" / "InfiniDepth",
]
for vendored_path in reversed(VENDORED_IMPORT_PATHS):
    vendored_path_str = str(vendored_path)
    if vendored_path.exists() and vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)

torch.backends.cuda.matmul.allow_tf32 = True

DEFAULT_DA3_MODEL_NAME = "depth-anything/DA3-GIANT-1.1"
DEFAULT_INFINIDEPTH_CKPT = "/home/it4i-anhquan/OccAny/checkpoints/infinidepth_depthsensor.ckpt"
IMAGE_SIZE = 512

DEPTH_METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3")
PERCENT_METRIC_KEYS = {"delta_1", "delta_2", "delta_3"}

from occany.loss.infinidepth_pseudo import (
    denormalize_imagenet,
    load_infinidepth_model,
    predict_infinidepth_view,
    sample_sparse_prompt,
)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DA3 or InfiniDepth_DepthSensor against GT LiDAR on nuScenes val.",
    )
    parser.add_argument("--dataset", "-ds", type=str, default="nuscenes", choices=["nuscenes"])
    parser.add_argument("--setting", type=str, default="surround", choices=["surround"])
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "da3"),
        choices=["da3", "infinidepth_depth_sensor"],
        help="Which depth model to evaluate.",
    )
    parser.add_argument(
        "--da3_model_name",
        type=str,
        default=os.environ.get("DA3_MODEL_NAME", DEFAULT_DA3_MODEL_NAME),
        help="HuggingFace model name/path for the DA3 backbone.",
    )
    parser.add_argument(
        "--infinidepth_ckpt",
        type=str,
        default=os.environ.get("INFINIDEPTH_CKPT", DEFAULT_INFINIDEPTH_CKPT),
        help="Checkpoint path for InfiniDepth_DepthSensor.",
    )
    parser.add_argument(
        "--infinidepth_num_prompt_samples",
        type=int,
        default=int(os.environ.get("INFINIDEPTH_NUM_PROMPT_SAMPLES", 1500)),
        help="Number of sparse prompt samples drawn from gt_depths for InfiniDepth.",
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=os.environ.get("EXP_NAME", "da3_gtscale_depth"),
        help="Experiment prefix for the output subdirectory.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=float(os.environ.get("DEPTH_MAX", 80.0)),
        help="Maximum GT depth (metres) for valid pixels and metric computation.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=float(os.environ.get("DEPTH_MIN", 1e-3)),
        help="Minimum valid GT/pred depth (metres) for scale fitting and metrics.",
    )
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default=os.path.join(os.environ.get("PROJECT", "."), "data/nuscenes"),
    )
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", 4)))
    parser.add_argument(
        "--silent",
        action="store_true",
        default=os.environ.get("SILENT", "").strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument("--world", type=int, default=int(os.environ.get("WORLD", 1)))
    parser.add_argument("--pid", type=int, default=int(os.environ.get("PID", 0)))
    parser.add_argument(
        "--lite",
        action="store_true",
        default=os.environ.get("LITE", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Evaluate on the first 200 samples only (quick smoke run).",
    )
    parser.add_argument(
        "--lite_num_samples",
        type=int,
        default=int(os.environ.get("LITE_NUM_SAMPLES", 200)),
        help="Number of samples kept when --lite is set.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=int(os.environ.get("SHUFFLE_SEED", 42)),
        help="Seed for the deterministic dataset permutation applied before iteration.",
    )
    parser.add_argument(
        "--num_save_samples",
        type=int,
        default=int(os.environ.get("NUM_SAVE_SAMPLES", 8)),
        help="Number of (shuffled) samples whose RGB+depth grid is dumped to depth_samples/.",
    )
    return parser


def get_output_resolution(dataset_name: str) -> Tuple[int, int]:
    if dataset_name == "nuscenes":
        return 518, 294
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_output_dir(args: argparse.Namespace) -> Path:
    parts = [args.exp_name, args.model, args.dataset, args.setting, f"img{IMAGE_SIZE}"]
    run_name = "_".join(part for part in parts if part)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_da3_model(device: str, da3_model_name: str) -> torch.nn.Module:
    from depth_anything_3.api import DepthAnything3

    print(f"Loading DA3 model: {da3_model_name}")
    da3 = DepthAnything3.from_pretrained(da3_model_name)
    da3 = da3.to(device)
    da3.eval()
    da3.requires_grad_(False)
    return da3


def finite_or_none(value):
    return float(value) if (value is not None and np.isfinite(value)) else None


def least_squares_scale(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    """Fit a single scalar s minimising ||s * pred - gt||^2."""
    num = torch.dot(gt, pred)
    den = torch.dot(pred, pred).clamp_min(eps)
    return float(num / den)


def compute_view_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_min: float,
    depth_max: float,
    fit_scale: bool = True,
) -> Tuple[Dict[str, Optional[float]], Optional[float], int]:
    """Compute Eigen depth metrics, optionally fitting a per-view LS scale first."""
    from occany.utils.recon_eval_helper import evaluate_depth_metrics

    pred_flat = pred_depth.reshape(-1).float()
    gt_flat = gt_depth.reshape(-1).float()
    valid = (
        (gt_flat > depth_min)
        & (gt_flat < depth_max)
        & (pred_flat > depth_min)
        & torch.isfinite(gt_flat)
        & torch.isfinite(pred_flat)
    )
    num_valid = int(valid.sum().item())

    none_metrics: Dict[str, Optional[float]] = {key: None for key in DEPTH_METRIC_KEYS}
    if num_valid < 50:
        return none_metrics, None, num_valid

    if fit_scale:
        scale = least_squares_scale(pred_flat[valid], gt_flat[valid])
        if not np.isfinite(scale) or scale <= 0:
            return none_metrics, None, num_valid
        metrics = evaluate_depth_metrics(pred_depth * scale, gt_depth, max_depth=depth_max)
        return metrics, scale, num_valid

    metrics = evaluate_depth_metrics(pred_depth, gt_depth, max_depth=depth_max)
    return metrics, 1.0, num_valid


def aggregate_metrics(per_view_results: List[Dict[str, object]]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {
        "num_views": len(per_view_results),
        "num_valid_views": sum(1 for item in per_view_results if item["abs_rel"] is not None),
        "mean_valid_pixels": float(
            np.mean([item["num_valid_pixels"] for item in per_view_results])
        ) if per_view_results else 0.0,
    }
    finite_scales = [item["scale"] for item in per_view_results if item.get("scale") is not None]
    aggregate["mean_scale"] = float(np.mean(finite_scales)) if finite_scales else None
    aggregate["median_scale"] = float(np.median(finite_scales)) if finite_scales else None

    for key in DEPTH_METRIC_KEYS:
        finite_values = [
            float(item[key])
            for item in per_view_results
            if item.get(key) is not None
        ]
        aggregate[key] = float(np.mean(finite_values)) if finite_values else None
    return aggregate


def print_aggregate_metrics(title: str, aggregate: Dict[str, object]) -> None:
    def _format_value(value: object, is_percent: bool = False) -> str:
        if value is None:
            return "None"
        if isinstance(value, (float, np.floating)):
            numeric_value = float(value) * 100.0 if is_percent else float(value)
            suffix = "%" if is_percent else ""
            return f"{numeric_value:.5f}{suffix}"
        return str(value)

    print("=" * 60)
    print(title)
    print(f"num_views:        {aggregate['num_views']}")
    print(f"num_valid_views:  {aggregate['num_valid_views']}")
    print(f"mean_scale:       {_format_value(aggregate['mean_scale'])}")
    print(f"median_scale:     {_format_value(aggregate['median_scale'])}")

    print("--- Depth Metrics ---")
    for key in DEPTH_METRIC_KEYS:
        print(f"{key:>16}: {_format_value(aggregate[key], is_percent=(key in PERCENT_METRIC_KEYS))}")
    print(f"mean_valid_pixels:{_format_value(aggregate['mean_valid_pixels'])}")


def save_depth_grid(
    save_path: Path,
    rgb_views: torch.Tensor,
    gt_depth_views: torch.Tensor,
    pred_depth_views: torch.Tensor,
    depth_min: float,
    depth_max: float,
) -> None:
    """Write a per-sample grid (rows=views, cols=[RGB, GT depth, Pred depth]) as PNG.

    rgb_views: (V, 3, H, W) in [0, 1]. gt/pred depths: (V, H, W) in metres
    (pred should already be scaled if a per-view LS scale was fitted).
    """
    import cv2
    from occany.utils.helpers import depth2rgb

    rgb_np = (rgb_views.clamp(0.0, 1.0).detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
    rgb_bgr = rgb_np[..., ::-1]
    rgb_col = np.concatenate(list(rgb_bgr), axis=0)

    gt_panels: List[np.ndarray] = []
    pred_panels: List[np.ndarray] = []
    for view_idx in range(rgb_views.shape[0]):
        gt_np = gt_depth_views[view_idx].detach().float().cpu().numpy()
        pred_np = pred_depth_views[view_idx].detach().float().cpu().numpy()
        gt_valid = (gt_np > depth_min) & (gt_np < depth_max) & np.isfinite(gt_np)
        pred_valid = (pred_np > depth_min) & np.isfinite(pred_np)
        gt_panels.append(
            depth2rgb(gt_np, valid_mask=gt_valid, min_depth=0.0, max_depth=depth_max)
        )
        pred_panels.append(
            depth2rgb(
                np.clip(pred_np, 0.0, depth_max),
                valid_mask=pred_valid,
                min_depth=0.0,
                max_depth=depth_max,
            )
        )
    gt_col = np.concatenate(gt_panels, axis=0)
    pred_col = np.concatenate(pred_panels, axis=0)
    grid = np.concatenate([rgb_col, gt_col, pred_col], axis=1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), grid)


def get_sample_id(data: Dict[str, List[str]], sample_idx: int) -> str:
    return f"{data['scene_name'][sample_idx]}_{data['begin_frame_token'][sample_idx]}"


def sanitize_sample_id(sample_id: str) -> str:
    safe_id = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in sample_id)
    safe_id = safe_id.strip("_")
    return safe_id if safe_id else "sample"


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    from occany.datasets.eval_helper import prepare_eval_setting

    output_resolution = get_output_resolution(args.dataset)
    output_dir = build_output_dir(args)

    if not args.silent:
        print(f"Output directory: {output_dir}")
        print(f"Output resolution: {output_resolution}")
        print(f"Sharding: pid={args.pid}, world={args.world}")

    if args.model == "da3":
        depth_model = load_da3_model(args.device, args.da3_model_name)
        fit_scale = True
    elif args.model == "infinidepth_depth_sensor":
        depth_model = load_infinidepth_model(args.device, args.infinidepth_ckpt)
        fit_scale = False
    else:
        raise ValueError(f"Unknown --model: {args.model}")

    dataset, collate_fn, recon_view_idx = prepare_eval_setting(
        dataset=args.dataset,
        setting=args.setting,
        image_size=IMAGE_SIZE,
        process_id=args.pid,
        num_worlds=args.world,
        split=args.split,
        base_model="da3",
        nuscenes_root=args.nuscenes_root,
    )

    rng = np.random.default_rng(args.shuffle_seed)
    shuffled_indices = rng.permutation(len(dataset)).tolist()
    if not args.silent:
        print(
            f"[shuffle] Deterministic permutation with seed={args.shuffle_seed} "
            f"over {len(dataset)} samples."
        )
    dataset = torch.utils.data.Subset(dataset, shuffled_indices)

    if args.lite:
        lite_n = min(args.lite_num_samples, len(dataset))
        print(f"[lite] Restricting evaluation to first {lite_n}/{len(dataset)} shuffled samples.")
        dataset = torch.utils.data.Subset(dataset, list(range(lite_n)))

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    per_view_results: List[Dict[str, object]] = []
    scale_mode = "per_view_least_squares_to_gt_lidar" if fit_scale else "metric_no_scaling"
    progress_desc = f"{args.model} depth eval ({scale_mode}, pid {args.pid}/{args.world})"

    samples_saved = 0
    samples_dir = output_dir / "depth_samples"

    for data in tqdm(data_loader, desc=progress_desc, disable=args.silent):
        imgs = data["imgs"].to(args.device)              # (B, V, 3, H, W)
        gt_depths = data["gt_depths"].to(args.device)    # (B, V, H, W)
        imgs = imgs[:, recon_view_idx]
        gt_depths = gt_depths[:, recon_view_idx]
        batch_size, num_views = imgs.shape[0], imgs.shape[1]
        need_save = samples_saved < args.num_save_samples

        if args.model == "da3":
            with torch.inference_mode():
                output = depth_model(imgs, export_feat_layers=[])
            pred_depths = output["depth"]                # (B, V, H, W)
        else:
            imgs_raw = denormalize_imagenet(imgs)
            pred_views = []
            with torch.inference_mode():
                for sample_idx in range(batch_size):
                    row = []
                    for view_idx in range(num_views):
                        pred_hw = predict_infinidepth_view(
                            depth_model,
                            imgs_raw[sample_idx, view_idx],
                            gt_depths[sample_idx, view_idx],
                            num_prompt_samples=args.infinidepth_num_prompt_samples,
                            depth_min=args.depth_min,
                            depth_max=args.depth_max,
                        )
                        row.append(pred_hw)
                    pred_views.append(torch.stack(row, dim=0))
            pred_depths = torch.stack(pred_views, dim=0)  # (B, V, H, W)
            output = None

        imgs_raw_for_save = (
            denormalize_imagenet(imgs).detach().cpu() if need_save else None
        )

        for sample_idx in range(batch_size):
            sample_id = get_sample_id(data, sample_idx)
            safe_sample_id = sanitize_sample_id(sample_id)
            scaled_pred_views: List[torch.Tensor] = []
            for view_idx in range(num_views):
                pred = pred_depths[sample_idx, view_idx].detach()
                gt = gt_depths[sample_idx, view_idx].detach()
                metrics, scale, num_valid = compute_view_metrics(
                    pred,
                    gt,
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                    fit_scale=fit_scale,
                )
                result: Dict[str, object] = {
                    "sample_id": safe_sample_id,
                    "view_idx": view_idx,
                    "num_valid_pixels": num_valid,
                    "scale": finite_or_none(scale),
                }
                for key in DEPTH_METRIC_KEYS:
                    result[key] = finite_or_none(metrics[key])
                per_view_results.append(result)

                if need_save:
                    view_scale = (
                        scale
                        if (
                            fit_scale
                            and scale is not None
                            and np.isfinite(scale)
                            and scale > 0
                        )
                        else 1.0
                    )
                    scaled_pred_views.append((pred.float() * view_scale).cpu())

            if need_save and samples_saved < args.num_save_samples:
                grid_path = samples_dir / f"{samples_saved:03d}_{safe_sample_id}.png"
                save_depth_grid(
                    grid_path,
                    imgs_raw_for_save[sample_idx],
                    gt_depths[sample_idx].detach().cpu(),
                    torch.stack(scaled_pred_views, dim=0),
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                )
                if not args.silent:
                    print(f"[save] depth_samples/{grid_path.name}")
                samples_saved += 1

        if not args.silent and len(per_view_results) and (
            len(per_view_results) % max(1, (len(dataset) * len(recon_view_idx)) // 10) == 0
        ):
            running_aggregate = aggregate_metrics(per_view_results)
            print_aggregate_metrics(
                f"Running metrics after {len(per_view_results)} view(s)",
                running_aggregate,
            )

        del output, pred_depths
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_metrics(per_view_results)
    results = {
        "config": {
            "model": args.model,
            "da3_model_name": args.da3_model_name if args.model == "da3" else None,
            "infinidepth_ckpt": (
                args.infinidepth_ckpt if args.model == "infinidepth_depth_sensor" else None
            ),
            "infinidepth_num_prompt_samples": (
                args.infinidepth_num_prompt_samples
                if args.model == "infinidepth_depth_sensor"
                else None
            ),
            "dataset": args.dataset,
            "setting": args.setting,
            "split": args.split,
            "image_size": IMAGE_SIZE,
            "depth_max": args.depth_max,
            "depth_min": args.depth_min,
            "scale_mode": scale_mode,
            "pid": args.pid,
            "world": args.world,
            "lite": args.lite,
            "lite_num_samples": args.lite_num_samples if args.lite else None,
        },
        "aggregate": aggregate,
        "per_view": per_view_results,
    }

    results_path = output_dir / f"depth_metrics_pid{args.pid}.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    print_aggregate_metrics("Aggregate depth metrics", aggregate)
    print(f"Saved metrics to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
