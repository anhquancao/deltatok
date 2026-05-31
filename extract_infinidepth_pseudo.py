#!/usr/bin/env python3
"""Extract InfiniDepth_DepthSensor pseudo-depth for preprocessed datasets.

For each ``<stem>.npz`` under the configured processed roots, loads the RGB
image and the LiDAR depthmap, runs InfiniDepth conditioned on LiDAR (full
sensor depth + every valid LiDAR pixel as the prompt), and writes a sibling
``<stem>.infinidepth.png`` containing the pseudo depth as a single-channel
16-bit PNG (KITTI-style fixed-point: ``depth_m * DEPTH_SCALE``, 0 = invalid).

Sky pixels are masked out using the frozen DA3 metric model (set to 0 in
the saved PNG, matching the existing invalid sentinel). Non-sky pixels are
not capped here; uint16 with scale=256 saturates at ~256 m, which is above
InfiniDepth's 200 m sky sentinel. The training loader is responsible for
any [depth_min, depth_max] masking. Pass ``--skip_sky_mask`` to disable.

Decoding at training time (KITTI convention, scale fixed at 256):

    import cv2
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # uint16
    pseudo_depthmap = raw.astype(np.float32) / 256.0
    invalid_mask = raw == 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from occany.utils.runtime_paths import prepend_vendored_import_paths

prepend_vendored_import_paths(REPO_ROOT)

torch.backends.cuda.matmul.allow_tf32 = True

from occany.loss.infinidepth_pseudo import (  # noqa: E402
    INFINIDEPTH_PSEUDO_SCALE as DEPTH_SCALE,
    INFINIDEPTH_PSEUDO_SUFFIX as OUTPUT_SUFFIX,
    load_infinidepth_model,
    predict_infinidepth_batched,
)
from occany.utils.helpers import depth2rgb  # noqa: E402


DEFAULT_PROCESSED_ROOTS = (
    "ddad_processed",
    "once_processed",
    "pandaset_processed",
    "waymo_processed",
    "vkitti_processed",
)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute InfiniDepth pseudo-depth for preprocessed temporal datasets.",
    )
    parser.add_argument(
        "--preprocess_data_root",
        type=str,
        default="/scratch/project/eu-25-92/data",
        help="Root containing processed dataset folders (e.g., ddad_processed).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output root. Pseudo-depth files are placed at the same relative path as the "
            "source .npz, with suffix '.infinidepth.npz'. Defaults to --preprocess_data_root "
            "(writes sibling files next to each source .npz)."
        ),
    )
    parser.add_argument(
        "--infinidepth_ckpt",
        type=str,
        default="checkpoints/infinidepth_depthsensor.ckpt",
        help="Path to the InfiniDepth_DepthSensor checkpoint.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=1e-3,
        help=(
            "Min LiDAR depth (m) used to filter LiDAR conditioning points "
            "(sensor depth + prompt) fed to InfiniDepth. Does NOT clip the output."
        ),
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=80.0,
        help=(
            "Max LiDAR depth (m) used to filter LiDAR conditioning points "
            "(sensor depth + prompt) fed to InfiniDepth. KITTI depth-eval "
            "standard; trims the unreliable LiDAR tail. Does NOT clip the output."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=(
            "Number of views per InfiniDepth forward (grouped by matching "
            "resolution). Default tuned for a 40 GB GPU at native processed "
            "resolutions (<= 518x294). Lower if you hit OOM."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help=(
            "Inner chunk size passed to predict_infinidepth_batched. Set equal "
            "to --batch_size to disable inner re-chunking."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=0,
        help="Shard index (0-based).",
    )
    parser.add_argument(
        "--world",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit on number of samples to process after sharding (-1 = all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Recompute even if the output .infinidepth.png already exists.",
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=5,
        help=(
            "Save colorized side-by-side (RGB | LiDAR | pseudo) PNGs for the "
            "first N samples processed, into --vis_dir. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="./results/infinidepth_pseudo_vis",
        help="Output directory for the verification visualizations.",
    )
    parser.add_argument(
        "--vis_max_depth",
        type=float,
        default=150.0,
        help="Max depth (m) used to normalize the depth colormap in the visualization.",
    )
    parser.add_argument(
        "--da3_metric_model_name",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
        help=(
            "HuggingFace name/path for the DA3 metric model used to predict the "
            "sky mask. Sky pixels are written as invalid (0) in the saved PNG."
        ),
    )
    parser.add_argument(
        "--sky_mask_threshold",
        type=float,
        default=0.3,
        help=(
            "Threshold on the DA3 metric sky prediction: pixels with sky score "
            "< threshold are kept; >= threshold are masked out as sky."
        ),
    )
    parser.add_argument(
        "--skip_sky_mask",
        action="store_true",
        default=False,
        help="Disable DA3 sky masking (saved depth then includes sky pixels).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.world < 1:
        raise ValueError(f"--world must be >= 1, got {args.world}")
    if not 0 <= args.pid < args.world:
        raise ValueError(f"--pid must satisfy 0 <= pid < world, got pid={args.pid}, world={args.world}")
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.depth_max <= args.depth_min:
        raise ValueError(f"--depth_max ({args.depth_max}) must exceed --depth_min ({args.depth_min})")


def iter_source_npz_paths(root: Path) -> List[Path]:
    """Find every preprocessed sample .npz under ``root``, excluding our own outputs and ``tmp/`` staging dirs."""
    paths: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "tmp"]
        for fn in filenames:
            if not fn.endswith(".npz"):
                continue
            if fn.endswith(OUTPUT_SUFFIX):
                continue
            paths.append(Path(dirpath) / fn)
    paths.sort()
    return paths


def encode_uint16_depth(depth: np.ndarray) -> np.ndarray:
    """Encode metric depth as uint16 (depth * DEPTH_SCALE, clipped to [0, 65535], 0 = invalid)."""
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0)
    scaled = np.round(depth * DEPTH_SCALE)
    scaled = np.clip(scaled, 0, 65535)
    return scaled.astype(np.uint16)


def save_side_by_side_vis(
    out_path: Path,
    image_hwc_rgb: np.ndarray,
    lidar_depth: np.ndarray,
    pseudo_depth: np.ndarray,
    max_depth: float,
) -> None:
    """Write [RGB | LiDAR-colorized | Pseudo-colorized] (H, 3W, 3) BGR PNG using depth2rgb."""
    rgb_bgr = cv2.cvtColor(image_hwc_rgb, cv2.COLOR_RGB2BGR)
    lidar_valid = np.isfinite(lidar_depth) & (lidar_depth > 0)
    pseudo_valid = np.isfinite(pseudo_depth) & (pseudo_depth > 0)
    lidar_vis = depth2rgb(lidar_depth, valid_mask=lidar_valid, min_depth=0.0, max_depth=max_depth)
    pseudo_vis = depth2rgb(pseudo_depth, valid_mask=pseudo_valid, min_depth=0.0, max_depth=max_depth)
    panel = np.concatenate([rgb_bgr, lidar_vis, pseudo_vis], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def output_path_for(
    src: Path,
    preprocess_root: Path,
    output_root: Path,
) -> Path:
    rel = src.relative_to(preprocess_root)
    return output_root / rel.with_name(rel.stem + OUTPUT_SUFFIX)


def load_sample(npz_path: Path):
    with np.load(npz_path) as data:
        image = data["image"]
        depthmap = data["depthmap"]
    return image, depthmap.astype(np.float32)


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_DA3_PATCH_SIZE = 14


def predict_non_sky_mask(
    da3_metric_model: torch.nn.Module,
    imgs_nchw_raw: torch.Tensor,
    sky_mask_threshold: float,
) -> torch.Tensor:
    """Run DA3 metric model on raw [0,1] images and return a non-sky bool mask.

    Returns shape (N, H, W) with True for non-sky pixels (to KEEP) and False
    for sky pixels (to mask out).
    """
    from depth_anything_3.utils.alignment import compute_sky_mask

    device = imgs_nchw_raw.device
    mean = _IMAGENET_MEAN.to(device=device, dtype=imgs_nchw_raw.dtype)
    std = _IMAGENET_STD.to(device=device, dtype=imgs_nchw_raw.dtype)
    imgs_norm = (imgs_nchw_raw - mean) / std

    h_orig, w_orig = imgs_norm.shape[-2], imgs_norm.shape[-1]
    pad_h = (-h_orig) % _DA3_PATCH_SIZE
    pad_w = (-w_orig) % _DA3_PATCH_SIZE
    if pad_h or pad_w:
        imgs_norm = F.pad(imgs_norm, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    imgs_bnchw = imgs_norm.unsqueeze(0)  # DA3 expects (B, N, 3, H, W).
    with torch.inference_mode():
        metric_output = da3_metric_model(imgs_bnchw, export_feat_layers=[])
    non_sky = compute_sky_mask(metric_output.sky, threshold=sky_mask_threshold)
    return non_sky.reshape(-1, *non_sky.shape[-2:])[..., :h_orig, :w_orig]


def process_paths(
    args: argparse.Namespace,
    paths: Sequence[Path],
    preprocess_root: Path,
    output_root: Path,
    model: torch.nn.Module,
    da3_metric_model: torch.nn.Module | None,
    device: torch.device,
    vis_remaining: int,
    vis_dir: Path,
) -> tuple[int, int]:
    todo: List[Path] = []
    for src in paths:
        out = output_path_for(src, preprocess_root, output_root)
        if out.exists() and not args.overwrite:
            continue
        todo.append(src)

    if not todo:
        print("[INFO] Nothing to do (all outputs exist; pass --overwrite to recompute).")
        return 0, vis_remaining

    saved = 0
    pbar = tqdm(total=len(todo), desc=f"shard {args.pid}/{args.world}")
    i = 0
    while i < len(todo):
        batch_paths: List[Path] = []
        images: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        ref_shape = None
        while i < len(todo) and len(batch_paths) < args.batch_size:
            src = todo[i]
            try:
                image, depth = load_sample(src)
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[WARN] Failed to load {src}: {exc}")
                pbar.update(1)
                i += 1
                continue
            shape = (image.shape[0], image.shape[1])
            if ref_shape is None:
                ref_shape = shape
            elif shape != ref_shape:
                break
            batch_paths.append(src)
            images.append(image)
            depths.append(depth)
            i += 1

        if not batch_paths:
            continue

        imgs_np = np.stack(images, axis=0)
        depths_np = np.stack(depths, axis=0)

        imgs_t = torch.from_numpy(imgs_np).to(device).permute(0, 3, 1, 2).float() / 255.0
        depths_t = torch.from_numpy(depths_np).to(device)

        with torch.inference_mode():
            preds = predict_infinidepth_batched(
                model=model,
                imgs_nchw_raw=imgs_t,
                gt_depth_nhw=depths_t,
                num_prompt_samples=None,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                chunk_size=args.chunk_size,
            )

        if da3_metric_model is not None:
            non_sky_mask = predict_non_sky_mask(
                da3_metric_model=da3_metric_model,
                imgs_nchw_raw=imgs_t,
                sky_mask_threshold=args.sky_mask_threshold,
            )
            preds = preds * non_sky_mask.to(preds.dtype)

        preds_np = preds.float().cpu().numpy()
        for src, pred, image, lidar in zip(batch_paths, preds_np, images, depths):
            out_path = output_path_for(src, preprocess_root, output_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(out_path), encode_uint16_depth(pred))
            if not ok:
                tqdm.write(f"[WARN] Failed to write PNG: {out_path}")
                continue
            saved += 1
            pbar.update(1)

            if vis_remaining > 0:
                rel = src.relative_to(preprocess_root)
                vis_name = rel.with_suffix("").as_posix().replace("/", "__") + ".vis.png"
                vis_path = vis_dir / vis_name
                try:
                    save_side_by_side_vis(
                        out_path=vis_path,
                        image_hwc_rgb=image,
                        lidar_depth=lidar,
                        pseudo_depth=pred,
                        max_depth=args.vis_max_depth,
                    )
                    tqdm.write(f"[VIS] Saved {vis_path}")
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"[WARN] Failed to write vis for {src}: {exc}")
                vis_remaining -= 1

    pbar.close()
    return saved, vis_remaining


def main() -> None:
    args = get_args_parser().parse_args()
    validate_args(args)

    preprocess_root = Path(args.preprocess_data_root).expanduser().resolve()
    if not preprocess_root.is_dir():
        raise FileNotFoundError(f"Preprocess data root not found: {preprocess_root}")
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else preprocess_root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_infinidepth_model(device=str(device), ckpt_path=args.infinidepth_ckpt)

    da3_metric_model = None
    if not args.skip_sky_mask:
        from depth_anything_3.api import DepthAnything3
        print(f"[INFO] Loading DA3 metric model for sky masking: {args.da3_metric_model_name}")
        da3_metric_model = DepthAnything3.from_pretrained(args.da3_metric_model_name)
        da3_metric_model = da3_metric_model.to(device)
        da3_metric_model.eval()
        da3_metric_model.requires_grad_(False)

    vis_dir = Path(args.vis_dir).expanduser().resolve()
    vis_remaining = max(0, int(args.num_vis))
    if vis_remaining > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving up to {vis_remaining} verification visualizations to {vis_dir}")

    total_saved = 0
    t0 = time.time()
    for root_name in DEFAULT_PROCESSED_ROOTS:
        root_path = preprocess_root / root_name
        if not root_path.is_dir():
            print(f"[WARN] Skipping {root_name}: not found at {root_path}")
            continue
        all_paths = iter_source_npz_paths(root_path)
        sharded = all_paths[args.pid :: args.world]
        if args.max_samples > 0:
            sharded = sharded[: args.max_samples]
        print(
            f"[INFO] {root_name}: {len(all_paths)} npz total, shard {args.pid}/{args.world} -> {len(sharded)} samples"
        )
        saved, vis_remaining = process_paths(
            args=args,
            paths=sharded,
            preprocess_root=preprocess_root,
            output_root=output_root,
            model=model,
            da3_metric_model=da3_metric_model,
            device=device,
            vis_remaining=vis_remaining,
            vis_dir=vis_dir,
        )
        total_saved += saved
        print(f"[INFO] {root_name}: saved {saved} files")

    dt = time.time() - t0
    print(f"[INFO] Done. Saved {total_saved} files in {dt:.1f}s under {output_root}")


if __name__ == "__main__":
    main()