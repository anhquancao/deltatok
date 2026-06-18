#!/usr/bin/env python3
"""Test OccRAE encode → decode roundtrip and save output for vis_viser.py.

Usage
-----
# Encode+decode demo images, save for visualisation:
python test_occ_rae.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --input_dir ./demo_data/input \
    --output_dir results/occrae_viser_output

# Then visualise:
python vis_viser.py --input_folder results/occrae_viser_output

# Optional: also run the normal inference pipeline side-by-side for comparison:
python test_occ_rae.py --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --compare

# Decode a pre-extracted latent (.pth from extract_occany_features.py):
python test_occ_rae.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --latent_path /gpfs/scratch/ehpc558/quan/occrae_emb/ddad_processed/val_8/000046_5.pth \
    --output_dir results/occrae_viser_output
    
python test_occ_rae.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --latent_path /gpfs/scratch/ehpc558/quan/occrae_output/overfit_occrae_tokens-20260416-152550/validation_latents/step_0010000/ddad_processed/val_0/000000_0.pth \
    --output_dir results/occrae_viser_output

# Encode+decode with the trained image decoder (saves reconstructed RGB PNGs
# and uses them as point colours in pts3d_render.npy):
python test_occ_rae.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --input_dir ./demo_data/input \
    --output_dir results/occrae_viser_output \
    --img_decoder_ckpt /mnt/proj1/eu-25-92/occrae_output/occrae_img_decoder/ckpts/current.pt
"""

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(
    Path(__file__).resolve().parent,
    extra=[
        "third_party/pyTorchChamferDistance",
        "third_party/GLD/src",
    ],
)

torch.backends.cuda.matmul.allow_tf32 = True

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test OccRAE encode/decode roundtrip and save vis_viser-compatible output.",
    )
    parser.add_argument(
        "--occany_recon_ckpt",
        type=str,
        default="checkpoints/occany_plus_recon_1B.pth",
        help="Path to the OccAny reconstruction checkpoint.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./demo_data/input",
        help="Path to demo RGB input directory (same structure as inference.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/occrae_viser_output",
        help="Directory where vis_viser-compatible .npy files are written.",
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        default=None,
        help="Path to a .pth latent file produced by extract_occany_features.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--encode_layer",
        type=int,
        default=12,
        choices=[12, 18],
        help="Backbone layer at which to capture tokens (12 pre-fusion, 18 post-fusion).",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Frame interval (for timestep computation).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Also run normal inference_batch and save side-by-side output for comparison.",
    )
    parser.add_argument(
        "--point_from_depth_and_pose",
        action="store_true",
        default=False,
        help="Compute pointmap from depth, intrinsics and c2w.",
    )
    parser.add_argument(
        "--img_decoder_ckpt",
        type=str,
        default=None,
        help="Path to a trained OccRAE image-decoder checkpoint (current.pt). "
             "If set, RGB is reconstructed and used as point colours.",
    )
    parser.add_argument(
        "--img_decoder_config_path",
        type=str,
        default="third_party/GLD/configs/decoder/ViTXL",
        help="HF config dir for GeneralDecoder_Variable (matches training).",
    )
    parser.add_argument(
        "--img_decoder_hidden_size",
        type=int,
        default=9216,
        help="Decoder hidden_size (3 * 3072 for DA3-Giant 3-level cat).",
    )
    parser.add_argument(
        "--img_decoder_patch_size",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--img_decoder_view_chunk_size",
        type=int,
        default=0,
        help="Chunk size over views during decode (0 = no chunking).",
    )
    parser.add_argument(
        "--no_ema_decoder",
        action="store_true",
        default=False,
        help="Use the live decoder weights instead of the EMA copy.",
    )
    return parser


def extract_demo_rgb_images(frame_dir: str):
    """Return sorted image paths and frame identifier from a demo directory."""
    frame_dir = Path(frame_dir)
    image_paths = sorted(
        str(p) for p in frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    frame_id = frame_dir.name
    return image_paths, frame_id


def denormalize_da3_imgs_to_minus1_1(imgs: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised (B,S,C,H,W) → [-1,1] RGB."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=imgs.dtype, device=imgs.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=imgs.dtype, device=imgs.device).view(1, 1, 3, 1, 1)
    imgs = (imgs * std) + mean
    return imgs.clamp(0.0, 1.0).mul(2.0).sub(1.0)


def maybe_load_image_decoder(occ_rae, args: argparse.Namespace):
    """Load the trained MAE image decoder onto OccRAE if --img_decoder_ckpt given."""
    if not args.img_decoder_ckpt:
        return False
    occ_rae.load_image_decoder(
        ckpt_path=args.img_decoder_ckpt,
        config_path=args.img_decoder_config_path,
        hidden_size=args.img_decoder_hidden_size,
        patch_size=args.img_decoder_patch_size,
        use_ema=not args.no_ema_decoder,
    )
    return True


def save_reconstructed_pngs(rec_01: torch.Tensor, scene_dir: str, prefix: str = "recon"):
    """Save reconstructed images (B, S, 3, H, W) in [0,1] as PNGs."""
    from PIL import Image

    arr = (rec_01.clamp(0, 1).cpu().float().numpy() * 255.0).astype(np.uint8)
    b, s = arr.shape[0], arr.shape[1]
    for bi in range(b):
        for si in range(s):
            img = arr[bi, si].transpose(1, 2, 0)  # H, W, 3
            out_path = os.path.join(scene_dir, f"{prefix}_b{bi}_v{si:02d}.png")
            Image.fromarray(img).save(out_path)


def rec01_to_minus1_1(rec_01: torch.Tensor) -> torch.Tensor:
    """(B, S, 3, H, W) in [0,1] → [-1,1] for build_save_dict colour input."""
    return rec_01.clamp(0, 1).mul(2.0).sub(1.0)


def build_save_dict(
    output: Dict[str, torch.Tensor],
    imgs_minus1_1: Optional[torch.Tensor],
    batch_idx: int,
) -> dict:
    """Build a vis_viser-compatible save dict from DA3Wrapper output.

    Args:
        output: dict from OccRAE.decode() or inference_batch() (pointmap, depth_conf, c2w, intrinsics, …).
        imgs_minus1_1: optional (B, S, C, H, W) colours in [-1, 1].
        batch_idx: index into the batch dimension.
    """
    from depth_anything_3.utils.geometry import affine_inverse
    from dust3r.utils.geometry import geotrf

    pts3d = output["pointmap"][batch_idx]          # (S, H, W, 3)
    conf = output["depth_conf"][batch_idx]          # (S, H, W)
    c2w = output.get("c2w")
    if c2w is not None:
        c2w = c2w[batch_idx]                        # (S, 3, 4) or (S, 4, 4)
        if c2w.shape[-2:] == (3, 4):
            bottom = torch.tensor([0., 0., 0., 1.], device=c2w.device, dtype=c2w.dtype)
            bottom = bottom.view(1, 1, 4).expand(c2w.shape[0], 1, 4)
            c2w = torch.cat([c2w, bottom], dim=-2)  # (S, 4, 4)
    else:
        S = pts3d.shape[0]
        c2w = torch.eye(4, device=pts3d.device, dtype=pts3d.dtype).unsqueeze(0).expand(S, 4, 4).clone()

    intrinsics = output.get("intrinsics")
    if intrinsics is not None:
        intrinsics_b = intrinsics[batch_idx]        # (S, 3, 3)
        focal = torch.stack([intrinsics_b[:, 0, 0], intrinsics_b[:, 1, 1]], dim=-1)  # (S, 2)
    else:
        S = pts3d.shape[0]
        focal = torch.ones((S, 2), device=pts3d.device, dtype=pts3d.dtype)

    # pts3d_local
    w2c = affine_inverse(c2w)
    S, H, W, _ = pts3d.shape
    pts3d_local = geotrf(w2c, pts3d.reshape(S, -1, 3)).reshape(S, H, W, 3)

    # colours: (S, C, H, W) → (S, H, W, C) in [-1, 1].
    # When decoding from latents only, RGBs are unavailable, so emit zeros.
    if imgs_minus1_1 is not None:
        colors_hwc = imgs_minus1_1[batch_idx].permute(0, 2, 3, 1).cpu().numpy()
    else:
        colors_hwc = np.zeros((S, H, W, 3), dtype=np.float32)

    return {
        "pts3d": pts3d.cpu().numpy(),
        "pts3d_local": pts3d_local.cpu().numpy(),
        "colors": colors_hwc,
        "conf": conf.cpu().numpy(),
        "focal": focal.cpu().numpy(),
        "c2w": c2w.cpu().numpy(),
    }


def resolve_resolution_from_payload(payload: Dict[str, object]) -> Tuple[int, int]:
    output_resolution = payload.get("output_resolution")
    if output_resolution is None:
        raise KeyError("Latent file is missing required key: output_resolution")
    if not isinstance(output_resolution, (list, tuple)) or len(output_resolution) != 2:
        raise ValueError(
            "Expected 'output_resolution' in latent payload to be a 2-item sequence "
            f"(width, height), got: {output_resolution}"
        )
    resolution_w, resolution_h = int(output_resolution[0]), int(output_resolution[1])
    return resolution_h, resolution_w


def run_latent_mode(args: argparse.Namespace, output_dir: str) -> str:
    if args.compare:
        print("[WARN] --compare is not supported with --latent_path. Skipping comparison branch.")

    latent_path = Path(args.latent_path).expanduser().resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    payload = torch.load(str(latent_path), map_location="cpu")
    required_keys = ("tokens", "timesteps")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise KeyError(f"Latent file is missing required keys: {missing_keys}")
    
    tokens = payload["tokens"]
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Expected 'tokens' to be a torch.Tensor, got {type(tokens)}")

    if tokens.ndim == 3:
        tokens = tokens.unsqueeze(0)
    elif tokens.ndim != 4:
        raise ValueError(f"Expected token shape [S,N,D] or [B,S,N,D], got {tuple(tokens.shape)}")

    h, w = resolve_resolution_from_payload(payload)

    print(f"[INFO] Latent input: {latent_path}")
    print(f"[INFO] Inferred resolution (H, W)=({h}, {w}), token shape={tuple(tokens.shape)}")

    from occany.model.occ_rae import OccRAE

    model_input_size = max(h, w)
    occ_rae = OccRAE(
        weights_path=args.occany_recon_ckpt,
        output_resolution=(model_input_size, model_input_size),
        device=args.device,
        encode_layer=args.encode_layer,
    )
    occ_rae.eval()

    has_img_decoder = maybe_load_image_decoder(occ_rae, args)

    latents_for_decode = {
        "tokens": tokens.to(device=args.device, dtype=torch.float32),
        "H": h,
        "W": w,
    }

    with torch.inference_mode():
        decoded_output = occ_rae.decode(
            latents_for_decode,
            pose_from_depth_ray=True,
            point_from_depth_and_pose=args.point_from_depth_and_pose,
        )

        colors_minus1_1 = None
        if has_img_decoder:
            rec_01 = occ_rae.decode_to_image(
                latents_for_decode, view_chunk_size=args.img_decoder_view_chunk_size
            )
            colors_minus1_1 = rec01_to_minus1_1(rec_01)

    scene_output_dir = os.path.join(output_dir, f"{latent_path.stem}_occ_rae")
    os.makedirs(scene_output_dir, exist_ok=True)

    if has_img_decoder:
        save_reconstructed_pngs(rec_01, scene_output_dir, prefix="recon")
        print(f"[INFO] Saved reconstructed PNGs → {scene_output_dir}")

    save_path = os.path.join(scene_output_dir, "pts3d_render.npy")
    sd = build_save_dict(decoded_output, imgs_minus1_1=colors_minus1_1, batch_idx=0)
    np.save(save_path, sd)
    print(f"[INFO] Saved OccRAE decode output → {save_path}")
    return output_dir


def run_image_mode(args: argparse.Namespace, input_dir: str, output_dir: str) -> str:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frame_dirs = sorted(p for p in Path(input_dir).iterdir() if p.is_dir())
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {input_dir}")

    print(f"[INFO] Input:  {input_dir}  ({len(frame_dirs)} frame dirs)")
    print(f"[INFO] Output: {output_dir}")

    # ---- model ----
    from occany.model.occ_rae import OccRAE
    from occany.utils.inference_helper import build_demo_reconstruction_views
    from occany.utils.resolution import get_output_resolution
    from PIL import Image

    # Determine resolution from the first image
    sample_paths, _ = extract_demo_rgb_images(str(frame_dirs[0]))
    with Image.open(sample_paths[0]) as img:
        output_resolution = get_output_resolution(img.size, model_family="da3")
    model_input_size = max(output_resolution)
    print(f"[INFO] Output resolution: {output_resolution}, model_input_size: {model_input_size}")

    occ_rae = OccRAE(
        weights_path=args.occany_recon_ckpt,
        device=args.device,
        encode_layer=args.encode_layer,
    )
    occ_rae.eval()

    has_img_decoder = maybe_load_image_decoder(occ_rae, args)

    # ---- per-frame processing ----
    for frame_dir in tqdm(frame_dirs, desc="Processing frames"):
        image_paths, frame_id = extract_demo_rgb_images(str(frame_dir))
        if not image_paths:
            print(f"[WARN] No images in {frame_dir}, skipping")
            continue

        with Image.open(image_paths[0]) as img:
            frame_resolution = get_output_resolution(img.size, model_family="da3")

        recon_views = build_demo_reconstruction_views(
            image_paths=image_paths,
            output_resolution=frame_resolution,
            model_family="da3",
            semantic_family=None,
            frame_interval=args.frame_interval,
            sam3_resolution=1008,
            device=args.device,
        )

        # Stack images → (B, S, C, H, W)
        imgs = torch.stack([v["img"] for v in recon_views], dim=1)  # (B, S, C, H, W)
        B, S, C, H, W = imgs.shape

        scene_output_dir = os.path.join(output_dir, f"{frame_id}_occ_rae")
        os.makedirs(scene_output_dir, exist_ok=True)

        with torch.inference_mode():
            # ------ encode → decode roundtrip ------
            latents = occ_rae.encode(imgs)
            decoded_output = occ_rae.decode(
                latents,
                pose_from_depth_ray=True,
                point_from_depth_and_pose=args.point_from_depth_and_pose,
            )

            imgs_vis = denormalize_da3_imgs_to_minus1_1(imgs)

            colors_minus1_1 = imgs_vis
            if has_img_decoder:
                rec_01 = occ_rae.decode_to_image(
                    latents, view_chunk_size=args.img_decoder_view_chunk_size
                )
                save_reconstructed_pngs(rec_01, scene_output_dir, prefix="recon")
                colors_minus1_1 = rec01_to_minus1_1(rec_01).to(imgs_vis.dtype)
                print(f"  Saved {rec_01.shape[0] * rec_01.shape[1]} reconstructed PNGs → {scene_output_dir}")

            for j in range(B):
                sd = build_save_dict(decoded_output, colors_minus1_1, j)
                save_path = os.path.join(scene_output_dir, "pts3d_render.npy")
                np.save(save_path, sd)
                print(f"  Saved OccRAE decode output → {save_path}")

            # ------ optional: normal inference for comparison ------
            if args.compare:
                direct_output = occ_rae.model.inference_batch(
                    imgs,
                    pose_from_depth_ray=True,
                    point_from_depth_and_pose=args.point_from_depth_and_pose,
                )
                for j in range(B):
                    sd_direct = build_save_dict(direct_output, imgs_vis, j)
                    save_path_direct = os.path.join(scene_output_dir, "pts3d_direct.npy")
                    np.save(save_path_direct, sd_direct)
                    print(f"  Saved direct inference output → {save_path_direct}")

                # Report numerical difference
                pts_dec = decoded_output["pointmap"].float()
                pts_dir = direct_output["pointmap"].float()
                diff = (pts_dec - pts_dir).abs()
                print(f"  Pointmap diff:  mean={diff.mean().item():.6f}  max={diff.max().item():.6f}")

                depth_dec = decoded_output["depth"].float()
                depth_dir = direct_output["depth"].float()
                ddiff = (depth_dec - depth_dir).abs()
                print(f"  Depth diff:     mean={ddiff.mean().item():.6f}  max={ddiff.max().item():.6f}")

    return output_dir


def main():
    args = get_args_parser().parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if args.latent_path:
        if args.input_dir:
            print(f"[INFO] --latent_path was provided; ignoring --input_dir={input_dir}")
        vis_input_folder = run_latent_mode(args, output_dir)
    else:
        vis_input_folder = run_image_mode(args, input_dir, output_dir)

    vis_input_folder_rel = os.path.relpath(vis_input_folder)
    vis_cmd = f"python vis_viser.py --input_folder {shlex.quote(vis_input_folder_rel)}"
    print(f"\n[INFO] Done. Visualise with:\n{vis_cmd}")


if __name__ == "__main__":
    main()
