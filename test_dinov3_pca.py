"""Standalone PCA-feature visualization for the DINOv3 backbone.

Follows third_party/dinov3/notebooks/pca.ipynb:
- aspect-preserving long-side resize so both dims are multiples of patch_size
- `model.get_intermediate_layers(..., reshape=True, norm=True)` for dense
  patch features in (B, C, H_patch, W_patch) layout (cls/storage stripped)
- sklearn `PCA(n_components=3, whiten=True)` on foreground patches
- `sigmoid(2 * pca_proj)` for vibrant colors
- optional foreground mask (`--foreground_mask`) zeros out the background
  via a PCA-1-sign proxy (median-filtered), in lieu of the notebook's
  pretrained `fg_classifier.pkl`; off by default so the PCA covers the
  whole image.

Output is a side-by-side PNG: original image (left) and PCA viz (right,
displayed at native patch grid via matplotlib, no upsampling).

Usage:
    python test_dinov3_pca.py \\
        --weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \\
        --input_dir demo_data/input \\
        --output_dir results/dinov3_pca
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import signal
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from occany.utils.runtime_paths import prepend_vendored_import_paths

prepend_vendored_import_paths()

from occany.model.dinov3_backbone import build_dinov3_backbone  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="occany/configs/dinov3/occany_dinov3_vith16plus.yaml")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True,
                   help="Directory of images, or a single image path.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="Tap index to PCA; default = last value of out_layers.")
    p.add_argument("--image_size", type=int, default=512,
                   help="Long-side target (multiple of patch_size).")
    p.add_argument("--foreground_mask", action="store_true",
                   help="Apply the PCA-1 foreground mask: fit PCA-3 only on the "
                        "foreground subset and zero out the background.")
    p.add_argument("--device", type=str, default="cuda")
    return p


def resize_for_patches(img: Image.Image, image_size: int, patch_size: int) -> Image.Image:
    """Aspect-preserving resize: long side becomes image_size, both multiples of patch_size."""
    w, h = img.size
    if h >= w:
        h_patches = image_size // patch_size
        w_patches = int(round(w * image_size / (h * patch_size)))
    else:
        w_patches = image_size // patch_size
        h_patches = int(round(h * image_size / (w * patch_size)))
    return img.resize((w_patches * patch_size, h_patches * patch_size), Image.BICUBIC)


def list_images(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    out: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def pca_visualize(
    feats: np.ndarray,
    h_patches: int,
    w_patches: int,
    use_foreground_mask: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project `feats` (N, C) to RGB via PCA-3 with whitening + sigmoid.

    If use_foreground_mask, fit PCA only on the foreground subset (PCA-1 sign,
    median-filtered) and zero out the background in the output.

    Returns:
        rgb_img: (H_patch, W_patch, 3) float32 in [0, 1]
        fg_mask: (H_patch, W_patch) bool, or None if mask disabled
    """
    fg_mask = None
    if use_foreground_mask:
        pca1 = PCA(n_components=1, whiten=True).fit_transform(feats)[:, 0]
        score = pca1.reshape(h_patches, w_patches)
        score_mf = signal.medfilt2d(score.astype(np.float32), kernel_size=3)
        fg_flat = score_mf.reshape(-1) > 0
        # The sign of PCA-1 is arbitrary — pick whichever half is the minority
        # as foreground (objects are usually less than half the scene).
        if fg_flat.mean() > 0.5:
            fg_flat = ~fg_flat
            score_mf = -score_mf
        fg_mask = fg_flat.reshape(h_patches, w_patches)
        fit_feats = feats[fg_flat]
        if fit_feats.shape[0] < 3:
            # Degenerate: fall back to fitting on all patches.
            fit_feats = feats
            fg_mask = None
    else:
        fit_feats = feats

    pca = PCA(n_components=3, whiten=True).fit(fit_feats)
    proj = pca.transform(feats)
    rgb = 1.0 / (1.0 + np.exp(-2.0 * proj))  # sigmoid(2x)
    rgb_img = rgb.reshape(h_patches, w_patches, 3).astype(np.float32)

    if fg_mask is not None:
        rgb_img = rgb_img * fg_mask[..., None].astype(np.float32)

    return rgb_img, fg_mask


def main():
    args = get_args_parser().parse_args()
    patch_size = 16
    assert args.image_size % patch_size == 0, \
        f"--image_size must be a multiple of {patch_size}, got {args.image_size}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bb = build_dinov3_backbone(args.config, weights_path=args.weights)
    bb = bb.eval().to(args.device)

    out_layers = bb.out_layers
    if args.layer is None:
        tap_layer = out_layers[-1]
    else:
        if args.layer not in out_layers:
            raise SystemExit(f"--layer {args.layer} not in out_layers {out_layers}")
        tap_layer = args.layer

    for img_path in list_images(Path(args.input_dir)):
        pil = Image.open(img_path).convert("RGB")
        pil_resized = resize_for_patches(pil, args.image_size, patch_size)
        W_resized, H_resized = pil_resized.size
        h_patches = H_resized // patch_size
        w_patches = W_resized // patch_size

        x = TF.to_tensor(pil_resized)
        x_norm = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        x_in = x_norm.unsqueeze(0).to(args.device)

        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.float32):
            feats_list = bb.vit.get_intermediate_layers(
                x_in, n=(tap_layer,), reshape=True, norm=True
            )
        feat_map = feats_list[-1].squeeze(0).detach().float().cpu()  # (C, H_patch, W_patch)
        C = feat_map.shape[0]
        feats = feat_map.view(C, -1).permute(1, 0).numpy()  # (N_patch, C)

        feat_norm = np.linalg.norm(feats, axis=-1).mean()

        rgb_img, fg_mask = pca_visualize(
            feats,
            h_patches=h_patches,
            w_patches=w_patches,
            use_foreground_mask=args.foreground_mask,
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
        axes[0].imshow(pil_resized)
        axes[0].set_title(f"image  {pil_resized.size}")
        axes[0].axis("off")
        axes[1].imshow(rgb_img)
        axes[1].set_title(
            f"PCA  tap={tap_layer}  grid=({h_patches},{w_patches})"
            + ("  +fg" if fg_mask is not None else "")
        )
        axes[1].axis("off")
        fig.tight_layout()
        out_path = output_dir / f"pca_{img_path.stem}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        print(
            f"{img_path} -> {out_path}  tap={tap_layer}  "
            f"patch_grid=({h_patches}, {w_patches})  feat_norm_mean={feat_norm:.3f}"
        )


if __name__ == "__main__":
    main()
