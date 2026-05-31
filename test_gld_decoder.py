#!/usr/bin/env python3
"""Test pretrained GLD MAE decoder with DA3-Base encoder features.

Extracts 4-level raw features from DA3-Base backbone at layers [5,7,9,11]
via OccRAE, strips CLS tokens, concatenates to 6144-dim, then decodes to
RGB via the pretrained GLD MAE decoder.

Usage
-----
python test_gld_decoder.py \
    --gld_ckpt checkpoints/GLD_pretrained_models/mae_decoder.pt \
    --input_dir ./demo_data/input \
    --output_dir ./results/output_gld
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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
    REPO_ROOT / "third_party" / "GLD" / "src",
]
for vendored_path in reversed(VENDORED_IMPORT_PATHS):
    vendored_path_str = str(vendored_path)
    if vendored_path.exists() and vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)

torch.backends.cuda.matmul.allow_tf32 = True

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
PATCH_SIZE = 14
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description="Test pretrained GLD MAE decoder with DA3-Base features.")
    parser.add_argument("--da3_pretrained_path", type=str,
                        default="checkpoints/GLD_pretrained_models/da3/model.safetensors",
                        help="Path to DA3-Base pretrained weights (safetensors).")
    parser.add_argument("--gld_ckpt", type=str, default="checkpoints/GLD_pretrained_models/mae_decoder.pt")
    parser.add_argument("--input_dir", type=str, default="./demo_data/input")
    parser.add_argument("--output_dir", type=str, default="./results/output_gld")
    parser.add_argument("--input_size", type=int, default=504,
                        help="Longer side resized to this (must be divisible by 14).")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_gld_decoder(ckpt_path: str, hidden_size: int, patch_size: int, device: str):
    """Load pretrained GLD GeneralDecoder_Variable from checkpoint."""
    from transformers import AutoConfig
    from stage1.decoders.decoder import GeneralDecoder_Variable

    config_path = str(REPO_ROOT / "third_party" / "GLD" / "configs" / "decoder" / "ViTXL")
    dec_config = AutoConfig.from_pretrained(config_path)
    dec_config.hidden_size = hidden_size
    dec_config.patch_size = patch_size

    decoder = GeneralDecoder_Variable(dec_config, base_image_size=(504, 504))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    if "ema_decoder" in ckpt:
        decoder_state = ckpt["ema_decoder"]
    elif "decoder" in ckpt:
        decoder_state = ckpt["decoder"]
    elif "ema" in ckpt:
        decoder_state = ckpt["ema"]
    elif "model" in ckpt:
        decoder_state = ckpt["model"]
    else:
        decoder_state = ckpt

    cleaned = {}
    for k, v in decoder_state.items():
        if k.startswith("decoder."):
            cleaned[k.replace("decoder.", "", 1)] = v
        else:
            cleaned[k] = v
    decoder_state = cleaned

    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    decoder.to(device).eval()
    print(f"[INFO] Loaded GLD decoder from {ckpt_path} (hidden_size={hidden_size})")
    return decoder


def extract_multilevel_features(occ_rae, images: torch.Tensor) -> torch.Tensor:
    """Extract 4-level raw features [local_x|current_x] from DA3-Base backbone.

    Uses the backbone's export_feat_layers to collect raw (x, local_x) at
    each out_layer, concatenates [local_x, current_x] per level (1536-dim
    for DA3-Base), strips CLS, and concatenates all 4 levels → 6144-dim.

    Args:
        occ_rae: OccRAE instance with DA3-Base backbone.
        images: (B, S, C, H, W) ImageNet-normalised.

    Returns:
        z: (B*S, N_patches, 6144)
    """
    B, S, C, H, W = images.shape
    out_layers = occ_rae.model.config["net"]["out_layers"]

    with torch.no_grad():
        _feats, aux_feats = occ_rae.model.model.backbone(
            images,
            cam_token=None,
            export_feat_layers=out_layers,
            ref_view_strategy="first",
        )

    level_feats = []
    for i in range(len(out_layers)):
        # raw_state = (current_x, local_x), both (B, S, N, embed_dim) with CLS
        current_x, local_x = aux_feats[i][1]
        raw = torch.cat([local_x, current_x], dim=-1)  # (B, S, N, 1536)
        raw = raw[:, :, 1:, :]  # strip CLS token
        level_feats.append(raw.reshape(-1, *raw.shape[2:]))  # (B*S, N_patches, 1536)

    z = torch.cat(level_feats, dim=-1)  # (B*S, N_patches, 6144)
    print(f"[INFO] Extracted features: {len(out_layers)} levels at {out_layers}, "
          f"concat shape={tuple(z.shape)}")
    return z


def load_images(image_paths, input_size, device):
    """Load images, resize so both dims are divisible by 14, ImageNet-normalize."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    target_size = None
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if target_size is None:
            w, h = img.size
            scale = input_size / max(w, h)
            new_w = int(round(w * scale / PATCH_SIZE)) * PATCH_SIZE
            new_h = int(round(h * scale / PATCH_SIZE)) * PATCH_SIZE
            target_size = (new_h, new_w)
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
        tensors.append(transform(img))

    return torch.stack(tensors).to(device)


def save_visualization(recon_01, gt_01, output_dir, frame_id):
    """Save reconstructed and GT images side-by-side as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    for v in range(recon_01.shape[0]):
        rec_np = (recon_01[v].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_np = (gt_01[v].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        combined = np.concatenate([gt_np, rec_np], axis=1)
        path = os.path.join(output_dir, f"{frame_id}_view{v:02d}.png")
        Image.fromarray(combined).save(path)
        print(f"  Saved {path} (left=GT, right=recon)")


def main():
    args = parse_args()

    from occany.model.occ_rae import OccRAE

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    frame_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories in {input_dir}")

    occ_rae = OccRAE(
        weights_path=None,
        output_resolution=(args.input_size, args.input_size),
        device=args.device,
        encode_layer=11,
        model_name="da3-base",
        da3_pretrained_path=args.da3_pretrained_path,
    )
    occ_rae.eval()

    embed_dim = occ_rae.model.config["net"]["out_layers"].__len__()
    hidden_size = 1536 * len(occ_rae.model.config["net"]["out_layers"])
    decoder = load_gld_decoder(args.gld_ckpt, hidden_size=hidden_size,
                               patch_size=PATCH_SIZE, device=args.device)

    mean = IMAGENET_MEAN.to(args.device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(args.device).view(1, 3, 1, 1)

    for frame_dir in frame_dirs:
        frame_id = frame_dir.name
        image_paths = sorted(
            str(p) for p in frame_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            continue

        imgs = load_images(image_paths, args.input_size, args.device)
        N, C, H, W = imgs.shape
        # Add batch dim: (1, N, C, H, W) — backbone expects (B, S, C, H, W)
        imgs_5d = imgs.unsqueeze(0)
        print(f"\n[INFO] Frame '{frame_id}': {N} views, resolution {H}x{W}")

        with torch.inference_mode():
            z = extract_multilevel_features(occ_rae, imgs_5d)

            logits = decoder(z, input_size=(H, W), drop_cls_token=False).logits
            recon = decoder.unpatchify(logits, (H, W))

            recon_01 = (recon * std + mean).clamp(0, 1)
            gt_01 = (imgs * std + mean).clamp(0, 1)

            l1 = (recon_01 - gt_01).abs().mean().item()
            print(f"  L1 loss (recon vs GT): {l1:.4f}")

        save_visualization(recon_01, gt_01, str(output_dir), frame_id)

    print(f"\n[INFO] Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
