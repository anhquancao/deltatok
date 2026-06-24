#!/usr/bin/env python3
"""Evaluate a trained OccRAE flow-matching model on pre-extracted token dumps.

Loads a checkpoint produced by train_deltatok_flow.py (via bsc_train_deltatok_flow.slurm),
runs flow-matching Euler sampling to generate target-view tokens conditioned
on the first ``cond_num`` reference views, decodes the sampled tokens through
OccRAE into 3D outputs, and saves vis_viser-compatible .npy files.

Usage
-----
# Sample from a trained model and decode to 3D:
python eval_occ_rae.py \
    --config configs/train_deltatok_flow_overfit.yaml \
    --ckpt /gpfs/scratch/ehpc551/occrae_exps/overfit/ckpts/current.pth \
    --latent_path /gpfs/scratch/ehpc558/quan/occrae_emb_overfit/ddad_processed/val_0/000000_0.pth \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --output_dir /gpfs/scratch/ehpc558/quan/occrae_output

# Also decode ground-truth tokens for side-by-side comparison:
python eval_occ_rae.py \\
    --config configs/train_deltatok_flow_overfit.yaml \\
    --ckpt /path/to/checkpoints/current.pth \\
    --latent_path /path/to/occrae_emb/dataset/scene/sample.pth \\
    --save_target

# Visualise:
python vis_viser.py --input_folder ./demo_data/output_occ_rae_eval/<stem>_eval
"""

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

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

VIT_SIZES = {
    "tiny": (384, 6, 6),
    "small": (384, 12, 6),
    "base": (768, 12, 12),
    "large": (1024, 24, 16),
    "xlarge": (1152, 28, 16),
    "xxlarge": (1536, 28, 16),
    "2b": (2048, 32, 16),
    "giant": (2048, 32, 16),
    "xxxl": (2048, 32, 16),
}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OccRAE model: sample tokens and decode to 3D.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config used for training (e.g. configs/train_deltatok_flow_overfit.yaml).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the training checkpoint (.pth).",
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        required=True,
        help="Path to a .pth token dump from extract_occany_features.py.",
    )
    parser.add_argument(
        "--occany_recon_ckpt",
        type=str,
        default="checkpoints/occany_plus_recon_1B.pth",
        help="Path to the OccAny reconstruction checkpoint for decoding.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_data/output_occ_rae_eval",
        help="Directory where vis_viser-compatible .npy files are written.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of Euler sampling steps.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling noise.")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Per-view timestep offset strength.",
    )
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="cosine",
        choices=["cosine", "square", "linear"],
    )
    parser.add_argument(
        "--save_target",
        action="store_true",
        default=False,
        help="Also decode the ground-truth tokens and save for side-by-side comparison.",
    )
    parser.add_argument(
        "--point_from_depth_and_pose",
        action="store_true",
        default=False,
        help="Compute pointmap from depth, intrinsics and c2w.",
    )
    return parser


# ---------------------------------------------------------------------------
# vis_viser-compatible output builder (adapted from test_occ_rae.py)
# ---------------------------------------------------------------------------

def build_save_dict(
    output: Dict[str, torch.Tensor],
    batch_idx: int = 0,
) -> dict:
    """Build a vis_viser-compatible save dict from OccRAE decode output (no RGB)."""
    from depth_anything_3.utils.geometry import affine_inverse
    from dust3r.utils.geometry import geotrf

    pts3d = output["pointmap"][batch_idx]
    conf = output["depth_conf"][batch_idx]
    c2w = output.get("c2w")
    if c2w is not None:
        c2w = c2w[batch_idx]
        if c2w.shape[-2:] == (3, 4):
            bottom = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=c2w.device, dtype=c2w.dtype,
            )
            bottom = bottom.view(1, 1, 4).expand(c2w.shape[0], 1, 4)
            c2w = torch.cat([c2w, bottom], dim=-2)
    else:
        S = pts3d.shape[0]
        c2w = (
            torch.eye(4, device=pts3d.device, dtype=pts3d.dtype)
            .unsqueeze(0)
            .expand(S, 4, 4)
            .clone()
        )

    intrinsics = output.get("intrinsics")
    if intrinsics is not None:
        intrinsics_b = intrinsics[batch_idx]
        focal = torch.stack(
            [intrinsics_b[:, 0, 0], intrinsics_b[:, 1, 1]], dim=-1,
        )
    else:
        S = pts3d.shape[0]
        focal = torch.ones((S, 2), device=pts3d.device, dtype=pts3d.dtype)

    w2c = affine_inverse(c2w)
    S, H, W, _ = pts3d.shape
    pts3d_local = geotrf(w2c, pts3d.reshape(S, -1, 3)).reshape(S, H, W, 3)
    colors_hwc = np.zeros((S, H, W, 3), dtype=np.float32)

    return {
        "pts3d": pts3d.cpu().numpy(),
        "pts3d_local": pts3d_local.cpu().numpy(),
        "colors": colors_hwc,
        "conf": conf.cpu().numpy(),
        "focal": focal.cpu().numpy(),
        "c2w": c2w.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Latent loading
# ---------------------------------------------------------------------------

def load_latent(
    latent_path: Path,
    output_resolution: Tuple[int, int],
) -> Tuple[torch.Tensor, dict]:
    """Load a token dump and validate it.

    Returns:
        packed_tokens: ``(V, C, S, 1)`` matching ``OccRAETokenDataset.__getitem__``.
        payload: raw dict from the ``.pth`` file.
    """
    payload = torch.load(str(latent_path), map_location="cpu", weights_only=False)
    for key in ("tokens", "timesteps", "output_resolution"):
        if key not in payload:
            raise KeyError(f"Latent file is missing required key: {key}")

    tokens = payload["tokens"]
    if tokens.ndim != 3:
        raise ValueError(
            f"Expected tokens with shape (V, seq_len, dim), got {tuple(tokens.shape)}"
        )

    num_views, seq_len, embed_dim = tokens.shape
    res_w, res_h = output_resolution
    expected_seq_len = (res_w // 14) * (res_h // 14) + 1
    if seq_len != expected_seq_len:
        raise ValueError(
            f"Token seq_len mismatch: got {seq_len}, expected {expected_seq_len} "
            f"from output_resolution={output_resolution}"
        )

    # Pack to (V, C, S, 1) matching OccRAETokenDataset.__getitem__
    packed = tokens.permute(0, 2, 1).unsqueeze(-1).contiguous()
    return packed, payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args_parser().parse_args()

    latent_path = Path(args.latent_path).expanduser().resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    output_dir = os.path.abspath(args.output_dir)

    # ---- Load config ----
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    # output_resolution may live under dataset (train_occrae_overfit) or model (train_deltatok_flow_overfit)
    _res = cfg.model.get("output_resolution")
    if _res is None:
        raise ValueError("output_resolution not found in cfg.model")
    output_resolution = list(_res)  # [W, H]
    cond_num = int(cfg.dataset.cond_num)
    num_views_cfg = int(cfg.model.get("num_views"))
    pred_mode = str(cfg.model.get("pred_mode", "x"))
    vit_size = str(cfg.model.get("vit_size", "base"))
    use_bf16 = str(cfg.training.get("dtype", "float32")).lower() == "bfloat16"

    res_w, res_h = output_resolution
    # patch_h/patch_w follow the trainer's convention (output_resolution[0]//14, [1]//14)
    patch_h = res_w // 14
    patch_w = res_h // 14

    print(f"[INFO] Config: {args.config}")
    print(
        f"[INFO] output_resolution={output_resolution}, cond_num={cond_num}, "
        f"pred_mode={pred_mode}, vit_size={vit_size}"
    )

    # ---- Load latent tokens ----
    packed_tokens, payload = load_latent(latent_path, (res_w, res_h))
    num_views = packed_tokens.shape[0]
    if num_views > num_views_cfg:
        raise ValueError(
            f"Latent has {num_views} views but the model supports at most {num_views_cfg}"
        )
    print(
        f"[INFO] Latent: {latent_path}, num_views={num_views}, "
        f"shape={tuple(packed_tokens.shape)}"
    )

    # ---- Build model ----
    from occrae.network.efficient_transformer import Transformer

    size_key = vit_size.lower()
    if size_key not in VIT_SIZES:
        print(f"[WARN] Unknown vit_size={vit_size}, defaulting to 'base'")
        size_key = "base"
    hidden_dim, depth, heads = VIT_SIZES[size_key]

    model = Transformer(
        out_dim=1536,
        num_views=num_views_cfg,
        hidden_dim=hidden_dim,
        proj=1,
        depth=depth,
        heads=heads,
        mlp_dim=hidden_dim * 4,
        dropout=float(cfg.model.get("dropout", 0.0)),
        is_causal=bool(cfg.model.get("is_causal", False)),
        use_trajectory_cond=False,
        trajectory_length=0,
        ref_spatial_size=(patch_h, patch_w),
        use_camera_embed=bool(cfg.model.get("vit_use_camera_embed", False)),
        attn_mode=str(cfg.model.get("attn_mode", "factorized")),
    )
    model = model.to(args.device)

    # ---- Load checkpoint (EMA weights) ----
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)

    ema_state = checkpoint.get("ema_state")
    if ema_state is not None and ema_state:
        from occrae.network.ema import EMA

        ema = EMA(model)
        ema.load_state_dict(ema_state, model)
        ema.copy_to(model)
        print(f"[INFO] Applied EMA weights (decay={ema.decay})")
    else:
        print("[INFO] No EMA state in checkpoint, using raw model weights")

    train_iter = checkpoint.get("iter", "?")
    print(f"[INFO] Checkpoint iteration: {train_iter}")

    model.eval()

    # ---- Set seed ----
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

    # ---- Prepare tokens for sampling ----
    # packed_tokens: (V, C, S, 1)
    # → (1, V, C, S, 1) → spatial layout (1, C, V, S, 1)
    x_tokens = packed_tokens.unsqueeze(0).to(
        device=args.device, dtype=torch.float32,
    )
    x_spatial = x_tokens.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, V, S, 1)

    print(
        f"[INFO] Sampling: num_steps={args.num_steps}, alpha={args.alpha}, "
        f"scheduler={args.scheduler_mode}, seed={args.seed}"
    )

    # ---- Sample ----
    from occrae.generation_helper import flow_euler_sample

    z = torch.randn_like(x_spatial)
    if cond_num > 0:
        z[:, :, :cond_num] = x_spatial[:, :, :cond_num]

    use_autocast = use_bf16 and args.device.startswith("cuda")
    sampled_spatial = flow_euler_sample(
        model, z,
        pred_mode=pred_mode,
        context=cond_num,
        num_steps=args.num_steps,
        alpha=args.alpha,
        scheduler_mode=args.scheduler_mode,
        autocast_ctx=torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast,
        ),
    )

    # ---- Compute MSE metrics ----
    diff_sq = (sampled_spatial.float() - x_spatial.float()).pow(2)
    diff_sq_views = diff_sq.permute(0, 2, 1, 3, 4)  # (B, V, C, S, 1)
    view_mse = diff_sq_views.mean(dim=(2, 3, 4))  # (B, V)
    ref_mse = view_mse[:, :cond_num].mean().item()
    tgt_mse = view_mse[:, cond_num:].mean().item()
    total_mse = view_mse.mean().item()
    print(
        f"[INFO] Sampling MSE: total={total_mse:.6f}, "
        f"ref={ref_mse:.6f}, tgt={tgt_mse:.6f}"
    )

    # ---- Free flow model memory before loading decoder ----
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- Convert sampled tokens for OccRAE decode ----
    # sampled_spatial: (B, C, V, S, 1) → (B, V, S, C) = (B, num_views, seq_len, embed_dim)
    tokens_for_decode = (
        sampled_spatial.squeeze(-1).permute(0, 2, 3, 1).contiguous()
    )

    # ---- Decode via OccRAE ----
    from occany.model.occ_rae import OccRAE

    model_input_size = max(res_w, res_h)
    occ_rae = OccRAE(
        weights_path=args.occany_recon_ckpt,
        output_resolution=(model_input_size, model_input_size),
        device=args.device,
        encode_layer=12,
    )
    occ_rae.eval()

    latents_for_decode = {
        "tokens": tokens_for_decode.to(device=args.device, dtype=torch.float32),
        "H": res_h,
        "W": res_w,
    }

    print("[INFO] Decoding sampled tokens via OccRAE...")
    with torch.inference_mode():
        decoded_output = occ_rae.decode(
            latents_for_decode,
            pose_from_depth_ray=True,
            point_from_depth_and_pose=args.point_from_depth_and_pose,
        )

    # ---- Save output ----
    scene_output_dir = os.path.join(output_dir, f"{latent_path.stem}_eval")
    os.makedirs(scene_output_dir, exist_ok=True)

    save_path = os.path.join(scene_output_dir, "pts3d_render.npy")
    sd = build_save_dict(decoded_output, batch_idx=0)
    np.save(save_path, sd)
    print(f"[INFO] Saved sampled output → {save_path}")

    # ---- Optionally decode and save ground-truth tokens ----
    if args.save_target:
        gt_tokens = (
            x_spatial.squeeze(-1).permute(0, 2, 3, 1).contiguous()
        )
        gt_latents = {
            "tokens": gt_tokens.to(device=args.device, dtype=torch.float32),
            "H": res_h,
            "W": res_w,
        }
        with torch.inference_mode():
            gt_decoded = occ_rae.decode(
                gt_latents,
                pose_from_depth_ray=True,
                point_from_depth_and_pose=args.point_from_depth_and_pose,
            )
        gt_save_path = os.path.join(scene_output_dir, "pts3d_target.npy")
        sd_gt = build_save_dict(gt_decoded, batch_idx=0)
        np.save(gt_save_path, sd_gt)
        print(f"[INFO] Saved ground-truth output → {gt_save_path}")

    vis_cmd = f"python vis_viser.py --input_folder {shlex.quote(scene_output_dir)}"
    print(f"\n[INFO] Done. Visualise with:\n{vis_cmd}")


if __name__ == "__main__":
    main()
