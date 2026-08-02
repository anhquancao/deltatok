#!/usr/bin/env python3
"""DeltaTok delta-token Flow-Matching forecast → vis_viser.py input.

Loads a trained DeltaTok *flow* checkpoint (the flow transformer, the only
trainable net in ``DeltaTokFlowMatchingTrainer``), runs the same eval path as
``DeltaTokFlowMatchingTrainer.eval_one_epoch`` (sample delta tokens from noise,
then autoregressive DeltaTok decode rollout) on the config's test datasets,
decodes the predicted tokens back to 3D with OccRAE, and saves per-scene
``pts3d_<setting>.npy`` dicts (keys: pts3d, pts3d_local, colors, conf, focal,
c2w) that ``vis_viser.py`` consumes directly.

Settings written per scene:
  * ``pts3d_orig.npy`` — decode of the unmodified OccRAE tokens (upper bound).
  * ``pts3d_tok.npy``  — rollout from the GT first frame driven by the *GT*
                         delta tokens (tokenizer-only error; the flow model's
                         upper bound).
  * ``pts3d_flow.npy`` — the forecast: delta tokens are *sampled from noise* by
                         the flow transformer (cross-attention on the frame-0
                         patch grids), then decoded autoregressively from the GT
                         first frame (timestep 0 is GT context).

Unlike the trainer's per-timestep ``_decode_tokens`` (loss bookkeeping), all V
views are decoded jointly here so the whole sequence shares one coordinate
frame (DA3 ``ref_view_strategy="first"``), which is what viser needs.

Usage (on Jean Zay, from a GPU node; defaults target the
deltatok_flow_full_deltaCtx_global_fullT_wd05_raymapoff run's last checkpoint):
  source env_jz_h100.sh &&
  python visualize_deltatok_flow.py \
      --output_dir results/deltatok_flow_viser \
      --num_scenes 2 --num_steps 50 --cfg training.bsize=2
  # Defaults: --config-name train_deltatok_flow_jeanzay, --ckpt the run's
  # current.pth, the JZ MAE image decoder, and the run's arch flags
  # (cond_mode=delta_ctx, attn_mode=global, vit_use_camera_embed=true) so the
  # built flow ViT matches the saved weights. Pass --cfg training.bsize=2 to
  # keep RGB decode within memory (the config's bsize=16 decodes the whole
  # batch before saving only --num_scenes). Flow generation always decodes
  # per-variant RGB with the OccRAE MAE image decoder (point colours + PNGs).

Then locally:
  python vis_viser.py --input_folder results/deltatok_flow_viser/<test_name>
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(
    Path(__file__).resolve().parent,
    extra=[
        "third_party/pyTorchChamferDistance",
        "third_party/GLD/src",
        "third_party/deltatok",
    ],
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from occrae.deltatok_flow_trainer import DeltaTokFlowMatchingTrainer  # noqa: E402
from occrae.generation_helper import flow_euler_sample  # noqa: E402
from occany.datasets import get_data_loader  # noqa: E402
from test_occ_rae import build_save_dict, denormalize_da3_imgs_to_minus1_1  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeltaTok flow-matching forecast → vis_viser-compatible .npy output."
    )
    parser.add_argument("--config-dir", type=str, default="configs/deltatok_flow")
    parser.add_argument("--config-name", type=str, default="train_deltatok_flow_jeanzay")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Optional Hydra-style overrides, e.g. training.bsize=1",
    )
    parser.add_argument(
        "--ckpt", type=str,
        default="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log/"
                "deltatok_flow_full_deltaCtx_global_fullT_wd05_raymapoff/ckpts/current.pth",
        help="Trained flow-transformer checkpoint (current.pth / iter_*.pth from "
             "DeltaTokFlowMatchingTrainer). Defaults to the JZ "
             "deltatok_flow_full_deltaCtx_global_fullT_wd05_raymapoff run's last checkpoint.",
    )
    parser.add_argument("--occany_recon_ckpt", type=str, default=None)
    parser.add_argument("--encode_layer", type=int, default=12)
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Euler integration steps for the flow sampler.",
    )
    parser.add_argument("--output_dir", type=str, default="results/deltatok_flow_viser")
    parser.add_argument(
        "--num_scenes", type=int, default=2,
        help="Number of scenes (batch items) to save per test dataset.",
    )
    parser.add_argument(
        "--test_name_filter", type=str, default=None,
        help="Only run test datasets whose name contains this substring "
             "(e.g. 'Kitti' or 'Nuscenes').",
    )
    # Image-decoder spec. Flow generation ALWAYS decodes RGB with the OccRAE MAE
    # image decoder (per-variant point colours + saved PNGs); the flow config has
    # no img_decoder block (unlike the tokenizer config), so it is supplied here.
    # These mirror the img_decoder block of configs/deltatok/train_deltatok.yaml /
    # test_occ_rae.py flags.
    parser.add_argument(
        "--img_decoder_ckpt", type=str,
        default="checkpoints/occrae_img_decoder.pt",
        help="MAE image-decoder checkpoint. Defaults to the JZ repo-local "
             "checkpoints/occrae_img_decoder.pt (same as the JZ flow config).",
    )
    parser.add_argument(
        "--img_decoder_config_path", type=str,
        default="third_party/GLD/configs/decoder/ViTXL",
    )
    parser.add_argument("--img_decoder_hidden_size", type=int, default=9216)
    parser.add_argument("--img_decoder_patch_size", type=int, default=14)
    parser.add_argument(
        "--no_ema_decoder", action="store_true", default=False,
        help="Use the raw (non-EMA) image-decoder weights.",
    )
    parser.add_argument(
        "--img_decoder_view_chunk", type=int, default=0,
        help="Decode RGB in chunks of this many views (0 = all at once). "
             "Raise it if surround decode_to_image OOMs.",
    )
    return parser


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")


def _decode_variant(trainer, tokens, height, width):
    """Decode full tokens jointly (single coordinate frame) and cast to float32."""
    with torch.no_grad(), trainer.autocast:
        decoded = trainer.occ_rae.decode(
            {"tokens": tokens, "H": int(height), "W": int(width)},
            pose_from_depth_ray=True,
        )
    return {
        k: (v.float() if isinstance(v, torch.Tensor) else v)
        for k, v in decoded.items()
    }


def _save_rgb_pngs(rgb_01: torch.Tensor, scene_dir: str, prefix: str) -> None:
    """Save (V, 3, H, W) RGB in [0, 1] as PNGs."""
    from PIL import Image

    arr = (rgb_01.clamp(0, 1).cpu().float().numpy() * 255.0).astype(np.uint8)
    for vi in range(arr.shape[0]):
        Image.fromarray(arr[vi].transpose(1, 2, 0)).save(
            os.path.join(scene_dir, f"{prefix}_v{vi:02d}.png")
        )


def main() -> None:
    args = get_args_parser().parse_args()
    device = torch.device("cuda")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.cfg)

    # Keys the user set explicitly via --cfg (e.g. "model.cond_mode") take
    # precedence over the run defaults injected below.
    cfg_override_keys = {o.split("=", 1)[0] for o in args.cfg}

    with open_dict(cfg):
        if args.occany_recon_ckpt:
            cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
        cfg.model.encode_layer = int(args.encode_layer)
        # Replicate the run's training EXTRA_CFG arch flags (the JZ config defaults
        # to cross/factorized/no-camera-embed) so the flow ViT built in __init__
        # matches the saved deltatok_flow_full_deltaCtx_global_fullT_wd05_raymapoff
        # weights (strict load). Override via --cfg model.<key>=... for other runs.
        if "model.cond_mode" not in cfg_override_keys:
            cfg.model.cond_mode = "delta_ctx"          # first delta token as clean in-seq context
        if "model.attn_mode" not in cfg_override_keys:
            cfg.model.attn_mode = "global"             # single global attention over all delta tokens
        if "model.vit_use_camera_embed" not in cfg_override_keys:
            cfg.model.vit_use_camera_embed = True      # absolute per-camera identity in the flow ViT
        # Flow generation always decodes RGB: the flow config has no img_decoder
        # block (unlike the tokenizer config), so inject one from the CLI args —
        # OccRAE._build_occ_rae then loads the MAE decoder and decode_to_image
        # yields per-variant RGB. Defaults mirror configs/deltatok/train_deltatok.yaml.
        cfg.model.img_decoder = {
            "ckpt_path": args.img_decoder_ckpt,
            "config_path": args.img_decoder_config_path,
            "hidden_size": int(args.img_decoder_hidden_size),
            "patch_size": int(args.img_decoder_patch_size),
            "use_ema": not args.no_ema_decoder,
            "view_chunk_size": int(args.img_decoder_view_chunk),
        }
        # No TensorBoard; vit_folder is only used by get_network's makedirs and
        # the (unused, resume=False) resume path — keep it inside output_dir.
        cfg.training.writer_log = ""
        cfg.training.vit_folder = os.path.join(output_dir, "ckpts") + "/"

    trainer_args = argparse.Namespace(
        resume=False,
        ckpt=args.ckpt,
        test_only=True,
        eval_only=True,
        debug=False,
        is_multi_gpus=False,
    )
    # The flow transformer is loaded from args.ckpt inside __init__ ->
    # get_network("vit") (get_pretrained_checkpoint_path reads args.ckpt because
    # resume=False). The frozen DeltaTok + OccRAE come from
    # cfg.model.{deltatok_ckpt, occany_recon_ckpt}.
    trainer = DeltaTokFlowMatchingTrainer(
        args=trainer_args, cfg=cfg, device=device,
        rank=0, world_size=1, distributed=False,
    )
    trainer.vit.eval()

    test_dataset_str = cfg.dataset.get("test_dataset", None)
    if not test_dataset_str:
        raise RuntimeError("Config has no dataset.test_dataset to draw scenes from.")

    saved_test_dirs = []

    have_img_dec = getattr(trainer.occ_rae, "img_decoder", None) is not None

    for sub in str(test_dataset_str).split("+"):
        sub = sub.strip()
        if not sub:
            continue
        test_name = sub.split("(")[0].strip()
        if args.test_name_filter and args.test_name_filter not in test_name:
            print(f"[INFO] Skipping {test_name} (filter: {args.test_name_filter})")
            continue
        print(f"[INFO] Building test dataset: {test_name}")
        loader = get_data_loader(
            sub,
            batch_size=int(cfg.training.bsize),
            num_workers=int(cfg.training.get("val_num_workers", 2)),
            shuffle=False,
            drop_last=False,
        )
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        ds = getattr(loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(0)

        test_dir = os.path.join(output_dir, _sanitize(test_name))
        scenes_saved = 0

        for batch in loader:
            if scenes_saved >= args.num_scenes:
                break
            batch = trainer._normalize_batch(batch)

            imgs = batch["imgs"].to(device, non_blocking=True)
            B, V = imgs.shape[:2]
            num_cameras = int(batch.get("num_cameras", 1))

            with torch.no_grad(), trainer.ema_scope():
                # Frozen OccRAE encode -> full tokens + spatial-only patch feats.
                tokens, feats, _, _, H, W = trainer._extract_pair_feats(
                    imgs, num_cameras=num_cameras, return_pairs=False
                )  # tokens (B, V, N_tok, C); feats (B, T, N, P, C)

                # Frozen DeltaTok encode of GT pairs -> GT delta tokens (the flow
                # target / tokenizer upper bound).
                z = trainer._encode_deltas(feats, H, W)                       # (B, T-1, N, K, C)
                x_spatial = trainer._z_to_flow_latent(z)                      # (B, C, T-1, N, K)
                cross_cond = trainer._build_cross_cond(feats[:, 0], H, W) if trainer.build_frame0_ctx else None  # (B, N, Hp, Wp, C) or None

                # Sample delta tokens from pure noise. build_frame0_ctx: frame-0 conditioning
                # grids (cross-attn kv or in-sequence concat). n_ctx>0: the first n_ctx delta
                # tokens are the clean context (GT, kept fixed by flow_euler_sample's handling).
                zr = trainer._sample_noise(x_spatial)                        # (B, C, T-1, N, K) matches eval prior (honors fixed_noise_mode)
                if trainer.n_ctx > 0:
                    zr[:, :, :trainer.n_ctx] = x_spatial[:, :, :trainer.n_ctx]  # GT first delta = clean context
                gen = flow_euler_sample(
                    trainer._ema_model(), zr,
                    pred_mode=cfg.model.pred_mode,
                    context=trainer.n_ctx,
                    num_steps=int(args.num_steps),
                    # Sampler knobs from cfg, else viz silently uses flow_euler_sample's
                    # cosine/alpha=0.5/ode defaults and stops matching the run's eval.
                    step_mode=str(cfg.model.get("sampler_step_mode", "ode")),
                    scheduler_mode=str(cfg.model.get("sampler_scheduler_mode", "cosine")),
                    alpha=float(cfg.model.get("sampler_alpha", 0.5)),
                    cross_cond=cross_cond,
                    autocast_ctx=trainer.autocast,
                )
                z_hat = trainer._flow_latent_to_z(gen)                       # (B, T-1, N, K, C)

                # Autoregressive DeltaTok decode from the GT first frame:
                # flow-sampled deltas (forecast) + GT deltas (tokenizer bound).
                with trainer.autocast:
                    x_hat_flow = trainer._rollout_from_z(
                        trainer.deltatok, feats[:, 0], z_hat, H, W, num_cameras
                    )  # (B*(T-1), N, P, C)
                    x_hat_tok = trainer._rollout_from_z(
                        trainer.deltatok, feats[:, 0], z, H, W, num_cameras
                    )  # (B*(T-1), N, P, C)

                full_tok = trainer._reconstruct_full_tokens(
                    tokens, x_hat_tok, B, V, num_cameras=num_cameras
                )   # (B, V, N_tok, C)
                full_flow = trainer._reconstruct_full_tokens(
                    tokens, x_hat_flow, B, V, num_cameras=num_cameras
                )   # (B, V, N_tok, C)

            height, width = batch["output_resolution_hw"]
            imgs_vis = denormalize_da3_imgs_to_minus1_1(imgs.float())

            variants = {"orig": tokens, "tok": full_tok, "flow": full_flow}
            decoded_by_variant = {
                name: _decode_variant(trainer, tok, height, width)
                for name, tok in variants.items()
            }
            rgb_by_variant = {}
            if have_img_dec:
                with torch.no_grad(), trainer.autocast:
                    for name, tok in variants.items():
                        rgb_by_variant[name] = trainer.occ_rae.decode_to_image(
                            {"tokens": tok, "H": int(height), "W": int(width)},
                            view_chunk_size=trainer._img_decoder_view_chunk,
                        ).float()

            for j in range(B):
                if scenes_saved >= args.num_scenes:
                    break
                stems = [str(s) for s in batch["frame_stems"][j]]
                scene_id = _sanitize(
                    f"{batch['scene_name'][j]}_{stems[0]}_to_{stems[-1]}"
                )
                scene_dir = os.path.join(test_dir, scene_id)
                os.makedirs(scene_dir, exist_ok=True)

                for name, decoded in decoded_by_variant.items():
                    colors = imgs_vis
                    if name in rgb_by_variant:
                        colors = rgb_by_variant[name].mul(2.0).sub(1.0)
                        _save_rgb_pngs(
                            rgb_by_variant[name][j], scene_dir, prefix=f"rgb_{name}"
                        )
                    sd = build_save_dict(decoded, colors, j)
                    save_path = os.path.join(scene_dir, f"pts3d_{name}.npy")
                    np.save(save_path, sd)
                print(f"[INFO] Saved scene {scene_id} → {scene_dir} "
                      f"(V={V}, num_cameras={num_cameras}, res={height}x{width})")
                scenes_saved += 1

        print(f"[INFO] {test_name}: saved {scenes_saved} scene(s) under {test_dir}")
        if scenes_saved > 0:
            saved_test_dirs.append(test_dir)

    print("\n[INFO] Done. Visualise with:")
    for test_dir in saved_test_dirs:
        print(f"python vis_viser.py --input_folder {test_dir}")


if __name__ == "__main__":
    main()
