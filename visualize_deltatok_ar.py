#!/usr/bin/env python3
"""Autoregressive DeltaTok next-frame prediction → vis_viser.py input.

Loads a trained DeltaTok checkpoint, runs the same eval path as
``DeltaTokTrainer.eval_one_epoch`` (teacher-forced reconstruction + joint
multi-camera autoregressive rollout) on the config's test datasets, decodes
the predicted tokens back to 3D with OccRAE, and saves per-scene
``pts3d_<setting>.npy`` dicts (keys: pts3d, pts3d_local, colors, conf, focal,
c2w) that ``vis_viser.py`` consumes directly.

Settings written per scene:
  * ``pts3d_orig.npy`` — decode of the unmodified OccRAE tokens (upper bound).
  * ``pts3d_tf.npy``   — teacher-forced DeltaTok reconstruction.
  * ``pts3d_ar.npy``   — autoregressive rollout (timestep 0 is GT context,
                         later timesteps feed predictions back).

Unlike the trainer's per-timestep ``_decode_tokens`` (loss bookkeeping), all V
views are decoded jointly here so the whole sequence shares one coordinate
frame (DA3 ``ref_view_strategy="first"``), which is what viser needs.

Usage (on Karolina, from ~/deltatok):
  conda activate occany
  python test_deltatok_ar.py \
      --config-name train_deltatok_karolina \
      --encode_layer 12 \
      --ckpt /mnt/proj1/eu-25-92/deltatok_log/deltatok_surround_constGlobalRope_layer12/ckpts/current.pth \
      --output_dir results/deltatok_ar_viser \
      --num_scenes 2

Then locally:
  python vis_viser.py --input_folder results/deltatok_ar_viser/<test_name>
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

from occrae.deltatok_trainer import DeltaTokTrainer  # noqa: E402
from occany.datasets import get_data_loader  # noqa: E402
from test_occ_rae import build_save_dict, denormalize_da3_imgs_to_minus1_1  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeltaTok autoregressive rollout → vis_viser-compatible .npy output."
    )
    parser.add_argument("--config-dir", type=str, default="configs/deltatok")
    parser.add_argument("--config-name", type=str, default="train_deltatok_karolina")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Optional Hydra-style overrides, e.g. training.bsize=1",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Trained DeltaTok checkpoint (current.pth from DeltaTokTrainer).",
    )
    parser.add_argument("--occany_recon_ckpt", type=str, default=None)
    parser.add_argument("--encode_layer", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="results/deltatok_ar_viser")
    parser.add_argument(
        "--num_scenes", type=int, default=2,
        help="Number of scenes (batch items) to save per test dataset.",
    )
    parser.add_argument(
        "--test_name_filter", type=str, default=None,
        help="Only run test datasets whose name contains this substring "
             "(e.g. 'Kitti' or 'Nuscenes').",
    )
    parser.add_argument(
        "--use_img_decoder", action="store_true", default=False,
        help="Keep the config's pretrained image decoder and use the RGB it "
             "reconstructs from each token variant as point colours (also saves "
             "PNGs). Off by default: colours come from the GT input images.",
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

    with open_dict(cfg):
        if args.occany_recon_ckpt:
            cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
        cfg.model.encode_layer = int(args.encode_layer)
        if not args.use_img_decoder and cfg.model.get("img_decoder", None) is not None:
            cfg.model.img_decoder.ckpt_path = None
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
    trainer = DeltaTokTrainer(
        args=trainer_args, cfg=cfg, device=device,
        rank=0, world_size=1, distributed=False,
    )
    trainer._load_checkpoint(args.ckpt, restore_train_state=False)
    trainer.tokenizer.eval()

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
            num_cameras = batch.get("num_cameras", 1)

            with torch.no_grad():
                tokens, feats, x_prev, x, H, W = trainer._extract_pair_feats(
                    imgs, num_cameras=num_cameras
                )
                with trainer.autocast:
                    x_hat_tf = trainer.tokenizer(
                        x_prev, x, H, W, num_cameras=num_cameras
                    )
                    x_hat_ar = trainer._autoregressive_rollout(
                        feats, H, W, num_cameras
                    )
                full_tf = trainer._reconstruct_full_tokens(
                    tokens, x_hat_tf, B, V, num_cameras=num_cameras
                )
                full_ar = trainer._reconstruct_full_tokens(
                    tokens, x_hat_ar, B, V, num_cameras=num_cameras
                )

            height, width = batch["output_resolution_hw"]
            imgs_vis = denormalize_da3_imgs_to_minus1_1(imgs.float())

            variants = {"orig": tokens, "tf": full_tf, "ar": full_ar}
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
