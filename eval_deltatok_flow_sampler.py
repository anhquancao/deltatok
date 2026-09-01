#!/usr/bin/env python3
"""Run the trainer's full generative eval with selectable sampler step modes.

Loads a trained DeltaTok flow checkpoint and calls the trainer's own
``eval_one_epoch`` (sample-from-noise -> DeltaTok rollout -> decode metrics), so
MSEToken / LossDepth / LossPointmap are bit-comparable with the training logs.
Runs once per ``--step_modes`` entry (see ``flow_euler_sample``):
  * ``ode``    — v-conversion Euler (the training-time eval; reproduces the
                 logged numbers = ckpt-load sanity).
  * ``damped`` — VGGT-World blend-to-x_hat update (fm.py:479): never divides by
                 the noise level, so the x-pred error floor near t=1 is not
                 amplified into the sample. Mean-seeking (initial noise discarded
                 at step 1). Result on the dit run: see the sampler-amplification
                 journal (0.293 -> 0.264 only; the wall is the decoder).

Tip: ``--step_modes ode --cfg training.eval_num_steps=1`` makes the ode sampler
output exactly x_hat(noise, t=0) — the pure regression baseline.

Usage (on Jean Zay, from a GPU node; defaults target the
..._50k_whitenpos_xxl_dit run's last ckpt):
  source env_jz_h100.sh &&
  python eval_deltatok_flow_sampler.py --output_dir results/deltatok_flow_sampler_eval
  # Defaults: --config-name train_deltatok_flow_waymo_jeanzay, --ckpt the dit run's
  # current.pth, and the run's arch flags from slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_dit_jz.slurm
  # (dit AdaLN 1536x20, per-position whitening, delta_ctx_cross, dtok64 non-affine
  # tokenizer). Override any with --cfg model.<key>=... for a different flow run.
"""

import argparse
import os
import re
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

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
from occany.datasets import get_data_loader  # noqa: E402

# Default target run. Its arch flags live in slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_dit_jz.slurm
# (EXTRA_CFG), NOT in the config — replicated in `_RUN_DEFAULTS` below.
_RUN_ROOT = ("/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log/"
             "deltatok_flow_waymo_dtok64_deltaCtxCross_global_fullT_wd05_raymapoff_50k_whitenpos_xxl_dit")
_DTOK64_ROOT = ("/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/"
                "deltatok_surround_layer12_bsize2_dtok64_cos1e3_1e5_b95_gradskip_noaff")

_RUN_DEFAULTS = {
    "model.cond_mode": "delta_ctx_cross",      # clean in-seq context + frame-0 cross-attn
    "model.attn_mode": "global",               # single global attention over all delta tokens
    "model.vit_use_camera_embed": True,        # per-camera identity in the flow ViT
    "model.deltatok.num_delta_tokens": 64,     # K=64, must match the frozen dtok64 ckpt
    "model.deltatok.norm_affine": False,       # dtok64 tokenizer is non-affine z-norm
    "model.deltatok_ckpt": _DTOK64_ROOT + "/ckpts/current.pth",
    "model.whiten_stats": _DTOK64_ROOT + "/ckpts/latent_stats_waymo_pos.pth",  # per-position whitening (whitenpos runs)
    "model.vit_size": "xxlarge_d20",           # dit run: width 1536 == C, depth 20
    "model.vit_dit_adaln": True,               # per-block AdaLN-Zero + Fourier t-embed
}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trainer-faithful generative eval with selectable sampler step modes."
    )
    parser.add_argument("--config-dir", type=str, default="configs/deltatok_flow")
    parser.add_argument("--config-name", type=str, default="train_deltatok_flow_waymo_jeanzay")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Optional Hydra-style overrides, e.g. training.bsize=8. Takes precedence "
             "over the injected run defaults.",
    )
    parser.add_argument(
        "--ckpt", type=str, default=os.path.join(_RUN_ROOT, "ckpts", "current.pth"),
        help="Trained flow-transformer checkpoint (current.pth / iter_*.pth from "
             "DeltaTokFlowMatchingTrainer). Loaded with strict=True.",
    )
    parser.add_argument("--occany_recon_ckpt", type=str, default=None)
    parser.add_argument("--encode_layer", type=int, default=12)
    parser.add_argument(
        "--step_modes", type=str, default="ode,damped",
        help="Comma-separated flow_euler_sample step modes; each gets one full "
             "eval_one_epoch pass. 'ode' first = ckpt-load sanity (reproduces the "
             "run's logged numbers).",
    )
    parser.add_argument(
        "--num_steps", type=str, default="",
        help="Comma-separated sampler step counts to sweep; one full eval_one_epoch "
             "pass per (num_steps, step_mode). Empty = use training.eval_num_steps once.",
    )
    parser.add_argument(
        "--viz_rgb", action="store_true",
        help="Keep the MAE image decoder loaded so the eval panels get RGB columns. "
             "Needs training.eval_num_visualizations > 0 to produce anything.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/deltatok_flow_sampler_eval")
    return parser


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")


def _build_test_loaders(cfg):
    """{test_name: loader} for cfg.dataset.test_dataset — mirrors how
    `DeltaTokFlowMatchingTrainer.fit` builds eval loaders, so `eval_one_epoch`
    sees the exact samples the training logs evaluated."""
    expr = cfg.dataset.get("test_dataset", None)
    if not expr:
        raise RuntimeError("Config has no dataset.test_dataset.")
    loaders = {}
    for sub in str(expr).split("+"):
        sub = sub.strip()
        if not sub:
            continue
        test_name = sub.split("(")[0].strip()
        loaders[test_name] = get_data_loader(
            sub,
            # val_bsize caps eval decode memory (0 = train bsize) — as the trainer does
            batch_size=int(cfg.training.get("val_bsize", 0)) or int(cfg.training.bsize),
            num_workers=int(cfg.training.get("val_num_workers", 2)),
            shuffle=False,
            drop_last=False,
        )
    return loaders


def main() -> None:
    args = get_args_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    run_name = Path(args.ckpt).resolve().parent.parent.name   # <run>/ckpts/current.pth -> <run>
    output_dir = os.path.join(os.path.abspath(args.output_dir), _sanitize(run_name))
    os.makedirs(output_dir, exist_ok=True)

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.cfg)

    # Keys the user set explicitly via --cfg take precedence over the run defaults.
    # lstrip("+~"): Hydra's add/remove prefixes are not part of the key name, and an
    # unmatched key here would silently reset the user's override to the default.
    cfg_override_keys = {o.split("=", 1)[0].lstrip("+~") for o in args.cfg}

    with open_dict(cfg):
        if args.occany_recon_ckpt:
            cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
        cfg.model.encode_layer = int(args.encode_layer)
        # Replicate the run's training EXTRA_CFG arch flags (the config defaults differ),
        # so the flow ViT built in __init__ matches the saved weights.
        for key, val in _RUN_DEFAULTS.items():
            if key not in cfg_override_keys:
                OmegaConf.update(cfg, key, val, merge=False)
        # --viz_rgb keeps the MAE image decoder for the panels' RGB columns; else
        # drop it (_build_occ_rae treats a falsy ckpt_path as "no decoder").
        if cfg.model.get("img_decoder", None) is not None and not args.viz_rgb:
            cfg.model.img_decoder.ckpt_path = None
        # No TensorBoard; vit_folder is only touched by get_network's makedirs.
        cfg.training.writer_log = ""
        cfg.training.vit_folder = os.path.join(output_dir, "ckpts") + "/"

    # ckpt=None: build a FRESH ViT here. get_network loads with strict=False, which would
    # silently drop every cross-attn weight on a wrong cond_mode; the explicit strict=True
    # load below fails loudly instead.
    trainer_args = argparse.Namespace(
        resume=False, ckpt=None, test_only=True, eval_only=True,
        debug=False, is_multi_gpus=False,
    )
    trainer = DeltaTokFlowMatchingTrainer(
        args=trainer_args, cfg=cfg, device=device,
        rank=0, world_size=1, distributed=False,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in ckpt["model_state_dict"].items()
    }
    trainer.vit.load_state_dict(state_dict, strict=True)
    print(f"[INFO] Loaded flow ViT from: {args.ckpt}\n"
          f"[INFO]   iter={ckpt.get('iter')} global_epoch={ckpt.get('global_epoch')}")
    if cfg.training.use_ema and ckpt.get("ema_state") is not None:
        trainer.ema.load_state_dict(ckpt["ema_state"], trainer._ema_model())
        print("[INFO]   loaded EMA weights")
    trainer.vit.eval()

    # eval_one_epoch iterates trainer.test_loaders (normally built in fit()).
    trainer.test_loaders = _build_test_loaders(cfg)

    modes = [m.strip() for m in args.step_modes.split(",") if m.strip()]
    # Empty --num_steps keeps the single-pass behaviour at the config's eval_num_steps.
    steps = [int(s) for s in args.num_steps.split(",") if s.strip()] or \
            [int(cfg.training.get("eval_num_steps", 50))]
    for n_steps in steps:
        cfg.training.eval_num_steps = n_steps  # re-read by eval_one_epoch each pass
        for mode in modes:
            cfg.model.sampler_step_mode = mode  # read by eval_one_epoch's flow_euler_sample call
            with open_dict(cfg):
                # panel filenames carry no step/mode, so one dir per pass or they overwrite
                cfg.training.eval_viz_dir = os.path.join(output_dir, "eval_viz", f"{mode}_steps{n_steps}")
            print(f"\n[INFO] ===== sampler_step_mode={mode} "
                  f"(eval_num_steps={n_steps}, "
                  f"eval_num_items={cfg.training.get('eval_num_items', 256)}) =====", flush=True)
            loss = trainer.eval_one_epoch()  # prints the [Eval/...] metric line itself
            print(f"[INFO] steps={n_steps} sampler_step_mode={mode}: "
                  f"Eval loss (flow) = {float(loss):.4f}")

    print(f"\n[INFO] Done. Viz (if any) under {output_dir}")


if __name__ == "__main__":
    main()
