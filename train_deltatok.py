#!/usr/bin/env python3
"""Train the DeltaTok tokenizer over OccRAE/DA3 layer-18 frame features."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
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


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DeltaTok tokenizer.")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="train_deltatok")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Optional Hydra-style overrides, e.g. dataset.max_train_samples=8",
    )
    parser.add_argument(
        "--log-and-ckpt-dir", type=str, default="results",
        help="Training root: TB logs at <dir>/<run-name>/tb_logs/, checkpoints at "
             "<dir>/<run-name>/ckpts/. Under --eval-only, the checkpoint is loaded "
             "from this same path.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Where to dump eval panels under --eval-only "
             "(<results-dir>/<run-name>/eval_only/). Defaults to --log-and-ckpt-dir.",
    )
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--run-name", type=str, default="deltatok")
    parser.add_argument("--resume", action="store_true", help="Resume from current.pth in ckpts dir.")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume / pretrained checkpoint.")
    parser.add_argument("--occany_recon_ckpt", type=str, default=None)
    parser.add_argument("--encode_layer", type=int, default=None)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training: load checkpoint from <log-and-ckpt-dir>/<run-name>/ckpts/, "
             "run eval_one_epoch once, save panels under <results-dir>/<run-name>/eval_only/, "
             "and print metrics instead of writing TensorBoard.",
    )
    return parser


def get_world_size() -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return 1


def setup_cuda_distributed() -> tuple[torch.device, int, int]:
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    elif "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % max(torch.cuda.device_count(), 1)))
        master_addr = os.environ.get("MASTER_ADDR")
        if not master_addr:
            node_list = os.environ.get("SLURM_NODELIST")
            if not node_list:
                raise RuntimeError(
                    "Distributed SLURM launch requires MASTER_ADDR or SLURM_NODELIST to be set."
                )
            master_addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1").strip()
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        raise RuntimeError("Distributed launch requested but no torchrun or SLURM rank environment was found.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    return torch.device("cuda", local_rank), rank, world_size


def main() -> None:
    args = get_args_parser().parse_args()

    world_size = get_world_size()
    distributed = world_size > 1
    args.is_multi_gpus = distributed

    if distributed:
        device, rank, world_size = setup_cuda_distributed()
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda")

    try:
        config_dir = Path(args.config_dir).expanduser().resolve()
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=args.config_name, overrides=args.cfg)

        with open_dict(cfg):
            if args.occany_recon_ckpt:
                cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
            if args.encode_layer:
                cfg.model.encode_layer = args.encode_layer

            run_root = Path(args.log_and_ckpt_dir).expanduser().resolve() / args.run_name
            cfg.exp_name = args.run_name
            cfg.training.vit_folder = str((run_root / "ckpts").resolve()) + "/"
            cfg.training.writer_log = str((run_root / "tb_logs").resolve()) + "/"
            cfg.training.eval_viz_dir = str((run_root / "results" / "eval_viz").resolve())

            if args.eval_only:
                # Print metrics, dump panels to disk, no TensorBoard.
                cfg.training.writer_log = ""
                eval_root = (
                    Path(args.results_dir).expanduser().resolve() / args.run_name
                    if args.results_dir
                    else run_root
                )
                cfg.training.eval_viz_dir = str((eval_root / "eval_only").resolve())
                # Reuse the existing test_only knob so the optimizer state isn't loaded.
                args.test_only = True

        trainer = DeltaTokTrainer(
            args=args,
            cfg=cfg,
            device=device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )
        trainer.run()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()