#!/usr/bin/env python3
"""Train a token-only multiview flow model over OccRAE layer-18 feature dumps."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import open_dict


REPO_ROOT = Path(__file__).resolve().parent
VENDORED_IMPORT_PATHS = [
    REPO_ROOT / "third_party",
    REPO_ROOT / "third_party" / "dust3r",
    REPO_ROOT / "third_party" / "croco" / "models" / "curope",
    # REPO_ROOT / "third_party" / "Grounded-SAM-2",
    # REPO_ROOT / "third_party" / "Grounded-SAM-2" / "grounding_dino",
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
torch.backends.cudnn.allow_tf32 = True

from occrae.deltatok_flow_trainer import DeltaTokFlowMatchingTrainer  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train token-only OccRAE flow matching.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Hydra config directory (default: configs).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="train_deltatok_flow_overfit",
        help="Hydra config name (without .yaml, default: train_deltatok_flow_overfit).",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        nargs="*",
        default=[],
        help="Optional Hydra-style overrides, e.g. dataset.max_train_samples=8",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to store run outputs.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Compute precision for training.",
    )
    parser.add_argument("--run-name", type=str, default="occrae_token_flow", help="Run name prefix.")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from a full training checkpoint.")
    parser.add_argument(
        "--preprocess_data_root",
        type=str,
        default=None,
        help="Override dataset.preprocess_data_root.",
    )
    parser.add_argument(
        "--occany_recon_ckpt",
        type=str,
        default=None,
        help="Override model.occany_recon_ckpt.",
    )
    parser.add_argument(
        "--encode_layer",
        type=int,
        default=None,
        help="Override model.encode_layer.",
    )
    parser.add_argument("--overfit", action="store_true", help="Overfit on a single training example.")
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
            if args.preprocess_data_root:
                cfg.dataset.preprocess_data_root = args.preprocess_data_root
            if args.occany_recon_ckpt:
                cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
            if args.encode_layer:
                cfg.model.encode_layer = args.encode_layer
            if args.overfit:
                cfg.dataset.max_train_samples = 1
                cfg.dataset.max_val_samples = 1

            run_root = Path(args.results_dir).expanduser().resolve() / args.run_name
            cfg.exp_name = args.run_name
            cfg.training.vit_folder = str((run_root / "ckpts").resolve()) + "/"
            cfg.training.writer_log = str((run_root / "tb_logs").resolve()) + "/"

        trainer = DeltaTokFlowMatchingTrainer(
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
