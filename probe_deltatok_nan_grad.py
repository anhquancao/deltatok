#!/usr/bin/env python3
"""Find where a DeltaTok training step turns a finite loss into NaN gradients.

Loads each --ckpts checkpoint into a single-GPU DeltaTokTrainer and replays the triplet
training step (train_one_epoch's forward, no optimizer step) on the same micro-batches,
with the default SDPA backends and with cuDNN attention off. Per term (recon, SIGReg) it lists the params whose grad is non-finite,
and on the first bad micro-batch it reruns backward under detect_anomaly to name the op.
Nothing is written to the run dir.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch.nn.attention import SDPBackend, sdpa_kernel

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

from occany.datasets import get_data_loader  # noqa: E402
from occrae.deltatok_trainer import DeltaTokTrainer  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locate NaN gradients in a DeltaTok training step.")
    parser.add_argument("--config-dir", type=str, default="configs/deltatok")
    parser.add_argument("--config-name", type=str, default="train_deltatok")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Hydra overrides: the run's EXTRA_CFG.",
    )
    parser.add_argument("--ckpts", type=str, nargs="+", required=True, help="Checkpoints to probe, in order.")
    parser.add_argument("--num-batches", type=int, default=16, help="Micro-batches per checkpoint and precision.")
    parser.add_argument("--seed", type=int, default=0, help="Same micro-batches for every checkpoint.")
    return parser


def main() -> None:
    args = get_args_parser().parse_args()
    device = torch.device("cuda")

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.cfg)

    with open_dict(cfg):
        run_root = Path(tempfile.mkdtemp(prefix="nan_probe_"))           # never the run dir
        cfg.training.vit_folder = str(run_root / "ckpts") + "/"
        cfg.training.writer_log = ""
        cfg.model.img_decoder.ckpt_path = None                           # no RGB decode here

    args.ckpt = None
    args.is_multi_gpus = False
    args.test_only = True                                                # weights + iter only, no optimizer state
    trainer = DeltaTokTrainer(args=args, cfg=cfg, device=device, rank=0, world_size=1, distributed=False)
    trainer.train_loader = get_data_loader(
        str(cfg.dataset.train_dataset), batch_size=cfg.training.bsize,
        num_workers=cfg.training.num_workers, shuffle=True, drop_last=True)
    net = trainer._unwrapped_tokenizer()
    trainer.tokenizer.train()
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    params = [p for _, p in named]

    def bad_grads(term):
        """Names of params whose grad of `term` is non-finite."""
        gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        return [n for n, g in zip(names, gs) if g is not None and not torch.isfinite(g).all()]

    def block_summary(bad):
        """Which encoder / decoder blocks hold non-finite grads, plus any other params."""
        blocks = {"encoder_blocks": set(), "decoder_blocks": set()}
        other = []
        for n in bad:
            m = re.match(r"(encoder_blocks|decoder_blocks)\.(\d+)\.", n)
            if m:
                blocks[m.group(1)].add(int(m.group(2)))
            else:
                other.append(n)
        return (f"bad enc {sorted(blocks['encoder_blocks'])} dec {sorted(blocks['decoder_blocks'])} "
                f"other {other[:6]}")

    def step_terms(imgs, num_cameras):
        """Triplet forward as train_one_epoch (recon + SIGReg on the 3 pairs); terms and z stats."""
        loss, _, _, _, z_tri, _, _ = trainer._compose_forward(imgs, num_cameras)
        z_bneck = torch.cat([z_tri[:, 0], z_tri[:, 1], z_tri[:, 2]], dim=0)    # (3B, N, K, Cz)
        live, pool, scale = trainer._sigreg_pooled(z_bneck.float())
        with torch.autocast(device_type="cuda", enabled=False):
            sig = trainer.sigreg(live, pool, seed=int(cfg.training.iter))
        terms = {"recon": trainer._clean_decode_weight * loss,
                 "sigreg": trainer._sigreg_weight * scale * sig}
        return terms, float(z_bneck.abs().max()), float(net._enc_row_absmax.max())

    for path in args.ckpts:
        tag = Path(path).name
        trainer._load_checkpoint(path, restore_train_state=True)
        bad_params = [n for n, p in named if not torch.isfinite(p).all()]
        print(f"== {tag}: iter {cfg.training.iter}, non-finite params {bad_params[:5]}, "
              f"max |param| {max(float(p.abs().max()) for p in params):.4g}", flush=True)
        no_cudnn = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        for mode, make_ctx in (("default_sdpa", nullcontext), ("no_cudnn_sdpa", lambda: sdpa_kernel(no_cudnn))):
            trainer._sigreg_pool = None
            torch.manual_seed(args.seed)
            trainer._set_train_loader_epoch(0)                               # same batches per ckpt and mode
            n_bad, traced = 0, False
            for b, batch in zip(range(args.num_batches), trainer.train_loader):
                batch = trainer._normalize_batch(batch)
                imgs = batch["imgs"].to(device, non_blocking=True)
                num_cameras = batch.get("num_cameras", 1)
                with make_ctx():                                             # backend picked at forward drives backward
                    terms, zmax, hmax = step_terms(imgs, num_cameras)
                    bad = {k: bad_grads(v) for k, v in terms.items()}
                n_bad += int(any(bad.values()))
                print(f"[{tag} {mode} b{b}] " + "  ".join(
                    f"{k}={float(v):.4g} bad={len(bad[k])}/{len(names)}" for k, v in terms.items())
                    + f"  |z|max={zmax:.4g} |h|max={hmax:.4g}  {block_summary(bad['recon'] + bad['sigreg'])}",
                    flush=True)
                if any(bad.values()) and not traced:
                    traced = True                                            # forward stack of the first NaN op
                    with torch.autograd.detect_anomaly(check_nan=True), make_ctx():
                        terms, _, _ = step_terms(imgs, num_cameras)
                        try:
                            sum(terms.values()).backward()
                        except RuntimeError as e:
                            print(f"[{tag} {mode}] anomaly: {e}", flush=True)
                    net.zero_grad(set_to_none=True)
            print(f"== {tag} {mode}: {n_bad}/{args.num_batches} micro-batches with non-finite grads", flush=True)


if __name__ == "__main__":
    main()
