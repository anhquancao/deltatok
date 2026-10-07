#!/usr/bin/env python3
"""Do cuDNN and flash SDPA give the DeltaTok tokenizer the same gradients?

Loads each --ckpts checkpoint into a single-GPU DeltaTokTrainer and, per micro-batch, runs
the triplet training step (recon + SIGReg, no optimizer step) three times on identical
inputs: flash, flash again (the flash-vs-flash noise floor) and cuDNN. Reports loss, grad
cosine and relative L2 difference, globally and for the worst blocks. The kernel is
switched by routing gated_attn.py's sdpa_kernel() call to the chosen backend.
Nothing is written to the run dir.
"""

from __future__ import annotations

import argparse
import math
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.attention as tna
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch.nn.attention import SDPBackend

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

# gated_attn.py imports sdpa_kernel when the tokenizer is built: route it to BACKEND.
BACKEND = [SDPBackend.FLASH_ATTENTION]
_real_sdpa_kernel = tna.sdpa_kernel
tna.sdpa_kernel = lambda *_a, **_k: _real_sdpa_kernel(list(BACKEND))

from occany.datasets import get_data_loader  # noqa: E402
from occrae.deltatok_trainer import DeltaTokTrainer  # noqa: E402


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare cuDNN vs flash SDPA gradients on a DeltaTok step.")
    parser.add_argument("--config-dir", type=str, default="configs/deltatok")
    parser.add_argument("--config-name", type=str, default="train_deltatok")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Hydra overrides: the run's EXTRA_CFG.",
    )
    parser.add_argument("--ckpts", type=str, nargs="+", required=True, help="Checkpoints to probe, in order.")
    parser.add_argument("--num-batches", type=int, default=4, help="Micro-batches per checkpoint.")
    parser.add_argument("--seed", type=int, default=0, help="Same micro-batches and triplets for every pass.")
    return parser


def sdpa_nodes(out: torch.Tensor) -> Counter:
    """Count ScaledDotProduct* autograd nodes reachable from `out`."""
    seen, stack, names = set(), [out.grad_fn], Counter()
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        if "ScaledDotProduct" in fn.name():
            names[fn.name().replace("ScaledDotProduct", "").replace("AttentionBackward0", "")] += 1
        stack.extend(nxt for nxt, _ in fn.next_functions)
    return names


def main() -> None:
    args = get_args_parser().parse_args()
    device = torch.device("cuda")

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.cfg)

    with open_dict(cfg):
        run_root = Path(tempfile.mkdtemp(prefix="sdpa_grad_probe_"))     # never the run dir
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
    groups = []                                                          # block id per param
    for n, _ in named:
        m = re.match(r"(encoder|decoder)_blocks\.(\d+)\.", n)
        groups.append(f"{m.group(1)[:3]}{m.group(2)}" if m else "other")
    params = [p for _, p in named]

    def grads(imgs, num_cameras, backend, seed):
        """(loss terms, grads, SDPA kernels) of one triplet step with `backend`."""
        BACKEND[:] = [backend]
        torch.manual_seed(seed)                                          # same triplet draw every pass
        trainer._sigreg_pool = None                                      # same SIGReg pool every pass
        loss, _, _, _, z_tri, _, _ = trainer._compose_forward(imgs, num_cameras)
        z_bneck = torch.cat([z_tri[:, 0], z_tri[:, 1], z_tri[:, 2]], dim=0)    # (3B, N, K, Cz)
        live, pool, scale = trainer._sigreg_pooled(z_bneck.float())
        with torch.autocast(device_type="cuda", enabled=False):
            sig = trainer.sigreg(live, pool, seed=int(cfg.training.iter))
        total = trainer._clean_decode_weight * loss + trainer._sigreg_weight * scale * sig
        kernels = sdpa_nodes(total)
        gs = torch.autograd.grad(total, params, allow_unused=True)
        gs = [torch.zeros_like(p) if g is None else g.detach().float() for p, g in zip(params, gs)]
        return (float(loss), float(sig)), gs, dict(kernels)

    def compare(ga, gb):
        """Global cosine, relative L2 diff ||gb - ga|| / ||ga||, and the 3 worst blocks."""
        dot = na = nb = nd = 0.0
        blk_d, blk_a = defaultdict(float), defaultdict(float)
        for g_a, g_b, grp in zip(ga, gb, groups):
            d2 = float((g_b - g_a).double().square().sum())
            a2 = float(g_a.double().square().sum())
            dot += float((g_a.double() * g_b.double()).sum())
            na += a2
            nb += float(g_b.double().square().sum())
            nd += d2
            blk_d[grp] += d2
            blk_a[grp] += a2
        worst = sorted(((math.sqrt(blk_d[k] / max(blk_a[k], 1e-30)), k) for k in blk_d), reverse=True)[:3]
        cos = dot / math.sqrt(max(na * nb, 1e-30))
        return cos, math.sqrt(nd / max(na, 1e-30)), ", ".join(f"{k} {r:.2e}" for r, k in worst)

    for path in args.ckpts:
        tag = f"{Path(path).parent.parent.name[-12:]}/{Path(path).name}"
        trainer._load_checkpoint(path, restore_train_state=True)
        print(f"== {tag}: iter {cfg.training.iter}", flush=True)
        torch.manual_seed(args.seed)
        trainer._set_train_loader_epoch(0)
        for b, batch in zip(range(args.num_batches), trainer.train_loader):
            batch = trainer._normalize_batch(batch)
            imgs = batch["imgs"].to(device, non_blocking=True)
            num_cameras = batch.get("num_cameras", 1)
            seed = args.seed * 1000 + b
            lf, gf, kf = grads(imgs, num_cameras, SDPBackend.FLASH_ATTENTION, seed)
            lf2, gf2, _ = grads(imgs, num_cameras, SDPBackend.FLASH_ATTENTION, seed)
            cos_ff, rel_ff, worst_ff = compare(gf, gf2)
            del gf2
            lc, gc, kc = grads(imgs, num_cameras, SDPBackend.CUDNN_ATTENTION, seed)
            finite = all(bool(torch.isfinite(g).all()) for g in gc)
            print(f"[{tag} b{b}] kernels flash={kf} cudnn={kc}", flush=True)
            print(f"[{tag} b{b}] recon flash {lf[0]:.6f} / {lf2[0]:.6f} cudnn {lc[0]:.6f}  "
                  f"sigreg flash {lf[1]:.6f} cudnn {lc[1]:.6f}", flush=True)
            print(f"[{tag} b{b}] flash vs flash: cos {cos_ff:.6f} rel {rel_ff:.3e}  worst: {worst_ff}", flush=True)
            if finite:
                cos_fc, rel_fc, worst_fc = compare(gf, gc)
                print(f"[{tag} b{b}] flash vs cudnn: cos {cos_fc:.6f} rel {rel_fc:.3e}  worst: {worst_fc}",
                      flush=True)
            else:
                print(f"[{tag} b{b}] flash vs cudnn: cudnn grads non-finite", flush=True)
            del gf, gc


if __name__ == "__main__":
    main()
