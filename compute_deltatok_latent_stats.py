#!/usr/bin/env python3
"""Per-channel mean/std of the frozen DeltaTok delta tokens (RAE-style whitening stats).

DeltaTok ends in a parameter-free ``nn.LayerNorm``, which normalizes each token ACROSS
channels: every ROW of the (M, C) delta-token matrix gets mean 0 / RMS 1. Columns are
unconstrained — channel c's std over the dataset is free, the massive-activation pattern
DINOv3 blocks are known for. The per-element bound is only |x_i| <= sqrt(C-1) (~39 at
C=1536), hence the observed max of 36.7.

The flow adds ISOTROPIC noise e ~ N(0, I) (``flow_noising``), so SNR_c(t) = t*std_c/(1-t):
one t means a different noise level per channel, and the quadratic loss is dominated by
the high-std ones. Measuring the moments lets ``model.whiten_stats`` normalize each
channel to std 1 — the step RAE describes but never ablates.

Uses only the frozen OccRAE + DeltaTok encode: no flow ViT weights, no flow ckpt.
float64 over ALL slots (context included — whitening applies to the whole tensor).

Gate: ``std.max()/std.min()``. ~1-2x -> whitening is a no-op (itself the unpublished
ablation result); >= ~10x -> the isotropic noise is mismatched, Phase 1 is worth it.

Usage (Jean Zay GPU node; defaults = the dtok64 non-affine tokenizer the waymo flow
runs freeze). Writes latent_stats_<tag>.pth beside the tokenizer ckpt:
  source env_jz_h100.sh && python compute_deltatok_latent_stats.py
  # Smoke first: --num_batches 2
"""

import argparse
import os
import re
import tempfile
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

# The stats belong to the TOKENIZER, not to any flow run, so only the keys that pin
# which frozen tokenizer is built are replicated here (the flow-ViT arch flags in
# slurm/jz_train_deltatok_flow_waymo.slurm cannot change z). Defaults = the dtok64
# non-affine tokenizer the waymo flow runs freeze.
_DTOK64_CKPT = ("/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/"
                "deltatok_surround_layer12_bsize2_dtok64_cos1e3_1e5_b95_gradskip_noaff/ckpts/current.pth")

_TOKENIZER_DEFAULTS = {
    "model.deltatok.num_delta_tokens": 64,     # K=64, must match the frozen dtok64 ckpt
    "model.deltatok.norm_affine": False,       # dtok64 tokenizer is non-affine z-norm
    "model.deltatok_ckpt": _DTOK64_CKPT,
}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-channel mean/std of frozen DeltaTok delta tokens (whitening stats)."
    )
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="train_deltatok_flow_jeanzay_waymo")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Optional Hydra-style overrides, e.g. training.bsize=8. Takes precedence "
             "over the injected tokenizer defaults.",
    )
    parser.add_argument("--occany_recon_ckpt", type=str, default=None)
    parser.add_argument("--encode_layer", type=int, default=12)
    parser.add_argument(
        "--num_batches", type=int, default=500,
        help="Batches to accumulate (500 ~ 4.2M token vectors, ~12 min). Scene coverage "
             "is the point, not estimator variance: 100 -> 500 moves per-channel std by "
             "<1% median, so 500 is comfortably converged.",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=("train", "val"),
        help="Whitening is a train-set statistic; val is for checking drift only.",
    )
    parser.add_argument(
        "--tag", type=str, default="waymo",
        help="Names the output: latent_stats_<tag>.pth, written beside the tokenizer ckpt.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Explicit output path; overrides the --tag placement.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")


def _build_split_loaders(cfg, split):
    """Return [(label, loader)] for a split.

    train: the whole `+`-joined expression in ONE loader with per_dataset_sampling;
    val: one loader per `+`-separated sub-dataset. Both mirror how
    `DeltaTokFlowMatchingTrainer.fit` builds them, so the samples match training/eval.
    """
    if split == "train":
        loader = get_data_loader(
            str(cfg.dataset.train_dataset),
            batch_size=int(cfg.training.bsize),
            num_workers=int(cfg.training.num_workers),
            shuffle=True,   # the first --num_batches batches should span scenes, not just the first ones
            drop_last=True,
            per_dataset_sampling=bool(cfg.dataset.get("per_dataset_sampling", False)),
        )
        return [("train", loader)]

    expr = cfg.dataset.get("test_dataset", None)
    if not expr:
        raise RuntimeError("Config has no dataset.test_dataset for the val split.")
    subs = [s.strip() for s in str(expr).split("+") if s.strip()]
    out = []
    for sub in subs:
        test_name = sub.split("(")[0].strip()
        loader = get_data_loader(
            sub,
            batch_size=int(cfg.training.bsize),
            num_workers=int(cfg.training.get("val_num_workers", 2)),
            shuffle=False,
            drop_last=False,
        )
        out.append(("val" if len(subs) == 1 else f"val_{_sanitize(test_name)}", loader))
    return out


def _pin_loader_epoch(loader) -> None:
    """Pin sampler/dataset to epoch 0 so repeat invocations see the same samples."""
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "set_epoch"):
        ds.set_epoch(0)


@torch.no_grad()
def accumulate_channel_stats(trainer, loader, args):
    """Per-channel moments of the delta tokens over `--num_batches` batches.

    Rows = tokens (M = B*(T-1)*N*K), columns = channels (C). LayerNorm normalizes the
    ROWS; this measures the COLUMNS. float64: bf16 sums over ~600k rows would lose the
    small per-channel means.

    Returns (mean (C,), std (C,), n, row_ms) — row_ms is the mean row mean-square, the
    LayerNorm sanity check (~1.0).
    """
    total = None       # (C,) float64 running per-channel sum
    total_sq = None    # (C,) float64 running per-channel sum of squares
    n = 0              # M token vectors seen
    row_ms_sum = 0.0   # running sum over rows of mean(x^2)

    batches_done = 0
    for batch in loader:
        if batches_done >= args.num_batches:
            break
        batch = trainer._normalize_batch(batch)

        imgs = batch["imgs"].to(trainer.device, non_blocking=True)  # (B, V, 3, H, W) with V = T*N views
        num_cameras = int(batch.get("num_cameras", 1))

        # Frozen OccRAE + DeltaTok encode -> z, the flow target and the thing measured.
        # want_tokens=False: no decode here, and feat0 (conditioning) is irrelevant to
        # the tokenizer's output distribution. The ~1B OccRAE forward dominates the cost.
        # NOTE: z is the RAW tokenizer output — _z_to_flow_latent is deliberately NOT
        # called, so these stats never depend on the whitening flag they configure.
        _, _, z, H, W = trainer._encode_inputs(batch, imgs, num_cameras, want_tokens=False)  # z (B, T-1, N, K, C)

        C = z.shape[-1]                                            # channel count (backbone.embed_dim)
        # Channels are already last -> one row per delta token. EVERY slot counts,
        # context included: whitening is applied to the whole tensor, not to a subset.
        m = z.reshape(-1, C).double()                              # (M_b, C) M_b = B*(T-1)*N*K token vectors
        if total is None:
            total = torch.zeros(C, dtype=torch.float64, device=m.device)
            total_sq = torch.zeros(C, dtype=torch.float64, device=m.device)
        total += m.sum(0)                                          # (C,) per-channel sum
        total_sq += m.square().sum(0)                              # (C,) per-channel sum of squares
        row_ms_sum += float(m.square().mean(1).sum())              # LayerNorm check: each row's mean(x^2)
        n += m.shape[0]

        batches_done += 1
        print(f"[INFO]   batch {batches_done}/{args.num_batches}  "
              f"tokens={n} C={C} res={H}x{W}", flush=True)

    if n == 0:
        raise RuntimeError("No usable batches: nothing to measure.")

    mean = total / n                                               # (C,) per-channel mean
    var = (total_sq / n) - mean.square()                           # (C,) E[x^2] - E[x]^2
    std = var.clamp_min(0).sqrt()                                  # (C,) clamp: round-off can make var slightly negative
    return mean.cpu(), std.cpu(), n, row_ms_sum / n


def report_stats(mean, std, n, row_ms, label, num_batches) -> float:
    """Print the distribution of per-channel stds and return the gate ratio."""
    C = mean.numel()
    smin, smax = float(std.min()), float(std.max())
    ratio = smax / max(smin, 1e-12)

    qs = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=torch.float64)
    pct = torch.quantile(std, qs)
    topk = torch.topk(std, min(10, C))

    print(f"\n[STATS] split={label}  C={C}  n={n} token vectors ({num_batches} batches)")
    # LayerNorm normalizes each row to mean 0 / var 1, so mean(x^2) per row must be 1.
    # A deviation here means the measured tensor is NOT the tokenizer's normed output.
    print(f"[STATS] LayerNorm check: mean row mean-square = {row_ms:.4f}  (must be ~1.0)")
    print(f"[STATS] |mean_c| max = {float(mean.abs().max()):.4f}")
    print(f"[STATS] std: min={smin:.4f}  max={smax:.4f}  ratio={ratio:.1f}x")
    print(f"[STATS] std percentiles: p0={pct[0]:.4f} p1={pct[1]:.4f} p50={pct[2]:.4f} "
          f"p99={pct[3]:.4f} p100={pct[4]:.4f}")
    print(f"[STATS] top-{topk.values.numel()} channels by std:")
    for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"[STATS]   c={i:5d}  std={v:.4f}")

    print(f"\n[GATE] std.max()/std.min() = {ratio:.1f}x")
    if ratio < 2.0:
        print("[GATE] ~1-2x: channels are already balanced -> whitening is a NO-OP.")
        print("[GATE] That IS the ablation result the RAE paper never published. Stop here.")
    elif ratio < 10.0:
        print("[GATE] 2-10x: ambiguous. Whitening may help a little; judge against the "
              "std percentiles above (a heavy low tail matters more than the max).")
    else:
        print("[GATE] >=10x: one t means a badly different SNR per channel -> proceed to "
              "Phase 1 (set model.whiten_stats to the file below).")
    return ratio


def main() -> None:
    args = get_args_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.cfg)

    # Keys the user set explicitly via --cfg take precedence over the tokenizer defaults.
    cfg_override_keys = {o.split("=", 1)[0] for o in args.cfg}

    with open_dict(cfg):
        if args.occany_recon_ckpt:
            cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
        cfg.model.encode_layer = int(args.encode_layer)
        # Pin the frozen tokenizer (the config defaults name a different one).
        for key, val in _TOKENIZER_DEFAULTS.items():
            if key not in cfg_override_keys:
                OmegaConf.update(cfg, key, val, merge=False)
        # Measure the RAW tokenizer output: whitening must never be on while producing
        # the very stats that configure it, or it would measure an already-whitened space.
        cfg.model.whiten_stats = None
        # No RGB is decoded here, so skip loading the MAE image decoder entirely
        # (_build_occ_rae treats a falsy ckpt_path as "no decoder").
        if cfg.model.get("img_decoder", None) is not None:
            cfg.model.img_decoder.ckpt_path = None
        # Nothing is written per-run: no TensorBoard, no ckpt (this script never trains).
        # vit_folder exists only because get_network unconditionally makedirs it.
        cfg.training.writer_log = ""
        cfg.training.vit_folder = tempfile.mkdtemp(prefix="deltatok_latent_stats_") + "/"

    out_path = args.out or os.path.join(
        os.path.dirname(str(cfg.model.deltatok_ckpt)), f"latent_stats_{_sanitize(args.tag)}.pth"
    )

    # resume=False + ckpt=None: NO flow ckpt is loaded — these stats depend only on the
    # frozen OccRAE + DeltaTok. __init__ still builds a fresh (unused) flow ViT; that is
    # the price of reusing the trainer's encode plumbing instead of re-deriving it.
    trainer_args = argparse.Namespace(
        resume=False, ckpt=None, test_only=True, eval_only=True,
        debug=False, is_multi_gpus=False,
    )
    trainer = DeltaTokFlowMatchingTrainer(
        args=trainer_args, cfg=cfg, device=device,
        rank=0, world_size=1, distributed=False,
    )

    loaders = _build_split_loaders(cfg, args.split)
    if len(loaders) > 1:
        print(f"[WARN] {args.split} has {len(loaders)} sub-datasets; measuring only the "
              f"first ({loaders[0][0]}). Whitening stats must describe ONE latent space.")
    label, loader = loaders[0]
    print(f"\n[INFO] Accumulating over {label} ({len(loader)} batches available, "
          f"using {args.num_batches})")
    _pin_loader_epoch(loader)

    mean, std, n, row_ms = accumulate_channel_stats(trainer, loader, args)
    report_stats(mean, std, n, row_ms, label, args.num_batches)

    payload = {
        "mean": mean.float(),                                   # (C,) per-channel mean
        "std": std.float(),                                     # (C,) per-channel std, UNfloored (the loader floors)
        "count": int(n),
        "row_mean_square": float(row_ms),
        # Provenance: which tokenizer these moments belong to. _load_whiten_stats warns
        # when deltatok_ckpt does not match the run's frozen tokenizer — the moments are
        # meaningless for any other one.
        "deltatok_ckpt": str(cfg.model.deltatok_ckpt),
        "encode_layer": int(cfg.model.encode_layer),
        "num_delta_tokens": int(cfg.model.deltatok.num_delta_tokens),
        "split": args.split,
        "dataset": str(cfg.dataset.train_dataset if args.split == "train"
                       else cfg.dataset.test_dataset),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)

    print(f"\n[INFO] Wrote {out_path}")
    print(f"[INFO] Feed it to the flow trainer with:\n"
          f"[INFO]   model.whiten_stats={out_path}")


if __name__ == "__main__":
    main()
