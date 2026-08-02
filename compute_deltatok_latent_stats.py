#!/usr/bin/env python3
"""Per-channel or per-position mean/std of frozen DeltaTok delta tokens (whitening stats).

Two modes (``--stats_mode``):
  channel (default): stats (C,) pooled over every slot — whitening equalizes channels.
  position: stats (K, C) — per delta-slot k — pooled over batch, transitions AND
    cameras. Mirrors RAE's calculate_stat.py, which keeps stats per latent ELEMENT
    (C, H, W), not per channel: pooled-(C,) whitening folds slot-dependent offsets
    into std, so within-slot variation stays non-isotropic after whitening. The
    camera axis is pooled, not resolved: the train sampler varies views_per_timestep
    (observed N=1 batches), so a fixed per-camera axis can neither be measured from
    train batches nor applied to them.

DeltaTok ends in a parameter-free ``nn.LayerNorm``, which normalizes each token ACROSS
channels: every ROW of the (M, C) delta-token matrix gets mean 0 / RMS 1. Columns are
unconstrained — channel c's std over the dataset is free, the massive-activation pattern
DINOv3 blocks are known for. The per-element bound is only |x_i| <= sqrt(C-1) (~39 at
C=1536), hence the observed max of 36.7.

The flow adds ISOTROPIC noise e ~ N(0, I) (``flow_noising``), so SNR(t) = t*std/(1-t):
one t means a different noise level per channel (and, in position mode, per camera/slot),
and the quadratic loss is dominated by the high-std ones. Measuring the moments lets
``model.whiten_stats`` normalize each measured unit to std 1.

Uses only the frozen OccRAE + DeltaTok encode: no flow ViT weights, no flow ckpt.
float64 over ALL slots (context included — whitening applies to the whole tensor).

Gate: ``std.max()/std.min()``. ~1-2x -> whitening is a no-op (itself the unpublished
ablation result); >= ~10x -> the isotropic noise is mismatched, Phase 1 is worth it.

Usage (Jean Zay GPU node; defaults = the dtok64 non-affine tokenizer the waymo flow
runs freeze). Writes latent_stats_<tag>.pth beside the tokenizer ckpt:
  source env_jz_h100.sh && python compute_deltatok_latent_stats.py
  # Smoke first: --num_batches 2
  # RAE-style per-position stats: --stats_mode position
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
# slurm/deltatok_flow/train_deltatok_flow_waymo_jz.slurm cannot change z). Defaults = the dtok64
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
    parser.add_argument("--config-dir", type=str, default="configs/deltatok_flow")
    parser.add_argument("--config-name", type=str, default="train_deltatok_flow_waymo_jeanzay")
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
        "--stats_mode", type=str, default="channel", choices=("channel", "position"),
        help="channel: (C,) pooled over all slots (original). position: (K, C) per "
             "delta-slot, RAE-style per-element stats (camera axis pooled).",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Names the output: latent_stats_<tag>.pth, written beside the tokenizer ckpt. "
             "Default: 'waymo' (channel mode) / 'waymo_pos' (position mode) — distinct so "
             "one mode never clobbers the other's file.",
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
    """Per-channel or per-position moments of the delta tokens over `--num_batches` batches.

    channel mode: rows = tokens (M = B*(T-1)*N*K), stats over the (C,) columns.
    position mode: rows = transition-cameras (M = B*(T-1)*N), stats keep the (K, C)
    slot layout — camera axis pooled so variable-N batches stay compatible. LayerNorm
    normalizes token rows; this measures across them. float64: bf16 sums over ~600k
    rows would lose the small means.

    Returns (mean, std, n, row_ms) — mean/std are (C,) or (K, C) by mode, n counts
    rows (samples per stat element), row_ms is the mean row mean-square, the LayerNorm
    sanity check (~1.0).
    """
    total = None       # (C,) or (K, C) float64 running sum
    total_sq = None    # same shape, running sum of squares
    n = 0              # rows seen (samples per stat element)
    n_tokens = 0       # token vectors seen (denominator of the LayerNorm row check)
    row_ms_sum = 0.0   # running sum over token rows of mean(x^2)

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
        # Channels are already last. EVERY slot counts, context included: whitening is
        # applied to the whole tensor, not to a subset.
        if args.stats_mode == "position":
            m = z.reshape(-1, *z.shape[3:]).double()               # (M_b, K, C) M_b = B*(T-1)*N transition-cameras
        else:
            m = z.reshape(-1, C).double()                          # (M_b, C) M_b = B*(T-1)*N*K token vectors
        if total is None:
            total = torch.zeros(m.shape[1:], dtype=torch.float64, device=m.device)
            total_sq = torch.zeros_like(total)
        assert total.shape == m.shape[1:], (
            f"batch stats shape {tuple(m.shape[1:])} != accumulated {tuple(total.shape)}; "
            "position mode needs a fixed delta-slot count across batches"
        )
        total += m.sum(0)                                          # per-element sum, (C,) or (N, K, C)
        total_sq += m.square().sum(0)                              # per-element sum of squares
        row_ms = m.square().mean(-1)                               # (M_b,) or (M_b, N, K) per-token-row mean(x^2)
        row_ms_sum += float(row_ms.sum())                          # LayerNorm check accumulator
        n_tokens += row_ms.numel()
        n += m.shape[0]

        batches_done += 1
        print(f"[INFO]   batch {batches_done}/{args.num_batches}  "
              f"tokens={n_tokens} C={C} res={H}x{W}", flush=True)

    if n == 0:
        raise RuntimeError("No usable batches: nothing to measure.")

    mean = total / n                                               # (C,) or (K, C) per-element mean
    var = (total_sq / n) - mean.square()                           # E[x^2] - E[x]^2
    std = var.clamp_min(0).sqrt()                                  # clamp: round-off can make var slightly negative
    return mean.cpu(), std.cpu(), n, row_ms_sum / n_tokens


def report_stats(mean, std, n, row_ms, label, num_batches) -> float:
    """Print the distribution of stds (per channel or per position) and return the gate ratio."""
    C = mean.shape[-1]
    flat = std.reshape(-1)                                         # all whitening divisors
    smin, smax = float(flat.min()), float(flat.max())
    ratio = smax / max(smin, 1e-12)

    qs = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=torch.float64)
    pct = torch.quantile(flat, qs)
    # position mode: rank channels by their worst (max-over-position) std.
    per_channel = std if std.dim() == 1 else std.amax(dim=tuple(range(std.dim() - 1)))
    topk = torch.topk(per_channel, min(10, C))

    shape_str = f"C={C}" if mean.dim() == 1 else f"(K,C)={tuple(mean.shape)}"
    print(f"\n[STATS] split={label}  {shape_str}  n={n} rows ({num_batches} batches)")
    # LayerNorm normalizes each row to mean 0 / var 1, so mean(x^2) per row must be 1.
    # A deviation here means the measured tensor is NOT the tokenizer's normed output.
    print(f"[STATS] LayerNorm check: mean row mean-square = {row_ms:.4f}  (must be ~1.0)")
    print(f"[STATS] |mean| max = {float(mean.abs().max()):.4f}")
    print(f"[STATS] std: min={smin:.4f}  max={smax:.4f}  ratio={ratio:.1f}x")
    print(f"[STATS] std percentiles: p0={pct[0]:.4f} p1={pct[1]:.4f} p50={pct[2]:.4f} "
          f"p99={pct[3]:.4f} p100={pct[4]:.4f}")
    print(f"[STATS] top-{topk.values.numel()} channels by std:")
    for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"[STATS]   c={i:5d}  std={v:.4f}")

    if mean.dim() > 1:
        # Law of total variance over positions: pooled (C,) var = E_pos[var] + Var_pos[mean].
        # The between share is exactly what pooled-(C,) whitening cannot remove.
        pos_dims = tuple(range(mean.dim() - 1))                    # all axes but channel
        within_var = std.square().mean(dim=pos_dims)               # (C,) E_pos[var_c]
        between_var = mean.var(dim=pos_dims, unbiased=False)       # (C,) Var_pos[mean_c]
        pooled_var = (within_var + between_var).clamp_min(1e-12)   # (C,) the channel-mode variance
        pooled_ratio = float((pooled_var.max() / pooled_var.min()).sqrt())
        between_share = between_var / pooled_var                   # (C,) slot-structure share
        print(f"[STATS] pooled-(C,) view: ratio={pooled_ratio:.1f}x  "
              f"between-position share of var: median={float(between_share.median()):.2f} "
              f"max={float(between_share.max()):.2f}")

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

    # Mode-specific default tag: channel keeps the original file name, position gets
    # its own so the two modes never overwrite each other.
    tag = args.tag or ("waymo" if args.stats_mode == "channel" else "waymo_pos")
    out_path = args.out or os.path.join(
        os.path.dirname(str(cfg.model.deltatok_ckpt)), f"latent_stats_{_sanitize(tag)}.pth"
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
        "mean": mean.float(),                                   # (C,) or (K, C) mean
        "std": std.float(),                                     # matching std, UNfloored (the loader floors)
        "stats_mode": args.stats_mode,                          # loader distinguishes by ndim; this is provenance
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
