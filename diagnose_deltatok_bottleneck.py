#!/usr/bin/env python3
"""Diagnose the DeltaTok channel bottleneck: is 768 a CAPACITY or an OPTIMIZATION limit?

The tc768 run (K=64, Cz=768) plateaus ~7x worse than uncompressed (K=64, Cz=1536) at
equal float budget, and can't make the uncompressed model's step-3-5k breakthrough. This
script decides WHY, without training the main model:

  Part A (train split): SVD the frozen dtok64 delta tokens z (post final LayerNorm, 1536-d).
    Report cumulative explained variance at candidate ranks + participation-ratio effective
    rank. Save {mean, eigvecs, eigvals} -> the PCA basis a retrofit fix would init from.

  Part B (recon split, the decisive test): for each rank r, project z onto its top-r PCA
    subspace P_r(z) and feed it through the FROZEN decoder; measure log-cosh recon vs the
    full-z recon. Explained variance is a proxy; recon-after-roundtrip is the real thing.

Verdict:
  recon_768 ~= recon_full (~0.01)  -> OPTIMIZATION-limited: 768 dims hold the code, SGD just
                                      can't find/invert the subspace -> retrofit-PCA will work.
  recon_768 -> ~0.06               -> CAPACITY-limited: no shared linear map suffices ->
                                      nonlinear bottleneck or gentler Cz.

Whitening (Lambda^-1/2) is a per-axis rescale the decoder's up-projection inverts, so it does
NOT change the recon test — Part B uses the orthonormal projection only. Uses only the frozen
OccRAE + DeltaTok (encode + decode); no flow ViT weights, no flow ckpt.

Usage (Jean Zay GPU node; defaults = the dtok64 non-affine tokenizer the waymo flow freezes):
  source env_jz_h100.sh && python diagnose_deltatok_bottleneck.py
  # Smoke first: --num_batches 4 --recon_num_batches 2
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
from occrae.deltatok_trainer import _log_cosh  # noqa: E402  (same recon loss the tokenizer trained on)
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

# Reference plateaus from the matched TB runs, printed alongside the measured recon so the
# verdict is anchored (uncompressed = the target, tc768 = the broken bottleneck).
_REF_UNCOMPRESSED = 0.0098   # K=64 Cz=1536 train recon plateau
_REF_TC768 = 0.065           # K=64 Cz=768  train recon plateau


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose whether the DeltaTok 768 channel bottleneck is capacity- or optimization-limited."
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
        "--num_batches", type=int, default=300,
        help="Part A: train batches for the PCA covariance (300 ~ 2.5M token vectors). "
             "Scene coverage matters more than estimator variance for the eigenspectrum.",
    )
    parser.add_argument(
        "--recon_num_batches", type=int, default=64,
        help="Part B: batches for the roundtrip-through-frozen-decoder recon (each runs "
             "1 decode per rank + 1 full decode; decode is cheap vs the ~1B OccRAE encode).",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=("train", "val"),
        help="Part A split: the PCA basis is a train-set statistic.",
    )
    parser.add_argument(
        "--recon_split", type=str, default="val", choices=("train", "val"),
        help="Part B split: recon roundtrip. val checks the 768-subspace generalizes; "
             "falls back to train if the config has no val set.",
    )
    parser.add_argument(
        "--ranks", type=str, default="256,384,512,640,768,896,1024,1280",
        help="Comma-separated PCA ranks to probe. 768 (the tc768 target) is always added.",
    )
    parser.add_argument(
        "--tag", type=str, default="waymo",
        help="Names the output: pca_basis_<tag>.pth, written beside the tokenizer ckpt.",
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

    train: the whole `+`-joined expression in ONE loader;
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


def _encode_batch(trainer, net, batch):
    """Frozen OccRAE + DeltaTok encode of one batch -> (x_prev, x, z, rope_local, rope_global).

    z is the RAW tokenizer output (post final non-affine LayerNorm), the exact tensor the
    channel bottleneck would compress. x_prev/x are the decode inputs/target for Part B.
    """
    batch = trainer._normalize_batch(batch)
    imgs = batch["imgs"].to(trainer.device, non_blocking=True)                  # (B, V, 3, H, W)
    num_cameras = int(batch.get("num_cameras", 1))
    _, _, x_prev, x, H, W = trainer._extract_pair_feats(imgs, num_cameras=num_cameras)  # x_prev,x (M, N, P, C)
    rope_local = net._compute_rope(H, W, x_prev.device, x_prev.dtype)           # length-P single-camera rope
    rope_global = net._compute_global_rope(H, W, num_cameras, x_prev.device, x_prev.dtype)  # length-(N*P) rope
    with trainer.autocast:
        z = net.encode(x_prev, x, rope_local, rope_global)                     # (M, N, K, C) delta tokens
    return x_prev, x, z, rope_local, rope_global


@torch.no_grad()
def accumulate_pca(trainer, net, loader, num_batches):
    """Mean + covariance of the frozen delta tokens over `num_batches`, then eigendecompose.

    Rows = token vectors (M = B*(T-1)*N*K), stats over the (C,) columns. float64: bf16 sums
    over millions of rows would lose the small means / off-diagonal covariance.

    Returns (mean (C,), eigvals (C,) desc, eigvecs (C, C) cols desc, n rows, row_mean_square).
    """
    total = None       # (C,) float64 running sum
    total_cov = None   # (C, C) running sum of outer products z z^T
    n = 0              # token rows seen
    n_tokens = 0       # denominator of the LayerNorm row check (== n here)
    row_ms_sum = 0.0   # running sum over rows of mean(z^2) — LayerNorm sanity (~1.0)

    done = 0
    for batch in loader:
        if done >= num_batches:
            break
        _, _, z, _, _ = _encode_batch(trainer, net, batch)          # z (M, N, K, C)
        C = z.shape[-1]                                             # channels (backbone.embed_dim)
        m = z.reshape(-1, C).float()                              # (Mb, C) all token vectors (TF32 matmul ~30x faster than fp64)
        if total is None:
            total = torch.zeros(C, dtype=torch.float64, device=m.device)
            total_cov = torch.zeros(C, C, dtype=torch.float64, device=m.device)
        total += m.sum(0).double()                                # (C,) per-channel sum
        total_cov += (m.t() @ m).double()                         # (C, C) sum of outer products (TF32 matmul, fp64 accum)
        row_ms = m.square().mean(-1)                               # (Mb,) per-row mean(z^2)
        row_ms_sum += float(row_ms.sum())
        n_tokens += row_ms.numel()
        n += m.shape[0]
        done += 1
        print(f"[PCA]   batch {done}/{num_batches}  rows={n}  C={C}", flush=True)

    if n == 0:
        raise RuntimeError("No usable batches for PCA.")

    mean = total / n                                              # (C,)
    cov = (total_cov / n) - torch.outer(mean, mean)              # (C, C) covariance E[zz^T]-E[z]E[z]^T
    evals, evecs = torch.linalg.eigh(cov.cpu())                  # ascending; evecs columns = eigenvectors
    evals = evals.flip(0).clamp_min(0)                          # (C,) descending, floor round-off negatives
    evecs = evecs.flip(1)                                       # (C, C) columns reordered to match desc evals
    return mean.cpu(), evals, evecs, n, row_ms_sum / n_tokens


def _project_z(z, mean, Vr):
    """Project delta tokens onto the top-r PCA subspace: P_r(z) = mean + (z-mean) Vr Vr^T.

    z: (M, N, K, C) any dtype. mean: (C,) float32 device. Vr: (C, r) float32 device (top-r
    orthonormal eigenvectors). Returns (M, N, K, C) in z's dtype.
    """
    C = z.shape[-1]
    zc = z.float() - mean                                        # (M, N, K, C) centered
    flat = zc.reshape(-1, C)                                     # (*, C)
    proj = (flat @ Vr) @ Vr.t()                                 # (*, C) = P_r applied to (z-mean)
    return (proj.reshape_as(zc) + mean).to(z.dtype)             # (M, N, K, C) recentered, original dtype


@torch.no_grad()
def roundtrip_recon(trainer, net, loader, mean, evecs, ranks, num_batches):
    """Feed P_r(z) through the frozen decoder and measure log-cosh recon, per rank.

    Returns (full_loss, {rank: loss}) — full_loss decodes the un-projected z (the ceiling).
    """
    device = trainer.device
    mean_d = mean.to(device).float()                            # (C,)
    Vr = {r: evecs[:, :r].to(device).float() for r in ranks}   # r -> (C, r) top-r eigenvectors

    full_sum = 0.0
    rank_sum = {r: 0.0 for r in ranks}
    done = 0
    for batch in loader:
        if done >= num_batches:
            break
        x_prev, x, z, rope_local, rope_global = _encode_batch(trainer, net, batch)  # z (M, N, K, C)
        with trainer.autocast:
            x_hat = net.decode(z, x_prev, rope_local, rope_global)                 # (M, N, P, C) full-z recon
        full_sum += float(_log_cosh(x_hat.float(), x.float()).mean())
        for r in ranks:
            z_r = _project_z(z, mean_d, Vr[r])                                     # (M, N, K, C) P_r(z)
            with trainer.autocast:
                x_hat_r = net.decode(z_r, x_prev, rope_local, rope_global)         # (M, N, P, C)
            rank_sum[r] += float(_log_cosh(x_hat_r.float(), x.float()).mean())
        done += 1
        print(f"[RECON]  batch {done}/{num_batches}", flush=True)

    if done == 0:
        raise RuntimeError("No usable batches for recon roundtrip.")
    return full_sum / done, {r: rank_sum[r] / done for r in ranks}


def report_pca(mean, evals, ranks, n, row_ms) -> None:
    """Print the eigenspectrum: participation-ratio effective rank + cumulative explained variance."""
    C = mean.shape[0]
    tot = float(evals.sum())
    pr = float((evals.sum() ** 2) / evals.square().sum())       # participation-ratio effective rank
    cum = torch.cumsum(evals, 0) / max(tot, 1e-12)             # (C,) cumulative variance fraction

    print(f"\n[PCA] n={n} token rows, C={C}")
    # LayerNorm normalizes each row to mean 0 / var 1, so mean(z^2) per row must be ~1.0.
    print(f"[PCA] LayerNorm check: mean row mean-square = {row_ms:.4f}  (must be ~1.0)")
    print(f"[PCA] participation-ratio effective rank = {pr:.1f} / {C}")
    print(f"[PCA] cumulative explained variance by PCA rank:")
    for r in ranks:
        print(f"[PCA]   r={r:5d}: {float(cum[r - 1]):.4f}")


def report_recon(full_loss, recon, ranks) -> None:
    """Print recon-after-roundtrip per rank against the full-z ceiling and the known plateaus."""
    print(f"\n[RECON] full-z (rank=C, no projection): loss={full_loss:.4f}  "
          f"(ref uncompressed plateau ~{_REF_UNCOMPRESSED})")
    print(f"[RECON] roundtrip recon vs PCA rank  (ref tc768 plateau ~{_REF_TC768}):")
    for r in ranks:
        print(f"[RECON]   r={r:5d}: {recon[r]:.4f}")


def print_verdict(full_loss, recon, ranks) -> None:
    """Classify capacity vs optimization from recon_768 vs the full-z ceiling."""
    l768 = recon[768]
    opt_thresh = max(0.02, 1.5 * full_loss)                    # "close to the ceiling"
    # smallest probed rank whose roundtrip recon still hugs the ceiling = the usable knee.
    knee = next((r for r in ranks if recon[r] <= opt_thresh), None)
    print(f"\n[VERDICT] full-z recon={full_loss:.4f}  recon@768={l768:.4f}  (opt threshold={opt_thresh:.4f})")
    if l768 <= opt_thresh:
        print("[VERDICT] OPTIMIZATION-limited: a 768-d PCA subspace preserves recon; the from-scratch")
        print("[VERDICT] bottleneck just can't find/invert it. -> retrofit PCA init (z_proj_down=Lambda^-1/2 U_768^T,")
        print("[VERDICT] z_proj_up=U_768 Lambda^1/2) + short fine-tune should recover uncompressed quality.")
    elif l768 >= 0.05:
        print("[VERDICT] CAPACITY-limited: 768 linear dims lose real recon quality. -> a shared linear")
        print("[VERDICT] bottleneck can't work; try a nonlinear (MLP) bottleneck or a gentler Cz.")
        if knee is not None:
            print(f"[VERDICT] knee: rank {knee} is the smallest probed rank still near the ceiling.")
    else:
        print("[VERDICT] PARTIAL: 768 is borderline. Consider a gentler Cz;")
        if knee is not None:
            print(f"[VERDICT] knee: rank {knee} is the smallest probed rank near the ceiling ({opt_thresh:.4f}).")


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
        # Measure the RAW tokenizer output: whitening must never be on while producing the
        # PCA basis, or it would rotate/scale the space these stats describe.
        cfg.model.whiten_stats = None
        # No RGB is decoded here, so skip loading the MAE image decoder entirely.
        if cfg.model.get("img_decoder", None) is not None:
            cfg.model.img_decoder.ckpt_path = None
        # Nothing is written per-run: no TensorBoard, no ckpt (this script never trains).
        cfg.training.writer_log = ""
        cfg.training.vit_folder = tempfile.mkdtemp(prefix="deltatok_bottleneck_diag_") + "/"

    out_path = args.out or os.path.join(
        os.path.dirname(str(cfg.model.deltatok_ckpt)), f"pca_basis_{_sanitize(args.tag)}.pth"
    )

    # resume=False + ckpt=None: NO flow ckpt is loaded — the diagnostic depends only on the
    # frozen OccRAE + DeltaTok. __init__ still builds a fresh (unused) flow ViT.
    trainer_args = argparse.Namespace(
        resume=False, ckpt=None, test_only=True, eval_only=True,
        debug=False, is_multi_gpus=False,
    )
    trainer = DeltaTokFlowMatchingTrainer(
        args=trainer_args, cfg=cfg, device=device,
        rank=0, world_size=1, distributed=False,
    )
    net = getattr(trainer.deltatok, "_orig_mod", trainer.deltatok)  # frozen DeltaTokModule (peel any compile wrap)

    ranks = sorted({int(r) for r in args.ranks.split(",") if r.strip()} | {768})

    # --- Part A: PCA eigenspectrum on the train split -------------------------------------
    a_loaders = _build_split_loaders(cfg, args.split)
    label_a, loader_a = a_loaders[0]
    print(f"\n[INFO] Part A (PCA) over {label_a} using {args.num_batches} batches")
    _pin_loader_epoch(loader_a)
    mean, evals, evecs, n, row_ms = accumulate_pca(trainer, net, loader_a, args.num_batches)
    C = mean.shape[0]
    ranks = [r for r in ranks if 0 < r < C]                    # a rank-C projection is identity (== full-z)
    report_pca(mean, evals, ranks, n, row_ms)

    # --- Part B: roundtrip-through-frozen-decoder recon on the recon split ----------------
    try:
        b_loaders = _build_split_loaders(cfg, args.recon_split)
    except RuntimeError:
        print(f"[WARN] no '{args.recon_split}' split available; falling back to train for recon.")
        b_loaders = _build_split_loaders(cfg, "train")
    label_b, loader_b = b_loaders[0]
    if len(b_loaders) > 1:
        print(f"[WARN] recon split has {len(b_loaders)} sub-datasets; using only {label_b}.")
    print(f"\n[INFO] Part B (recon roundtrip) over {label_b} using {args.recon_num_batches} batches")
    _pin_loader_epoch(loader_b)
    full_loss, recon = roundtrip_recon(trainer, net, loader_b, mean, evecs, ranks, args.recon_num_batches)
    report_recon(full_loss, recon, ranks)
    print_verdict(full_loss, recon, ranks)

    # --- Save the PCA basis (the retrofit fix's init) + the diagnostic results ------------
    payload = {
        "mean": mean.float(),                                   # (C,)
        "eigvals": evals.float(),                               # (C,) descending
        "eigvecs": evecs.float(),                               # (C, C) columns = eigenvectors, descending
        "ranks": ranks,
        "recon_by_rank": {int(r): float(recon[r]) for r in ranks},
        "recon_full": float(full_loss),
        "count": int(n),
        "row_mean_square": float(row_ms),
        "deltatok_ckpt": str(cfg.model.deltatok_ckpt),
        "encode_layer": int(cfg.model.encode_layer),
        "num_delta_tokens": int(cfg.model.deltatok.num_delta_tokens),
        "pca_split": args.split,
        "recon_split": label_b,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"\n[INFO] Wrote PCA basis + results to {out_path}")


if __name__ == "__main__":
    main()
