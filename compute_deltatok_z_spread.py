#!/usr/bin/env python3
"""Paired isotropy/spread comparison of two DeltaTok tokenizers' z on the SAME batches.

Variant of ``compute_deltatok_latent_stats.py``: that script measures ONE tokenizer's
per-channel moments to build whitening stats; this one loads N tokenizer checkpoints of
the SAME architecture, runs the frozen OccRAE encode ONCE per batch, and encodes those
identical features with every arm. Paired by construction — the only difference between
the reported numbers is the tokenizer weights, not the data.

Built for the question "did SIGReg spread the variance better?", i.e.
``..._prenorm_sigreg0.05_pool_nozn`` vs ``..._prenorm_nozn``. Two views of the variance:

  per-channel std   min/max/ratio + percentiles of sqrt(diag(cov)) — axis-aligned spread.
  eigenspectrum     eigvals of the (Cz, Cz) covariance. Scale-free shares and effective
                    ranks (participation ratio, spectral entropy, n90/n99) — this is the
                    "rank ~290 << 768" collapse SIGReg exists to fix, and it is basis
                    -free where per-channel std is not.

These runs are `_nozn` (no final LayerNorm), so z is raw ``z_proj_down`` output: rows are
NOT unit-RMS and the absolute scale carries no meaning across arms. Every share, ratio and
effective rank below is invariant to an overall rescale of z, so those are the cross-arm
verdict; total variance and the raw std percentiles are within-arm context only.

The moments and the spread scalars come from ``occrae.z_spread.ZSpreadStats`` -- the same
class the trainer logs ``Eval/<test>/Z*`` from, so an offline number and a TensorBoard
number for one checkpoint can never disagree. Only the extra reporting lives here.

z is encoded in bf16 (the training dtype), so the smallest eigenvalues sit on a
quantization floor — read the top of the spectrum and the effective ranks, not l_min.
`--cfg training.dtype=float32` removes that floor at ~2x the cost.

Usage (Jean Zay GPU node, one GPU; defaults = the dtok64/tc768/prenorm/nozn arm pair):
  source env_jz_h100.sh && python compute_deltatok_z_spread.py
  # Smoke first: --num_batches 2
  # Other arms: --ckpt label=/path/to/ckpt (repeatable, replaces the defaults)
"""

import argparse
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

from occrae.deltatok_trainer import DeltaTokTrainer  # noqa: E402
from occrae.z_spread import ZSpreadStats  # noqa: E402  shared with the trainer's Eval/*/Z* logging
from occany.datasets import get_data_loader  # noqa: E402

_LOG_ROOT = "/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
_ARM = "deltatok_surround_layer12_bsize2_dtok64_tc768_cos1e3_1e5_b95_gradskip_noaff_prenorm"

# Matched epoch (both arms have epoch_40.pth) so training length is not a confound.
_DEFAULT_CKPTS = [
    f"nosigreg={_LOG_ROOT}/{_ARM}_nozn/ckpts/epoch_40.pth",
    f"sigreg0.05={_LOG_ROOT}/{_ARM}_sigreg0.05_pool_nozn/ckpts/epoch_40.pth",
]

# Architecture of BOTH arms (they differ only in the sigreg loss term, which is not a
# parameter). Mirrors slurm/deltatok/archive/train_deltatok_multitoken_mlp_sigreg_jz.slurm's EXTRA_CFG.
_TOKENIZER_DEFAULTS = {
    "model.deltatok.num_delta_tokens": 64,       # K=64 delta tokens/camera
    "model.deltatok.target_channels": 768,       # Cz=768 channel bottleneck
    "model.deltatok.norm_affine": False,         # _noaff
    "model.deltatok.z_norm": False,              # _nozn: no final LayerNorm, z is raw
    "model.deltatok.bottleneck_mlp": False,      # no _mlp tag in either run name
    "model.deltatok.use_camera_rope": False,     # layer-12 no-camera-rope run
}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired z spread/isotropy comparison of two DeltaTok checkpoints."
    )
    parser.add_argument("--config-dir", type=str, default="configs/deltatok")
    parser.add_argument("--config-name", type=str, default="train_deltatok_jeanzay")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Hydra-style overrides, e.g. training.bsize=4. Takes precedence over the "
             "injected tokenizer defaults.",
    )
    parser.add_argument(
        "--ckpt", type=str, action="append", default=None, metavar="LABEL=PATH",
        help="Tokenizer checkpoint to score; repeatable. Replaces the default arm pair.",
    )
    parser.add_argument("--occany_recon_ckpt", type=str, default=None)
    parser.add_argument("--encode_layer", type=int, default=12)
    parser.add_argument(
        "--num_batches", type=int, default=100,
        help="Batches to accumulate. 100 ~ 128k z rows at Cz=768, enough for a stable "
             "(768, 768) covariance.",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=("train", "val"),
        help="train: the distribution SIGReg actually regularized. val checks drift.",
    )
    parser.add_argument(
        "--whiten_stats", type=str, default=None, metavar="PATH",
        help="Apply this run's whitening (latent_stats_*.pth) before measuring. Required to "
             "compare against a flow run trained with model.whiten_stats: the flow sees the "
             "WHITENED code, so the marginal-only floor must be measured on that, not on raw z.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_whiten(path: str, device):
    """(mean, std) from a latent_stats file, shaped to broadcast over z's (M, N, K, Cz)."""
    blob = torch.load(Path(path).expanduser(), map_location="cpu")
    mean, std = blob["mean"].to(device).float(), blob["std"].to(device).float()  # (Cz,) or (K, Cz)
    if mean.shape != std.shape:
        raise ValueError(f"whiten stats mean {tuple(mean.shape)} != std {tuple(std.shape)}")
    if mean.dim() not in (1, 2):
        raise ValueError(f"whiten stats must be (Cz,) or (K, Cz), got {tuple(mean.shape)}")
    print(f"[INFO] whitening with {path}  shape={tuple(mean.shape)}")
    return mean, std


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")


def _parse_ckpt_specs(specs):
    """['label=path', ...] -> [(label, path)], with the label defaulting to the run dir."""
    out = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            # No label given: name the arm after its run directory (…/<run>/ckpts/x.pth).
            path, label = label, Path(label).parent.parent.name
        out.append((label, str(Path(path).expanduser())))
    return out


def _build_loader(cfg, split):
    """One loader for the split: train as the single `+`-joined expression (as training
    does), val as its FIRST sub-dataset only — one latent space per report."""
    if split == "train":
        return "train", get_data_loader(
            str(cfg.dataset.train_dataset),
            batch_size=int(cfg.training.bsize),
            num_workers=int(cfg.training.num_workers),
            shuffle=True,      # the first --num_batches batches should span scenes
            drop_last=True,
            per_dataset_sampling=bool(cfg.dataset.get("per_dataset_sampling", False)),
        )

    subs = [s.strip() for s in str(cfg.dataset.test_dataset).split("+") if s.strip()]
    if len(subs) > 1:
        print(f"[WARN] val has {len(subs)} sub-datasets; measuring only the first.")
    return f"val_{_sanitize(subs[0].split('(')[0].strip())}", get_data_loader(
        subs[0],
        batch_size=int(cfg.training.bsize),
        num_workers=int(cfg.training.get("val_num_workers", 2)),
        shuffle=False,
        drop_last=False,
    )


def _pin_loader_epoch(loader) -> None:
    """Pin sampler/dataset to epoch 0: ResizedDataset asserts set_epoch was called, and
    it also makes repeat invocations see the same samples."""
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "set_epoch"):
        ds.set_epoch(0)


def _load_tokenizer_weights(net, path: str) -> None:
    """Load a DeltaTok trainer checkpoint's model weights into ``net`` (strict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in ckpt["model_state_dict"].items()
    }
    # Key-exact (the trainer loads with strict=False for cross-arch init): a silently
    # random bottleneck would look exactly like a spread difference.
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (
        f"{path} does not match the configured architecture: "
        f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
    )
    net.eval()
    print(f"[INFO] loaded {path}  (iter={ckpt.get('iter')}, epoch={ckpt.get('global_epoch')})")


def flow_loss_floor(evals, t_clamp: float = 0.05, n_t: int = 4096) -> float:
    """Flow loss a model reaches on this code using ZERO conditional information.

    Rectified flow adds isotropic N(0, I) noise to every one of the Cz directions, but the
    code's variance is not isotropic -- a direction holding ~no variance has target ~0, a
    constant the denoiser nails immediately and near-freely. This integrates that free part
    out: per eigen-direction i, the Gaussian marginal-only residual under the trainer's
    pred_mode=x / loss_mode=v (see occrae/deltatok_flow_trainer.flow_loss),

        Var(x_i | z_t) = s_i^2 (1-t)^2 / (t^2 s_i^2 + (1-t)^2),  loss weight 1/clamp(1-t)^2

    averaged over the configured t ~ U(0,1). Gaussian is the weak assumption, and it is
    exactly what SIGReg buys -- on a sigreg arm this floor is tight, which is what makes
    the comparison against the run's achieved Eval/Loss meaningful.

    Read it as: achieved >> floor means the model is not using its conditioning at all;
    achieved well below floor means real conditional structure was learned. SCALE-dependent
    (unlike everything else in this report), so it is only comparable to a flow run trained
    on this exact code -- an unwhitened arm at row mean-square 108 will read enormous.
    """
    t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)                 # (n_t,) flow time
    var = evals.double().clamp_min(0).unsqueeze(1)                         # (C, 1) per-direction variance
    om = (1.0 - t).unsqueeze(0)                                            # (1, n_t) 1-t
    resid = var * om.square() / (t.unsqueeze(0).square() * var + om.square()).clamp_min(1e-300)  # (C, n_t)
    integrand = resid / om.clamp_min(t_clamp).square()                     # (C, n_t) v-space weighting
    return float(torch.trapz(integrand, t, dim=1).mean())                  # mean over directions


def summarize(label, s):
    """Print one arm's spread report and return the numbers the comparison table uses.

    ``s`` is a ``ZSpreadStats.summary(full=True)`` dict — the SAME implementation the
    trainer logs from, so an offline number and a TensorBoard number can never disagree.
    Everything below is presentation plus the extras only this report shows.
    """
    C = s["cz"]
    std = s["std"]                                                     # (C,) per-channel std
    smin, smax = float(std.min()), float(std.max())
    pct = torch.quantile(std, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=torch.float64))

    evals, p = s["evals"], s["shares"]                                 # (C,) descending, and their shares
    tot, pr, ent, n90 = s["total_var"], s["part_rank"], s["ent_rank"], s["n90"]
    n, row_ms = s["rows"], s["row_mean_square"]
    n99 = int((p.cumsum(0) < 0.99).sum()) + 1                          # 99% companion to n90
    lo = float(evals[-1])

    print(f"\n===== {label} =====")
    print(f"[SPREAD] rows={n}  Cz={C}  mean row mean-square={row_ms:.4f}  "
          f"|mean| max={s['mean_abs_max']:.4f}  total var={tot:.4f}")
    print(f"[SPREAD] per-channel std: min={smin:.4f} max={smax:.4f} ratio={smax / max(smin, 1e-12):.2f}x")
    # Row-scale tail. z_norm=true pins every row to unit RMS -> tail 1.00x; these _nozn arms
    # leave it free, and a large tail means an MSE over these rows is carried by few tokens.
    print(f"[SPREAD] row mean-square: p50={s['row_ms_p50']:.4f} p90={s['row_ms_p90']:.4f} "
          f"p99={s['row_ms_p99']:.4f} max={s['row_ms_max']:.4f}  "
          f"p99/p50={s['row_ms_tail']:.2f}x (norm ratio {s['row_ms_tail'] ** 0.5:.2f}x)  "
          f"top1%share={s['row_ms_top1pct']:.3f}")
    print(f"[SPREAD] std percentiles: p0={pct[0]:.4f} p1={pct[1]:.4f} p50={pct[2]:.4f} "
          f"p99={pct[3]:.4f} p100={pct[4]:.4f}")
    print(f"[SPREAD] eigenvalues: l1={float(evals[0]):.4f} l_med={float(evals[C // 2]):.6f} "
          f"l_min={lo:.3e}  l1/l_min={float(evals[0]) / max(lo, 1e-30):.3e}")
    # top290: the measured participation rank of the tc768 squeeze — if SIGReg spread the
    # code, this share should now fall clearly short of 1.0.
    print(f"[SPREAD] variance share: top1={float(p[0]):.3f} top10={float(p[:10].sum()):.3f} "
          f"top50={float(p[:50].sum()):.3f} top290={float(p[:min(290, C)].sum()):.3f}")
    print(f"[SPREAD] effective rank: participation={pr:.1f}/{C}  entropy={ent:.1f}/{C}  "
          f"n90={n90}  n99={n99}")
    # Finite-sample ceiling: even a perfect N(0,I) code reads below C when rows ~ C.
    # PR only means "collapsed" when it is far under THIS, not under C.
    print(f"[SPREAD] PR ceiling at n={n}: {C / (1.0 + C / n):.1f}/{C} "
          f"(a perfect N(0,I) code would read this)")
    floor = flow_loss_floor(evals)
    print(f"[SPREAD] marginal-only flow loss floor={floor:.4f}  "
          f"(what a flow reaches on this code using NO conditioning; scale-dependent)")
    return {
        "label": label, "C": C, "total_var": tot, "std_ratio": smax / max(smin, 1e-12),
        "pr": pr, "entropy_rank": ent, "n90": n90, "n99": n99,
        "top1": float(p[0]), "top10": float(p[:10].sum()), "row_ms_tail": s["row_ms_tail"],
    }


def main() -> None:
    args = get_args_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    arms = _parse_ckpt_specs(args.ckpt or _DEFAULT_CKPTS)

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.cfg)

    cfg_override_keys = {o.split("=", 1)[0] for o in args.cfg}
    with open_dict(cfg):
        if args.occany_recon_ckpt:
            cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
        cfg.model.encode_layer = int(args.encode_layer)
        for key, val in _TOKENIZER_DEFAULTS.items():
            if key not in cfg_override_keys:
                OmegaConf.update(cfg, key, val, merge=False)
        cfg.training.sigreg_weight = 0.0  # nothing trains here; keeps the trainer single-output
        # No RGB decode -> skip the MAE image decoder entirely.
        if cfg.model.get("img_decoder", None) is not None:
            cfg.model.img_decoder.ckpt_path = None
        # Nothing is written: no TensorBoard, no checkpoint. vit_folder exists only
        # because get_network unconditionally makedirs it.
        cfg.training.writer_log = ""
        cfg.training.vit_folder = tempfile.mkdtemp(prefix="deltatok_z_spread_") + "/"

    # resume=False + ckpt=None: the trainer builds the frozen OccRAE and ONE tokenizer;
    # every arm's weights are loaded explicitly below.
    trainer_args = argparse.Namespace(
        resume=False, ckpt=None, test_only=True, eval_only=True,
        debug=False, is_multi_gpus=False,
    )
    trainer = DeltaTokTrainer(
        args=trainer_args, cfg=cfg, device=device,
        rank=0, world_size=1, distributed=False,
    )

    # One DeltaTok module per arm, same architecture, encoded on the same features.
    backbone = trainer.occ_rae.model._get_pretrained_backbone()
    nets = []
    for i, (label, path) in enumerate(arms):
        net = trainer._unwrapped_tokenizer() if i == 0 else \
            trainer._make_deltatok_module(backbone).to(device)
        _load_tokenizer_weights(net, path)
        nets.append((label, net))

    whiten = _load_whiten(args.whiten_stats, device) if args.whiten_stats else None
    split_label, loader = _build_loader(cfg, args.split)
    print(f"\n[INFO] {split_label}: {len(loader)} batches available, using {args.num_batches}")
    _pin_loader_epoch(loader)
    accs = [ZSpreadStats(device) for _ in nets]

    done = 0
    for batch in loader:
        if done >= args.num_batches:
            break
        batch = trainer._normalize_batch(batch)
        imgs = batch["imgs"].to(device, non_blocking=True)              # (B, V, 3, H, W)
        num_cameras = int(batch.get("num_cameras", 1))

        with torch.no_grad():
            # Frozen OccRAE encode, ONCE — every arm then sees identical (x_prev, x).
            _, _, x_prev, x, H, W = trainer._extract_pair_feats(imgs, num_cameras=num_cameras)  # (M, N, P, C)
            for (label, net), acc in zip(nets, accs):
                with trainer.autocast:
                    # encode only: the decoder plays no part in z's distribution.
                    rope_local = net._compute_rope(H, W, x_prev.device, x_prev.dtype)
                    rope_global = net._compute_global_rope(H, W, num_cameras, x_prev.device, x_prev.dtype)
                    z = net.encode(x_prev, x, rope_local, rope_global)  # (M, N, K, Cz)
                z = z.float()
                if whiten is not None:
                    # Same op as DeltaTokFlowTrainer._z_to_flow_latent: channels last, so
                    # (Cz,) stats broadcast per channel and (K, Cz) per slot.
                    assert whiten[0].dim() == 1 or z.shape[2:] == whiten[0].shape, \
                        f"z slots {tuple(z.shape[2:])} != whiten stats {tuple(whiten[0].shape)}"
                    z = (z - whiten[0]) / whiten[1]            # (M, N, K, Cz) z-scored
                acc.update(z)

        done += 1
        print(f"[INFO]   batch {done}/{args.num_batches}  rows/arm={accs[0].n} "
              f"N={num_cameras} res={H}x{W}", flush=True)

    if done == 0 or accs[0].n == 0:
        raise RuntimeError("No usable batches: nothing to measure.")

    reports = []
    for (label, _), acc in zip(nets, accs):
        # distributed=False: this script is single-GPU, so summary() runs no collectives.
        reports.append(summarize(label, acc.summary(distributed=False, full=True)))

    print(f"\n===== comparison ({split_label}, {done} batches, {accs[0].n} rows/arm) =====")
    print(f"{'arm':<14} {'totvar':>9} {'stdratio':>9} {'PR':>7} {'entrank':>8} "
          f"{'n90':>5} {'n99':>5} {'top1':>6} {'top10':>6} {'rowtail':>8}")
    for r in reports:
        print(f"{r['label']:<14} {r['total_var']:>9.3f} {r['std_ratio']:>9.2f} {r['pr']:>7.1f} "
              f"{r['entropy_rank']:>8.1f} {r['n90']:>5d} {r['n99']:>5d} {r['top1']:>6.3f} "
              f"{r['top10']:>6.3f} {r['row_ms_tail']:>7.2f}x")
    print("[NOTE] higher PR / entrank / n90 / n99 and lower top1, top10, std ratio = "
          "variance spread more evenly across the 768-d budget.")
    print("[NOTE] rowtail is a different axis: spread ACROSS rows, not across channels. "
          "1.00x = every row the same scale (what z_norm=true forces); large = an MSE over "
          "these rows is dominated by a few outlier tokens.")


if __name__ == "__main__":
    main()
