#!/usr/bin/env python3
"""DeltaTok flow-matching error resolved per noise level t (50 uniform bins).

``Eval/LossFlow`` is a single scalar averaged over t ~ sigmoid(N(mu, sigma)), so it
cannot say *where* in the noise schedule the model is weak: a model that is excellent
near t->1 and useless near t->0 logs the same number as one that is mediocre
everywhere. This sweeps t over ``--num_bins`` uniform bins of [0, 1] and reports the
teacher-forced prediction error in each bin, on the config's train and/or test datasets.

Per bin (centre t_c) the noising and the loss come from the trainer itself
(``flow_noising`` / ``flow_loss``), so the numbers are the training objective resolved
per t rather than a re-derivation of it. Three errors come out of one forward:
  * ``mse_x`` — raw denoising error ||x_pred - x||^2 in delta-token space. Directly
                comparable across t; the one to read first.
  * ``mse_v`` — the actual training objective (loss_mode=v) = mse_x / clamp(1-t, 0.05)^2,
                carrying an up-to-400x upweight as t->1.
  * ``mse_e`` — noise-prediction error.
Alongside them: ``x_var`` (error of the trivial "predict 0" predictor — the line the
model must beat, and also sigma_x^2), ``e_var``, ``snr`` (= t^2*sigma_x^2/((1-t)^2*E[e^2]),
which equals t^2/(1-t)^2 only if whitening delivered sigma_x^2 = 1) and ``train_mass``
(analytic training density in that bin, following ``model.t_dist`` — which bins were
actually trained). Per split it also prints the per-channel and per-delta-slot SNR
spreads: whitening exists to make these ~1, and a large spread means one scalar t cannot
describe the noise level of every channel/slot.

The frozen OccRAE (~1B) encode dominates the cost, so each batch is encoded ONCE and
every bin is swept on it; the flow ViT sees only (T-1)*N*K delta tokens per item.

Usage (on Jean Zay, from a GPU node; defaults target the
deltatok_flow_waymo_dtok64_deltaCtxCross_global_fullT_wd05_raymapoff_50k run's last ckpt):
  source env_jz_h100.sh &&
  python eval_deltatok_flow_t_bins.py --output_dir results/deltatok_flow_t_bins
  # Defaults: --config-name train_deltatok_flow_waymo_jeanzay, --ckpt the _50k run's
  # current.pth, and the run's arch flags from slurm/deltatok_flow/train_deltatok_flow_waymo_jz.slurm
  # (cond_mode=delta_ctx_cross, attn_mode=global, vit_use_camera_embed=true, dtok64
  # non-affine tokenizer) so the built flow ViT matches the saved weights. Override any
  # of them with --cfg model.<key>=... for a different flow run.
  # Smoke first: --num_bins 4 --num_batches 1 --noise_repeats 1 --splits val
"""

import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np
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

# Default target run. Its arch flags live in slurm/deltatok_flow/train_deltatok_flow_waymo_jz.slurm
# (EXTRA_CFG), NOT in the config — replicated in `_RUN_DEFAULTS` below.
_RUN_ROOT = ("/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log/"
             "deltatok_flow_waymo_dtok64_deltaCtxCross_global_fullT_wd05_raymapoff_50k")
_DTOK64_CKPT = ("/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/"
                "deltatok_surround_layer12_bsize2_dtok64_cos1e3_1e5_b95_gradskip_noaff/ckpts/current.pth")

_RUN_DEFAULTS = {
    "model.cond_mode": "delta_ctx_cross",      # clean in-seq context + frame-0 cross-attn
    "model.attn_mode": "global",               # single global attention over all delta tokens
    "model.vit_use_camera_embed": True,        # per-camera identity in the flow ViT
    "model.deltatok.num_delta_tokens": 64,     # K=64, must match the frozen dtok64 ckpt
    "model.deltatok.norm_affine": False,       # dtok64 tokenizer is non-affine z-norm
    "model.deltatok_ckpt": _DTOK64_CKPT,
    "model.whiten_stats": None,                # baseline is unwhitened; pass --cfg model.whiten_stats=<path> for a whitened ckpt
}

_MODES = ("x", "v", "e")  # error flavours, all from one forward via flow_loss's loss_mode


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeltaTok flow-matching error per noise-level (t) bin."
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
        "--num_bins", type=int, default=50,
        help="Uniform bins over t in [0, 1]; t is swept at the bin centres.",
    )
    parser.add_argument(
        "--noise_repeats", type=int, default=4,
        help="Independent noise draws e per (batch, bin); each is one cheap ViT forward.",
    )
    parser.add_argument(
        "--num_batches", type=int, default=8,
        help="Batches per split (8 x bsize 16 = the val set's 128 seqs).",
    )
    parser.add_argument(
        "--splits", type=str, default="train,val",
        help="Comma-separated: train, val, or both.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/deltatok_flow_t_bins")
    return parser


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")


def _train_t_mass(lo: float, hi: float, mu: float, sigma: float,
                  t_dist: str = "logitnormal") -> float:
    """Training-time probability mass of t in [lo, hi], analytically.

    logitnormal: t = sigmoid(s), s ~ N(mu, sigma) (see `flow_noising`), so
    P(t <= q) = P(s <= logit(q)) = Phi((logit(q) - mu) / sigma).
    uniform: the mass is just the bin width. No sampling needed either way.
    """
    if t_dist == "uniform":
        return max(0.0, min(1.0, hi) - max(0.0, lo))
    if t_dist != "logitnormal":
        raise ValueError(f"model.t_dist must be uniform|logitnormal, got {t_dist!r}")

    def _cdf(q):
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return 1.0
        s = math.log(q / (1.0 - q))                                   # logit(q)
        return 0.5 * (1.0 + math.erf((s - mu) / (sigma * math.sqrt(2.0))))
    return _cdf(hi) - _cdf(lo)


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
def sweep_split(trainer, cfg, loader, bin_centers, args):
    """Accumulate per-bin squared error over `--num_batches` batches of one loader.

    Returns (sums, counts, diag): sums[mode] / sums["x_var"] / sums["e_var"] are
    element-weighted sums per bin; counts holds the element count per bin; diag holds
    the t-independent SNR-uniformity ratios (one value per batch).
    """
    n_bins = len(bin_centers)
    sums = {m: np.zeros(n_bins, dtype=np.float64) for m in _MODES}
    sums["x_var"] = np.zeros(n_bins, dtype=np.float64)
    sums["e_var"] = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    diag = {"spread_c": [], "spread_k": []}

    batches_done = 0
    for batch in loader:
        if batches_done >= args.num_batches:
            break
        batch = trainer._normalize_batch(batch)

        imgs = batch["imgs"].to(trainer.device, non_blocking=True)  # (B, V, 3, H, W) with V = T*N views
        num_cameras = int(batch.get("num_cameras", 1))

        # Frozen OccRAE + DeltaTok encode -> the flow target. ONCE per batch: this is the
        # ~1B forward that dominates, and every bin below reuses it. Tokens are only
        # needed for a decode rollout, which this diagnostic does not do.
        _, feat0, z, H, W = trainer._encode_inputs(batch, imgs, num_cameras, want_tokens=False)  # feat0 (B, N, P, C); z (B, T-1, N, K, C)
        if z.shape[1] <= trainer.n_ctx:
            print(f"[WARN] Skipping batch: T-1={z.shape[1]} <= n_ctx={trainer.n_ctx}, nothing to predict")
            continue
        x_spatial = trainer._z_to_flow_latent(z)  # (B, C, T-1, N, K) flow latent layout; whitened iff model.whiten_stats
        cross_cond = trainer._build_cross_cond(feat0, H, W) if trainer.build_frame0_ctx else None  # (B, N, Hp, Wp, C) or None

        # flow_loss masks the n_ctx clean context slots, so each reported mean is over
        # predicted slots only. Element-weighted because the train split's T-1 varies
        # per batch (max_memory_num_views=24).
        b_, c_, t_, n_, k_ = x_spatial.shape                        # B, C, T-1 transitions, N cameras, K delta tokens
        n_elem = float(b_ * c_ * (t_ - trainer.n_ctx) * n_ * k_)    # == flow_loss's mask.sum()
        # Error of the trivial "predict 0" predictor on the same slots: delta tokens are
        # ~unit-scale after DeltaTok's LayerNorm, so this is the bar mse_x must clear.
        # Doubles as sigma_x^2 in SNR(t) = t^2*sigma_x^2 / ((1-t)^2*E[e^2]).
        xs = x_spatial[:, :, trainer.n_ctx:].float()                # (B, C, T-1-n_ctx, N, K)
        x_var = xs.pow(2).mean().item()
        # Whitening is meant to equalise signal power so one scalar t describes every
        # channel/slot. snr_c is proportional to var_c, so these ratios are t-free.
        var_c = xs.pow(2).mean(dim=(0, 2, 3, 4))                    # (C,) per-channel signal power
        var_k = xs.pow(2).mean(dim=(0, 1, 2, 3))                    # (K,) per delta-slot signal power
        diag["spread_c"].append(float(var_c.max() / var_c.clamp_min(1e-12).min()))
        diag["spread_k"].append(float(var_k.max() / var_k.clamp_min(1e-12).min()))

        for bi, t_c in enumerate(bin_centers):
            for _ in range(args.noise_repeats):
                # Driving trainer.flow_noising / trainer.flow_loss through the cfg knobs
                # (rather than reimplementing them) keeps this EXACTLY the training
                # objective: context pinning at t=1, _sample_noise's fixed_noise_mode and
                # the clamp(1-t, min=0.05) x/v/e conversions all come along.
                cfg.model.train_fixed_t = float(t_c)  # pins every non-context slot to this bin's t
                z_t, e, t = trainer.flow_noising(
                    x_spatial, context=trainer.n_ctx, mu=cfg.model.mu, sigma=cfg.model.sigma
                )  # z_t (B, C, T-1, N, K); e (B, C, T-1, N, K); t (B, T-1)

                with trainer.autocast:
                    pred = trainer._ema_model()(
                        x=z_t, ada_cond=t, cross_cond=cross_cond, return_feat=False,
                    )  # (B, C, T-1, N, K) x-prediction (pred_mode=x)
                    for mode in _MODES:
                        cfg.model.loss_mode = mode   # same pred, three error flavours
                        loss = trainer.flow_loss(
                            pred=pred, x=x_spatial, z_t=z_t, e=e, t=t, context=trainer.n_ctx,
                        )
                        sums[mode][bi] += loss.item() * n_elem
                sums["x_var"][bi] += x_var * n_elem
                # Measured, not assumed: fixed_noise_mode probes make E[e^2] != 1.
                sums["e_var"][bi] += e[:, :, trainer.n_ctx:].float().pow(2).mean().item() * n_elem
                counts[bi] += n_elem

        batches_done += 1
        print(f"[INFO]   batch {batches_done}/{args.num_batches}  "
              f"B={b_} T-1={t_} N={n_} K={k_} res={H}x{W}", flush=True)

    if batches_done == 0:
        raise RuntimeError("No usable batches: nothing to measure.")
    return sums, counts, diag


def write_csv(path, rows) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["bin_lo", "bin_hi", "t", "n_elem", "mse_x", "mse_v", "mse_e",
                        "x_var", "e_var", "snr", "train_mass"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_split(path, label, rows) -> None:
    """x-MSE (raw error, vs the predict-0 baseline) and v-MSE (the objective) vs t."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t"] for r in rows]
    mass = np.array([r["train_mass"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for ax, keys, title in (
        (axes[0], [("mse_x", "x-MSE (pred vs GT delta)"), ("x_var", "predict-0 baseline")],
         f"{label}: raw denoising error vs noise level"),
        (axes[1], [("mse_v", "v-MSE (training objective)")],
         f"{label}: training objective vs noise level"),
    ):
        # Training density behind the curves: which bins the model actually saw.
        ax_t = ax.twinx()
        ax_t.fill_between(t, 0, mass / max(mass.max(), 1e-12), color="tab:gray", alpha=0.15, lw=0)
        ax_t.set_ylabel("train density (normalized)", color="tab:gray")
        ax_t.set_ylim(0, 1.05)
        ax_t.tick_params(axis="y", colors="tab:gray")
        for key, name in keys:
            ax.plot(t, [r[key] for r in rows], marker="o", ms=3, label=name,
                    ls="--" if key == "x_var" else "-")
        ax.set_yscale("log")
        ax.set_ylabel("MSE (log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        ax.set_zorder(ax_t.get_zorder() + 1)  # curves above the density fill
        ax.patch.set_visible(False)
    axes[1].set_xlabel("t  (0 = pure noise, 1 = clean)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_overlay(path, per_split) -> None:
    """train vs val x-MSE and v-MSE on shared axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in (
        (axes[0], "mse_x", "raw denoising error (x-MSE)"),
        (axes[1], "mse_v", "training objective (v-MSE)"),
    ):
        for label, rows in per_split.items():
            ax.plot([r["t"] for r in rows], [r[key] for r in rows], marker="o", ms=3, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("t  (0 = pure noise, 1 = clean)")
        ax.set_ylabel("MSE (log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


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
        # No RGB is decoded here, so skip loading the MAE image decoder entirely
        # (_build_occ_rae treats a falsy ckpt_path as "no decoder").
        if cfg.model.get("img_decoder", None) is not None:
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

    edges = np.linspace(0.0, 1.0, int(args.num_bins) + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0          # 0.01, 0.03, ... 0.99 at 50 bins
    mu, sigma = float(cfg.model.mu), float(cfg.model.sigma)
    # train_mass must follow the run's actual prior, else the [CHECK] reconciliation below
    # is integrated against the wrong density.
    t_dist = str(cfg.model.get("t_dist", "logitnormal"))
    # Up front: the sweep drives the train_fixed_t path, so an invalid t_dist would only
    # surface in _train_t_mass after the whole GPU sweep has already run.
    if t_dist not in ("uniform", "logitnormal"):
        raise ValueError(f"model.t_dist must be uniform|logitnormal, got {t_dist!r}")
    prior = "U(0,1)" if t_dist == "uniform" else f"sigmoid(N({mu}, {sigma}))"
    print(f"[INFO] {args.num_bins} bins over t in [0,1]; {args.noise_repeats} noise draw(s) "
          f"per (batch, bin); train t ~ {prior}")

    per_split_rows = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in ("train", "val"):
            raise ValueError(f"--splits must contain only train/val, got {split!r}")
        for label, loader in _build_split_loaders(cfg, split):
            print(f"\n[INFO] Sweeping {label} ({len(loader)} batches available)")
            _pin_loader_epoch(loader)
            # ema_scope so a use_ema run measures the EMA weights eval_one_epoch would
            # have used (no-op when use_ema is false, as on the default run).
            with trainer.ema_scope():
                sums, counts, diag = sweep_split(trainer, cfg, loader, bin_centers, args)

            rows = []
            for bi, t_c in enumerate(bin_centers):
                n = max(counts[bi], 1.0)
                rows.append({
                    "bin_lo": round(float(edges[bi]), 6),
                    "bin_hi": round(float(edges[bi + 1]), 6),
                    "t": round(float(t_c), 6),
                    "n_elem": int(counts[bi]),
                    "mse_x": float(sums["x"][bi] / n),
                    "mse_v": float(sums["v"][bi] / n),
                    "mse_e": float(sums["e"][bi] / n),
                    "x_var": float(sums["x_var"][bi] / n),
                    "e_var": float(sums["e_var"][bi] / n),
                    # SNR of z_t = t*x + (1-t)*e. Equals t^2/(1-t)^2 iff whitening
                    # delivered sigma_x^2 = 1, so the gap to that is the whitening check.
                    "snr": float((t_c ** 2 * sums["x_var"][bi])
                                 / max((1.0 - t_c) ** 2 * sums["e_var"][bi], 1e-12)),
                    "train_mass": _train_t_mass(float(edges[bi]), float(edges[bi + 1]),
                                                mu, sigma, t_dist),
                })
            per_split_rows[label] = rows

            csv_path = os.path.join(output_dir, f"t_bins_{label}.csv")
            write_csv(csv_path, rows)
            plot_split(os.path.join(output_dir, f"t_bins_{label}.png"), label, rows)

            print(f"\n[{label}]  {'t':>6} {'mse_x':>10} {'mse_v':>12} {'mse_e':>10} "
                  f"{'x_var':>8} {'snr':>10} {'train_mass':>10}")
            for r in rows:
                print(f"[{label}]  {r['t']:6.3f} {r['mse_x']:10.4f} {r['mse_v']:12.4f} "
                      f"{r['mse_e']:10.4f} {r['x_var']:8.4f} {r['snr']:10.3f} "
                      f"{r['train_mass']:10.5f}")

            # Whitening check. sigma_x^2 should be 1 (else every t sits at a different
            # noise level than the schedule intends); the spreads should be ~1 (else one
            # scalar t cannot describe every channel / delta slot, and the premise of the
            # t-schedule work is broken).
            sigma_x2 = float(np.mean([r["x_var"] for r in rows]))
            print(f"[{label}] sigma_x^2 = {sigma_x2:.4f} (1.0 iff whitening holds)   "
                  f"per-channel SNR spread = {float(np.mean(diag['spread_c'])):.1f}x   "
                  f"per-slot SNR spread = {float(np.mean(diag['spread_k'])):.1f}x")

            # Reconciliation: Eval/LossFlow is exactly this v-MSE averaged over the training
            # t density, so re-integrating the curve against train_mass must reproduce the
            # run's logged scalar (within its MC noise). A large gap means the load or the
            # noising is wrong, not that the model is interesting.
            mass = np.array([r["train_mass"] for r in rows])
            v = np.array([r["mse_v"] for r in rows])
            print(f"[CHECK] {label}: density-weighted v-MSE = {float((mass * v).sum() / mass.sum()):.4f}"
                  f"   (compare with this run's logged Eval/LossFlow)")
            print(f"[INFO] Wrote {csv_path}")

    if len(per_split_rows) > 1:
        overlay = os.path.join(output_dir, "t_bins.png")
        plot_overlay(overlay, per_split_rows)
        print(f"[INFO] Wrote {overlay}")

    print(f"\n[INFO] Done. Outputs under {output_dir}")


if __name__ == "__main__":
    main()
