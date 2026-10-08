#!/usr/bin/env python3
"""Where does the DeltaTok flow lose the camera motion?

Each window's 8 forecast deltas are decoded in six versions, and a camera is fitted to each:
  gt      the true deltas (control: the tokenizer's own motion)
  zero    z = 0
  repeat  the context delta z_1 in every forecast slot (constant velocity in token space)
  donor   another window's true deltas (does the camera follow the donor's speed?)
  flow1   flow sample, 1 ODE step (= the flow's x-prediction from pure noise)
  flowN   flow sample, --num_steps ODE steps
  half         0.5 x the true deltas (a shrunk token, like flow1's)
  noise@flow1  true deltas + isotropic noise at flow1's token MSE (same error size, random direction)
  noise@flowN  same at flowN's token MSE
The context slot holds the true delta in every version. Per version: camera path vs GT
(step length, heading, unaligned ATE), token MSE to the true forecast deltas, and the
layer-12 feature error of the decode vs OccAny's own (rollout and teacher-forced).

Usage (on BSC, 1 GPU; the flow run's arch flags come in through --cfg):
  sbatch slurm/probe_flow_motion_bsc.slurm
  # Smoke first: EXTRA_ARGS="--num_batches 1 --splits val"
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
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

from occrae.deltatok_flow_trainer import DeltaTokFlowMatchingTrainer  # noqa: E402
from occrae.deltatok_trainer import _log_cosh  # noqa: E402
from occrae.generation_helper import flow_euler_sample  # noqa: E402
from occany.datasets import get_data_loader  # noqa: E402

_VARIANTS = ("gt", "zero", "repeat", "donor", "flow1", "flowN", "half", "noise@flow1", "noise@flowN", "flow1_zgain2")
_POSE_VARIANTS = ("flow1_nopose", "flowN_nopose")  # pose-cond ckpt only: pose zeroed, the train-time drop value
_FEAT_VARIANTS = ("feat_true", "feat_partway25", "feat_partway50", "feat_partway75", "feat_noise@partway50",
                  "feat_noise@flow1", "flow1_progress", "flow1_rest", "flowN_progress", "flowN_rest",
                  "flow1_gain2", "flow1_gain4", "flowN_gain2", "flowN_gain3")  # future feats fed to DA3, no DeltaTok
_FEAT_GAINS = (("flow1", 2), ("flow1", 4), ("flowN", 2), ("flowN", 3))  # flow's predicted change since the last observed frame x g


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeltaTok flow: camera motion of decoded forecast deltas, six versions."
    )
    parser.add_argument("--config-dir", type=str, default="configs/deltatok_flow")
    parser.add_argument("--config-name", type=str, default="train_deltatok_flow_alldata_ctx2fwd8_bsc")
    parser.add_argument(
        "--cfg", type=str, nargs="*", default=[],
        help="Hydra-style overrides; must carry the flow run's arch flags.",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Trained flow-transformer checkpoint (current.pth / iter_*.pth from "
             "DeltaTokFlowMatchingTrainer). Loaded with strict=True.",
    )
    parser.add_argument("--occany_recon_ckpt", type=str, default=None)
    parser.add_argument("--encode_layer", type=int, default=12)
    parser.add_argument("--num_steps", type=int, default=20, help="ODE steps of the flowN sample.")
    parser.add_argument(
        "--num_batches", type=int, default=16,
        help="Batches per split and per val set.",
    )
    parser.add_argument(
        "--splits", type=str, default="train,val",
        help="Comma-separated: train, val, or both.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/probe_flow_motion")
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


def _flow_sample(trainer, x_spatial, cross_cond, num_steps, pose_cond):
    """Same draw and call as eval_one_epoch (deltatok_flow_trainer.py:925-943)."""
    cfg = trainer.cfg
    zr = trainer._sample_noise(x_spatial)                             # (B, C, T-1, N, K) seeded prior
    if trainer.n_ctx > 0:
        zr[:, :, :trainer.n_ctx] = x_spatial[:, :, :trainer.n_ctx]    # GT context delta stays clean
    gen = flow_euler_sample(
        trainer._ema_model(), zr,
        pred_mode=cfg.model.pred_mode,
        context=trainer.n_ctx,
        num_steps=num_steps,
        step_mode=str(cfg.model.get("sampler_step_mode", "ode")),
        scheduler_mode=str(cfg.model.get("sampler_scheduler_mode", "cosine")),
        alpha=float(cfg.model.get("sampler_alpha", 0.5)),
        cross_cond=cross_cond,
        pose_cond=pose_cond,
        autocast_ctx=trainer.autocast,
    )
    return trainer._flow_latent_to_z(gen)                             # (B, T-1, N, K, C) tokenizer scale


def _camera_centres(trainer, tokens, feat0, z_seq, H, W, height, width, num_cameras):
    """Roll out z_seq from GT frame 0, decode all views at once, fit cameras.
    -> centres (B, T, 3) of cam 0, rolled-out layer-12 patch feats (B, T-1, N, P, C)."""
    B = tokens.shape[0]
    with trainer.autocast:
        x_hat = trainer._rollout_from_z(trainer.deltatok, feat0, z_seq, H, W, num_cameras)  # (B*(T-1), N, P, C)
    centres = _fit_centres(trainer, tokens, x_hat, height, width, num_cameras)             # (B, T, 3)
    return centres, x_hat.view(B, -1, *x_hat.shape[1:])                                   # (B, T, 3), (B, T-1, N, P, C)


def _fit_centres(trainer, tokens, x_hat, height, width, num_cameras):
    """Layer-12 patch feats of frames 1..T-1 (B*(T-1), N, P, C) after observed frame 0 -> DA3 decode -> cam-0 centres (B, T, 3)."""
    B, V = tokens.shape[:2]
    full = trainer._reconstruct_full_tokens(tokens, x_hat, B, V, num_cameras=num_cameras)    # (B, V, N_tok, C)
    with trainer.autocast:
        dec = trainer._decode_tokens(full, height, width, num_cameras=num_cameras)
    with torch.autocast("cuda", enabled=False):
        c2w, _ = trainer.occ_rae.model._process_ray_pose_estimation(
            dec["ray"].float(), dec["ray_conf"].float(), height, width)                   # (B, V, 3, 4) frame-0 coords
    return c2w[..., :3, 3].reshape(B, V // num_cameras, num_cameras, 3)[:, :, 0]           # (B, T, 3) camera centres


def _append_cam(r, P, G, dG, n_ctx, fc, B):
    """One batch's camera stats into a version's record: steps (B, T-1), ATE (B,), heading cos (B, F)."""
    dP = P[:, 1:] - P[:, :-1]                                                 # (B, T-1, 3)
    r["step"].append(dP.norm(dim=-1).cpu())
    r["ate"].append((P[:, n_ctx + 1:] - G[:, n_ctx + 1:]).norm(dim=-1).pow(2).mean(1).sqrt().cpu())  # (B,)
    r["head"].append(_cos(dP[:, fc].flatten(0, 1), dG[:, fc].flatten(0, 1)).view(B, -1).cpu())     # (B, F)


def _teacher_forced(trainer, feats, z_seq, H, W, num_cameras):
    """Decode each forecast delta from the GT previous frame, as feat_loss does. -> (B, F, N, P, C)."""
    n_ctx = trainer.n_ctx
    B, T = feats.shape[:2]
    x_prev = feats[:, n_ctx:T - 1].flatten(0, 1)                                   # (B*F, N, P, C) GT frame t
    z_f = z_seq[:, n_ctx:].flatten(0, 1).to(x_prev.dtype)                           # (B*F, N, K, Cz)
    with trainer.autocast:
        x_tf = trainer.deltatok(x_prev, None, H, W, num_cameras=num_cameras, z_input=z_f)  # (B*F, N, P, C)
    return x_tf.view(B, T - 1 - n_ctx, *x_tf.shape[1:])                             # (B, F, N, P, C)


def _feat_err(pred, tgt):
    """Log-cosh vs OccAny's layer-12 feats, LossFeat units: (B, F, N, P, C) x2 -> (B, F)."""
    with torch.autocast("cuda", enabled=False):
        return _log_cosh(pred.float(), tgt.float()).mean(dim=(2, 3, 4))


def _cos(a, b):
    """Row cosine of (M, D) tensors -> (M,)."""
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def _split_progress(x, f_last_obs, f_future):
    """Split the predicted change since the last observed frame along the true change.
    x, f_future: (B, F, N, P, C) predicted / true future feats; f_last_obs: (B, N, P, C).
    -> progress (B, F) (0 = copy, 1 = truth), progress_only, rest_only (B, F, N, P, C)."""
    true_change = (f_future - f_last_obs.unsqueeze(1)).float()                    # (B, F, N, P, C)
    pred_change = (x - f_last_obs.unsqueeze(1)).float()                           # (B, F, N, P, C)
    progress = (pred_change * true_change).sum(dim=(2, 3, 4)) / true_change.pow(2).sum(dim=(2, 3, 4)).clamp_min(1e-12)  # (B, F)
    along = progress[..., None, None, None] * true_change                         # (B, F, N, P, C) progress part of the change
    progress_only = f_last_obs.unsqueeze(1).float() + along                       # (B, F, N, P, C) last observed + progress
    rest_only = f_future.float() + (pred_change - along)                          # (B, F, N, P, C) truth + the rest
    return progress, progress_only, rest_only


@torch.no_grad()
def motion_split(trainer, loader, args):
    """Per window and version: camera steps (B, T-1), ATE, heading cos; plus token stats."""
    n_ctx = trainer.n_ctx
    z_names: list[str] = list(_VARIANTS)
    if trainer.pose_cond:
        z_names.extend(_POSE_VARIANTS)
    keys = ("step", "ate", "head", "mse", "feat_ar", "feat_tf", "prog", "znorm")
    rec = {v: {k: [] for k in keys} for v in z_names + list(_FEAT_VARIANTS)}
    rec_gt = {"step": [], "step_donor": [], "cos_next": [], "cos_other": [], "feat_copy": []}
    noise_gen = torch.Generator(device=trainer.device).manual_seed(args.seed + 1)  # off the flow prior's stream
    feat_noise_gen = torch.Generator(device=trainer.device).manual_seed(args.seed + 2)  # own stream: noise_gen draws stay as in v2

    batches_done = 0
    for batch in loader:
        if batches_done >= args.num_batches:
            break
        batch = trainer._normalize_batch(batch)
        imgs = batch["imgs"].to(trainer.device, non_blocking=True)  # (B, V, 3, H, W) with V = T*N views
        num_cameras = int(batch.get("num_cameras", 1))
        height, width = batch["output_resolution_hw"]
        B = imgs.shape[0]
        if B < 2:
            print(f"[WARN] Skipping batch of {B}: donor needs 2 windows")
            continue

        tokens, feat0, z, H, W = trainer._encode_inputs(batch, imgs, num_cameras, want_tokens=True)  # tokens (B, V, N_tok, C); feat0 (B, N, P, C); z (B, T-1, N, K, C)
        x_spatial = trainer._z_to_flow_latent(z)                                      # (B, C, T-1, N, K)
        cross_cond = trainer._build_cross_cond(feat0, H, W) if trainer.build_frame0_ctx else None  # (B, N, Hp, Wp, C) or None

        fc = slice(n_ctx, None)                                                       # forecast delta slots
        versions = {v: z.clone() for v in ("gt", "zero", "repeat", "donor")}         # each (B, T-1, N, K, C)
        versions["zero"][:, fc] = 0
        versions["repeat"][:, fc] = z[:, n_ctx - 1:n_ctx]                             # context delta z_1 in every slot
        versions["donor"][:, fc] = z.roll(1, dims=0)[:, fc]                           # window b takes window b-1's deltas
        pose_cond = trainer._build_pose_cond(batch, num_cameras)                      # (B, T-1, 7) or None
        for name, steps in (("flow1", 1), ("flowN", args.num_steps)):
            state = trainer._eval_noise_gen.get_state()                               # rewind: same prior with and without pose
            versions[name] = _flow_sample(trainer, x_spatial, cross_cond, steps, pose_cond)
            if pose_cond is not None:
                trainer._eval_noise_gen.set_state(state)
                versions[f"{name}_nopose"] = _flow_sample(trainer, x_spatial, cross_cond, steps, torch.zeros_like(pose_cond))
        versions["half"] = z.clone()
        versions["half"][:, fc] = 0.5 * z[:, fc]                                      # shrunk toward z=0, like a mean
        versions["flow1_zgain2"] = versions["flow1"].clone()
        versions["flow1_zgain2"][:, fc] = 2 * versions["flow1"][:, fc]               # 1-step sample at ~true token size
        for src in ("flow1", "flowN"):
            mse = (versions[src][:, fc].float() - z[:, fc].float()).pow(2).mean()     # () this batch's token MSE
            eps = torch.randn(z[:, fc].shape, generator=noise_gen, device=z.device, dtype=torch.float32)  # (B, F, N, K, C)
            versions[f"noise@{src}"] = z.clone()
            versions[f"noise@{src}"][:, fc] = (z[:, fc].float() + mse.sqrt() * eps).to(z.dtype)  # same MSE, random direction

        prefix = trainer._num_prefix_tokens
        feats = tokens[:, :, prefix:].reshape(B, -1, num_cameras, tokens.shape[2] - prefix, tokens.shape[3])  # (B, T, N, P, C) OccAny layer-12
        f_future = feats[:, n_ctx + 1:]                                               # (B, F, N, P, C) true future feats
        f_last_obs = feats[:, n_ctx]                                                  # (B, N, P, C) last observed frame
        rec_gt["feat_copy"].append(_feat_err(feats[:, n_ctx:-1], f_future).cpu())     # (B, F) "frame t as frame t+1"

        G = batch["gt_c2w"].to(trainer.device).float()[..., :3, 3]                    # (B, V, 3) GT centres
        G = G.reshape(B, -1, num_cameras, 3)[:, :, 0]                                 # (B, T, 3) cam 0
        dG = G[:, 1:] - G[:, :-1]                                                     # (B, T-1, 3) GT steps
        rec_gt["step"].append(dG.norm(dim=-1).cpu())
        rec_gt["step_donor"].append(dG.roll(1, dims=0).norm(dim=-1).cpu())

        zf = z[:, fc].float().flatten(2)                                              # (B, F, N*K*C) GT forecast deltas
        rec_gt["cos_next"].append(_cos(zf[:, :-1].flatten(0, 1), zf[:, 1:].flatten(0, 1)).cpu())          # t vs t+1, same window
        rec_gt["cos_other"].append(_cos(zf.flatten(0, 1), zf.roll(1, dims=0).flatten(0, 1)).cpu())       # same t, other window

        flow_feats = {}                                                               # flow rollout future feats, split below
        for v in z_names:
            z_seq = versions[v]                                                       # (B, T-1, N, K, C)
            P, x_ar = _camera_centres(trainer, tokens, feat0, z_seq, H, W, height, width, num_cameras)  # (B, T, 3), (B, T-1, N, P, C)
            _append_cam(rec[v], P, G, dG, n_ctx, fc, B)
            rec[v]["mse"].append((z_seq[:, fc].float() - z[:, fc].float()).pow(2).mean(dim=(2, 3, 4)).cpu())    # (B, F) per frame
            rec[v]["feat_ar"].append(_feat_err(x_ar[:, n_ctx:], f_future).cpu())                               # (B, F) rollout
            rec[v]["feat_tf"].append(_feat_err(_teacher_forced(trainer, feats, z_seq, H, W, num_cameras), f_future).cpu())  # (B, F)
            progress, _, _ = _split_progress(x_ar[:, n_ctx:], f_last_obs, f_future)  # (B, F)
            rec[v]["prog"].append(progress.cpu())
            row_ratio = z_seq[:, fc].float().norm(dim=-1) / z[:, fc].float().norm(dim=-1).clamp_min(1e-12)  # (B, F, N, K) token row norms vs true deltas
            rec[v]["znorm"].append(row_ratio.mean(dim=(2, 3)).cpu())                  # (B, F)
            if v in ("flow1", "flowN"):
                flow_feats[v] = x_ar[:, n_ctx:]                                       # (B, F, N, P, C)

        # Feature versions: future frames given as layer-12 feats, DA3 decodes them with no DeltaTok.
        f_future32 = f_future.float()                                                 # (B, F, N, P, C)
        f_last32 = f_last_obs.unsqueeze(1).float()                                    # (B, 1, N, P, C)
        feat_versions = {"feat_true": f_future32}                                     # each (B, F, N, P, C)
        for pct in (25, 50, 75):
            feat_versions[f"feat_partway{pct}"] = f_last32 + pct / 100 * (f_future32 - f_last32)  # pct% of the way to the truth
        for name, src in (("feat_noise@partway50", feat_versions["feat_partway50"]), ("feat_noise@flow1", flow_feats["flow1"].float())):
            mse = (src - f_future32).pow(2).mean()                                    # () feat MSE to match
            eps = torch.randn(f_future32.shape, generator=feat_noise_gen, device=f_future32.device, dtype=torch.float32)  # (B, F, N, P, C)
            feat_versions[name] = f_future32 + mse.sqrt() * eps                       # same feat MSE, random direction
        for src in ("flow1", "flowN"):
            _, progress_only, rest_only = _split_progress(flow_feats[src], f_last_obs, f_future)
            gap = (progress_only + rest_only - f_future32 - flow_feats[src].float()).abs().max()
            assert gap < 1e-2, f"{src}: progress_only + rest_only - truth != rollout (max gap {gap:.3g})"
            feat_versions[f"{src}_progress"] = progress_only
            feat_versions[f"{src}_rest"] = rest_only
        for src, g in _FEAT_GAINS:
            feat_versions[f"{src}_gain{g}"] = f_last32 + g * (flow_feats[src].float() - f_last32)  # (B, F, N, P, C)

        obs_feats = feats[:, 1:n_ctx + 1]                                             # (B, n_ctx, N, P, C) observed frames 1..n_ctx
        no_token = torch.full((B, f_future.shape[1]), float("nan"))                   # (B, F) these versions have no z
        for v, x_future in feat_versions.items():
            x_seq = torch.cat([obs_feats, x_future.to(obs_feats.dtype)], dim=1).flatten(0, 1)  # (B*(T-1), N, P, C) frames 1..T-1
            P = _fit_centres(trainer, tokens, x_seq, height, width, num_cameras)     # (B, T, 3)
            _append_cam(rec[v], P, G, dG, n_ctx, fc, B)
            rec[v]["feat_ar"].append(_feat_err(x_future, f_future).cpu())            # (B, F)
            progress, _, _ = _split_progress(x_future, f_last_obs, f_future)        # (B, F)
            rec[v]["prog"].append(progress.cpu())
            for k in ("mse", "feat_tf", "znorm"):
                rec[v][k].append(no_token)

        batches_done += 1
        print(f"[INFO]   batch {batches_done}/{args.num_batches}  B={B} T-1={z.shape[1]} "
              f"N={num_cameras} res={H}x{W}", flush=True)

    if batches_done == 0:
        raise RuntimeError("No usable batches: nothing to measure.")
    cat = lambda xs: torch.cat(xs).numpy()                                            # noqa: E731
    return ({v: {k: cat(x) for k, x in r.items()} for v, r in rec.items()},
            {k: cat(x) for k, x in rec_gt.items()})


def summarize(rec, rec_gt, n_ctx):
    """One row per version. Paths and ratios cover the forecast steps only."""
    gt_path = rec_gt["step"][:, n_ctx:].sum(1)                                        # (W,) GT forecast path, m
    donor_path = rec_gt["step_donor"][:, n_ctx:].sum(1)                               # (W,) donor's GT forecast path
    moving = gt_path > 2.0                                                            # windows where the car drives
    rows = {}
    for v, r in rec.items():
        path = r["step"][:, n_ctx:].sum(1)                                            # (W,) predicted forecast path
        rows[v] = {
            "median_step_per_frame_m": np.median(r["step"], 0).round(3).tolist(),     # frames 1..T-1
            "path_ratio_median": float(np.median(path[moving] / gt_path[moving])),
            "corr_path_own_gt": float(np.corrcoef(path, gt_path)[0, 1]),
            "corr_path_donor_gt": float(np.corrcoef(path, donor_path)[0, 1]),
            "heading_cos_median": float(np.median(r["head"][moving])),
            "ate_mean_m": float(r["ate"].mean()),
            "token_mse": float(r["mse"].mean()),
            "token_mse_per_frame": r["mse"].mean(0).round(4).tolist(),                # forecast frames n_ctx+1..T-1
            "feat_err_rollout": float(r["feat_ar"].mean()),
            "feat_err_rollout_per_frame": r["feat_ar"].mean(0).round(4).tolist(),
            "feat_err_tf": float(r["feat_tf"].mean()),
            "feat_err_tf_per_frame": r["feat_tf"].mean(0).round(4).tolist(),
            "progress_median": float(np.median(r["prog"])),                           # 0 = copy of the last observed frame, 1 = truth
            "progress_per_frame": np.median(r["prog"], 0).round(3).tolist(),
            "z_norm_ratio": float(r["znorm"].mean()),                                 # nan for feature versions
        }
    rows["_gt"] = {
        "windows": int(len(gt_path)), "moving": int(moving.sum()),
        "gt_median_step_per_frame_m": np.median(rec_gt["step"], 0).round(3).tolist(),
        "cos_next_delta_mean": float(rec_gt["cos_next"].mean()),
        "cos_other_window_mean": float(rec_gt["cos_other"].mean()),
        "feat_err_copy": float(rec_gt["feat_copy"].mean()),                           # frame t as frame t+1
        "feat_err_copy_per_frame": rec_gt["feat_copy"].mean(0).round(4).tolist(),
    }
    return rows


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

    with open_dict(cfg):
        if args.occany_recon_ckpt:
            cfg.model.occany_recon_ckpt = args.occany_recon_ckpt
        cfg.model.encode_layer = int(args.encode_layer)
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
    print(f"[INFO] n_ctx={trainer.n_ctx} pred_mode={cfg.model.pred_mode} "
          f"scheduler={cfg.model.get('sampler_scheduler_mode')} alpha={cfg.model.get('sampler_alpha')} "
          f"flowN steps={args.num_steps}", flush=True)

    results = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in ("train", "val"):
            raise ValueError(f"--splits must contain only train/val, got {split!r}")
        for label, loader in _build_split_loaders(cfg, split):
            print(f"\n[INFO] Probing {label} ({len(loader)} batches available)")
            _pin_loader_epoch(loader)
            # Seeded flow prior per set, so reruns sample the same noise.
            trainer._eval_noise_gen = torch.Generator(device=device).manual_seed(args.seed)
            with trainer.ema_scope():
                rec, rec_gt = motion_split(trainer, loader, args)
            rows = summarize(rec, rec_gt, trainer.n_ctx)
            results[label] = rows

            g = rows["_gt"]
            print(f"[{label}] windows {g['windows']} (moving {g['moving']})  GT median step/frame m "
                  f"{g['gt_median_step_per_frame_m']}")
            print(f"[{label}] GT delta cos: next transition {g['cos_next_delta_mean']:.3f}  "
                  f"other window {g['cos_other_window_mean']:.3f}")
            print(f"[{label}] copy (frame t as t+1) feat err {g['feat_err_copy']:.4f}  per frame {g['feat_err_copy_per_frame']}")
            print(f"[{label}] {'version':20s} {'path/GT':>8s} {'corrOwn':>8s} {'corrDonor':>9s} "
                  f"{'headCos':>8s} {'ATE m':>7s} {'tokMSE':>7s} {'zNorm':>6s} {'featAR':>7s} {'featTF':>7s} "
                  f"{'prog':>6s}  median step/frame m")
            for v, r in rows.items():
                if v == "_gt":
                    continue
                print(f"[{label}] {v:20s} {r['path_ratio_median']:8.3f} {r['corr_path_own_gt']:8.2f} "
                      f"{r['corr_path_donor_gt']:9.2f} {r['heading_cos_median']:8.2f} {r['ate_mean_m']:7.2f} "
                      f"{r['token_mse']:7.3f} {r['z_norm_ratio']:6.3f} {r['feat_err_rollout']:7.4f} {r['feat_err_tf']:7.4f} "
                      f"{r['progress_median']:6.3f}  {r['median_step_per_frame_m']}", flush=True)
            for v, r in rows.items():
                if v == "_gt":
                    continue
                print(f"[{label}]   {v:20s} featAR/frame {r['feat_err_rollout_per_frame']}  "
                      f"featTF/frame {r['feat_err_tf_per_frame']}  tokMSE/frame {r['token_mse_per_frame']}  "
                      f"prog/frame {r['progress_per_frame']}", flush=True)

    out = os.path.join(output_dir, "motion.json")
    with open(out, "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    print(f"\n[INFO] Wrote {out}")


if __name__ == "__main__":
    main()
