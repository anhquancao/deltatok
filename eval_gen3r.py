#!/usr/bin/env python3
"""Dump pretrained Gen3R forecasts (depth, points, c2w at DA3 scale, GT-mask pixels
only) on the DeltaTok flow-eval windows; score with eval_forecast_metrics.py.

Frames 0-1 are context, 2..T-1 are forecast. Window frame k sits in slot k * slot_step
of one Gen3R clip (4k+1 slots; 13 at step 1); only the context slots are given, with
zero Plücker (camera-free) and an empty prompt (text-free). Its geometry adapter and VGGT
heads decode the slots in frame-0 coordinates, and DA3METRIC-LARGE on the context frames
(predicted K) sets the metric scale. Writes <output_dir>/<run>/<set>/<n:05d>.npz.
Plan: docs/research/plan/2026-10-04_flow_gen3r_baseline_eval.md

Usage (BSC GPU node):
  source env_bsc.sh && python eval_gen3r.py \
      --config-name eval_deltatok_flow_alldata_ctx2fwd8_kitti_bsc \
      --ckpt /gpfs/scratch/ehpc1001/quan/gen3r/checkpoints
"""

import argparse
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(
    Path(__file__).resolve().parent,
    extra=[
        "third_party/Gen3R",
    ],
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from gen3r.pipeline import Gen3RPipeline  # noqa: E402
from gen3r.utils.common_utils import convert_to_token_list  # noqa: E402
from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.alignment import (  # noqa: E402
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
)

from occany.datasets import get_data_loader  # noqa: E402
from occany.utils.helpers import convert_depth_to_point_cloud  # noqa: E402
from occrae.deltatok_shared import DeltaTokSharedMixin  # noqa: E402
from occrae.forecast_dump import save_windows  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
N_CTX = 2  # frames 0-1 given
GEN3R_HW = {(168, 518): (336, 1008), (266, 518): (448, 896)}  # loader (H, W) -> multiples of 112, ~560² px
NEG_PROMPT = "bad detailed"  # infer.py


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Gen3R on the DeltaTok flow eval", add_help=True)
    parser.add_argument("--config-dir", type=str, default="configs/deltatok_flow")
    parser.add_argument("--config-name", type=str, default="eval_deltatok_flow_alldata_ctx2fwd8_kitti_bsc")
    parser.add_argument("--ckpt", type=str, default="/gpfs/scratch/ehpc1001/quan/gen3r/checkpoints")
    parser.add_argument("--num_items", type=int, default=None,
                        help="Windows per set; above the split takes it whole (window_stride sets the size).")
    parser.add_argument("--test_filter", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)  # infer.py
    parser.add_argument("--slot_step", type=int, default=1,
                        help="Clip slots per window frame: 1 = 13-slot clip, 5 = 49 slots at 10 Hz.")
    parser.add_argument("--ctx_mode", choices=["both", "first"], default="both",
                        help="first = ctx 0 only, Gen3R's trained 1view mask (smoke comparison).")
    parser.add_argument("--prompt", type=str, default="")  # text-free, as VGGT-World Table 3; trained w/ 20% drop
    parser.add_argument("--da3_metric_model", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--verbose_batches", type=int, default=1,
                        help="Print per-window scales for the first N batches of each set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", type=int, default=0, help="Slurm array task: windows n with n %% num_shards == shard.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/gen3r_alldata_ctx2fwd8")
    return parser


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-")


def _build_test_loaders(cfg, num_items=None, test_filter=None):
    """{test_name: loader} for cfg.dataset.test_dataset — mirrors how
    `DeltaTokFlowMatchingTrainer.fit` builds eval loaders, so `eval_one_epoch`
    sees the exact samples the training logs evaluated."""
    expr = cfg.dataset.get("test_dataset", None)
    if not expr:
        raise RuntimeError("Config has no dataset.test_dataset.")
    kw = dict(
        # val_bsize caps eval decode memory (0 = train bsize) — as the trainer does
        batch_size=int(cfg.training.get("val_bsize", 0)) or int(cfg.training.bsize),
        num_workers=int(cfg.training.get("val_num_workers", 2)),
        shuffle=False,
        drop_last=False,
    )
    loaders = {}
    for sub in str(expr).split("+"):
        sub = sub.strip()
        if not sub:
            continue
        if test_filter is not None and test_filter not in sub:
            continue
        if num_items is not None:
            ds_expr = sub.split("@", 1)[-1].strip()           # drop the config's "<n> @"
            loader = get_data_loader(f"{num_items} @ {ds_expr}", **kw)
            pool = len(loader.dataset.dataset)                # windows in the full split
            if pool < num_items:
                # ResizedDataset would repeat windows; take the split whole instead
                loader = get_data_loader(f"{pool} @ {ds_expr}", **kw)
            sub = f"{min(num_items, pool)} @ {ds_expr}"
        else:
            loader = get_data_loader(sub, **kw)
        test_name = sub.split("(")[0].strip()                 # tag carries the real count
        m = re.search(r"window_stride=(\d+)", sub)
        if m:
            test_name += f" stride{m.group(1)}"   # an unstrided set and its strided twin log under separate names
        loaders[test_name] = loader
    return loaders


def _load_gen3r(path, device):
    pipe = Gen3RPipeline.from_pretrained(path)
    pipe.to(device).to(torch.bfloat16)                                                 # as infer.py
    for name, comp in pipe.components.items():
        print(f"[INFO] Gen3R {name}: {type(comp).__name__}", flush=True)
    return pipe


@torch.no_grad()
def _generate(pipe, x, ctx_mode, slot_step, n_slots, hw, args, gen):
    """One window: context frames at their slots, camera-free; returns the denoised latents."""
    x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)  # (2, 3, h, w) ctx in [0, 1]
    ctrl = torch.zeros(1, n_slots, 3, *hw, device=x.device)                             # (1, S, 3, h, w)
    idx = [0, slot_step] if ctx_mode == "both" else [0]
    ctrl[0, idx] = x[: len(idx)]
    cams = torch.zeros(1, n_slots, 6, *hw, device=x.device)                              # (1, S, 6, h, w) camera-free
    return pipe(prompt=args.prompt, negative_prompt=NEG_PROMPT, control_cameras=cams.to(torch.bfloat16),
                control_images=ctrl.to(torch.bfloat16), control_index=idx, num_frames=n_slots,
                height=hw[0], width=hw[1], num_inference_steps=args.steps, guidance_scale=args.guidance,
                generator=gen, output_type="latent", return_dict=False)[0]               # (1, 16, (S-1)/4+1, h/8, 2w/8)


@torch.no_grad()
def _decode(pipe, latents, slots, out_hw):
    """Geometry adapter + VGGT heads; depth, conf and K of the window's slots at out_hw, c2w."""
    geo = latents.chunk(2, dim=-1)[1]                                                  # (1, 16, f, h/8, w/8)
    tok = pipe.geo_adapter.decode(geo).sample                                          # (1, 5C, S, h/14, w/14)
    agg, frames = convert_to_token_list(rearrange(tok, "b c f h w -> b f h w c"), pipe.vggt.aggregator.patch_size)  # 4 x (1, S, 5+P, C)
    h, w = frames.shape[-2:]
    H, W = out_hw
    T = len(slots)
    # bf16 heads, as Gen3RPipeline.decode_latents
    pose_enc = pipe.vggt.camera_head(agg)[-1].float()                                  # (1, S, 9) trunk attends over all S
    w2c, K = pose_encoding_to_extri_intri(pose_enc[:, slots], (h, w))                  # (1, T, 3, 4), (1, T, 3, 3)
    depth, conf = pipe.vggt.depth_head([a[:, slots] for a in agg], frames[:, slots], 5)  # (1, T, h, w, 1), (1, T, h, w)
    depth, conf = depth[..., 0].float(), conf.float()                                  # (1, T, h, w)
    if (h, w) != (H, W):
        depth = F.interpolate(depth.flatten(0, 1)[:, None], size=(H, W), mode="bilinear",
                              align_corners=False).view(1, T, H, W)                    # (1, T, H, W)
        conf = F.interpolate(conf.flatten(0, 1)[:, None], size=(H, W), mode="bilinear",
                             align_corners=False).view(1, T, H, W)                     # (1, T, H, W)
        K = K.clone()
        K[..., 0, :] *= W / w
        K[..., 1, :] *= H / h
    w2c44 = torch.eye(4, device=w2c.device, dtype=torch.float64).repeat(1, T, 1, 1)   # (1, T, 4, 4)
    w2c44[..., :3, :] = w2c.double()
    c2w = torch.linalg.inv(w2c44).float()                                              # (1, T, 4, 4)
    return depth.float(), conf.float(), K.float(), c2w


def _da3_scale(m_depth, m_sky, depth, conf, K):
    """Per-window LSQ scale of VGGT depth to DA3-metric depth on the context frames
    (extract_recon.py:404-434). Returns (B,) scale and (B,) fallback flag."""
    md = apply_metric_scaling(m_depth, K[:, :N_CTX])                                  # (B, 2, H, W)
    non_sky = compute_sky_mask(m_sky, threshold=0.3)                                   # (B, 2, H, W)
    B = depth.shape[0]
    scale = torch.ones(B, device=depth.device)
    fallback = torch.ones(B, dtype=torch.bool, device=depth.device)
    for b in range(B):
        ns = non_sky[b]
        if ns.sum() <= 10:
            continue
        conf_ns = conf[b, :N_CTX][ns]
        if conf_ns.numel() == 0:
            continue
        median_conf = torch.quantile(sample_tensor_for_quantile(conf_ns, max_samples=100000), 0.5)
        align = compute_alignment_mask(conf[b, :N_CTX], ns, depth[b, :N_CTX], md[b], median_conf)
        if align.sum() == 0:
            continue
        s = least_squares_scale_scalar(md[b][align], depth[b, :N_CTX][align])
        if torch.isfinite(s):
            scale[b] = s
            fallback[b] = False
    return scale, fallback


def main() -> None:
    args = get_args_parser().parse_args()
    device = torch.device("cuda")

    output_dir = os.path.join(os.path.abspath(args.output_dir), f"{args.ctx_mode}_step{args.slot_step}_native")
    os.makedirs(output_dir, exist_ok=True)

    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)
    with open_dict(cfg):
        cfg.training.val_bsize = 1  # the pipeline's camera reshape is batch-1 (pipeline_gen3r.py:684)

    pipe = _load_gen3r(args.ckpt, device)
    da3_metric = DepthAnything3.from_pretrained(args.da3_metric_model).to(device).eval()
    da3_metric.requires_grad_(False)
    batcher = DeltaTokSharedMixin()  # only _normalize_batch
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    loaders = _build_test_loaders(cfg, args.num_items, args.test_filter)
    for test_name, loader in loaders.items():
        # Pin data and noise per set, as eval_one_epoch does (rank 0).
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        ds = getattr(loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(0)
        torch.manual_seed(args.seed)

        set_dir = os.path.join(output_dir, _sanitize(test_name))
        os.makedirs(set_dir, exist_ok=True)
        n_fallback, n_items = 0, 0
        t0 = time.time()
        for it, raw in enumerate(loader):
            if it % args.num_shards != args.shard:  # another array task's window
                continue
            torch.manual_seed(args.seed + it)  # DA3's quantile draw per window, same whatever the split
            batch = batcher._normalize_batch(raw)
            imgs = batch["imgs"].to(device, non_blocking=True)                          # (1, T, 3, H, W) DA3-normed
            B, T, _, H, W = imgs.shape
            assert B == 1 and batch["num_cameras"] == 1, "one cam-0 window per batch"
            x01 = (imgs * std + mean).clamp(0, 1)                                       # (1, T, 3, H, W)
            hw = GEN3R_HW[(H, W)]
            slots = [k * args.slot_step for k in range(T)]                              # window frame -> clip slot
            n_slots = 4 * ((slots[-1] + 3) // 4) + 1                                    # 4k+1: 13 at step 1, 49 at 5

            gen = torch.Generator(device).manual_seed(args.seed + it)                   # per-window noise
            lat = _generate(pipe, x01[0, :N_CTX], args.ctx_mode, args.slot_step, n_slots, hw, args, gen)
            depth, conf, K, c2w = _decode(pipe, lat, slots, (H, W))                     # (1, T, H, W) x 2, (1, T, 3, 3), (1, T, 4, 4)

            with torch.no_grad():
                m = da3_metric(imgs[:, :N_CTX], export_feat_layers=[])
            m_depth, m_sky = m["depth"].float(), m["sky"].float()                        # (B, 2, H, W)

            scale, fallback = _da3_scale(m_depth, m_sky, depth, conf, K)                 # (1,), (1,)
            depth = depth * scale[:, None, None, None]                                     # (1, T, H, W) metres
            c2w[..., :3, 3] *= scale[:, None, None]                                        # (1, T, 4, 4)
            point = convert_depth_to_point_cloud(depth, K, c2w)                            # (1, T, H, W, 3) frame-0 coords
            save_windows(set_dir, it, depth, point, c2w, batch)                           # B = 1, so it = window index
            n_items += B
            n_fallback += int(fallback.sum())

            if n_items <= args.verbose_batches:
                print(f"[DBG/{test_name}] x01 [{x01.min():.3f}, {x01.max():.3f}]  gen3r in {hw}  slots {slots} of "
                      f"{n_slots}  ctx {args.ctx_mode}  peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                      flush=True)
                print(f"[DBG/{test_name}] s_da3 {scale.tolist()}  fallback {fallback.tolist()}", flush=True)
            if n_items % 50 == 1:
                print(f"[INFO] {test_name}: {n_items} windows, {(time.time() - t0) / n_items:.2f} s/window",
                      flush=True)

        print(f"[INFO] {test_name}: wrote {n_items} windows, {n_fallback} DA3 fallbacks, to {set_dir}", flush=True)


if __name__ == "__main__":
    main()
