#!/usr/bin/env python3
"""Standard error of eval means vs window stride, from a per-window CSV.

The CSV comes from eval_deltatok_flow_sampler.py --dump_per_window with
training.val_bsize=1 (one row per window: test, scene, start stem, losses).
For each test set and stride, windows are thinned exactly like
BaseSeqDatasetMultiView(window_stride=...), and the SE of the mean is a block
bootstrap within scenes: overlapping windows are resampled together, so they
are not counted as independent.

  python analyze_eval_windows.py per_window_ode_steps20_sigmaNone.csv
"""
import argparse
import math

import numpy as np
import pandas as pd

SPAN = {"Kitti": 45, "Waymo": 45, "Occ3dNuscenes": 9}  # frame-id span of one window (sub5 x 9, sub1 x 9)


def thin(starts, stride):
    """Indices of windows kept by the window_stride rule (starts sorted within one scene)."""
    keep, next_allowed = [], -math.inf
    for i, s in enumerate(starts):
        if s >= next_allowed:
            keep.append(i)
            next_allowed = s + stride
    return keep


def block_se(values_per_scene, block, n_boot, rng):
    """SE of the pooled mean; resample contiguous blocks of `block` windows (within scenes) with replacement."""
    blocks = [v[i:i + block] for v in values_per_scene for i in range(0, len(v), block)]
    sums = np.array([b.sum() for b in blocks])                    # (n_blocks,)
    lens = np.array([len(b) for b in blocks])                     # (n_blocks,)
    idx = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))  # (n_boot, n_blocks)
    means = sums[idx].sum(1) / lens[idx].sum(1)                   # (n_boot,)
    return means.std()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--metrics", default="LossDepth,LossPointmap,LossRaymap,MSEToken")
    p.add_argument("--strides", default="1,2,5,10,23,46,92")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    df = pd.read_csv(args.csv)
    df["start_id"] = df["start"].str.split("_").str[0].astype(int)  # stem "<frame_id>_<cam>" -> frame id
    metrics = [m for m in args.metrics.split(",") if m in df.columns]

    for test, dt in df.groupby("test"):
        span = next(v for k, v in SPAN.items() if k in str(test))
        print(f"\n## {test}  ({len(dt)} windows, {dt.scene.nunique()} scenes)")
        print("stride | windows | " + " | ".join(f"{m} mean ± SE (SE %)" for m in metrics))
        for stride in [int(s) for s in args.strides.split(",")]:
            kept = []
            for _, ds in dt.sort_values("start_id").groupby("scene"):
                kept.append(ds.iloc[thin(ds["start_id"].tolist(), stride)])
            block = max(1, math.ceil((span + 1) / stride))             # windows that share frames go in one block
            cells = []
            for m in metrics:
                vals = [k[m].to_numpy() for k in kept]
                mean = np.concatenate(vals).mean()
                se = block_se(vals, block, args.n_boot, rng)
                cells.append(f"{mean:.4f} ± {se:.4f} ({100 * se / mean:.2f}%)")
            print(f"{stride} | {sum(len(k) for k in kept)} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
