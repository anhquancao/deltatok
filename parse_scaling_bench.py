#!/usr/bin/env python3
"""Turn the Jean Zay scaling-benchmark logs into the EuroHPC Section 2.6 tables and plots.

Reads the `.out` files produced by slurm/bench_scaling_deltatok{,_flow}_jz.slurm, keeps the
post-warmup `Training (Epoch ...)` lines only, and writes a CSV, markdown tables and a log-log
speedup plot.

    python parse_scaling_bench.py --logs ../monitor_jobs/data/logs/Jeanzay

Timings use the MEAN over the timed window (node-hours = mean x steps); the median is reported
beside it as the steady-state cost.

The two trainers print different meters and the difference is load-bearing:
  * tokenizer  `time:` = seconds per LOADER iteration (micro-step); bracket index counts loader
    iterations. s/optimizer-step = time * grad_cum.  It also prints `data:` (loader wait).
  * flow       `time:` = seconds per OPTIMIZER update (deltatok_flow_trainer.py `update_time`);
    bracket index counts updates. s/optimizer-step = time. No `data:` meter.
"""
import argparse
import csv
import glob
import math
import os
import re
import statistics
import sys

BENCH_RE = re.compile(
    r"BENCH MODE=(\w+) NGPU=(\d+) BSIZE=(\d+) EFF_BSIZE=(\d+) NODES=(\S+)")
EFF_RE = re.compile(
    r"effective_bsize=(\d+) = (\d+) rank\(s\) x bsize (\d+) x grad_cum (\d+)")
RUN_RE = re.compile(r"^RUN_NAME=bench_(tok|flow)_")
# 'Training (Epoch 0)  [ 340/1000]  ...  time: 0.8412  data: 0.0010  max gpu mem: 44684 (81559)'
TRAIN_RE = re.compile(
    r"^Training \(Epoch (\d+)\)\s+\[\s*(\d+)/(\d+)\]"          # epoch, index, total
    r".*?\btime:\s*([\d.]+)"                                    # rolling mean step time
    r"(?:.*?\bdata:\s*([\d.]+))?"                               # tokenizer only
    r".*?\bmax gpu mem:\s*([\d.]+)")


def parse_log(path, warmup):
    """One benchmark point, or None if the file is not a benchmark log."""
    with open(path, errors="replace") as fh:
        text = fh.read()

    m = BENCH_RE.search(text)
    if not m:
        return None
    mode, ngpu, bsize, eff = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))

    rm = RUN_RE.search(text, re.M) or re.search(r"^RUN_NAME=bench_(tok|flow)_", text, re.M)
    model = rm.group(1) if rm else ("flow" if "deltatok_flow" in text else "tok")

    em = EFF_RE.search(text)
    if not em:
        return {"path": path, "model": model, "mode": mode, "ngpu": ngpu, "bsize": bsize,
                "global": eff, "error": "no effective_bsize line — job died before setup"}
    world, ck_bsize, grad_cum = int(em.group(2)), int(em.group(3)), int(em.group(4))
    if (world, ck_bsize, int(em.group(1))) != (ngpu, bsize, eff):
        return {"path": path, "model": model, "mode": mode, "ngpu": ngpu, "bsize": bsize,
                "global": eff,
                "error": f"ran {world}x{ck_bsize} global {em.group(1)}, expected "
                         f"{ngpu}x{bsize} global {eff}"}

    # Warmup is expressed in optimizer steps; the tokenizer's bracket counts loader iterations.
    cut = warmup * grad_cum if model == "tok" else warmup
    epochs, times, datas, mems, last_idx = set(), [], [], [], 0
    for line in text.splitlines():
        tm = TRAIN_RE.match(line)
        if not tm:
            continue
        epochs.add(int(tm.group(1)))
        idx = int(tm.group(2))
        last_idx = max(last_idx, idx)
        mems.append(float(tm.group(6)))
        if idx < cut:
            continue
        times.append(float(tm.group(4)))
        if tm.group(5) is not None:
            datas.append(float(tm.group(5)))

    if not times:
        return {"path": path, "model": model, "mode": mode, "ngpu": ngpu, "bsize": bsize,
                "global": eff, "grad_cum": grad_cum,
                "error": f"no post-warmup Training lines (last index {last_idx}, cut {cut})"}

    # MEAN is the headline: node-hours = mean step time x steps, so a run that stalls really
    # does cost more and the resource request must carry that. The median is kept beside it as
    # the steady-state cost — stragglers, Lustre hiccups and DDP jitter only ever make a step
    # slower, so median < mean always. A large gap between the two is itself the finding.
    t_mean = statistics.fmean(times)
    t_med = statistics.median(times)
    scale = grad_cum if model == "tok" else 1        # tokenizer times a loader iteration
    return {
        "path": path, "model": model, "mode": mode, "nodes": ngpu // 4, "ngpu": ngpu,
        "bsize": bsize, "global": eff, "grad_cum": grad_cum,
        "n_prints": len(times),
        "t_micro_s": t_mean if model == "tok" else t_mean / grad_cum,
        "t_step_s": t_mean * scale,
        "t_step_median_s": t_med * scale,
        "stall_ratio": t_mean / t_med,               # >1.05 = the timed window was not clean
        # spread across the timed window: a wide band means the point is not steady state
        "t_step_p10_s": (statistics.quantiles(times, n=10)[0] if len(times) > 3 else t_med) * scale,
        "t_step_p90_s": (statistics.quantiles(times, n=10)[-1] if len(times) > 3 else t_med) * scale,
        "data_frac": (statistics.fmean(datas) / t_mean) if datas else None,
        "peak_gpu_mb": max(mems) if mems else None,
        "samples_per_s": eff / (t_mean * scale),
        "epochs_seen": len(epochs),
        "last_idx": last_idx,
        "error": None,
    }


def ladder(points, model, mode):
    """Points of one ladder, ascending in GPUs. 'both' belongs to strong and weak alike."""
    # copies: a "both" point sits in two ladders and score() writes per-ladder columns onto it
    rows = [dict(p) for p in points
            if p["model"] == model and p["mode"] in (mode, "both") and not p["error"]]
    return sorted(rows, key=lambda p: p["ngpu"])


INT_FIELDS = ("nodes", "ngpu", "bsize", "global", "grad_cum", "n_prints",
               "epochs_seen", "last_idx")


def load_extra_csv(path):
    """Points measured by another session, in this script's own CSV schema (e.g. OccAny/WP2)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            p = {"error": None, "path": path}
            for k, v in r.items():
                if v in (None, ""):
                    p[k] = None
                elif k in INT_FIELDS:
                    p[k] = int(float(v))
                elif k in ("model", "mode"):
                    p[k] = v
                elif k == "path":
                    p["path"] = v
                else:
                    p[k] = float(v)
            rows.append(p)
    return rows


def score(rows):
    """Add per-epoch time, speed-up and parallel efficiency, relative to the smallest job.

    Speed-up is taken on TIME PER EPOCH, not per step. A step is a different amount of work at
    each weak-scaling point (the batch grows with the allocation), so a per-step ratio there is
    not a speed-up; a fixed EPOCH_SAMPLES epoch is the same work everywhere and makes the two
    ladders directly comparable.
    """
    if not rows:
        return rows
    for r in rows:
        r["epoch_s"] = EPOCH_SAMPLES / r["samples_per_s"]
    base_e, base_n = rows[0]["epoch_s"], rows[0]["ngpu"]
    for r in rows:
        r["epoch_speedup"] = base_e / r["epoch_s"]
        r["epoch_efficiency"] = r["epoch_speedup"] * base_n / r["ngpu"]
        # kept for the plot annotations / CSV
        r["speedup"] = rows[0]["t_step_s"] / r["t_step_s"]
        r["efficiency"] = r["speedup"] * base_n / r["ngpu"]
        r["weak_efficiency"] = rows[0]["t_step_s"] / r["t_step_s"]
    return rows


# One table per model, strong ladder above weak, distinguished by the leading column.
# Time per epoch normalises the two ladders onto the same fixed work (EPOCH_SAMPLES), which is
# what makes speed-up meaningful for weak scaling too: the batch grows, the epoch does not.
EPOCH_SAMPLES = 64000

MD_COLS = [("_ladder", "ladder", "{}"), ("ngpu", "GPUs", "{}"), ("nodes", "nodes", "{}"),
           ("global", "Total Batch Size", "{}"), ("bsize", "Batch/GPU", "{}"),
           ("_epoch_s", "Training Time/Epoch (s)", "{:.0f}"),
           ("_speedup", "Speed-up (vs. {base} GPUs)", "{:.2f}"),  # {base} filled per table
           ("_eff_pct", "Efficiency (%)", "{:.0f}")]


def markdown(strong_rows, weak_rows, title, production_ngpu=None):
    base = (strong_rows or weak_rows)[0]["ngpu"]      # speed-up is relative to the smallest job
    out = [f"### {title}", "",
           "| " + " | ".join(c[1].format(base=base) for c in MD_COLS) + " |",
           "|" + "|".join("---" for _ in MD_COLS) + "|"]
    for kind, rows in (("strong", strong_rows), ("weak", weak_rows)):
        for r in rows:
            r = dict(r, _ladder=kind, _epoch_s=r["epoch_s"],
                     _speedup=r["epoch_speedup"], _eff_pct=100.0 * r["epoch_efficiency"])
            bold = r["ngpu"] == _prod_ngpu(production_ngpu, kind)
            cells = []
            for key, _, fmt in MD_COLS:
                v = r.get(key)
                c = "—" if v is None else fmt.format(v)
                cells.append(f"**{c}**" if bold else c)
            out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# Job size expected to carry the main load, from the proposal's milestone table. An int applies
# to both ladders; a {"strong": n, "weak": n} dict marks a different size in each — WP2 (occany)
# is tabled at two sizes, architecture + rig-aware at 4 nodes and data scaling at 8 nodes, and
# the ladder lands one in each.
PRODUCTION_NGPU = {"tok": 16, "flow": 32, "occany": {"strong": 16, "weak": 32}}

# key, table heading, figure suffix
MODELS = (("tok", "DeltaTok tokenizer", "tokenizer"),
          ("flow", "DeltaTok-flow world model", "flow"),
          ("occany", "OccAny geometry foundation model", "occany"))


def _prod_ngpu(spec, kind):
    """The production GPU count for one ladder. Accepts an int or a per-ladder dict."""
    return spec.get(kind) if isinstance(spec, dict) else spec


def plot(groups, path_stem):
    """One figure per model, three panels: training time per epoch, speed-up, efficiency.

    All three against GPU count, strong and weak ladders overlaid. Dashed lines are the ideal:
    1/N for time, N for speed-up, 100% for efficiency. All axes are linear.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib missing — skipping the plots", file=sys.stderr)
        return
    LADDERS = (("strong", "strong scaling (total batch fixed)", "o", "#1f77b4"),
               ("weak", "weak scaling (batch/GPU fixed)", "s", "#d95f02"))
    # the epoch size behind the time axis is stated in the table caption, not on the axis
    PANELS = (("epoch_s", "Training time (s)", False, "Training time vs GPUs"),
              ("epoch_speedup", "Speed-up", False, "Speed-up vs GPUs"),
              ("_eff_pct", "Efficiency (%)", False, "Efficiency vs GPUs"))

    for model, _label, tag in MODELS:
        if not (groups.get((model, "strong")) or groups.get((model, "weak"))):
            continue
        # readable when the figure is dropped into a proposal at half page width
        plt.rcParams.update({"font.size": 17, "axes.titlesize": 20, "axes.labelsize": 18,
                             "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 14, "lines.linewidth": 2.5,
                             "lines.markersize": 10})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        base_n = None
        for ax, (key, ylab, logy, title) in zip(axes, PANELS):
            ylo = yhi = None                # data range, for the y limits below
            for kind, label, marker, color in LADDERS:
                rows = groups.get((model, kind)) or []
                if not rows:
                    continue
                base_n = rows[0]["ngpu"]
                x = [r["ngpu"] for r in rows]
                y = [100.0 * r["epoch_efficiency"] if key == "_eff_pct" else r[key] for r in rows]
                ylo = min(y) if ylo is None else min(ylo, min(y))
                yhi = max(y) if yhi is None else max(yhi, max(y))
                ax.plot(x, y, marker=marker, color=color, label=label, zorder=3)
            # ideal reference, anchored at the smallest job
            if base_n is not None:
                xs = [4, 8, 16, 32] if base_n >= 4 else sorted({r["ngpu"] for k in ("strong", "weak")
                                                                for r in groups.get((model, k)) or []})
                ref0 = (groups.get((model, "strong")) or groups.get((model, "weak")))[0]
                ideal = {"epoch_s": [ref0["epoch_s"] * base_n / n for n in xs],
                         "epoch_speedup": [n / base_n for n in xs],
                         "_eff_pct": [100.0] * len(xs)}[key]
                ax.plot(xs, ideal, ls="--", lw=1, color="0.45", zorder=2, label="ideal")
                # the ideal can overshoot the data (speed-up reaches N/N0); keep it on-axes
                if key != "_eff_pct":
                    yhi = max(yhi or 0, max(ideal))
            # Linear on both axes: ideal speed-up is then a straight line and every tick
            # interval is uniform. Ideal time stays a 1/N hyperbola — that is the function,
            # not the axis.
            ax.set_xlim(0, 34)
            # Linear y with evenly spaced ticks: the range spans well under one decade, so a log
            # axis buys no span and costs uniform tick intervals.
            if key == "_eff_pct":
                # start just under the data, not at 0 — the points sit in the top half
                ax.set_ylim(max(0, 10 * math.floor((ylo - 5) / 10)) if ylo else 0, 104)
            else:
                ax.set_ylim(0, (yhi or 1) * 1.08)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
            # bare GPU counts: on a linear axis the 4 and 8 ticks sit close enough that a
            # two-line "4\n(1 node)" label collides with its neighbour
            ax.set_xticks([4, 8, 16, 32])
            ax.set_xticklabels(["4", "8", "16", "32"])
            ax.set_xlabel("GPUs (4 per node)")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
        # one shared legend under the panels: at this text size an in-axes legend covers the
        # 4-GPU points in the time panel
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02), frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(f"{path_stem}_{tag}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path_stem}_{tag}.png / .svg")


def main():
    global EPOCH_SAMPLES
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="../monitor_jobs/data/logs/Jeanzay",
                    help="directory of cached SLURM .out files")
    ap.add_argument("--glob", default="*.out")
    ap.add_argument("--warmup", type=int, default=100,
                    help="optimizer steps discarded before timing (default 100 of 500)")
    ap.add_argument("--epoch-samples", type=int, default=EPOCH_SAMPLES,
                    help="samples per epoch used for the time-per-epoch column (default 64000)")
    ap.add_argument("--out-stem", default="docs/journal/deltatok_scaling_jz",
                    help="path stem for the .csv / .md / .png / .svg outputs")
    ap.add_argument("--extra-csv", action="append", default=[],
                    help="CSV of points measured elsewhere, same schema (repeatable)")
    args = ap.parse_args()
    EPOCH_SAMPLES = args.epoch_samples

    files = sorted(glob.glob(os.path.join(args.logs, args.glob)))
    points, bad = [], []
    for f in files:
        p = parse_log(f, args.warmup)
        if p is None:
            continue
        (bad if p["error"] else points).append(p)

    for extra in args.extra_csv:
        rows = load_extra_csv(extra)
        points.extend(rows)
        print(f"merged {len(rows)} points from {extra}")

    if not points and not bad:
        sys.exit(f"no benchmark logs found under {args.logs}/{args.glob}")

    # Newest job id wins when a point was re-run.
    points.sort(key=lambda p: p["path"])
    dedup = {(p["model"], p["mode"], p["ngpu"], p["bsize"], p["global"]): p for p in points}
    points = list(dedup.values())

    groups = {}
    for model, _, _ in MODELS:
        for kind in ("strong", "weak"):
            groups[(model, kind)] = score(ladder(points, model, kind))

    # score() works on copies; carry the epoch columns back for the CSV
    for rows in groups.values():
        for r in rows:
            for k in ("epoch_s", "epoch_speedup", "epoch_efficiency"):
                dedup[(r["model"], r["mode"], r["ngpu"], r["bsize"], r["global"])][k] = r[k]

    os.makedirs(os.path.dirname(args.out_stem) or ".", exist_ok=True)
    csv_path = f"{args.out_stem}.csv"
    fields = ["model", "mode", "nodes", "ngpu", "bsize", "global", "grad_cum", "n_prints",
              "t_micro_s", "t_step_s", "t_step_median_s", "stall_ratio",
              "t_step_p10_s", "t_step_p90_s", "data_frac",
              "peak_gpu_mb", "samples_per_s", "epoch_s", "epoch_speedup", "epoch_efficiency",
              "epochs_seen", "last_idx", "path"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for p in sorted(points, key=lambda p: (p["model"], p["mode"], p["ngpu"])):
            w.writerow(p)
    print(f"wrote {csv_path} ({len(points)} points)")

    md = ["# Scaling on Jean Zay H100 (gpu_p6)", "",
          f"{args.warmup} warmup steps discarded; step time is the **mean** of the rest, since "
          f"node-hours are mean x steps. Epoch = {EPOCH_SAMPLES:,} samples.", ""]
    for model, label, _ in MODELS:
        st, wk = groups[(model, "strong")], groups[(model, "weak")]
        if st or wk:
            gb = st[0]["global"] if st else "—"
            md.append(markdown(st, wk, f"{label} — strong (global batch {gb}) and weak scaling",
                               production_ngpu=PRODUCTION_NGPU[model]))
    md.append(f"Bold = the production job size from the milestone table. Strong holds the total "
              f"batch fixed; weak holds batch/GPU fixed. Time per epoch assumes an epoch of "
              f"{EPOCH_SAMPLES:,} samples, which is what lets speed-up and efficiency be quoted "
              f"for both ladders on the same footing.")
    md_path = f"{args.out_stem}.md"
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))
    print(f"wrote {md_path}")
    print("\n".join(md))

    plot(groups, args.out_stem)

    for b in bad:
        print(f"[skip] {os.path.basename(b['path'])}: {b['error']}", file=sys.stderr)
    # Epoch boundaries inside the measurement mean a checkpoint write stalled every rank.
    for p in points:
        if p["epochs_seen"] > 1:
            print(f"[warn] {os.path.basename(p['path'])}: {p['epochs_seen']} epochs seen — "
                  "an epoch boundary landed inside the 500 steps", file=sys.stderr)
        if p["stall_ratio"] > 1.05:
            print(f"[warn] {os.path.basename(p['path'])}: mean/median = "
                  f"{p['stall_ratio']:.2f} — the timed window carries stalls, not steady state",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
