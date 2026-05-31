#!/usr/bin/env python3
"""Parallel rsync Karolina -> BSC, fanning out per second-level entry.

Mirrors sh/verify_karolina_to_bsc.sh but runs WORKERS rsync subprocesses in
parallel. Each task syncs one entry directly under a dataset root (e.g. a
single `scene-1062/` directory, or a loose top-level file like
`annotations.json`). That granularity matters because the shell version
runs one rsync per dataset and goes serial as soon as a single big scene
dominates.

Run ON KAROLINA. Requires the `bsc` ssh alias in ~/.ssh/config there.

Usage:
    python scripts/sync_karolina_to_bsc.py
    python scripts/sync_karolina_to_bsc.py occ3d_nuscenes_processed kitti_processed
    DRY_RUN=1 python scripts/sync_karolina_to_bsc.py
    WORKERS=8 python scripts/sync_karolina_to_bsc.py occ3d_nuscenes_processed

Env knobs:
    SRC_ROOT  default /scratch/project/eu-25-92/data
    DST_HOST  default bsc
    DST_ROOT  default /gpfs/scratch/ehpc793/occany_data
    WORKERS   default 16  (concurrent rsync subprocesses; each opens its own ssh)
    DRY_RUN   set to 1 to add rsync -n
    ITEMIZE   set to 1 to add --itemize-changes (noisy; off by default)

Notes:
  - Uses --size-only, matching the shell version. Files that exist on BSC
    with the same byte count are skipped. Does NOT detect silent bit-rot.
  - 16 concurrent ssh sessions normally fits inside sshd's MaxStartups
    (10:30:60 by default). Lower WORKERS if BSC starts dropping you.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC_ROOT = Path(os.environ.get("SRC_ROOT", "/scratch/project/eu-25-92/data"))
DST_HOST = os.environ.get("DST_HOST", "bsc")
DST_ROOT = os.environ.get("DST_ROOT", "/gpfs/scratch/ehpc793/occany_data")
WORKERS = int(os.environ.get("WORKERS", "16"))
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
ITEMIZE = os.environ.get("ITEMIZE", "0") == "1"

DEFAULT_DATASETS = [
    "waymo_processed",
    "vkitti_processed",
    "ddad_processed",
    "pandaset_processed",
    "once_processed",
    "kitti_processed",
    "occ3d_nuscenes_processed",
]

SSH_CMD = "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3"


def rsync_cmd() -> list[str]:
    cmd = [
        "rsync", "-aH", "-h", "--stats", "--size-only",
        "--exclude=tmp/", "--exclude=.tmp/",
        "-e", SSH_CMD,
    ]
    if DRY_RUN:
        cmd.append("-n")
    if ITEMIZE:
        cmd.append("--itemize-changes")
    return cmd


def run_rsync(label: str, src: str, dst: str) -> tuple[str, int, float, str]:
    cmd = rsync_cmd() + [src, dst]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    # Keep the trailing --stats block + any stderr; drop per-file chatter.
    tail = "\n".join(out.splitlines()[-8:]) if out else ""
    msg_parts = []
    if tail:
        msg_parts.append(tail)
    if err:
        msg_parts.append("STDERR: " + err)
    return label, proc.returncode, elapsed, "\n".join(msg_parts)


def tasks_for_dataset(dataset: str) -> list[tuple[str, str, str]]:
    src_dir = SRC_ROOT / dataset
    if not src_dir.exists():
        print(f"!! skip {dataset}: {src_dir} not present on Karolina",
              file=sys.stderr, flush=True)
        return []

    dst_base = f"{DST_HOST}:{DST_ROOT}/{dataset}"
    out: list[tuple[str, str, str]] = []
    for entry in sorted(src_dir.iterdir()):
        if entry.is_dir():
            label = f"{dataset}/{entry.name}/"
            out.append((label, f"{entry}/", f"{dst_base}/{entry.name}/"))
        elif entry.is_file():
            label = f"{dataset}/{entry.name}"
            out.append((label, str(entry), f"{dst_base}/{entry.name}"))
    return out


def main() -> int:
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DATASETS

    tasks: list[tuple[str, str, str]] = []
    for d in datasets:
        tasks.extend(tasks_for_dataset(d))

    if not tasks:
        print("no tasks to run")
        return 0

    print(
        f"==> {len(tasks)} tasks across {len(datasets)} dataset(s), "
        f"WORKERS={WORKERS}, DRY_RUN={int(DRY_RUN)}, ITEMIZE={int(ITEMIZE)}\n"
        f"    src: {SRC_ROOT}\n"
        f"    dst: {DST_HOST}:{DST_ROOT}",
        flush=True,
    )

    failures: list[tuple[str, str]] = []
    t0 = time.time()
    n = len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(run_rsync, label, src, dst): label
            for label, src, dst in tasks
        }
        try:
            for fut in as_completed(futures):
                done += 1
                label, rc, elapsed, msg = fut.result()
                status = "ok" if rc == 0 else f"FAIL rc={rc}"
                header = f"[{done}/{n}] {status} {elapsed:6.1f}s  {label}"
                if msg:
                    print(header + "\n" + msg + "\n", flush=True)
                else:
                    print(header, flush=True)
                if rc != 0:
                    failures.append((label, msg))
        except KeyboardInterrupt:
            print("\n!! interrupted, cancelling pending tasks", flush=True)
            for f in futures:
                f.cancel()
            raise

    dt = time.time() - t0
    print(f"\n== finished in {dt:.1f}s: {n - len(failures)} ok, "
          f"{len(failures)} failed")
    for label, _ in failures:
        print(f"   FAIL  {label}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
