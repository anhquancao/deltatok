#!/usr/bin/env python3
"""Push per-scene tar stores from the cougar DAF store to BSC using parallel rsync workers.

Counterpart of OccAny's sync/pull_tars_from_jeanzay.py: same batching, resume manifest
and transfer window, with the direction reversed (local source, remote destination).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:  # cougar may not have tzdata; fall back to the machine's local clock
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
except Exception:
    PARIS_TZ = None

try:  # progress bar is optional so the script runs on a bare python3
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Explicit user@host, so the transfer does not depend on an ssh config alias on cougar.
# BSC has transfer1..4.bsc.es; pass several to --remote to spread batches over them.
DEFAULT_REMOTE = "vale352205@transfer1.bsc.es"

# Never the login nodes: a multi-hour rsync receiving on glogin* is killed by the watchdog.
DEFAULT_DST = "/gpfs/scratch/ehpc1001/occany_data_tar"

SSH_OPTS = [
    "-o", "Compression=no",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
]

FILES_FROM_BATCH_SIZE = 2


def now() -> datetime:
    return datetime.now(PARIS_TZ) if PARIS_TZ else datetime.now()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push occany_data_tar from cougar to BSC using parallel rsync workers."
    )
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument("-j", "--jobs", type=int, default=8)
    # One or more user@host; batches are handed out round-robin.
    parser.add_argument("--remote", nargs="+", default=[DEFAULT_REMOTE])
    parser.add_argument(
        "--src-dir",
        default=os.path.join(
            os.environ.get("HOME", ""), "daf/datasets_daf/quan/occany_data_tar"
        ),
    )
    parser.add_argument("--dst", default=DEFAULT_DST)
    # Required: no implicit default, so nobody starts a multi-TB push by accident.
    # "--datasets all" takes every dir in the store.
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--list", action="store_true", help="print the store's datasets and exit")
    parser.add_argument("--batch-size", type=int, default=FILES_FROM_BATCH_SIZE)
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--manifest", default="", help="resume manifest path (default: per destination)")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if bool(args.start_time) != bool(args.end_time):
        parser.error("Both --start-time and --end-time must be provided together")

    time_re = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
    if args.start_time:
        if not time_re.match(args.start_time):
            parser.error(f"Invalid --start-time format: {args.start_time}. Expected HH:MM (24-hour).")
        if not time_re.match(args.end_time):
            parser.error(f"Invalid --end-time format: {args.end_time}. Expected HH:MM (24-hour).")
        if args.start_time == args.end_time:
            parser.error("--start-time and --end-time cannot be equal")

    return args


def time_to_seconds(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 3600 + int(m) * 60


def current_seconds_of_day() -> int:
    t = now()
    return t.hour * 3600 + t.minute * 60 + t.second


def is_in_transfer_window(now_s: int, start_s: int, end_s: int) -> bool:
    if start_s < end_s:
        return start_s <= now_s < end_s
    else:
        return now_s >= start_s or now_s < end_s


def wait_for_transfer_window(start_time: str, end_time: str, log_fn):
    if not start_time:
        return
    start_s = time_to_seconds(start_time)
    end_s = time_to_seconds(end_time)
    while True:
        now_s = current_seconds_of_day()
        if is_in_transfer_window(now_s, start_s, end_s):
            return
        if start_s < end_s:
            if now_s < start_s:
                sleep_s = start_s - now_s
            else:
                sleep_s = 86400 - now_s + start_s
        else:
            sleep_s = start_s - now_s
        sleep_s = max(sleep_s, 1)
        log_fn(
            f"Outside allowed transfer window {start_time}-{end_time}. "
            f"Sleeping {sleep_s}s until next window opens."
        )
        time.sleep(sleep_s)


class DoneManifest:
    """Thread-safe tracker of completed files, persisted to disk."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                self.done = {line.strip() for line in f if line.strip()}

    def is_done(self, rel_path: str) -> bool:
        return rel_path in self.done

    def mark_done(self, rel_path: str):
        with self.lock:
            self.done.add(rel_path)
            with open(self.path, "a") as f:
                f.write(rel_path + "\n")


class Logger:
    def __init__(self, master_log_path: str):
        self.master_log_path = master_log_path
        self.lock = threading.Lock()

    def log(self, msg: str):
        line = f"[{now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.lock:
            print(line, flush=True)
            with open(self.master_log_path, "a") as f:
                f.write(line + "\n")


def available_datasets(src_dir: str) -> list:
    """Dataset dirs in the store. Loose files (the pull's done_manifest.txt) are not datasets."""
    return sorted(
        entry for entry in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, entry))
    )


def discover_local_files(src_dir: str, datasets: list) -> list:
    """Walk the selected dataset dirs, returning (path relative to src_dir, size) pairs."""
    files = []
    for dataset in datasets:
        root = os.path.join(src_dir, dataset)
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(full)
                except OSError:  # vanished or unreadable on the NFS mount
                    continue
                files.append((os.path.relpath(full, src_dir), size))
    files.sort()
    return files


def human_bytes(n: int) -> str:
    val = float(n)
    unit = "B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024:
            break
        if unit != "TiB":
            val /= 1024
    return f"{val:.1f} {unit}"


def _drain_pipe(pipe: object, out: list):
    for line in pipe:
        out.append(line)
    pipe.close()


def run_batch(
    batch_id: int,
    files: list,
    src_dir: str,
    dst_dir: str,
    remote: str,
    dry_run: bool,
    start_time: str,
    end_time: str,
    pbar,
    master_log_fn,
    manifest: DoneManifest,
    n_batches: int = 0,
    worker_num: int = 0,
) -> bool:
    if not files:
        return True

    master_log_fn(f"Dispatching batch #{batch_id + 1}/{n_batches} to worker #{worker_num} via {remote}")
    ssh_cmd = "ssh " + " ".join(SSH_OPTS)
    start_s = time_to_seconds(start_time) if start_time else None
    end_s = time_to_seconds(end_time) if end_time else None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix=f"rsync_b{batch_id}_"
    ) as tmp:
        for rel, _ in files:
            tmp.write(rel + "\n")
        filelist_path = tmp.name

    try:
        # --files-from implies --relative, so the dataset subdirs are recreated under dst.
        # dst itself is not implied, hence the mkdir -p in --rsync-path.
        flags = [
            "-a", "--partial", "--inplace",
            "-e", ssh_cmd,
            "--files-from", filelist_path,
        ]
        if dry_run:
            flags.append("--dry-run")
        else:
            flags.append("--rsync-path=mkdir -p '%s' && rsync" % dst_dir)

        cmd = [
            "rsync", *flags,
            f"{src_dir.rstrip('/')}/",
            f"{remote}:{dst_dir.rstrip('/')}/",
        ]

        while True:
            wait_for_transfer_window(start_time, end_time, master_log_fn)

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            stderr_lines = []
            stderr_thread = threading.Thread(
                target=_drain_pipe, args=(proc.stderr, stderr_lines), daemon=True
            )
            stderr_thread.start()

            killed_by_window = threading.Event()
            if start_s is not None:
                def _watchdog(p, ev, ss=start_s, es=end_s):
                    while p.poll() is None:
                        time.sleep(30)
                        if not is_in_transfer_window(current_seconds_of_day(), ss, es):
                            ev.set()
                            p.terminate()
                            return
                threading.Thread(
                    target=_watchdog, args=(proc, killed_by_window), daemon=True
                ).start()

            proc.stdout.read()
            proc.wait()
            stderr_thread.join(timeout=5)

            if killed_by_window.is_set():
                master_log_fn(f"Batch {batch_id + 1}: paused (outside transfer window)")
                continue

            if proc.returncode not in (0, 24):
                stderr = "".join(stderr_lines).strip()
                master_log_fn(f"Batch {batch_id + 1} FAILED (exit code: {proc.returncode})")
                if stderr:
                    master_log_fn(stderr)
                return False

            if not dry_run:  # a dry run must not poison the resume manifest
                for rel, _ in files:
                    manifest.mark_done(rel)
            if pbar is not None:
                pbar.update(sum(size for _, size in files))
            return True
    finally:
        os.unlink(filelist_path)


def main():
    args = parse_args()

    src_dir = os.path.expanduser(args.src_dir)
    if not os.path.isdir(src_dir):
        print(f"Source {src_dir} does not exist (is the DAF store mounted?)", file=sys.stderr)
        sys.exit(1)

    present = available_datasets(src_dir)

    if args.list:
        print(f"{src_dir}\n")
        for dataset in present:
            found = discover_local_files(src_dir, [dataset])
            total = sum(size for _, size in found)
            print(f"  {dataset:<26} {len(found):>7} files  {human_bytes(total):>10}")
        sys.exit(0)

    if not args.datasets:
        print(
            "--datasets is required. Available: " + " ".join(present)
            + "\nUse --list for sizes, or --datasets all for the whole store.",
            file=sys.stderr,
        )
        sys.exit(2)

    datasets = present if args.datasets == ["all"] else list(args.datasets)
    unknown = [d for d in datasets if d not in present]
    if unknown:
        print(
            f"Unknown dataset(s): {' '.join(unknown)}\nAvailable: {' '.join(present)}",
            file=sys.stderr,
        )
        sys.exit(1)

    log_dir = "logs/sync_bsc"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(log_dir, f"push_tars_{timestamp}.log")
    logger = Logger(master_log_path)
    log = logger.log

    # The manifest is per destination: resuming against a different --dst must not
    # skip files that were never sent there.
    manifest_path = args.manifest or os.path.join(
        log_dir, "done_" + re.sub(r"[^A-Za-z0-9]+", "_", args.dst).strip("_") + ".txt"
    )

    log(f"Scanning {src_dir} for {' '.join(datasets)} ...")
    all_files = discover_local_files(src_dir, datasets)

    if not all_files:
        log(f"No files found under {src_dir} for {' '.join(datasets)}")
        sys.exit(1)

    manifest = DoneManifest(manifest_path)
    pending_files = [(rel, size) for rel, size in all_files if not manifest.is_done(rel)]
    already_done = len(all_files) - len(pending_files)
    pending_bytes = sum(size for _, size in pending_files)

    log(
        f"Found {len(all_files)} files total, {already_done} already done, "
        f"{len(pending_files)} remaining ({human_bytes(pending_bytes)})"
    )

    if not pending_files:
        log("Nothing to transfer")
        sys.exit(0)

    batches = [
        pending_files[i : i + args.batch_size]
        for i in range(0, len(pending_files), args.batch_size)
    ]
    n_batches = len(batches)
    jobs = min(args.jobs, n_batches)

    log(f"Split {len(pending_files)} files into {n_batches} batches of up to {args.batch_size}, using {jobs} parallel workers")
    log(f"Source: {src_dir}")
    log(f"Destination: {' '.join(args.remote)}:{args.dst}")
    log(f"Datasets: {' '.join(datasets)}")
    log(f"Manifest: {manifest_path}")
    log(f"Logs directory: {log_dir}")
    if args.start_time:
        log(f"Transfer window: {args.start_time}-{args.end_time}. Transfers pause outside this window.")
    else:
        log("Transfer window: always on")
    if args.dry_run:
        log("Dry run enabled")

    pbar = (
        tqdm(total=pending_bytes, desc="Pushing tars", unit="B", unit_scale=True)
        if tqdm is not None
        else None
    )

    futures = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for batch_id, batch in enumerate(batches):
            worker_num = batch_id % jobs + 1
            fut = pool.submit(
                run_batch,
                batch_id=batch_id,
                files=batch,
                src_dir=src_dir,
                dst_dir=args.dst,
                remote=args.remote[batch_id % len(args.remote)],
                dry_run=args.dry_run,
                start_time=args.start_time,
                end_time=args.end_time,
                pbar=pbar,
                master_log_fn=log,
                manifest=manifest,
                n_batches=n_batches,
                worker_num=worker_num,
            )
            futures[fut] = batch_id + 1

    failed = 0
    for fut in as_completed(futures):
        batch_num = futures[fut]
        if not fut.result():
            log(f"Batch {batch_num} failed")
            failed += 1

    if pbar is not None:
        pbar.close()

    log("=" * 42)
    if failed == 0:
        log("All batches completed successfully")
    else:
        log(f"WARNING: {failed} batch(es) failed")
    log(f"Done manifest: {manifest_path}")
    log(f"Master log: {master_log_path}")

    sys.exit(failed)


if __name__ == "__main__":
    main()
