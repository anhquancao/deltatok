#!/usr/bin/env python3
"""Watch the repo for changes and rsync to Karolina on each event."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

REPO_ROOT = Path(__file__).resolve().parent.parent

RSYNC_EXCLUDES = [
    ".git/",
    "__pycache__/",
    "**/__pycache__/",
    "*.pyc",
    "results/",
    "slurm/*.out",
    "slurm/*.err",
    "slurm/output/",
    "outputs/",
    "pretrained_ckpts/",
    "checkpoint/",
    "**/checkpoint/",
    "checkpoints/",
    "**/checkpoints/",
]

IGNORE_PATTERNS = {".git", "__pycache__", ".pyc", "results", "slurm/output", "outputs",
                   "pretrained_ckpts", "checkpoint", "checkpoints"}


def should_ignore(path: str) -> bool:
    for pattern in IGNORE_PATTERNS:
        if pattern in path:
            return True
    return False


def build_rsync_cmd(src: str, dst: str, *, dry_run: bool = False) -> list[str]:
    cmd = [
        "rsync", "-az", "--delete",
        "--partial",
        "--info=stats2,progress2",
        "--human-readable",
        "--prune-empty-dirs",
    ]
    for exc in RSYNC_EXCLUDES:
        cmd += ["--exclude", exc]
    if dry_run:
        cmd += ["--dry-run", "-v"]
    cmd += [src, dst]
    return cmd


class SyncHandler(FileSystemEventHandler):
    def __init__(self, rsync_cmd: list[str], debounce: float = 1.0):
        super().__init__()
        self.rsync_cmd = rsync_cmd
        self.debounce = debounce
        self._last_sync = 0.0
        self._pending = False

    def on_any_event(self, event):
        if should_ignore(event.src_path):
            return
        self._pending = True

    def sync_if_pending(self):
        if not self._pending:
            return
        now = time.monotonic()
        if now - self._last_sync < self.debounce:
            return
        self._pending = False
        self._last_sync = now
        print(f"\n[sync] change detected, syncing...")
        subprocess.run(self.rsync_cmd)


def main():
    parser = argparse.ArgumentParser(description="Watch repo and rsync to Karolina.")
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument("--remote", default="karolina")
    parser.add_argument("--remote-dir", default="/home/it4i-anhquan")
    parser.add_argument("--debounce", type=float, default=1.0,
                        help="Seconds to wait after last change before syncing (default: 1.0)")
    args = parser.parse_args()

    src = str(REPO_ROOT) + "/"
    dst = f"{args.remote}:{args.remote_dir}/OccAny/"
    rsync_cmd = build_rsync_cmd(src, dst, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run enabled")

    # Initial sync
    print(f"Initial sync: {src} -> {dst}")
    subprocess.run(rsync_cmd)

    handler = SyncHandler(rsync_cmd, debounce=args.debounce)
    observer = Observer()
    observer.schedule(handler, str(REPO_ROOT), recursive=True)
    observer.start()
    print(f"Watching {REPO_ROOT} for changes (debounce={args.debounce}s)...")

    try:
        while True:
            handler.sync_if_pending()
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()