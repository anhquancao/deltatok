#!/usr/bin/env python3

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


DEFAULT_HOSTS = [
    "vale352205@glogin1.bsc.es",
    "vale352205@glogin2.bsc.es",
    "vale352205@alogin1.bsc.es",
    "vale352205@alogin2.bsc.es",
]
DEFAULT_DATA_DIRS = [
    "once_processed",
    "waymo_processed",
    "vkitti_processed",
    "ddad_processed",
    "pandaset_processed",
    "kitti_processed"
]
IGNORED_CHILD_DIR_NAMES = {"tmp"}


@dataclass(frozen=True)
class SyncTask:
    index: int
    source_path: str
    target_suffix: str
    label: str
    host: str


@dataclass(frozen=True)
class SyncResult:
    task: SyncTask
    success: bool
    output: str = ""


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean-like value, got {value!r}")


def split_shell_words(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return shlex.split(value)


def default_source_root() -> Path:
    scratch = os.environ.get("SCRATCH", os.getcwd())
    return Path(os.environ.get("SOURCE_ROOT", str(Path(scratch) / "data"))).expanduser()


def default_remote_root() -> str:
    return os.environ.get("REMOTE_ROOT", "/gpfs/scratch/ehpc558/quan/occrae_data")


def default_hosts() -> list[str]:
    env_hosts = split_shell_words(os.environ.get("HOSTS"))
    return env_hosts or list(DEFAULT_HOSTS)


def default_data_dirs() -> list[str]:
    env_data_dirs = split_shell_words(os.environ.get("DATA_DIRS"))
    return env_data_dirs or list(DEFAULT_DATA_DIRS)


def default_extra_rsync_args() -> list[str]:
    return split_shell_words(os.environ.get("EXTRA_RSYNC_ARGS"))





def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push local dataset folders to BSC with folder-level multiprocessing and tqdm progress."
    )
    parser.add_argument("--source-root", type=Path, default=default_source_root(), help="Local source root")
    parser.add_argument("--remote-root", type=str, default=default_remote_root(), help="Remote destination root")
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=default_hosts(),
        help="Remote hosts used for round-robin assignment",
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=default_data_dirs(),
        help="Top-level directories under SOURCE_ROOT to scan",
    )
    parser.add_argument(
        "--extra-rsync-args",
        nargs="*",
        default=default_extra_rsync_args(),
        help="Extra rsync arguments appended after the defaults",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=16,
        help="Worker process count. Defaults to one process per sync task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=parse_bool_env("DRY_RUN", False),
        help="Enable rsync --dry-run",
    )
    parser.add_argument(
        "--delete-remote",
        action="store_true",
        default=parse_bool_env("DELETE_REMOTE", False),
        help="Enable rsync --delete and top-level reconciliation",
    )
    return parser


def ensure_rsync_available() -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is not available")


def build_sync_plan(
    source_root: Path,
    data_dirs: Sequence[str],
    hosts: Sequence[str],
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path], list[SyncTask]]:
    available_roots: list[Path] = []
    missing_sources: list[Path] = []
    ignored_sources: list[Path] = []
    ignored_roots: list[Path] = []
    sync_tasks: list[SyncTask] = []
    empty_roots: list[Path] = []

    for data_dir in data_dirs:
        source_dir = source_root / data_dir
        if source_dir.is_dir():
            available_roots.append(source_dir)
        else:
            missing_sources.append(source_dir)

    if not available_roots:
        raise RuntimeError(f"none of the requested data directories exist under {source_root!s}")

    for source_dir in available_roots:
        data_dir = source_dir.name
        child_entries = sorted(source_dir.iterdir(), key=lambda path: path.name)
        syncable_entries = [entry for entry in child_entries if entry.name not in IGNORED_CHILD_DIR_NAMES]

        for ignored_entry in child_entries:
            if ignored_entry.name in IGNORED_CHILD_DIR_NAMES:
                ignored_sources.append(ignored_entry)

        if syncable_entries:
            for child_entry in syncable_entries:
                task_index = len(sync_tasks)
                sync_tasks.append(
                    SyncTask(
                        index=task_index,
                        source_path=str(child_entry),
                        target_suffix=f"{data_dir}/",
                        label=f"{data_dir}/{child_entry.name}",
                        host=hosts[task_index % len(hosts)],
                    )
                )
        elif child_entries:
            ignored_roots.append(source_dir)
        else:
            empty_roots.append(source_dir)
            task_index = len(sync_tasks)
            sync_tasks.append(
                SyncTask(
                    index=task_index,
                    source_path=str(source_dir),
                    target_suffix="",
                    label=f"{data_dir}/",
                    host=hosts[task_index % len(hosts)],
                )
            )

    return available_roots, missing_sources, ignored_sources, ignored_roots, empty_roots, sync_tasks


def base_rsync_args(*, dry_run: bool, delete_remote: bool, remote_root: str, extra_rsync_args: Sequence[str]) -> list[str]:
    args = ["rsync", "-a", "--human-readable"]
    if dry_run:
        args.append("--dry-run")
    if delete_remote:
        args.append("--delete")
    if extra_rsync_args:
        args.extend(extra_rsync_args)
    if not dry_run:
        args.append(f"--rsync-path=mkdir -p {shlex.quote(remote_root)} && rsync")
    return args


def run_subprocess(command: Sequence[str]) -> tuple[bool, str]:
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.returncode == 0, completed.stdout.strip()


def reconcile_remote_roots(
    *,
    available_roots: Sequence[Path],
    hosts: Sequence[str],
    remote_root: str,
    rsync_args: Sequence[str],
) -> None:
    for root_idx, root_source in enumerate(available_roots):
        host = hosts[root_idx % len(hosts)]
        root_name = root_source.name
        remote_target = f"{host}:{remote_root}/{root_name}/"
        command = [
            *rsync_args,
            *[f"--exclude={ignored_name}" for ignored_name in IGNORED_CHILD_DIR_NAMES],
            "--exclude=*/*",
            f"{root_source}/",
            remote_target,
        ]
        print(f"[{host}] Reconciling top-level entries for '{root_name}/' -> '{remote_target}'")
        success, output = run_subprocess(command)
        if not success:
            detail = output or "rsync exited with a non-zero status"
            raise RuntimeError(f"Failed to reconcile {root_name}/ on {host}:\n{detail}")


def sync_one(task: SyncTask, rsync_args: Sequence[str], remote_root: str) -> SyncResult:
    remote_target = f"{task.host}:{remote_root}/{task.target_suffix}"
    command = [
        *rsync_args,
        task.source_path,
        remote_target,
    ]
    success, output = run_subprocess(command)
    return SyncResult(task=task, success=success, output=output)


def sync_one_star(args: tuple[SyncTask, Sequence[str], str]) -> SyncResult:
    return sync_one(*args)


def main() -> int:
    args = get_args_parser().parse_args()

    source_root = args.source_root.expanduser()
    remote_root = args.remote_root
    hosts = list(args.hosts)
    data_dirs = list(args.data_dirs)
    extra_rsync_args = list(args.extra_rsync_args)

    if not source_root.is_dir():
        raise RuntimeError(f"SOURCE_ROOT '{source_root}' does not exist")
    if not hosts:
        raise RuntimeError("HOSTS is empty")
    if not data_dirs:
        raise RuntimeError("DATA_DIRS is empty")

    ensure_rsync_available()

    available_roots, missing_sources, ignored_sources, ignored_roots, empty_roots, sync_tasks = build_sync_plan(source_root, data_dirs, hosts)
    rsync_args = base_rsync_args(
        dry_run=args.dry_run,
        delete_remote=args.delete_remote,
        remote_root=remote_root,
        extra_rsync_args=extra_rsync_args,
    )

    print(f"SOURCE_ROOT: {source_root}")
    print(f"REMOTE_ROOT: {remote_root}")
    print(f"DATA_DIRS: {' '.join(data_dirs)}")
    print(f"AVAILABLE_HOSTS: {' '.join(hosts)}")
    print(f"SYNC_PATHS: {len(sync_tasks)}")

    for missing_source in missing_sources:
        print(f"WARNING: skipping missing source directory '{missing_source}'", file=sys.stderr)

    for ignored_source in ignored_sources:
        print(f"WARNING: skipping ignored path '{ignored_source}'", file=sys.stderr)

    for ignored_root in ignored_roots:
        print(f"WARNING: skipping '{ignored_root}' because it only contains ignored paths", file=sys.stderr)

    for empty_root in empty_roots:
        print(f"WARNING: '{empty_root}' is empty; syncing the root directory itself", file=sys.stderr)

    if not sync_tasks:
        print("No syncable paths were found after applying ignores; nothing to do.")
        return 0

    if args.delete_remote:
        reconcile_remote_roots(
            available_roots=available_roots,
            hosts=hosts,
            remote_root=remote_root,
            rsync_args=rsync_args,
        )

    worker_count = args.processes if args.processes is not None else len(sync_tasks)
    worker_count = max(1, min(worker_count, len(sync_tasks)))

    worker_inputs = [(task, rsync_args, remote_root) for task in sync_tasks]
    failures: list[SyncResult] = []

    with mp.get_context("spawn").Pool(processes=worker_count) as pool:
        progress_bar = tqdm(total=len(sync_tasks), desc="Pushing folders", unit="folder")
        try:
            for result in pool.imap_unordered(sync_one_star, worker_inputs):
                progress_bar.update(1)
                if result.success:
                    tqdm.write(f"[done] {result.task.label} -> {result.task.host}")
                else:
                    failures.append(result)
                    tqdm.write(f"[failed] {result.task.label} -> {result.task.host}")
                progress_bar.set_postfix(success=progress_bar.n - len(failures), failed=len(failures))
        finally:
            progress_bar.close()

    if failures:
        error_lines = [
            f"{result.task.label} ({result.task.host})\n{result.output or 'rsync exited with a non-zero status'}"
            for result in failures
        ]
        raise RuntimeError("One or more rsync tasks failed:\n\n" + "\n\n".join(error_lines))

    print(f"All {len(sync_tasks)} paths synced successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
