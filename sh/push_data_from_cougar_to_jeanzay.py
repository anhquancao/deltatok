#!/usr/bin/env python3
"""Push dataset folders from cougar (local) to jean-zay, with scene-level
multiprocessing and a tqdm progress bar.

Run *on cougar*. Mirrors the structure of
sh/pull_data_from_karolina_to_cluster.py but in the opposite direction:
the *source* is local (cougar's
/home/acao/daf/datasets_daf/quan/occany_preprocess_data) and the
*destination* is remote
(jean-zay:/lustre/fsn1/projects/rech/trg/uyl37fq/occany_data).
Each first-level entry under each requested data dir (typically a scene
directory) becomes its own rsync task. With --delete-remote, a final
non-recursive rsync per data dir removes any top-level entry on the
remote destination that no longer exists locally.

Jean-zay purges files older than 30 days. We hand rsync `--no-times`
so destination files inherit jean-zay's current time on creation rather
than the (potentially ancient) cougar mtime, and pair it with
`--size-only` so re-runs skip unchanged content instead of re-transferring
on every mtime mismatch. Caveat: this only stamps "now" on files that
actually get transferred — files already on jean-zay that rsync skips
(same size) keep their old remote mtime. Re-pushing new content won't
extend the lifetime of older skipped files; for that you'd need a
remote `ssh jean-zay 'find ... -exec touch'` pass.

Requires an SSH alias `jean-zay` configured on cougar.
"""

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


DEFAULT_HOSTS = ["jean-zay"]
DEFAULT_DATA_DIRS = [
    "waymo_processed",
    "vkitti_processed",
    "ddad_processed",
    "pandaset_processed",
    "once_processed",
    "kitti_processed",
    "occ3d_nuscenes_processed",
]
IGNORED_CHILD_DIR_NAMES = {"tmp"}

DEFAULT_SSH_CMD = (
    "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
)


@dataclass(frozen=True)
class SyncTask:
    index: int
    source_path: str          # absolute path on cougar
    remote_sub_dir: str       # sub-dir under remote_root (e.g. "kitti_processed")
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
    raise ValueError(f"Environment variable {name} must be boolean-like, got {value!r}")


def split_shell_words(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return shlex.split(value)


def default_local_root() -> Path:
    return Path(os.environ.get(
        "LOCAL_ROOT",
        "/home/acao/daf/datasets_daf/quan/occany_preprocess_data",
    )).expanduser()


def default_remote_root() -> str:
    return os.environ.get(
        "REMOTE_ROOT",
        "/lustre/fsn1/projects/rech/trg/uyl37fq/occany_data",
    )


def default_hosts() -> list[str]:
    env_hosts = split_shell_words(os.environ.get("HOSTS"))
    return env_hosts or list(DEFAULT_HOSTS)


def default_data_dirs() -> list[str]:
    env_data_dirs = split_shell_words(os.environ.get("DATA_DIRS"))
    return env_data_dirs or list(DEFAULT_DATA_DIRS)


def default_extra_rsync_args() -> list[str]:
    return split_shell_words(os.environ.get("EXTRA_RSYNC_ARGS"))


def default_ssh_cmd() -> str:
    return os.environ.get("SSH_CMD", DEFAULT_SSH_CMD)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push local dataset folders from cougar to jean-zay with "
                    "scene-level multiprocessing and tqdm progress."
    )
    parser.add_argument("--local-root", type=Path, default=default_local_root(),
                        help="Local source root on cougar")
    parser.add_argument("--remote-root", type=str, default=default_remote_root(),
                        help="Remote destination root on jean-zay")
    parser.add_argument("--hosts", nargs="+", default=default_hosts(),
                        help="Remote hosts used for round-robin assignment.")
    parser.add_argument("--data-dirs", nargs="+", default=default_data_dirs(),
                        help="Top-level directories under LOCAL_ROOT to push")
    parser.add_argument("--ssh-cmd", type=str, default=default_ssh_cmd(),
                        help="SSH command passed to rsync via -e")
    parser.add_argument("--extra-rsync-args", nargs="*",
                        default=default_extra_rsync_args(),
                        help="Extra rsync arguments appended after the defaults")
    parser.add_argument("--processes", type=int, default=16,
                        help="Worker process count")
    parser.add_argument("--dry-run", action="store_true",
                        default=parse_bool_env("DRY_RUN", False),
                        help="Enable rsync --dry-run")
    parser.add_argument("--delete-remote", action="store_true",
                        default=parse_bool_env("DELETE_REMOTE", False),
                        help="Enable top-level reconciliation: delete remote "
                             "entries no longer present locally")
    return parser


def ensure_rsync_available() -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is not available")


def build_sync_plan(
    *,
    local_root: Path,
    data_dirs: Sequence[str],
    hosts: Sequence[str],
) -> tuple[list[str], list[str], list[str], list[SyncTask]]:
    """Enumerate work units across all requested data dirs.

    Returns: (available_dirs, missing_dirs, ignored_entries, sync_tasks).
    """
    available_dirs: list[str] = []
    missing_dirs: list[str] = []
    ignored_entries: list[str] = []
    sync_tasks: list[SyncTask] = []

    for data_dir in data_dirs:
        local_data_dir = local_root / data_dir
        if not local_data_dir.is_dir():
            missing_dirs.append(data_dir)
            continue

        available_dirs.append(data_dir)
        entries = sorted(p.name for p in local_data_dir.iterdir())
        if not entries:
            continue

        for entry in entries:
            if entry in IGNORED_CHILD_DIR_NAMES:
                ignored_entries.append(f"{data_dir}/{entry}")
                continue
            task_index = len(sync_tasks)
            sync_tasks.append(SyncTask(
                index=task_index,
                source_path=str(local_data_dir / entry),
                remote_sub_dir=data_dir,
                label=f"{data_dir}/{entry}",
                host=hosts[task_index % len(hosts)],
            ))

    if not available_dirs:
        raise RuntimeError(
            f"none of the requested data directories exist under {local_root}"
        )

    return available_dirs, missing_dirs, ignored_entries, sync_tasks


def base_rsync_args(
    *, dry_run: bool, ssh_cmd: str, extra_rsync_args: Sequence[str],
) -> list[str]:
    # -W: skip rolling-checksum delta scan (pointless for initial bulk push).
    # --partial --inplace: resumable, no temp-file churn.
    # --no-times: destination files inherit jean-zay's "now" mtime instead
    # of cougar's (possibly ancient) source mtime -- dodges the 30-day purge.
    # --size-only: required companion to --no-times so re-runs skip on size
    # alone; otherwise the now/source mtime mismatch would force re-transfer.
    # No -z: dataset content is already compressed.
    args = [
        "rsync", "-aHW", "--human-readable",
        "--info=stats1",
        "--no-times",
        "--size-only",
        "--partial", "--inplace",
        "-e", ssh_cmd,
    ]
    for name in IGNORED_CHILD_DIR_NAMES:
        args.append(f"--exclude={name}/")
        args.append(f"--exclude={name}/**")
    args.extend(["--exclude=*.tmp", "--exclude=.tmp/", "--exclude=.tmp/**"])
    if dry_run:
        args.append("--dry-run")
    if extra_rsync_args:
        args.extend(extra_rsync_args)
    return args


def run_subprocess(command: Sequence[str]) -> tuple[bool, str]:
    completed = subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return completed.returncode == 0, completed.stdout.strip()


def remote_target_with_mkdir(
    *, host: str, remote_root: str, sub_dir: str,
) -> tuple[str, str]:
    """Return (remote_target_url, rsync_path_arg).

    Uses --rsync-path so rsync creates the destination directory on
    jean-zay before transferring, avoiding a separate ssh round-trip.
    """
    remote_dir = f"{remote_root}/{sub_dir}" if sub_dir else remote_root
    remote_target = f"{host}:{remote_dir}/"
    rsync_path = f"--rsync-path=mkdir -p {shlex.quote(remote_dir)} && rsync"
    return remote_target, rsync_path


def reconcile_remote_roots(
    *,
    available_dirs: Sequence[str],
    hosts: Sequence[str],
    remote_root: str,
    local_root: Path,
    rsync_args: Sequence[str],
    dry_run: bool,
) -> None:
    """Drop remote top-level entries that no longer exist locally.

    Uses `rsync -d --delete` (no recursion) so it only touches the top
    level of each data dir, leaving subdir contents (already handled by
    the parallel pass) alone.
    """
    for idx, data_dir in enumerate(available_dirs):
        host = hosts[idx % len(hosts)]
        local_source = f"{local_root / data_dir}/"
        remote_target, rsync_path = remote_target_with_mkdir(
            host=host, remote_root=remote_root, sub_dir=data_dir,
        )
        command = [
            *rsync_args, "-d", "--delete", rsync_path,
            *[f"--exclude={name}" for name in IGNORED_CHILD_DIR_NAMES],
            local_source, remote_target,
        ]
        print(f"[{host}] Reconciling top-level entries: "
              f"'{local_source}' -> '{remote_target}'")
        success, output = run_subprocess(command)
        if not success:
            detail = output or "rsync exited with a non-zero status"
            raise RuntimeError(f"Failed to reconcile {data_dir} on {host}:\n{detail}")


def sync_one(task: SyncTask, rsync_args: Sequence[str], remote_root: str) -> SyncResult:
    remote_target, rsync_path = remote_target_with_mkdir(
        host=task.host, remote_root=remote_root, sub_dir=task.remote_sub_dir,
    )
    command = [*rsync_args, rsync_path, task.source_path, remote_target]
    success, output = run_subprocess(command)
    return SyncResult(task=task, success=success, output=output)


def sync_one_star(args: tuple[SyncTask, Sequence[str], str]) -> SyncResult:
    return sync_one(*args)


def main() -> int:
    args = get_args_parser().parse_args()

    local_root = args.local_root.expanduser()
    hosts = list(args.hosts)
    data_dirs = list(args.data_dirs)
    extra_rsync_args = list(args.extra_rsync_args)
    ssh_cmd = args.ssh_cmd
    remote_root = args.remote_root

    if not local_root.is_dir():
        raise RuntimeError(f"LOCAL_ROOT '{local_root}' does not exist")
    if not hosts:
        raise RuntimeError("HOSTS is empty")
    if not data_dirs:
        raise RuntimeError("DATA_DIRS is empty")
    if not remote_root:
        raise RuntimeError("REMOTE_ROOT is empty")

    ensure_rsync_available()

    available_dirs, missing_dirs, ignored_entries, sync_tasks = build_sync_plan(
        local_root=local_root,
        data_dirs=data_dirs,
        hosts=hosts,
    )

    rsync_args = base_rsync_args(
        dry_run=args.dry_run,
        ssh_cmd=ssh_cmd,
        extra_rsync_args=extra_rsync_args,
    )

    print(f"LOCAL_ROOT:  {local_root}")
    print(f"REMOTE_ROOT: {hosts[0]}:{remote_root}")
    print(f"DATA_DIRS:   {' '.join(data_dirs)}")
    print(f"HOSTS:       {' '.join(hosts)}")
    print(f"SYNC_TASKS:  {len(sync_tasks)}")

    for missing_dir in missing_dirs:
        print(f"WARNING: skipping missing local dir '{missing_dir}'", file=sys.stderr)
    for ignored_entry in ignored_entries:
        print(f"WARNING: skipping ignored entry '{ignored_entry}'", file=sys.stderr)

    if not sync_tasks:
        print("No syncable entries were found; nothing to do.")
        return 0

    worker_count = max(1, min(args.processes, len(sync_tasks)))
    worker_inputs = [(task, rsync_args, remote_root) for task in sync_tasks]
    failures: list[SyncResult] = []

    with mp.get_context("spawn").Pool(processes=worker_count) as pool:
        bar = tqdm(total=len(sync_tasks), desc="Pushing folders", unit="folder")
        try:
            for result in pool.imap_unordered(sync_one_star, worker_inputs):
                bar.update(1)
                if result.success:
                    tqdm.write(f"[done] {result.task.label} -> {result.task.host}")
                else:
                    failures.append(result)
                    tqdm.write(f"[failed] {result.task.label} -> {result.task.host}")
                bar.set_postfix(success=bar.n - len(failures), failed=len(failures))
        finally:
            bar.close()

    if args.delete_remote:
        reconcile_remote_roots(
            available_dirs=available_dirs,
            hosts=hosts,
            remote_root=remote_root,
            local_root=local_root,
            rsync_args=rsync_args,
            dry_run=args.dry_run,
        )

    if failures:
        error_lines = [
            f"{r.task.label} ({r.task.host})\n"
            f"{r.output or 'rsync exited with a non-zero status'}"
            for r in failures
        ]
        raise RuntimeError("One or more rsync tasks failed:\n\n" + "\n\n".join(error_lines))

    print(f"All {len(sync_tasks)} entries synced successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
