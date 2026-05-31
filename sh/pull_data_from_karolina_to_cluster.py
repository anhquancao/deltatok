#!/usr/bin/env python3
"""Pull dataset folders from Karolina to a local destination, with
scene-level multiprocessing and a tqdm progress bar.

Mirrors the structure of sh/push_data_to_bsc.py but in the opposite
direction: the *source* is remote (karolina:$SCRATCH/data/...) and the
*destination* is local. Each first-level entry under each requested data
dir (typically a scene directory) becomes its own rsync task. With
--delete-local, a final non-recursive rsync per data dir removes any
top-level entry on the local destination that no longer exists on
karolina.

Run on the local box that has an SSH alias `karolina` configured.
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


DEFAULT_HOSTS = ["karolina"]
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

# Karolina's sshd only offers aes256-ctr / aes256-gcm. aes256-gcm uses
# AES-NI on modern CPUs and is the fastest of the two.
DEFAULT_SSH_CMD = (
    "ssh -c aes256-gcm@openssh.com,aes256-ctr "
    "-o ServerAliveInterval=30 -o ServerAliveCountMax=3"
)


@dataclass(frozen=True)
class SyncTask:
    index: int
    remote_path: str          # absolute path on Karolina
    local_target_dir: str     # local parent directory to drop the entry into
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
    return os.environ.get("REMOTE_ROOT", "")  # resolved on Karolina if empty


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
        description="Pull local dataset folders from Karolina with scene-level "
                    "multiprocessing and tqdm progress."
    )
    parser.add_argument("--local-root", type=Path, default=default_local_root(),
                        help="Local destination root")
    parser.add_argument("--remote-root", type=str, default=default_remote_root(),
                        help="Remote source root on Karolina. "
                             "If empty, resolved as $SCRATCH/data via SSH.")
    parser.add_argument("--hosts", nargs="+", default=default_hosts(),
                        help="Remote hosts used for round-robin assignment.")
    parser.add_argument("--data-dirs", nargs="+", default=default_data_dirs(),
                        help="Top-level directories under REMOTE_ROOT to pull")
    parser.add_argument("--ssh-cmd", type=str, default=default_ssh_cmd(),
                        help="SSH command passed to rsync via -e")
    parser.add_argument("--extra-rsync-args", nargs="*",
                        default=default_extra_rsync_args(),
                        help="Extra rsync arguments appended after the defaults")
    parser.add_argument("--processes", type=int, default=8,
                        help="Worker process count")
    parser.add_argument("--dry-run", action="store_true",
                        default=parse_bool_env("DRY_RUN", False),
                        help="Enable rsync --dry-run")
    parser.add_argument("--delete-local", action="store_true",
                        default=parse_bool_env("DELETE_LOCAL", False),
                        help="Enable top-level reconciliation: delete local "
                             "entries no longer present on Karolina")
    return parser


def ensure_rsync_available() -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is not available")


def ssh_capture(ssh_cmd: str, host: str, remote_cmd: str) -> str:
    """Run a remote command via ssh and return stdout (stripped)."""
    full = shlex.split(ssh_cmd) + [host, remote_cmd]
    completed = subprocess.run(
        full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    return completed.stdout.strip()


def resolve_remote_root(ssh_cmd: str, host: str, override: str) -> str:
    if override:
        return override
    out = ssh_capture(ssh_cmd, host, 'echo "$SCRATCH/data"')
    if not out:
        raise RuntimeError(f"could not resolve $SCRATCH/data on {host}")
    return out


def list_remote_entries(ssh_cmd: str, host: str, remote_dir: str) -> list[str]:
    """Return the names of first-level entries (files or dirs) under remote_dir.

    Returns an empty list if remote_dir is missing.
    """
    quoted = shlex.quote(remote_dir)
    cmd = (
        f"if [ -d {quoted} ]; then "
        f"find {quoted} -mindepth 1 -maxdepth 1 -printf '%f\\n'; "
        f"fi"
    )
    out = ssh_capture(ssh_cmd, host, cmd)
    return [name for name in out.splitlines() if name]


def build_sync_plan(
    *,
    ssh_cmd: str,
    primary_host: str,
    remote_root: str,
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
        remote_data_dir = f"{remote_root}/{data_dir}"
        entries = list_remote_entries(ssh_cmd, primary_host, remote_data_dir)
        if not entries:
            # could be missing OR empty -- check existence explicitly
            exists = ssh_capture(
                ssh_cmd, primary_host,
                f'[ -d {shlex.quote(remote_data_dir)} ] && echo yes || echo no',
            )
            if exists != "yes":
                missing_dirs.append(data_dir)
                continue
            # exists but empty -- nothing to do for this dir
            available_dirs.append(data_dir)
            continue

        available_dirs.append(data_dir)
        local_target_dir = str(local_root / data_dir)

        for entry in sorted(entries):
            if entry in IGNORED_CHILD_DIR_NAMES:
                ignored_entries.append(f"{data_dir}/{entry}")
                continue
            task_index = len(sync_tasks)
            sync_tasks.append(SyncTask(
                index=task_index,
                remote_path=f"{remote_data_dir}/{entry}",
                local_target_dir=local_target_dir,
                label=f"{data_dir}/{entry}",
                host=hosts[task_index % len(hosts)],
            ))

    if not available_dirs:
        raise RuntimeError(
            f"none of the requested data directories exist under "
            f"{primary_host}:{remote_root}"
        )

    return available_dirs, missing_dirs, ignored_entries, sync_tasks


def base_rsync_args(
    *, dry_run: bool, ssh_cmd: str, extra_rsync_args: Sequence[str],
) -> list[str]:
    # -W: skip rolling-checksum delta scan (pointless for initial bulk pull).
    # --partial --inplace: resumable, no temp-file churn.
    # No -z: dataset content is already compressed.
    args = [
        "rsync", "-aHW", "--human-readable",
        "--info=stats1",
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


def reconcile_local_roots(
    *,
    available_dirs: Sequence[str],
    hosts: Sequence[str],
    remote_root: str,
    local_root: Path,
    rsync_args: Sequence[str],
    dry_run: bool,
) -> None:
    """Drop local top-level entries that no longer exist on Karolina.

    Uses `rsync -d --delete` (no recursion) so it only touches the top
    level of each data dir, leaving subdir contents (already handled by
    the parallel pass) alone.
    """
    for idx, data_dir in enumerate(available_dirs):
        host = hosts[idx % len(hosts)]
        remote_source = f"{host}:{remote_root}/{data_dir}/"
        local_target = f"{local_root / data_dir}/"
        command = [
            *rsync_args, "-d", "--delete",
            *[f"--exclude={name}" for name in IGNORED_CHILD_DIR_NAMES],
            remote_source, local_target,
        ]
        print(f"[{host}] Reconciling top-level entries: "
              f"'{remote_source}' -> '{local_target}'")
        success, output = run_subprocess(command)
        if not success:
            detail = output or "rsync exited with a non-zero status"
            raise RuntimeError(f"Failed to reconcile {data_dir} on {host}:\n{detail}")


def sync_one(task: SyncTask, rsync_args: Sequence[str]) -> SyncResult:
    remote_source = f"{task.host}:{task.remote_path}"
    # Ensure parent exists on local side.
    os.makedirs(task.local_target_dir, exist_ok=True)
    command = [*rsync_args, remote_source, f"{task.local_target_dir}/"]
    success, output = run_subprocess(command)
    return SyncResult(task=task, success=success, output=output)


def sync_one_star(args: tuple[SyncTask, Sequence[str]]) -> SyncResult:
    return sync_one(*args)


def main() -> int:
    args = get_args_parser().parse_args()

    local_root = args.local_root.expanduser()
    hosts = list(args.hosts)
    data_dirs = list(args.data_dirs)
    extra_rsync_args = list(args.extra_rsync_args)
    ssh_cmd = args.ssh_cmd

    if not hosts:
        raise RuntimeError("HOSTS is empty")
    if not data_dirs:
        raise RuntimeError("DATA_DIRS is empty")

    ensure_rsync_available()
    local_root.mkdir(parents=True, exist_ok=True)

    primary_host = hosts[0]
    remote_root = resolve_remote_root(ssh_cmd, primary_host, args.remote_root)

    available_dirs, missing_dirs, ignored_entries, sync_tasks = build_sync_plan(
        ssh_cmd=ssh_cmd,
        primary_host=primary_host,
        remote_root=remote_root,
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
    print(f"REMOTE_ROOT: {primary_host}:{remote_root}")
    print(f"DATA_DIRS:   {' '.join(data_dirs)}")
    print(f"HOSTS:       {' '.join(hosts)}")
    print(f"SYNC_TASKS:  {len(sync_tasks)}")

    for missing_dir in missing_dirs:
        print(f"WARNING: skipping missing remote dir '{missing_dir}'", file=sys.stderr)
    for ignored_entry in ignored_entries:
        print(f"WARNING: skipping ignored entry '{ignored_entry}'", file=sys.stderr)

    if not sync_tasks:
        print("No syncable entries were found; nothing to do.")
        return 0

    worker_count = max(1, min(args.processes, len(sync_tasks)))
    worker_inputs = [(task, rsync_args) for task in sync_tasks]
    failures: list[SyncResult] = []

    with mp.get_context("spawn").Pool(processes=worker_count) as pool:
        bar = tqdm(total=len(sync_tasks), desc="Pulling folders", unit="folder")
        try:
            for result in pool.imap_unordered(sync_one_star, worker_inputs):
                bar.update(1)
                if result.success:
                    tqdm.write(f"[done] {result.task.label} <- {result.task.host}")
                else:
                    failures.append(result)
                    tqdm.write(f"[failed] {result.task.label} <- {result.task.host}")
                bar.set_postfix(success=bar.n - len(failures), failed=len(failures))
        finally:
            bar.close()

    if args.delete_local:
        reconcile_local_roots(
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
