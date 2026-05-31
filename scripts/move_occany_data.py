#!/usr/bin/env python3
"""Move occany_data from ehpc558 to ehpc793 via parallel copy + unlink.

The two paths sit on different filesystems, so rename(2) is unusable and we
must do real I/O. This script uses a thread pool to parallelise at the file
level (vs the original `xargs -P 16 rsync`, which parallelises at the
top-level subdir and goes single-threaded once one subdir dominates).

Resumes a half-finished `rsync --remove-source-files` run: files already at
the destination have the source verified by size and unlinked (logged); files
still at the source only are copied across and unlinked.


Crash-safety: each file is copied to a `<name>.part.<pid>.<tid>` sibling and
committed to its final name with `os.link` (which fails on EEXIST rather
than overwriting), then the source is unlinked. A killed run never leaves a
partial file at the final path. Orphan `.part.*` files from previous runs
are swept at startup.
"""
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

SRC = Path("/gpfs/scratch/ehpc558/quan/occrae_data")
DST = Path("/gpfs/scratch/ehpc793/occany_data")
CONFLICT_LOG = Path(__file__).with_name("move_occany_data.conflicts.log")
SRC_REMAINS_LOG = Path(__file__).with_name("move_occany_data.src_remains.log")
WORKERS = 16
MAX_IN_FLIGHT = WORKERS * 4
PROGRESS_EVERY = 200


def transfer_one(src_file: Path, dst_file: Path) -> dict:
    """Move src_file to dst_file.

    Returns a dict with:
        status: 'moved' | 'conflict' | 'moved_src_remains'
        size:   bytes (0 for conflict)
        src_size, dst_size: only when status == 'conflict'
        unlink_err: only when status == 'moved_src_remains'
    """
    if dst_file.exists():
        try:
            src_size = src_file.stat().st_size
            dst_size = dst_file.stat().st_size
        except OSError:
            src_size = dst_size = -1
        if src_size >= 0 and src_size == dst_size:
            try:
                os.unlink(src_file)
            except OSError:
                pass
        return {
            "status": "conflict",
            "size": 0,
            "src_size": src_size,
            "dst_size": dst_size,
        }

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_file.with_name(
        f"{dst_file.name}.part.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        shutil.copy2(src_file, tmp)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    try:
        os.link(tmp, dst_file)
    except FileExistsError:
        try:
            tmp.unlink()
        except OSError:
            pass
        try:
            src_size = src_file.stat().st_size
            dst_size = dst_file.stat().st_size
        except OSError:
            src_size = dst_size = -1
        if src_size >= 0 and src_size == dst_size:
            try:
                os.unlink(src_file)
            except OSError:
                pass
        return {
            "status": "conflict",
            "size": 0,
            "src_size": src_size,
            "dst_size": dst_size,
        }
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

    try:
        size = os.stat(dst_file).st_size
    except OSError:
        size = 0
    try:
        tmp.unlink()
    except OSError:
        pass

    try:
        os.unlink(src_file)
    except OSError as e:
        return {"status": "moved_src_remains", "size": size, "unlink_err": str(e)}

    return {"status": "moved", "size": size}


def sweep_orphan_parts(dst_root: Path) -> int:
    """Remove leftover *.part.* files from previous killed runs."""
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(dst_root):
        for name in filenames:
            if ".part." in name:
                try:
                    os.unlink(os.path.join(dirpath, name))
                    removed += 1
                except OSError:
                    pass
    return removed


def iter_pairs(src_root: Path, dst_root: Path):
    """Yield (src_file, dst_file) pairs by streaming os.walk."""
    for dirpath, _dirnames, filenames in os.walk(src_root):
        src_dir = Path(dirpath)
        rel_dir = src_dir.relative_to(src_root)
        dst_dir = dst_root / rel_dir
        for name in filenames:
            yield src_dir / name, dst_dir / name


def count_files(src_root: Path) -> int:
    n = 0
    for _dirpath, _dirnames, filenames in os.walk(src_root):
        n += len(filenames)
    return n


def main() -> int:
    if not SRC.is_dir():
        print(f"SRC does not exist or is not a directory: {SRC}", file=sys.stderr)
        return 2
    if not DST.is_dir():
        print(f"DST does not exist or is not a directory: {DST}", file=sys.stderr)
        return 2

    print(f"sweeping orphan .part.* files under {DST} ...", flush=True)
    t_sweep = time.time()
    swept = sweep_orphan_parts(DST)
    print(f"  removed {swept} orphan(s) in {time.time() - t_sweep:.1f}s", flush=True)

    print("counting source files...", flush=True)
    t_count = time.time()
    total_files = count_files(SRC)
    print(f"  {total_files} files in {time.time() - t_count:.1f}s", flush=True)
    if total_files == 0:
        print("nothing to move")
        return 0

    moved = 0
    conflicts = 0
    src_remains = 0
    errors = 0
    bytes_moved = 0
    touched_parents: set[Path] = set()
    t0 = time.time()
    is_tty = sys.stdout.isatty()
    counter_lock = threading.Lock()

    def render_progress(
        done: int, m: int, c: int, sr: int, e: int, bm: int, final: bool = False
    ) -> None:
        dt = time.time() - t0
        rate = done / dt if dt > 0 else 0.0
        mb_s = (bm / dt / 1e6) if dt > 0 else 0.0
        pct = 100.0 * done / total_files if total_files else 100.0
        remaining = (total_files - done) / rate if rate > 0 else 0.0
        line = (
            f"  {done}/{total_files} ({pct:5.1f}%) "
            f"moved={m} conflicts={c} src_remains={sr} errors={e} "
            f"[{rate:7.0f} files/s, {mb_s:7.1f} MB/s, "
            f"ETA {remaining/60:5.1f} min]"
        )
        end = "\n" if final or not is_tty else ""
        sep = "\r" if is_tty else ""
        print(f"{sep}{line}", end=end, flush=True)

    def handle_future(fut, pair) -> None:
        nonlocal moved, conflicts, src_remains, errors, bytes_moved
        src_file, dst_file = pair
        try:
            result = fut.result()
        except Exception as e:
            with counter_lock:
                if is_tty:
                    print("", flush=True)
                print(
                    f"ERROR moving {src_file} -> {dst_file}: {e}", file=sys.stderr
                )
                errors += 1
            return

        status = result["status"]
        if status == "moved":
            with counter_lock:
                moved += 1
                bytes_moved += result["size"]
        elif status == "moved_src_remains":
            with counter_lock:
                src_remains += 1
                bytes_moved += result["size"]
                src_remains_fp.write(
                    f"{src_file}\t{dst_file}\t{result['unlink_err']}\n"
                )
        elif status == "conflict":
            with counter_lock:
                conflicts += 1
                conflict_fp.write(
                    f"{src_file}\t{dst_file}\t"
                    f"src_size={result['src_size']}\t"
                    f"dst_size={result['dst_size']}\n"
                )

    with CONFLICT_LOG.open("w") as conflict_fp, SRC_REMAINS_LOG.open(
        "w"
    ) as src_remains_fp, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        in_flight: dict = {}

        def drain(min_done: int = 1) -> None:
            done_set, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done_set:
                pair = in_flight.pop(fut)
                handle_future(fut, pair)
            with counter_lock:
                done = moved + conflicts + src_remains + errors
                m, c, sr, e, bm = moved, conflicts, src_remains, errors, bytes_moved
            if done % PROGRESS_EVERY < len(done_set) or done == total_files:
                render_progress(done, m, c, sr, e, bm)

        for src_file, dst_file in iter_pairs(SRC, DST):
            touched_parents.add(src_file.parent)
            while len(in_flight) >= MAX_IN_FLIGHT:
                drain()
            fut = pool.submit(transfer_one, src_file, dst_file)
            in_flight[fut] = (src_file, dst_file)

        while in_flight:
            drain()

    with counter_lock:
        done = moved + conflicts + src_remains + errors
        m, c, sr, e, bm = moved, conflicts, src_remains, errors, bytes_moved
    render_progress(done, m, c, sr, e, bm, final=True)

    dirs_removed = 0
    for parent in sorted(touched_parents, key=lambda p: len(p.parts), reverse=True):
        cur = parent
        while cur != SRC and SRC in cur.parents:
            try:
                cur.rmdir()
                dirs_removed += 1
            except OSError:
                break
            cur = cur.parent

    dt = time.time() - t0
    print(
        f"done in {dt:.1f}s: moved={moved}, conflicts={conflicts}, "
        f"src_remains={src_remains}, errors={errors}, "
        f"empty_dirs_removed={dirs_removed}"
    )
    if conflicts:
        print(f"  conflict list: {CONFLICT_LOG}")
    if src_remains:
        print(
            f"  src-unlink-failed list (data IS safe at dst): {SRC_REMAINS_LOG}"
        )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
