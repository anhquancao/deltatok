#!/usr/bin/env python3
"""Verify that InfiniDepth pseudo-depth outputs exist for every source .npz.

Walks the same processed roots as ``extract_infinidepth_pseudo.py`` and, for
each ``<stem>.npz``, checks that a sibling ``<stem>.infinidepth.png`` exists
(and is non-empty / decodable as uint16 when ``--check_readable`` is passed).

Reports per-dataset coverage and the global totals. Optionally writes the
list of missing source paths to a text file so a follow-up shard can re-run
just those samples.

Example::

    conda activate occany && python verify_infinidepth_pseudo.py --check_readable
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Sequence, Tuple

from tqdm import tqdm

# Keep these in sync with extract_infinidepth_pseudo.py.
DEFAULT_PROCESSED_ROOTS = (
    "waymo_processed",
    "ddad_processed",
    "pandaset_processed",
    "vkitti_processed",
    "once_processed",
)
OUTPUT_SUFFIX = ".infinidepth.png"


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify InfiniDepth pseudo-depth coverage over processed datasets.",
    )
    parser.add_argument(
        "--preprocess_data_root",
        type=str,
        default="/scratch/project/eu-25-92/data",
        help="Root containing processed dataset folders (mirrors extract script).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root where .infinidepth.png files live. Defaults to --preprocess_data_root.",
    )
    parser.add_argument(
        "--roots",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Override processed root names (e.g. --roots waymo_processed once_processed). "
            f"Defaults to: {' '.join(DEFAULT_PROCESSED_ROOTS)}."
        ),
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=0,
        help="Shard index (0-based). Verification is sharded with the same scheme as the extractor.",
    )
    parser.add_argument(
        "--world",
        type=int,
        default=1,
        help="Total number of shards. Use 1 to verify everything.",
    )
    parser.add_argument(
        "--check_readable",
        action="store_true",
        default=False,
        help=(
            "Beyond existence, decode each .infinidepth.png with cv2 and confirm "
            "it is a non-empty uint16 image. Slower; uses cv2 if available."
        ),
    )
    parser.add_argument(
        "--list_missing",
        type=str,
        default="./results/infinidepth_pseudo_missing.txt",
        help=(
            "Path to write the list of missing source .npz paths (one per line). "
            "Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--max_print",
        type=int,
        default=20,
        help="Maximum number of missing/bad paths to print per dataset (full list still written to --list_missing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help=(
            "Number of worker processes for parallel verification. 1 = serial (default). "
            "Recommended: 16 when --check_readable is set (CPU-bound PNG decode). For "
            "existence-only checks 1-4 is usually enough."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="Per-worker chunk size for imap_unordered (only used when --workers > 1).",
    )
    return parser


def iter_source_npz_paths(root: Path, desc: str = "scanning") -> List[Path]:
    """Find source .npz files under root, pruning any ``tmp/`` subtree."""
    paths: List[Path] = []
    pbar = tqdm(desc=f"{desc} {root.name}", unit="file", leave=False)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "tmp"]
        for fn in filenames:
            if not fn.endswith(".npz"):
                continue
            if fn.endswith(OUTPUT_SUFFIX):
                continue
            paths.append(Path(dirpath) / fn)
            pbar.update(1)
    pbar.close()
    paths.sort()
    return paths


def output_path_for(src: Path, preprocess_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(preprocess_root)
    return output_root / rel.with_name(rel.stem + OUTPUT_SUFFIX)


def check_png(path: Path) -> Tuple[bool, str]:
    """Return (ok, reason). Existence-only when cv2/numpy aren't imported."""
    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        return False, f"cv2/numpy unavailable: {exc}"

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "cv2.imread returned None (corrupt PNG?)"
    if img.dtype != np.uint16:
        return False, f"unexpected dtype {img.dtype} (expected uint16)"
    if img.ndim != 2 or img.size == 0:
        return False, f"unexpected shape {img.shape}"
    return True, ""


# Worker-process globals — populated by _init_worker so we don't ship roots
# with every task. Also used in-process when --workers == 1.
_WORKER_PREPROCESS_ROOT: Path | None = None
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_CHECK_READABLE: bool = False


def _init_worker(preprocess_root: str, output_root: str, check_readable: bool) -> None:
    global _WORKER_PREPROCESS_ROOT, _WORKER_OUTPUT_ROOT, _WORKER_CHECK_READABLE
    _WORKER_PREPROCESS_ROOT = Path(preprocess_root)
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_CHECK_READABLE = check_readable


def _verify_one(src_str: str) -> Tuple[str, str, str]:
    """Return (status, src_str, reason). status ∈ {'present','missing','bad'}."""
    src = Path(src_str)
    out = output_path_for(src, _WORKER_PREPROCESS_ROOT, _WORKER_OUTPUT_ROOT)
    if not out.exists():
        return "missing", src_str, ""
    if _WORKER_CHECK_READABLE:
        ok, reason = check_png(out)
        if not ok:
            return "bad", src_str, reason
    return "present", src_str, ""


def verify_root(
    root_name: str,
    preprocess_root: Path,
    output_root: Path,
    pid: int,
    world: int,
    check_readable: bool,
    max_print: int,
    workers: int,
    chunksize: int,
) -> Tuple[int, int, int, List[Path], List[Tuple[Path, str]]]:
    root_path = preprocess_root / root_name
    if not root_path.is_dir():
        print(f"[WARN] {root_name}: not found at {root_path}")
        return 0, 0, 0, [], []

    all_paths = iter_source_npz_paths(root_path)
    sharded = all_paths[pid::world]

    missing: List[Path] = []
    bad: List[Tuple[Path, str]] = []
    present = 0
    pbar = tqdm(
        total=len(sharded),
        desc=f"verify {root_name} [{pid}/{world}]",
        unit="file",
        leave=True,
    )

    def _consume(results) -> None:
        nonlocal present
        for status, src_str, reason in results:
            if status == "missing":
                missing.append(Path(src_str))
            elif status == "bad":
                bad.append((Path(src_str), reason))
            else:
                present += 1
            pbar.update(1)
            pbar.set_postfix(
                present=present, missing=len(missing), bad=len(bad), refresh=False,
            )

    src_strs = (str(s) for s in sharded)
    init_args = (str(preprocess_root), str(output_root), check_readable)
    if workers > 1 and sharded:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=init_args,
        ) as pool:
            _consume(pool.map(_verify_one, src_strs, chunksize=chunksize))
    else:
        _init_worker(*init_args)
        _consume(_verify_one(s) for s in src_strs)
    pbar.close()

    total = len(sharded)
    pct = (present / total * 100.0) if total else 100.0
    status = "OK" if (not missing and not bad) else "INCOMPLETE"
    print(
        f"[{status}] {root_name}: present={present}/{total} "
        f"({pct:.2f}%) missing={len(missing)} bad={len(bad)} "
        f"[shard {pid}/{world}, total_unsharded={len(all_paths)}]"
    )

    if missing and max_print > 0:
        print(f"  first {min(max_print, len(missing))} missing:")
        for p in missing[:max_print]:
            print(f"    - {p}")
        if len(missing) > max_print:
            print(f"    ... and {len(missing) - max_print} more")
    if bad and max_print > 0:
        print(f"  first {min(max_print, len(bad))} bad:")
        for p, reason in bad[:max_print]:
            print(f"    - {p}  ({reason})")
        if len(bad) > max_print:
            print(f"    ... and {len(bad) - max_print} more")

    return total, present, len(missing), missing, bad


def main(argv: Sequence[str] | None = None) -> int:
    args = get_args_parser().parse_args(argv)

    if args.world < 1:
        print(f"[ERROR] --world must be >= 1, got {args.world}", file=sys.stderr)
        return 2
    if not 0 <= args.pid < args.world:
        print(
            f"[ERROR] --pid must satisfy 0 <= pid < world, got pid={args.pid}, world={args.world}",
            file=sys.stderr,
        )
        return 2

    preprocess_root = Path(args.preprocess_data_root).expanduser().resolve()
    if not preprocess_root.is_dir():
        print(f"[ERROR] preprocess_data_root not found: {preprocess_root}", file=sys.stderr)
        return 2
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else preprocess_root
    )

    if args.workers < 1:
        print(f"[ERROR] --workers must be >= 1, got {args.workers}", file=sys.stderr)
        return 2
    if args.chunksize < 1:
        print(f"[ERROR] --chunksize must be >= 1, got {args.chunksize}", file=sys.stderr)
        return 2

    roots = tuple(args.roots) if args.roots else DEFAULT_PROCESSED_ROOTS
    print(
        f"[INFO] Verifying InfiniDepth outputs under {output_root}\n"
        f"       source root={preprocess_root} shard={args.pid}/{args.world} "
        f"check_readable={args.check_readable} workers={args.workers}"
    )

    grand_total = 0
    grand_present = 0
    grand_missing = 0
    grand_bad = 0
    all_missing: List[Path] = []
    all_bad: List[Tuple[Path, str]] = []
    for root_name in roots:
        total, present, n_missing, missing, bad = verify_root(
            root_name=root_name,
            preprocess_root=preprocess_root,
            output_root=output_root,
            pid=args.pid,
            world=args.world,
            check_readable=args.check_readable,
            max_print=args.max_print,
            workers=args.workers,
            chunksize=args.chunksize,
        )
        grand_total += total
        grand_present += present
        grand_missing += n_missing
        grand_bad += len(bad)
        all_missing.extend(missing)
        all_bad.extend(bad)

    pct = (grand_present / grand_total * 100.0) if grand_total else 100.0
    print(
        "\n[SUMMARY] "
        f"present={grand_present}/{grand_total} ({pct:.2f}%) "
        f"missing={grand_missing} bad={grand_bad}"
    )

    if args.list_missing and (all_missing or all_bad):
        out_path = Path(args.list_missing).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for p in all_missing:
                fh.write(f"{p}\n")
            for p, _ in all_bad:
                fh.write(f"{p}\n")
        print(f"[INFO] Wrote {len(all_missing) + len(all_bad)} missing/bad paths to {out_path}")

    return 0 if (grand_missing == 0 and grand_bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
