from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys


_VENDORED_PATHS = (
    Path("third_party"),
    Path("third_party/dust3r"),
    Path("third_party/croco/models/curope"),
    Path("third_party/Grounded-SAM-2"),
    Path("third_party/Grounded-SAM-2/grounding_dino"),
    Path("third_party/sam3"),
    Path("third_party/Depth-Anything-3/src"),
    Path("third_party/InfiniDepth"),
    Path("third_party/dinov3"),
)


def prepend_vendored_import_paths(
    repo_root: Path | None = None,
    extra: Iterable[str | Path] = (),
) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    repo_root = repo_root.resolve()
    rel_paths = (*_VENDORED_PATHS, *(Path(p) for p in extra))
    vendored_paths = tuple(repo_root / rel_path for rel_path in rel_paths)
    for vendored_path in reversed(vendored_paths):
        vendored_path_str = str(vendored_path)
        if vendored_path.exists() and vendored_path_str not in sys.path:
            sys.path.insert(0, vendored_path_str)
    return repo_root
