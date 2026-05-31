from __future__ import annotations

import os
import pickle
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from occrae.util.seq_helper import generate_surround_temporal_sequences, resolve_cameras


_SEQ_CACHE_VERSION = 2


ResolutionList = List[Tuple[int, int]]


def _normalize_resolutions(resolutions: Any) -> ResolutionList:
    """Coerce a (H, W) tuple, list of (H, W), or OmegaConf ListConfig into List[Tuple[int, int]]."""
    if resolutions is None:
        raise ValueError("resolutions must not be None")

    # Materialise OmegaConf containers / iterators into a plain list.
    seq = list(resolutions)
    if not seq:
        raise ValueError("resolutions must be non-empty")

    # Single (H, W) pair (two ints) -> wrap in a list.
    if len(seq) == 2 and all(isinstance(x, (int, float)) for x in seq):
        return [(int(seq[0]), int(seq[1]))]

    out: ResolutionList = []
    for entry in seq:
        pair = list(entry)
        if len(pair) != 2:
            raise ValueError(f"Each resolution must be a (H, W) pair, got {entry!r}")
        out.append((int(pair[0]), int(pair[1])))
    return out


@dataclass(frozen=True)
class ProcessedDatasetConfig:
    processed_root: str
    frame_digits: int
    output_resolution: ResolutionList
    cameras: Tuple[str, ...]


@dataclass(frozen=True)
class SequenceRecord:
    split: str
    processed_root: str
    scene_name: str
    frame_stems: Tuple[str, ...]
    timesteps: Tuple[int, ...]
    output_resolution: ResolutionList
    num_cameras: int = 1
    camera_ids: Tuple[str, ...] = ()


DEFAULT_TRAIN_PROCESSED_ROOTS = (
    "ddad_processed",
    "once_processed",
    "pandaset_processed",
    "waymo_processed",
    "vkitti_processed",
)
DEFAULT_VAL_PROCESSED_ROOTS = (
    "kitti_08_processed",
    "occ3d_nuscenes_processed",
)
DEFAULT_PROCESSED_ROOTS = DEFAULT_TRAIN_PROCESSED_ROOTS + DEFAULT_VAL_PROCESSED_ROOTS


def get_processed_root_split(processed_root: str) -> str:
    if processed_root in DEFAULT_TRAIN_PROCESSED_ROOTS:
        return "train"
    if processed_root in DEFAULT_VAL_PROCESSED_ROOTS:
        return "val"
    return "train"


def _camera_tuple(dataset: str, camera: str) -> Tuple[str, ...]:
    return tuple(str(camera_id) for camera_id in resolve_cameras(dataset, camera, "temporal"))


PROCESSED_DATASET_CONFIGS: Dict[str, ProcessedDatasetConfig] = {
    "ddad_processed": ProcessedDatasetConfig(
        processed_root="ddad_processed",
        frame_digits=6,
        output_resolution=[(294, 518), (280, 518), (266, 518), (210, 518), (168, 518)],
        cameras=_camera_tuple("ddad", "all"),
    ),
    "once_processed": ProcessedDatasetConfig(
        processed_root="once_processed",
        frame_digits=6,
        output_resolution=[(294, 518), (280, 518), (266, 518), (210, 518), (168, 518)],
        cameras=_camera_tuple("once", "all"),
    ),
    "pandaset_processed": ProcessedDatasetConfig(
        processed_root="pandaset_processed",
        frame_digits=6,
        output_resolution=[(294, 518), (280, 518), (266, 518), (210, 518), (168, 518)],
        cameras=_camera_tuple("pandaset", "all"),
    ),
    "waymo_processed": ProcessedDatasetConfig(
        processed_root="waymo_processed",
        frame_digits=5,
        output_resolution=[(294, 518), (280, 518), (266, 518), (210, 518), (168, 518)],
        cameras=_camera_tuple("waymo", "all"),
    ),
    "vkitti_processed": ProcessedDatasetConfig(
        processed_root="vkitti_processed",
        frame_digits=5,
        output_resolution=[(168, 518)],
        cameras=_camera_tuple("vkitti", "all"),
    ),
    "kitti_08_processed": ProcessedDatasetConfig(
        processed_root="kitti_08_processed",
        frame_digits=6,
        output_resolution=[(168, 518)],
        cameras=_camera_tuple("kitti", "all"),
    ),
    "occ3d_nuscenes_processed": ProcessedDatasetConfig(
        processed_root="occ3d_nuscenes_processed",
        frame_digits=6,
        output_resolution=[(294, 518)],
        cameras=_camera_tuple("nuscenes", "all"),
    ),
}


def iter_scene_dirs(processed_root: Path) -> List[Path]:
    scenes = sorted(
        child
        for child in processed_root.iterdir()
        if child.is_dir() and "tmp" not in child.name
    )
    if not scenes and any(processed_root.glob("*.npz")):
        return [processed_root]
    return scenes


def load_scene_frame_stems(scene_dir: Path) -> List[str]:
    return sorted(path.stem for path in scene_dir.glob("*.npz"))


def _sequence_cache_path(
    root_path: Path,
    subsampling_rate: int,
    max_stride: int,
    frame_stride: Optional[int],
) -> Path:
    fs = "none" if frame_stride is None else str(frame_stride)
    name = f"v{_SEQ_CACHE_VERSION}_sub{subsampling_rate}_stride{max_stride}_fs{fs}_mode_surround_temporal"
    return root_path / f"{name}.pkl"


def _load_cached_root_records(
    cache_path: Path,
    expected_params: Dict[str, Any],
) -> Optional[Tuple[List[SequenceRecord], Dict[str, int]]]:
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to read sequence record cache {cache_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _SEQ_CACHE_VERSION:
        return None
    if payload.get("params") != expected_params:
        return None
    records = payload.get("records")
    stats = payload.get("stats")
    if not isinstance(records, list) or not isinstance(stats, dict):
        return None
    return records, stats


def _save_cached_root_records(
    cache_path: Path,
    records: List[SequenceRecord],
    stats: Dict[str, int],
    params: Dict[str, Any],
) -> None:
    try:
        payload = {
            "version": _SEQ_CACHE_VERSION,
            "params": params,
            "records": records,
            "stats": stats,
        }
        # Atomic write so concurrent DDP ranks never observe a partial file.
        fd, tmp_path = tempfile.mkstemp(
            prefix=cache_path.name + ".", suffix=".tmp", dir=str(cache_path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:
        print(f"[WARN] Failed to write sequence record cache {cache_path}: {exc}")


def _build_root_records(
    processed_root_name: str,
    split: str,
    config: ProcessedDatasetConfig,
    root_path: Path,
    output_resolution: ResolutionList,
    subsampling_rate: int,
    max_stride: int,
    frame_stride: Optional[int],
) -> Tuple[List[SequenceRecord], Dict[str, int]]:
    cameras = tuple(str(c) for c in config.cameras)

    scene_dirs = iter_scene_dirs(root_path)
    root_sequences: List[SequenceRecord] = []

    for scene_dir in tqdm(scene_dirs, desc=f"Indexing {processed_root_name}", leave=False):
        frame_stems = load_scene_frame_stems(scene_dir)
        if not frame_stems:
            continue
        scene_name = scene_dir.relative_to(root_path).as_posix()

        for frame_seq, timesteps in generate_surround_temporal_sequences(
            frame_stems=frame_stems,
            cameras=cameras,
            frame_digits=config.frame_digits,
            subsampling_rate=subsampling_rate,
            max_stride=max_stride,
            frame_stride=frame_stride,
        ):
            root_sequences.append(
                SequenceRecord(
                    split=split,
                    processed_root=processed_root_name,
                    scene_name=scene_name,
                    frame_stems=frame_seq,
                    timesteps=timesteps,
                    output_resolution=output_resolution,
                    num_cameras=len(cameras),
                    camera_ids=cameras,
                )
            )

    if root_sequences:
        res_h, res_w = root_sequences[0].output_resolution[0]
        stats = {
            "scene_count": len(scene_dirs),
            "sequence_count": len(root_sequences),
            "resolution_w": res_w,
            "resolution_h": res_h,
        }
    else:
        stats = {}
    return root_sequences, stats


def build_sequence_records(
    preprocess_data_root: Path,
    processed_roots: Sequence[str],
    subsampling_rate: int,
    max_stride: int,
    frame_stride: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[List[SequenceRecord], Dict[str, Dict[str, int]]]:
    all_records: List[SequenceRecord] = []
    stats: Dict[str, Dict[str, int]] = {}
    supported_root_found = False

    for processed_root_name in processed_roots:
        if processed_root_name not in PROCESSED_DATASET_CONFIGS:
            raise KeyError(
                f"Unknown processed root '{processed_root_name}'. Available: {sorted(PROCESSED_DATASET_CONFIGS)}"
            )

        split = get_processed_root_split(processed_root_name)
        config = PROCESSED_DATASET_CONFIGS[processed_root_name]
        supported_root_found = True
        root_path = preprocess_data_root / processed_root_name
        if not root_path.is_dir():
            print(f"[WARN] Skipping {processed_root_name}: processed root not found at {root_path}")
            continue

        output_resolution = list(config.output_resolution)

        cache_params = {
            "processed_root": processed_root_name,
            "split": split,
            "subsampling_rate": int(subsampling_rate),
            "max_stride": int(max_stride),
            "frame_stride": None if frame_stride is None else int(frame_stride),
            "output_resolution": [list(pair) for pair in output_resolution],
            "cameras": list(config.cameras),
            "frame_digits": config.frame_digits,
        }
        cache_path = _sequence_cache_path(
            root_path=root_path,
            subsampling_rate=subsampling_rate,
            max_stride=max_stride,
            frame_stride=frame_stride,
        )

        cached: Optional[Tuple[List[SequenceRecord], Dict[str, int]]] = None
        if use_cache:
            cached = _load_cached_root_records(cache_path, cache_params)

        if cached is not None:
            root_sequences, root_stats = cached
            print(
                f"[INFO] Loaded {len(root_sequences)} cached sequence records "
                f"for {processed_root_name} from {cache_path}"
            )
        else:
            root_sequences, root_stats = _build_root_records(
                processed_root_name=processed_root_name,
                split=split,
                config=config,
                root_path=root_path,
                output_resolution=output_resolution,
                subsampling_rate=subsampling_rate,
                max_stride=max_stride,
                frame_stride=frame_stride,
            )
            if use_cache and root_sequences:
                _save_cached_root_records(cache_path, root_sequences, root_stats, cache_params)
                print(
                    f"[INFO] Cached {len(root_sequences)} sequence records "
                    f"for {processed_root_name} to {cache_path}"
                )

        if root_stats:
            stats[processed_root_name] = root_stats
        all_records.extend(root_sequences)

    if not supported_root_found:
        raise ValueError("No processed roots support temporal extraction")
    if not all_records:
        raise ValueError("No sequences were discovered for the selected processed roots")

    return all_records, stats


class PreprocessedSequenceDataset(Dataset):
    def __init__(self, preprocess_data_root: Union[str, Path], records: Sequence[SequenceRecord], load_gt: bool = False):
        self.preprocess_data_root = Path(preprocess_data_root)
        self.records = list(records)
        self.load_gt = load_gt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        scene_dir = self.preprocess_data_root / record.processed_root / record.scene_name
        imgs: List[np.ndarray] = []
        depthmaps: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        cam2worlds: List[np.ndarray] = []

        for frame_stem in record.frame_stems:
            npz_path = scene_dir / f"{frame_stem}.npz"
            with np.load(npz_path) as data:
                imgs.append(data["image"])
                if self.load_gt:
                    depthmaps.append(data["depthmap"])
                    intrinsics.append(data["intrinsics"])
                    cam2worlds.append(data["cam2world"])

        out = {
            "split": record.split,
            "processed_root": record.processed_root,
            "scene_name": record.scene_name,
            "frame_stems": list(record.frame_stems),
            "timesteps": list(record.timesteps),
            "output_resolution": record.output_resolution,
            "num_cameras": record.num_cameras,
            "camera_ids": list(record.camera_ids),
            "imgs": imgs,
        }
        if self.load_gt:
            w2c_cam0 = np.linalg.inv(cam2worlds[0].astype(np.float64)).astype(np.float32)
            cam2worlds = [w2c_cam0 @ c2w for c2w in cam2worlds]
            out.update({
                "depthmaps": depthmaps,
                "intrinsics": intrinsics,
                "cam2worlds": cam2worlds,
            })
        return out

    def get_processed_root_index_groups(self) -> Dict[Tuple[str, int], List[int]]:
        grouped: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            grouped[(record.processed_root, record.num_cameras)].append(index)
        return dict(grouped)
