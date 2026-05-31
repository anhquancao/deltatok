"""Shared sequence generation helpers.

Used by both dataset_setup/base_make_seq.py and occrae/dataset/preprocessed_sequence.py.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


CAMERA_PRESETS: Dict[str, Dict[str, list]] = {
    "waymo": {
        "surround": list(range(1, 6)),
        "all": list(range(1, 6)),
    },
    "ddad": {
        "surround": list(range(6)),
        "all": list(range(6)),
    },
    "pandaset": {
        "surround": list(range(6)),
        "all": list(range(6)),
    },
    "once": {
        "surround": ["cam01", "cam05", "cam06", "cam07", "cam08"],
        "all": ["cam01", "cam05", "cam06", "cam07", "cam08"],
    },
    "vkitti": {
        "all": [0],
    },
    "kitti": {
        "all": [0],
    },
    "nuscenes": {
        "surround": list(range(6)),
        "all": list(range(6)),
    },
}


def resolve_cameras(dataset: str, camera: str, seq_mode: str) -> list:
    if camera == "all" and seq_mode not in ("temporal", "surround_temporal"):
        raise ValueError("Camera 'all' is only supported with temporal or surround_temporal sequence generation")

    camera_map = CAMERA_PRESETS.get(dataset)
    if camera_map is None:
        raise ValueError(f"Dataset {dataset} does not define camera presets")

    try:
        return list(camera_map[camera])
    except KeyError as exc:
        raise ValueError(f"Camera {camera} not supported for {dataset}") from exc


def parse_frame_stem(frame_stem: str) -> Tuple[int, str]:
    prefix, camera_id = frame_stem.split("_", 1)
    return int(prefix), camera_id


def format_frame_stem(frame_id: int, camera_id: str, frame_digits: int) -> str:
    return f"{frame_id:0{frame_digits}d}_{camera_id}"


def generate_surround_temporal_sequences(
    frame_stems: Sequence[str],
    cameras: Sequence[str],
    frame_digits: int,
    subsampling_rate: int,
    max_stride: int,
    frame_stride: Optional[int] = None,
) -> List[Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Temporal sequences of surround views: all cameras at each timestep.

    Matches base_make_seq.py::SeqMaker._generate_surround_temporal_seq.
    """
    if not cameras or len(cameras) < 2:
        return []

    strides = [stride * subsampling_rate for stride in range(max_stride + 1)]
    frame_ids_by_camera: Dict[str, set] = defaultdict(set)

    for frame_stem in frame_stems:
        frame_id, camera_id = parse_frame_stem(frame_stem)
        if camera_id in cameras:
            frame_ids_by_camera[camera_id].add(frame_id)

    if len(frame_ids_by_camera) < len(cameras):
        return []

    complete_frame_ids = sorted(
        set.intersection(*(frame_ids_by_camera[cam] for cam in cameras))
    )
    if not complete_frame_ids:
        return []

    complete_set = set(complete_frame_ids)
    num_cameras = len(cameras)

    sequences: List[Tuple[Tuple[str, ...], Tuple[int, ...]]] = []
    idx = 0
    while idx < len(complete_frame_ids):
        base_frame_id = complete_frame_ids[idx]
        flat_stems: List[str] = []
        flat_timesteps: List[int] = []
        valid = True
        for stride in strides:
            target_frame_id = base_frame_id + stride
            if target_frame_id not in complete_set:
                valid = False
                break
            for cam_id in cameras:
                flat_stems.append(
                    format_frame_stem(target_frame_id, cam_id, frame_digits)
                )
            flat_timesteps.extend([stride] * num_cameras)

        if valid and len(flat_stems) == len(strides) * num_cameras:
            sequences.append((tuple(flat_stems), tuple(flat_timesteps)))
            if frame_stride is not None:
                next_frame_id = base_frame_id + frame_stride
                idx = bisect_right(complete_frame_ids, next_frame_id - 1)
            else:
                sequence_end_frame_id = base_frame_id + strides[-1]
                idx = bisect_right(complete_frame_ids, sequence_end_frame_id)
        else:
            idx += 1

    return sequences
