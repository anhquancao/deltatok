#!/usr/bin/env python3
"""Extract OccAny backbone tokens from preprocessed temporal sequence roots.

This script scans processed dataset folders directly, builds temporal sequences
with the same camera presets used by dataset_setup/base_make_seq.py, and
extracts cached OccAny tokens for each sequence.
"""

import argparse
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

# ---------------------------------------------------------------------------
# Vendor imports (same as extract_recon.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
VENDORED_IMPORT_PATHS = [
    REPO_ROOT / "third_party",
    REPO_ROOT / "third_party" / "dust3r",
    REPO_ROOT / "third_party" / "croco" / "models" / "curope",
    REPO_ROOT / "third_party" / "Grounded-SAM-2",
    REPO_ROOT / "third_party" / "Grounded-SAM-2" / "grounding_dino",
    REPO_ROOT / "third_party" / "sam3",
    REPO_ROOT / "third_party" / "Depth-Anything-3" / "src",
    REPO_ROOT / "third_party" / "pyTorchChamferDistance",
]
for vendored_path in reversed(VENDORED_IMPORT_PATHS):
    vendored_path_str = str(vendored_path)
    if vendored_path.exists() and vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)

torch.backends.cuda.matmul.allow_tf32 = True

from occrae.util.seq_helper import resolve_cameras
from depth_anything_3.utils.io.input_processor import InputProcessor
from occany.utils.image_util import crop_resize_if_necessary


DEFAULT_TRAIN_PROCESSED_ROOTS = (
    "ddad_processed",
    "once_processed",
    "pandaset_processed",
    "waymo_processed",
    "vkitti_processed",
)
DEFAULT_VAL_PROCESSED_ROOTS = (
    "kitti_08_processed",
)
DEFAULT_PROCESSED_ROOTS = DEFAULT_TRAIN_PROCESSED_ROOTS + DEFAULT_VAL_PROCESSED_ROOTS


def get_processed_root_split(processed_root: str) -> str:
    if processed_root in DEFAULT_TRAIN_PROCESSED_ROOTS:
        return "train"
    if processed_root in DEFAULT_VAL_PROCESSED_ROOTS:
        return "val"
    # Keep unknown roots extractable; treat them as train by default.
    return "train"

@dataclass(frozen=True)
class ProcessedDatasetConfig:
    processed_root: str
    frame_digits: int
    output_resolution: Tuple[int, int]
    temporal_cameras: Optional[Tuple[str, ...]]


@dataclass(frozen=True)
class SequenceRecord:
    split: str
    processed_root: str
    scene_name: str
    frame_stems: Tuple[str, ...]
    timesteps: Tuple[int, ...]
    output_resolution: Tuple[int, int]


def _camera_tuple(dataset: str, camera: str) -> Tuple[str, ...]:
    return tuple(str(camera_id) for camera_id in resolve_cameras(dataset, camera, "temporal"))


PROCESSED_DATASET_CONFIGS: Dict[str, ProcessedDatasetConfig] = {
    "ddad_processed": ProcessedDatasetConfig(
        processed_root="ddad_processed",
        frame_digits=6,
        output_resolution=(518, 294),
        temporal_cameras=_camera_tuple("ddad", "all"),
    ),
    "once_processed": ProcessedDatasetConfig(
        processed_root="once_processed",
        frame_digits=6,
        output_resolution=(518, 294),
        temporal_cameras=_camera_tuple("once", "all"),
    ),
    "pandaset_processed": ProcessedDatasetConfig(
        processed_root="pandaset_processed",
        frame_digits=6,
        output_resolution=(518, 294),
        temporal_cameras=_camera_tuple("pandaset", "all"),
    ),
    "waymo_processed": ProcessedDatasetConfig(
        processed_root="waymo_processed",
        frame_digits=5,
        output_resolution=(518, 294),
        temporal_cameras=_camera_tuple("waymo", "all"),
    ),
    "vkitti_processed": ProcessedDatasetConfig(
        processed_root="vkitti_processed",
        frame_digits=5,
        output_resolution=(518, 168),
        temporal_cameras=("0",),
    ),
    "kitti_08_processed": ProcessedDatasetConfig(
        processed_root="kitti_08_processed",
        frame_digits=6,
        output_resolution=(518, 168),
        temporal_cameras=("0",),
    ),
}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract OccAny backbone tokens from scanned preprocessed temporal roots.",
    )
    parser.add_argument(
        "--occany_recon_ckpt",
        type=str,
        default="checkpoints/occany_plus_recon_1B.pth",
        help="Path to the OccAny reconstruction checkpoint.",
    )
    parser.add_argument(
        "--preprocess_data_root",
        type=str,
        default="/gpfs/scratch/ehpc558/quan/occrae_data",
        help="Root containing processed dataset folders such as ddad_processed and once_processed.",
    )
    parser.add_argument(
        "--processed_roots",
        type=str,
        nargs="+",
        default=list(DEFAULT_PROCESSED_ROOTS),
        help="Processed dataset subdirectories to scan under --preprocess_data_root.",
    )
    parser.add_argument(
        "--subsampling_rate",
        type=int,
        default=5,
        help="Stride multiplier used for temporal sequence generation.",
    )
    parser.add_argument(
        "--max_stride",
        type=int,
        default=9,
        help="Maximum stride index used for temporal sequence generation.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=None,
        help="Number of frames to jump after each sequence. If None, jumps to the end of the sequence (no overlap).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/scratch/ehpc558/quan/occrae_emb",
        help="Output root. Features are written under per-processed-root subdirectories.",
    )
    parser.add_argument(
        "--encode_layer",
        type=int,
        default=12,
        choices=[12, 18],
        help="Backbone layer at which to capture tokens (12 pre-fusion, 18 post-fusion).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Storage dtype for saved tokens (default: bfloat16).",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=0,
        help="Shard index (0-based).",
    )
    parser.add_argument(
        "--world",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit on number of samples to process after sharding (-1 = all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples loaded per dataloader batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.world < 1:
        raise ValueError(f"--world must be >= 1, got {args.world}")
    if args.pid < 0 or args.pid >= args.world:
        raise ValueError(f"--pid must satisfy 0 <= pid < world, got pid={args.pid}, world={args.world}")
    if args.subsampling_rate < 1:
        raise ValueError(f"--subsampling_rate must be >= 1, got {args.subsampling_rate}")
    if args.max_stride < 0:
        raise ValueError(f"--max_stride must be >= 0, got {args.max_stride}")


def unique_processed_roots(processed_roots: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for root_name in processed_roots:
        if root_name not in seen:
            seen.add(root_name)
            ordered.append(root_name)
    return ordered


def split_frame_stem(frame_stem: str) -> Tuple[int, str]:
    frame_id_str, camera_id = frame_stem.rsplit("_", 1)
    return int(frame_id_str), camera_id


def parse_frame_stem(frame_stem: str) -> Tuple[int, str]:
    prefix, camera_id = frame_stem.split("_", 1)
    return int(prefix), camera_id


def format_frame_stem(frame_id: int, camera_id: str, frame_digits: int) -> str:
    return f"{frame_id:0{frame_digits}d}_{camera_id}"


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


def generate_temporal_sequences(
    frame_stems: Sequence[str],
    config: ProcessedDatasetConfig,
    subsampling_rate: int,
    max_stride: int,
    frame_stride: Optional[int] = None,
) -> List[Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    strides = [stride * subsampling_rate for stride in range(max_stride + 1)]
    frame_stem_set = set(frame_stems)
    frame_ids_by_camera: Dict[str, List[int]] = defaultdict(list)

    for frame_stem in frame_stems:
        frame_id, camera_id = parse_frame_stem(frame_stem)
        if config.temporal_cameras is None or camera_id not in config.temporal_cameras:
            continue
        frame_ids_by_camera[camera_id].append(frame_id)

    sequences: List[Tuple[Tuple[str, ...], Tuple[int, ...]]] = []
    for camera_id in config.temporal_cameras or ():
        camera_frame_ids = sorted(frame_ids_by_camera.get(camera_id, []))
        frame_idx = 0
        while frame_idx < len(camera_frame_ids):
            frame_id = camera_frame_ids[frame_idx]
            seq = []
            for stride in strides:
                next_frame_stem = format_frame_stem(frame_id + stride, camera_id, config.frame_digits)
                if next_frame_stem not in frame_stem_set:
                    break
                seq.append(next_frame_stem)
            if len(seq) == len(strides):
                sequences.append((tuple(seq), tuple(strides)))
                if frame_stride is not None:
                    # Jump by frame_stride frames
                    next_frame_id = frame_id + frame_stride
                    frame_idx = bisect_right(camera_frame_ids, next_frame_id - 1)
                else:
                    # Default: Jump to the first frame strictly after this sequence so accepted windows do not overlap.
                    sequence_end_frame_id = frame_id + strides[-1]
                    frame_idx = bisect_right(camera_frame_ids, sequence_end_frame_id)
            else:
                frame_idx += 1
    return sequences


def build_sequence_records(
    preprocess_data_root: Path,
    processed_roots: Sequence[str],
    subsampling_rate: int,
    max_stride: int,
    frame_stride: Optional[int] = None,
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
        if config.temporal_cameras is None:
            print(f"[WARN] Skipping {processed_root_name}: temporal mode is not configured")
            continue

        supported_root_found = True
        root_path = preprocess_data_root / processed_root_name
        if not root_path.is_dir():
            print(f"[WARN] Skipping {processed_root_name}: processed root not found at {root_path}")
            continue

        scene_dirs = iter_scene_dirs(root_path)
        root_sequences: List[SequenceRecord] = []
        for scene_dir in tqdm(scene_dirs, desc=f"Indexing {processed_root_name}", leave=False):
            frame_stems = load_scene_frame_stems(scene_dir)
            if not frame_stems:
                continue

            scene_sequences = generate_temporal_sequences(
                frame_stems=frame_stems,
                config=config,
                subsampling_rate=subsampling_rate,
                max_stride=max_stride,
                frame_stride=frame_stride,
            )

            for frame_seq, timesteps in scene_sequences:
                root_sequences.append(
                    SequenceRecord(
                        split=split,
                        processed_root=processed_root_name,
                        scene_name=scene_dir.relative_to(root_path).as_posix(),
                        frame_stems=frame_seq,
                        timesteps=timesteps,
                        output_resolution=config.output_resolution,
                    )
                )

        stats[processed_root_name] = {
            "scene_count": len(scene_dirs),
            "sequence_count": len(root_sequences),
            "num_views": len(root_sequences[0].frame_stems) if root_sequences else 0,
            "resolution_w": config.output_resolution[0],
            "resolution_h": config.output_resolution[1],
        }
        all_records.extend(root_sequences)
    if not supported_root_found:
        raise ValueError("No processed roots support temporal extraction")
    if not all_records:
        raise ValueError("No sequences were discovered for the selected processed roots")

    return all_records, stats


class PreprocessedSequenceDataset(Dataset):
    def __init__(self, preprocess_data_root: Path, records: Sequence[SequenceRecord]):
        self.preprocess_data_root = preprocess_data_root
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        scene_dir = self.preprocess_data_root / record.processed_root / record.scene_name
        imgs: List[torch.Tensor] = []

        for frame_stem in record.frame_stems:
            npz_path = scene_dir / f"{frame_stem}.npz"
            with np.load(npz_path) as data:
                image = data["image"]

            resized_image = crop_resize_if_necessary(
                image,
                resolution=record.output_resolution,
                rng=None,
                info=f"{record.scene_name}/{frame_stem}",
            )
            imgs.append(InputProcessor.NORMALIZE(to_tensor(resized_image)))

        return {
            "split": record.split,
            "processed_root": record.processed_root,
            "scene_name": record.scene_name,
            "frame_stems": list(record.frame_stems),
            "timesteps": list(record.timesteps),
            "output_resolution": tuple(record.output_resolution),
            "imgs": imgs,
        }


def shard_indices(total: int, pid: int, world: int) -> List[int]:
    return list(range(pid, total, world))


def collate_identity(batch: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return batch


def build_occ_rae(weights_path: str, device: str, encode_layer: int, model_input_size: int):
    from occany.model.occ_rae import OccRAE

    occ_rae = OccRAE(
        weights_path=weights_path,
        output_resolution=(model_input_size, model_input_size),
        device=device,
        encode_layer=encode_layer,
    )
    occ_rae.eval()
    return occ_rae


def save_feature_sample(
    args: argparse.Namespace,
    sample: Dict[str, object],
    tokens: torch.Tensor,
    output_root: Path,
) -> None:
    torch_dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    split = str(sample["split"])
    processed_root = str(sample["processed_root"])
    scene_name = str(sample["scene_name"])
    first_frame_stem = str(sample["frame_stems"][0])
    first_frame_relpath = str(
        Path(processed_root) / scene_name / f"{first_frame_stem}.npz"
    )
    sample_output_dir = output_root / split / processed_root  / scene_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "1st_frame_relpath": first_frame_relpath,
        "timesteps": list(sample["timesteps"]),
        "output_resolution": tuple(sample["output_resolution"]),
    }
    out_path = sample_output_dir / f"{first_frame_stem}.pth"
    torch.save(
        {
            **metadata,
            "tokens": tokens.cpu().to(torch_dtype_map[args.dtype]),
        },
        out_path,
    )
    print(f"[INFO] Saved features at {out_path} with metadata {metadata}")


def extract_sequences(
    args: argparse.Namespace,
    dataset: Dataset,
    stats: Dict[str, Dict[str, int]],
) -> None:
    total = len(dataset)
    indices = shard_indices(total, args.pid, args.world)
    if args.max_samples > 0:
        indices = indices[: args.max_samples]

    print(
        f"[INFO] Extracting layer {args.encode_layer} tokens from {len(indices)} samples "
        f"(shard {args.pid}/{args.world}, total {total})"
    )
    for processed_root, root_stats in stats.items():
        print(
            f"[INFO] {processed_root}: scenes={root_stats['scene_count']} seqs={root_stats['sequence_count']} "
            f"views={root_stats['num_views']} resolution=({root_stats['resolution_w']}, {root_stats['resolution_h']})"
        )

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_identity,
    )

    occ_rae_cache: Dict[int, object] = {}
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    for batch in tqdm(loader, desc=f"extract shard {args.pid}"):
        grouped_samples: Dict[Tuple[Tuple[int, int], int], List[Dict[str, object]]] = defaultdict(list)
        for sample in batch:
            key = (tuple(sample["output_resolution"]), len(sample["imgs"]))
            grouped_samples[key].append(sample)

        for (output_resolution, _num_views), grouped_batch in grouped_samples.items():
            model_input_size = max(output_resolution)
            if model_input_size not in occ_rae_cache:
                print(f"[INFO] Building OccRAE for model_input_size={model_input_size}")
                occ_rae_cache[model_input_size] = build_occ_rae(
                    weights_path=args.occany_recon_ckpt,
                    device=args.device,
                    encode_layer=args.encode_layer,
                    model_input_size=model_input_size,
                )

            occ_rae = occ_rae_cache[model_input_size]
            imgs = torch.stack(
                [torch.stack(sample["imgs"], dim=0) for sample in grouped_batch],
                dim=0,
            ).to(args.device)
            latents = occ_rae.encode(imgs)
            tokens = latents["tokens"]

            for sample, sample_tokens in zip(grouped_batch, tokens):
                save_feature_sample(
                    args=args,
                    sample=sample,
                    tokens=sample_tokens,
                    output_root=output_root,
                )
                total_saved += 1

    print(f"[INFO] Done. Saved {total_saved} samples under {output_root}")


def main() -> None:
    args = get_args_parser().parse_args()
    validate_args(args)

    preprocess_data_root = Path(args.preprocess_data_root).expanduser().resolve()
    if not preprocess_data_root.is_dir():
        raise FileNotFoundError(f"Preprocess data root not found: {preprocess_data_root}")

    processed_roots = unique_processed_roots(args.processed_roots)

    records, stats = build_sequence_records(
        preprocess_data_root=preprocess_data_root,
        processed_roots=args.processed_roots,
        subsampling_rate=args.subsampling_rate,
        max_stride=args.max_stride,
        frame_stride=args.frame_stride,
    )

    dataset = PreprocessedSequenceDataset(preprocess_data_root=preprocess_data_root, records=records)
    extract_sequences(args=args, dataset=dataset, stats=stats)

if __name__ == "__main__":
    main()

