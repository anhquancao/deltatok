from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import BatchSampler, Dataset


def collate_identity(batch: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return batch


def _normalize_resolution(value: Sequence[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"Expected output_resolution to have length 2, got {value!r}")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid output_resolution={value!r}")
    return width, height


def _expected_seq_len(output_resolution: Tuple[int, int], patch_size: int = 14) -> int:
    width, height = output_resolution
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError(
            f"output_resolution={output_resolution} must be divisible by patch_size={patch_size}"
        )
    return (width // patch_size) * (height // patch_size) + 1


def _root_name(key: object) -> str:
    """Extract the processed_root string from a grouping key.

    ``OccRAETokenDataset`` groups by ``str``, while
    ``PreprocessedSequenceDataset`` groups by ``(str, int)``.
    """
    if isinstance(key, tuple):
        return key[0]
    return str(key)


class ProcessedRootBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: "OccRAETokenDataset",
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        root_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        super().__init__(range(len(dataset)), batch_size=batch_size, drop_last=drop_last)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.root_weights = root_weights

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches(self) -> List[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        grouped_indices = self.dataset.get_processed_root_index_groups()
        processed_roots = list(grouped_indices.keys())
        if self.shuffle and processed_roots:
            order = torch.randperm(len(processed_roots), generator=generator).tolist()
            processed_roots = [processed_roots[i] for i in order]

        batches: List[List[int]] = []
        batch_root_names: List[str] = []
        for processed_root in processed_roots:
            indices = list(grouped_indices[processed_root])
            if self.shuffle and indices:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]

            rname = _root_name(processed_root)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
                batch_root_names.append(rname)

        if not batches:
            return []

        if self.root_weights is not None:
            root_batch_counts: Dict[str, int] = defaultdict(int)
            for rn in batch_root_names:
                root_batch_counts[rn] += 1

            weights = torch.zeros(len(batches))
            for i, rn in enumerate(batch_root_names):
                weights[i] = self.root_weights.get(rn, 1.0) / root_batch_counts[rn]

            selected = torch.multinomial(
                weights, len(batches), replacement=True, generator=generator,
            )
            batches = [batches[i] for i in selected.tolist()]
        elif self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]

        if self.world_size == 1:
            return batches
        if self.drop_last:
            usable = len(batches) - (len(batches) % self.world_size)
            batches = batches[:usable]
        elif len(batches) % self.world_size != 0:
            batches = batches + batches[: self.world_size - (len(batches) % self.world_size)]
        return batches[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        total_batches = 0
        for indices in self.dataset.get_processed_root_index_groups().values():
            batch_count, remainder = divmod(len(indices), self.batch_size)
            total_batches += batch_count
            if remainder and not self.drop_last:
                total_batches += 1
        if self.world_size == 1:
            return total_batches
        if self.drop_last:
            return total_batches // self.world_size
        return (total_batches + self.world_size - 1) // self.world_size


class OccRAETokenDataset(Dataset):
    """Dataset over layer-18 token dumps produced by extract_occany_features.py."""

    def __init__(
        self,
        feature_root: str,
        *,
        cond_num: int = 2,
        processed_roots: Optional[Sequence[str]] = None,
        max_samples: int = -1,
    ) -> None:
        self.feature_root = Path(feature_root).expanduser().resolve()
        if not self.feature_root.is_dir():
            raise FileNotFoundError(f"Feature root not found: {self.feature_root}")
        if cond_num != 2:
            raise ValueError(f"This pipeline is fixed to cond_num=2, got {cond_num}")
        self.cond_num = cond_num
        self.allowed_processed_roots = set(processed_roots or [])

        feature_paths = sorted(self.feature_root.rglob("*.pth"))
        if self.allowed_processed_roots:
            feature_paths = [
                path
                for path in feature_paths
                if path.relative_to(self.feature_root).parts
                and path.relative_to(self.feature_root).parts[0] in self.allowed_processed_roots
            ]
        if max_samples > 0:
            feature_paths = feature_paths[:max_samples]
        if not feature_paths:
            raise ValueError(f"No feature dumps found under {self.feature_root}")
        self.feature_paths = feature_paths
        self.sample_processed_roots = [
            path.relative_to(self.feature_root).parts[0] for path in self.feature_paths
        ]

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        feature_path = self.feature_paths[idx]
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)

        if "tokens" not in payload:
            raise KeyError(f"Missing 'tokens' in feature dump: {feature_path}")
        if "timesteps" not in payload:
            raise KeyError(f"Missing 'timesteps' in feature dump: {feature_path}")
        if "output_resolution" not in payload:
            raise KeyError(f"Missing 'output_resolution' in feature dump: {feature_path}")
        if "1st_frame_relpath" not in payload:
            raise KeyError(f"Missing '1st_frame_relpath' in feature dump: {feature_path}")

        tokens = payload["tokens"]
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected tokens with shape (V, seq_len, dim), got {tuple(tokens.shape)} in {feature_path}"
            )

        num_views, seq_len, embed_dim = (int(dim) for dim in tokens.shape)
        if num_views <= self.cond_num:
            raise ValueError(
                f"Need more than {self.cond_num} views per sample, got {num_views} in {feature_path}"
            )
        output_resolution = _normalize_resolution(payload["output_resolution"])
        expected_seq_len = _expected_seq_len(output_resolution)
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Token sequence length mismatch in {feature_path}: got {seq_len}, "
                f"expected {expected_seq_len} from output_resolution={output_resolution}"
            )

        timesteps = tuple(int(step) for step in payload["timesteps"])
        if len(timesteps) != num_views:
            raise ValueError(
                f"timesteps length {len(timesteps)} does not match num_views {num_views} in {feature_path}"
            )

        packed_tokens = tokens.permute(0, 2, 1).unsqueeze(-1).contiguous()
        return {
            "path": str(feature_path),
            "processed_root": self.sample_processed_roots[idx],
            "first_frame_relpath": str(payload["1st_frame_relpath"]),
            "timesteps": timesteps,
            "output_resolution": output_resolution,
            "tokens": packed_tokens,
        }

    def get_processed_root_index_groups(self) -> Dict[str, List[int]]:
        # Build a lookup from processed_root to the dataset indices that belong to it.
        grouped: Dict[str, List[int]] = defaultdict(list)
        for index, processed_root in enumerate(self.sample_processed_roots):
            grouped[processed_root].append(index)
        return dict(grouped)



