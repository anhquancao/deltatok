# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# A dataset base class that you can easily resize and combine.
# --------------------------------------------------------
import numpy as np
# Bases from dust3r: a duplicate EasyDataset would shadow EasyDataset_MUSt3R.make_sampler via MRO.
from dust3r.datasets.base.easy_dataset import EasyDataset, CatDataset, MulDataset, ResizedDataset
from occany.datasets.batched_sampler import BatchedRandomSampleOccAny, DatasetAwareBatchSamplerOccAny


class _ForwardsCameraSampling:
    """Size wrappers hide the shard's camera-sampling range; the sampler needs it."""

    @property
    def min_views_per_timestep(self):
        return getattr(self.dataset, 'min_views_per_timestep', 1)

    @property
    def max_views_per_timestep(self):
        return getattr(self.dataset, 'max_views_per_timestep', None)


class EasyDataset_MUSt3R(EasyDataset):
    def __add__(self, other):
        left = self.datasets if isinstance(self, CatDataset_MUSt3R) else [self]
        right = other.datasets if isinstance(other, CatDataset_MUSt3R) else [other]
        return CatDataset_MUSt3R([*left, *right])

    def __rmul__(self, factor):
        return MulDataset_MUSt3R(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset_MUSt3R(factor, self)

    def _sampler_config(self):
        """One shard's per-batch draw ranges: aspect ratio, and cameras per timestep."""
        return {
            'num_of_aspect_ratios': len(self._resolutions),
            'min_views_per_timestep': getattr(self, 'min_views_per_timestep', 1),
            'max_views_per_timestep': getattr(self, 'max_views_per_timestep', None),
        }

    @property
    def dataset_configs(self):
        # A lone dataset is a single shard spanning the whole index range.
        return [self._sampler_config()], [len(self)]

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True,
                     per_dataset_sampling=False):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet

        configs, cum_sizes = self.dataset_configs
        if per_dataset_sampling:
            # Batches never span shards, and each pins one cameras-per-timestep count.
            return DatasetAwareBatchSamplerOccAny(self, batch_size, (configs, cum_sizes),
                world_size=world_size, rank=rank, drop_last=drop_last)

        # Without it the dataset never receives a count and silently returns the whole rig.
        assert all(c['max_views_per_timestep'] is None for c in configs), (
            "max_views_per_timestep requires dataset.per_dataset_sampling=true"
        )
        # Item shape is fixed (num_timesteps x named cams), so the aspect ratio is
        # the only thing a batch must agree on.
        return BatchedRandomSampleOccAny(self, batch_size,
            num_of_aspect_ratios=len(self._resolutions),
            world_size=world_size, rank=rank, drop_last=drop_last)


class CatDataset_MUSt3R(CatDataset, EasyDataset_MUSt3R):
    # Shards must agree on item shape (num_timesteps x len(cams)) unless batches are
    # per-shard; CatDataset._resolutions already asserts they agree on resolutions.

    @property
    def dataset_configs(self):
        return [d._sampler_config() for d in self.datasets], list(self._cum_sizes)

    def __getitem__(self, idx):
        other = None
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, 'right')
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)
        if other is not None:
            new_idx = (new_idx, *other)
        return dataset[new_idx]


class MulDataset_MUSt3R(MulDataset, _ForwardsCameraSampling, EasyDataset_MUSt3R):

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.dataset[idx[0] // self.multiplicator, *idx[1:]]
        else:
            return self.dataset[idx // self.multiplicator]


class ResizedDataset_MUSt3R(ResizedDataset, _ForwardsCameraSampling, EasyDataset_MUSt3R):

    def set_epoch(self, epoch):
        """Read one continuous shuffled stream instead of dust3r's fresh-permutation-
        per-epoch, which re-drew the first new_size and left new_size<<len(dataset)
        arms covering only 1-(1-new_size/n)^epochs of the data. Pass p is permuted with
        seed p+777, so epoch 0 is unchanged and every item is seen once per pass."""
        n = len(self.dataset)
        if n == 0:
            raise ZeroDivisionError(
                f"ResizedDataset error: self.dataset {repr(self.dataset)} has length 0. "
                f"Cannot resize to {self.new_size}.")
        chunks, taken = [], 0
        while taken < self.new_size:
            # Global stream position -> which pass, and how far into it.
            p, off = divmod(epoch * self.new_size + taken, n)
            perm = np.random.default_rng(seed=p + 777).permutation(n)
            chunk = perm[off:off + (self.new_size - taken)]
            chunks.append(chunk)
            taken += len(chunk)
        self._idxs_mapping = np.concatenate(chunks)

    def __getitem__(self, idx):
        assert hasattr(self, '_idxs_mapping'), 'You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()'
        if isinstance(idx, tuple):
            return self.dataset[self._idxs_mapping[idx[0]], *idx[1:]]
        else:
            return self.dataset[self._idxs_mapping[idx]]
