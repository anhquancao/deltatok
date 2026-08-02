# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# A dataset base class that you can easily resize and combine.
# --------------------------------------------------------
import numpy as np
# Bases from dust3r: a duplicate EasyDataset would shadow EasyDataset_OccAny.make_sampler via MRO.
from dust3r.datasets.base.easy_dataset import EasyDataset, CatDataset, MulDataset, ResizedDataset
from occany.datasets.batched_sampler import BatchedRandomSampleOccAny, DatasetAwareBatchSamplerOccAny


class EasyDataset_MUSt3R(EasyDataset):
    def __add__(self, other):
        left = self.datasets if isinstance(self, CatDataset_MUSt3R) else [self]
        right = other.datasets if isinstance(other, CatDataset_MUSt3R) else [other]
        return CatDataset_MUSt3R([*left, *right])

    def __rmul__(self, factor):
        return MulDataset_MUSt3R(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset_MUSt3R(factor, self)

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True, per_dataset_sampling=False):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet

        if per_dataset_sampling and hasattr(self, 'dataset_configs'):
            return DatasetAwareBatchSamplerOccAny(self, batch_size,
                                                  dataset_configs=self.dataset_configs,
                                                  world_size=world_size, rank=rank, drop_last=drop_last)

        num_of_aspect_ratios = len(self._resolutions)
        min_memory_num_views = self.min_memory_num_views
        max_memory_num_views = self.max_memory_num_views
        return BatchedRandomSampleOccAny(self, batch_size,
            num_of_aspect_ratios=num_of_aspect_ratios,
            min_memory_num_views=min_memory_num_views,
            max_memory_num_views=max_memory_num_views,
            world_size=world_size, rank=rank, drop_last=drop_last)


class CatDataset_MUSt3R(CatDataset, EasyDataset_MUSt3R):

    @property
    def min_memory_num_views(self):
        return self.datasets[0].min_memory_num_views

    @property
    def max_memory_num_views(self):
        return self.datasets[0].max_memory_num_views

    @property
    def dataset_configs(self):
        configs = []
        for dataset in self.datasets:
            config = {
                'min_memory_num_views': getattr(dataset, 'min_memory_num_views', 2),
                'max_memory_num_views': getattr(dataset, 'max_memory_num_views', 10),
                # Pin views-per-timestep per batch so all items collate to one size.
                'min_views_per_timestep': getattr(dataset, 'min_views_per_timestep', 1),
                'num_views_per_timestep': getattr(dataset, 'num_views_per_timestep', 1),
                # Sampling cap on cameras per timestep (<= num_views_per_timestep).
                'max_views_per_timestep': getattr(dataset, 'max_views_per_timestep', None),
                # When max_num_timesteps is set, sampler draws timesteps (not mem views).
                'min_num_timesteps': getattr(dataset, 'min_num_timesteps', 1),
                'max_num_timesteps': getattr(dataset, 'max_num_timesteps', None),
                'num_of_aspect_ratios': len(dataset._resolutions) if hasattr(dataset, '_resolutions') else 1,
                'resolutions': list(dataset._resolutions) if hasattr(dataset, '_resolutions') else [(512, 512)],
            }
            configs.append(config)
        return configs, self._cum_sizes

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


class MulDataset_MUSt3R(MulDataset, EasyDataset_MUSt3R):

    @property
    def min_memory_num_views(self):
        return self.dataset.min_memory_num_views

    @property
    def max_memory_num_views(self):
        return self.dataset.max_memory_num_views

    @property
    def min_views_per_timestep(self):
        return self.dataset.min_views_per_timestep

    @property
    def num_views_per_timestep(self):
        return self.dataset.num_views_per_timestep

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.dataset[idx[0] // self.multiplicator, *idx[1:]]
        else:
            return self.dataset[idx // self.multiplicator]


class ResizedDataset_MUSt3R(ResizedDataset, EasyDataset_MUSt3R):

    @property
    def min_memory_num_views(self):
        return self.dataset.min_memory_num_views

    @property
    def max_memory_num_views(self):
        return self.dataset.max_memory_num_views

    @property
    def min_views_per_timestep(self):
        return self.dataset.min_views_per_timestep

    @property
    def num_views_per_timestep(self):
        return self.dataset.num_views_per_timestep

    def __getitem__(self, idx):
        assert hasattr(self, '_idxs_mapping'), 'You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()'
        if isinstance(idx, tuple):
            return self.dataset[self._idxs_mapping[idx[0]], *idx[1:]]
        else:
            return self.dataset[self._idxs_mapping[idx]]
