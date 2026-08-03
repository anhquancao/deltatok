# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import numpy as np
import torch


class BatchedRandomSampler:
    """ Random sampling under a constraint: each sample in the batch has the same feature, 
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True):
        self.batch_size = batch_size
        self.pool_size = pool_size

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size*world_size) if drop_last else N
        assert world_size == 1 or drop_last, 'must drop the last batch in distributed mode'

        # distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random feat_idxs (same across each batch)
        n_batches = (self.total_size+self.batch_size-1) // self.batch_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches)
        feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        feat_idxs = feat_idxs.ravel()[:self.total_size]

        # put them together
        idxs = np.c_[sample_idxs, feat_idxs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size-1) // (self.world_size * self.batch_size))
        idxs = idxs[self.rank*size_per_proc: (self.rank+1)*size_per_proc]

        yield from (tuple(idx) for idx in idxs)


def round_by(total, multiple, up=False):
    if up:
        total = total + multiple-1
    return (total//multiple) * multiple


class BatchedRandomSampleOccAny(BatchedRandomSampler):
    """Pins one aspect ratio per batch. Item shape (num_timesteps x named cams) is a
    dataset constant, so the resolution is the only thing a batch must agree on."""

    def __init__(self, dataset,
                 batch_size,
                 num_of_aspect_ratios,
                 world_size=1, rank=0, drop_last=True):
        super().__init__(dataset, batch_size, pool_size=None, world_size=world_size, rank=rank, drop_last=drop_last)
        self.num_of_aspect_ratios = num_of_aspect_ratios

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        resolution_idxs = rng.integers(self.num_of_aspect_ratios, size=n_batches)
        resolution_idxs = np.broadcast_to(resolution_idxs[:, None], (n_batches, self.batch_size))
        resolution_idxs = resolution_idxs.ravel()[:self.total_size]

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size - 1) // (self.world_size * self.batch_size))
        idxs = np.arange(self.rank * size_per_proc, (self.rank + 1) * size_per_proc)

        for i in idxs:
            yield (sample_idxs[i], resolution_idxs[i])


class DatasetAwareBatchSamplerOccAny(BatchedRandomSampler):
    """Variable-camera batching: a batch never spans shards, and pins one aspect ratio
    AND one cameras-per-timestep count. _get_views draws WHICH cameras per item, so a
    shared count is what keeps the items in a batch collatable (num_timesteps is a
    dataset constant, so equal vpt => equal view count)."""

    def __init__(self, dataset, batch_size, dataset_configs,
                 world_size=1, rank=0, drop_last=True):
        super().__init__(dataset, batch_size, pool_size=None, world_size=world_size, rank=rank, drop_last=drop_last)
        self.configs, self.cum_sizes = dataset_configs

    def _shard_bounds(self):
        """(start, end) per shard, from the concatenated dataset's cumulative sizes."""
        starts = [0, *self.cum_sizes[:-1]]
        return list(zip(starts, self.cum_sizes))

    def _n_batches_per_rank(self):
        # Whole batches per shard only: a shard's remainder is dropped, never merged
        # into the next shard's batch (that would mix rigs inside one batch).
        total = sum((end - start) // self.batch_size for start, end in self._shard_bounds())
        return total // self.world_size

    def __len__(self):
        return self._n_batches_per_rank() * self.batch_size

    def __iter__(self):
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        all_batches = []
        for config, (start_idx, end_idx) in zip(self.configs, self._shard_bounds()):
            shard_idxs = np.arange(start_idx, end_idx)
            rng.shuffle(shard_idxs)
            n_batches = len(shard_idxs) // self.batch_size
            if n_batches == 0:
                continue
            shard_idxs = shard_idxs[:n_batches * self.batch_size].reshape(n_batches, self.batch_size)

            # Fixed-camera shards keep the 2-tuple: items identical to the plain sampler,
            # only the batch grouping (never spanning shards) differs. The dataset
            # validated min <= max <= num_views_per_timestep, so no clamping here.
            vpt_cap = config['max_views_per_timestep']
            for b in range(n_batches):
                res_idx = int(rng.integers(config['num_of_aspect_ratios']))
                if vpt_cap is None:
                    all_batches.append([(int(s), res_idx) for s in shard_idxs[b]])
                else:
                    vpt = int(rng.integers(config['min_views_per_timestep'], vpt_cap + 1))
                    all_batches.append([(int(s), res_idx, vpt) for s in shard_idxs[b]])

        # Shuffle batch ORDER (not membership) so shards interleave across the epoch.
        order = rng.permutation(len(all_batches))
        n_per_proc = len(all_batches) // self.world_size  # equal count per rank; tail dropped
        for b in order[self.rank * n_per_proc: (self.rank + 1) * n_per_proc]:
            yield from all_batches[b]

