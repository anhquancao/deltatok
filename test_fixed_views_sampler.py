"""Quick test for DatasetAwareBatchSamplerFixedViews.

Run on Karolina: python test_fixed_views_sampler.py
"""
import sys
sys.path.insert(0, 'third_party/dust3r')
sys.path.insert(0, 'third_party/Depth-Anything-3')
sys.path.insert(0, 'third_party/croco')
sys.path.insert(0, 'third_party/GLD/src')

import numpy as np
from dust3r.datasets.base.batched_sampler import DatasetAwareBatchSamplerFixedViews


class FakeDataset:
    def __len__(self):
        return 1000


def test_basic():
    dataset = FakeDataset()
    batch_size = 4

    # 3 datasets with different camera counts:
    # Dataset 0 (Waymo):    items [0, 400), 5 cameras, max_mem=20
    # Dataset 1 (DDAD):     items [400, 800), 6 cameras, max_mem=24
    # Dataset 2 (VKitti):   items [800, 1000), 1 camera, max_mem=10
    configs = [
        {
            'num_views_per_timestep': 5,
            'min_memory_num_views': 5,
            'max_memory_num_views': 20,
            'num_of_aspect_ratios': 3,
            'resolutions': [(518, 294), (518, 280), (518, 168)],
            'ray_map_prob': -1.0,
            'ray_map_idx': [],
            'min_num_timesteps': 2,
        },
        {
            'num_views_per_timestep': 6,
            'min_memory_num_views': 6,
            'max_memory_num_views': 24,
            'num_of_aspect_ratios': 3,
            'resolutions': [(518, 294), (518, 280), (518, 168)],
            'ray_map_prob': -1.0,
            'ray_map_idx': [],
            'min_num_timesteps': 2,
        },
        {
            'num_views_per_timestep': 1,
            'min_memory_num_views': 2,
            'max_memory_num_views': 10,
            'num_of_aspect_ratios': 3,
            'resolutions': [(518, 294), (518, 280), (518, 168)],
            'ray_map_prob': -1.0,
            'ray_map_idx': [],
            'min_num_timesteps': 2,
        },
    ]
    cum_sizes = [400, 800, 1000]

    sampler = DatasetAwareBatchSamplerFixedViews(
        dataset, batch_size,
        dataset_configs=(configs, cum_sizes),
        min_num_timesteps=2,
        world_size=1, rank=0, drop_last=True,
    )
    sampler.set_epoch(0)

    all_tuples = list(sampler)
    print(f"Total tuples emitted: {len(all_tuples)}")

    batch_count = 0
    for i in range(0, len(all_tuples), batch_size):
        batch = all_tuples[i:i + batch_size]
        if len(batch) < batch_size:
            break

        # All items in batch must have same memory_num_views and vpt
        mem_views = set(t[2] for t in batch)
        vpts = set(t[4] for t in batch)
        assert len(mem_views) == 1, f"Batch {batch_count}: inconsistent memory_num_views {mem_views}"
        assert len(vpts) == 1, f"Batch {batch_count}: inconsistent vpt {vpts}"

        vpt = batch[0][4]
        mv = batch[0][2]
        # memory_num_views must be divisible by vpt
        assert mv % vpt == 0, f"Batch {batch_count}: memory_num_views={mv} not divisible by vpt={vpt}"
        T = mv // vpt
        # T must respect min_num_timesteps
        assert T >= 2, f"Batch {batch_count}: T={T} < min_num_timesteps=2"

        # All items must come from datasets with cam_count >= vpt
        for t in batch:
            idx = t[0]
            if idx < 400:
                cam = 5  # Waymo
            elif idx < 800:
                cam = 6  # DDAD
            else:
                cam = 1  # VKitti
            assert cam >= vpt, (
                f"Batch {batch_count}: item {idx} has cam_count={cam} < vpt={vpt}"
            )

        batch_count += 1

    print(f"Total batches: {batch_count}")
    print(f"All batches have consistent (memory_num_views, vpt) ✓")
    print(f"All memory_num_views = T * vpt ✓")
    print(f"All T >= min_num_timesteps ✓")
    print(f"All items from eligible datasets ✓")

    # Print distribution of vpt values
    vpt_counts = {}
    for i in range(0, batch_count * batch_size, batch_size):
        vpt = all_tuples[i][4]
        vpt_counts[vpt] = vpt_counts.get(vpt, 0) + 1
    print(f"\nVpt distribution across batches: {dict(sorted(vpt_counts.items()))}")

    # Verify mixing: vpt=1 batches should contain items from all 3 datasets
    vpt1_datasets = set()
    for i in range(0, batch_count * batch_size, batch_size):
        if all_tuples[i][4] == 1:
            for j in range(batch_size):
                idx = all_tuples[i + j][0]
                if idx < 400:
                    vpt1_datasets.add('Waymo')
                elif idx < 800:
                    vpt1_datasets.add('DDAD')
                else:
                    vpt1_datasets.add('VKitti')
    print(f"Datasets present in vpt=1 batches: {vpt1_datasets}")
    assert 'VKitti' in vpt1_datasets, "VKitti should appear in vpt=1 batches"


if __name__ == "__main__":
    test_basic()
    print("\nAll tests passed!")
