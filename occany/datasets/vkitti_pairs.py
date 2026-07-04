# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WayMo
# dataset at https://github.com/waymo-research/waymo-open-dataset
# See datasets_preprocess/preprocess_waymo.py
# --------------------------------------------------------
from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


class VKittiSeqMultiView(BaseSeqDatasetMultiView):
    def __init__(self, *args, VKITTI_PROCESSED_ROOT, seq_pkl_name='vkitti_seq_video.pkl', num_views_per_timestep=1, **kwargs):
        # num_views_per_timestep = physical cameras per timestep (VKitti mono: 1).
        super().__init__(*args, ROOT=VKITTI_PROCESSED_ROOT, seq_pkl_name=seq_pkl_name, num_views_per_timestep=num_views_per_timestep, **kwargs)
        self.is_metric_scale = True
