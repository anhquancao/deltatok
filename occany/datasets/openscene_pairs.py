# OpenScene (nuPlan-derived, 8-cam rig: 0=CAM_F0). Preprocessed by ../OccAny
# dataset_setup/openscene/preprocess_openscene.py into one tar per scene (use_tar=True).
from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


class OpenSceneSeqMultiView(BaseSeqDatasetMultiView):
    def __init__(self, *args, OPENSCENE_PREPROCESSED_ROOT, seq_pkl_name,
                 num_views_per_timestep=8, **kwargs):
        # No split filter: trainval and test are separate roots (scene names collide).
        super().__init__(*args, ROOT=OPENSCENE_PREPROCESSED_ROOT, seq_pkl_name=seq_pkl_name,
                         num_views_per_timestep=num_views_per_timestep, **kwargs)
        self.is_metric_scale = True  # depths are projected nuPlan lidar
