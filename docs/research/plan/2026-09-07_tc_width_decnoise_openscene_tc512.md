# tc_width — decoder-noise tc512 arm + OpenScene trainval (front cam, tar store)

Created 2026-09-07 · thread `tc_width` · prior cycle:
[`../plan/2026-09-07_tc_width_decnoise_e2e_tc512.md`](../plan/2026-09-07_tc_width_decnoise_e2e_tc512.md)
· arm: `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8_openscene`
· jobs: _pending_

Adds `/gpfs/scratch/ehpc1001/occany_data_tar/openscene_trainval` to training and
`.../openscene_test` to `test_dataset`, both front camera only (`fixed_cams=[0]` = CAM_F0). OpenScene is a
tar store (one uncompressed `<scene>.tar.gz` + `.idx.npz` per scene); `../OccAny` already reads it, so the
reader is ported. Checked on BSC: pkl schema is `scenes/frames/seqs`, a record is 80 = 10 slots × 8 cams
timestep-major, members are `<scene>/<frame_id>.npz` with `image/depthmap/intrinsics/cam2world`.
trainval: 19376 scenes / 550506 seqs; test: 2026 scenes / 57185 seqs.

## 1 `occany/datasets/tar_store.py` (new)

```bash
cp ../OccAny/occany/datasets/tar_store.py occany/datasets/tar_store.py
```
Verbatim. Numpy-only. Reader: `get_store(scene_tar_path(root, scene)).read("<scene>/<file>")`.

## 2 `occany/datasets/base_seq_dataset.py`

Imports, after `:10 import pickle`:
```python
import io
from occany.datasets import tar_store
```

`__init__` signature `:29-32`: add `use_tar=False,` after `select_scenes=None, exclude_scenes=None,`.
After `:60 self.ROOT = ROOT`:
```python
        self.use_tar = use_tar  # frames from one uncompressed tar per scene (see tar_store)
```

`_get_views` `:166`, after `preprocessed_scene_dir = ...`:
```python
        store = tar_store.get_store(
            tar_store.scene_tar_path(self.ROOT, scene_name)) if self.use_tar else None
```

`_get_views` `:195-199`, replace the `npz_path` / `np.load` block with:
```python
            if store is None:
                npz_path = osp.join(preprocessed_scene_dir, f"{frame_id}.npz")
                try:
                    data = np.load(npz_path)
                except Exception:
                    raise RuntimeError(f"Failed to load dataset sample: {npz_path}")
            else:
                npz_path = f"{store.tar_path}::{scene_name}/{frame_id}.npz"
                try:
                    data = np.load(io.BytesIO(store.read(f"{scene_name}/{frame_id}.npz")))
                except Exception:
                    raise RuntimeError(f"Failed to load dataset sample: {npz_path}")
```
Same shape as `../OccAny/occany/datasets/base_seq_dataset.py:540-552`. `use_tar=False` path is unchanged.

## 3 `occany/datasets/openscene_pairs.py` (new)

```bash
cp occany/datasets/once_pairs.py occany/datasets/openscene_pairs.py
```
Edit to:
```python
# OpenScene (nuPlan-derived, 8-cam rig: 0=CAM_F0). Preprocessed by ../OccAny
# dataset_setup/openscene/preprocess_openscene.py into one tar per scene (use_tar=True).
from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


class OpenSceneSeqMultiView(BaseSeqDatasetMultiView):
    def __init__(self, *args, OPENSCENE_PREPROCESSED_ROOT,
                 seq_pkl_name='seq_surround_temporal_sub1_stride9_fs1_openscene_all.pkl',
                 num_views_per_timestep=8, **kwargs):
        super().__init__(*args, ROOT=OPENSCENE_PREPROCESSED_ROOT, seq_pkl_name=seq_pkl_name,
                         num_views_per_timestep=num_views_per_timestep, **kwargs)
        self.is_metric_scale = True  # depths are projected nuPlan lidar
        # No select_scene: trainval and test are separate roots (scene names collide across splits).
```

## 4 `occany/datasets/__init__.py`

After `from .once_pairs import OnceSeqMultiView  # noqa: F401`:
```python
from .openscene_pairs import OpenSceneSeqMultiView  # noqa: F401
```

## 5 `configs/deltatok/train_deltatok_nt10_openscene_bsc.yaml` (new)

```bash
cp configs/deltatok/train_deltatok_nt10_bsc.yaml configs/deltatok/train_deltatok_nt10_openscene_bsc.yaml
```
`defaults` → `- train_deltatok_nt10_bsc`. Hydra cannot append to a string, so restate both strings.
`train_dataset`: the five nt10 entries unchanged, plus
```yaml
      4000 @ OpenSceneSeqMultiView(OPENSCENE_PREPROCESSED_ROOT='/gpfs/scratch/ehpc1001/occany_data_tar/openscene_trainval', \
      seq_pkl_name='seq_surround_temporal_sub1_stride9_fs1_openscene_trainval_all.pkl', \
      num_timesteps=10, num_views_per_timestep=8, fixed_cams=[0], use_tar=True, \
      z_far=50, split='train', \
      resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)])"
```
`test_dataset`: the KITTI + nuScenes entries from `train_deltatok_bsc.yaml:39-48` unchanged, plus
```yaml
      206 @ OpenSceneSeqMultiView(OPENSCENE_PREPROCESSED_ROOT='/gpfs/scratch/ehpc1001/occany_data_tar/openscene_test', \
      seq_pkl_name='seq_surround_temporal_sub1_stride9_fs1_openscene_test_all.pkl', \
      num_timesteps=5, num_views_per_timestep=8, fixed_cams=[0], use_tar=True, \
      z_far=50, split='val', seed=42, \
      resolution=[(518, 294)])"
```
And
```yaml
training:
  max_iter: 137500  # 22000 items / 16 = 1375 iters/epoch x 100 (was 18000 -> 112500)
```
`max_iter` is both the hard stop and the cosine horizon (`train_deltatok_bsc.yaml:57-61`), so it moves with
the epoch length.

## 6 `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc.slurm` (new)

```bash
cp slurm/deltatok/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc.slurm \
   slurm/deltatok/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc.slurm
```
Change only:
```
--job-name=deltatok_decnoise_os_bsc
--output/--error = slurm/output/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc_%j.{out,err}
header: "+ OpenScene trainval (front cam, tar store) in train, openscene_test in eval"
export CONFIG_NAME=train_deltatok_nt10_openscene_bsc
RUN_NAME=...compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}_openscene
```

## 7 Pre-flight on BSC (user syncs), then submit

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n use_tar occany/datasets/base_seq_dataset.py && grep -n OpenScene occany/datasets/__init__.py && ls occany/datasets/tar_store.py occany/datasets/openscene_pairs.py && grep -n \"openscene\|max_iter\" configs/deltatok/train_deltatok_nt10_openscene_bsc.yaml && grep -E \"CONFIG_NAME=|RUN_NAME=|--time|--account\" slurm/deltatok/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc.slurm'"
```

Watch until the first `[KEpoch` line. The `.out` must show `Loaded ... OpenSceneSeqMultiView` with 550506 seqs
(train) and 57185 (test), three `Building test datasets` entries, and `[Eval/206 @ OpenSceneSeqMultiView...]`
after epoch 1.

## Assumptions

- Single camera applies to the OpenScene entries only; the other five train sets keep the twin's
  `min/max_views_per_timestep=1..2`.
- Epoch grows 18000 → 22000 items: at the twin's 0.61 h/epoch the 40 h wall reaches ~53 epochs, not 66.
- Eval viz for OpenScene falls back to a plain vertical stack (`occany/utils/vis_util.py:680`); fine for one cam.
