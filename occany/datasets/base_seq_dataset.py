# Based on the code from MUSt3R (https://github.com/naver/must3r)
import torch
import os.path as osp
import numpy as np
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, is_good_type, view_name
from occany.utils.image_util import get_SAM3_transforms
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from occany.utils.helpers import project_lidar_world2camera
from dust3r.utils.geometry import depthmap_to_camera_coordinates
import pickle
from occany.datasets.easy_dataset import EasyDataset_MUSt3R
from torchvision.transforms.functional import to_tensor
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.geometry import affine_inverse_np
from occany.utils.helpers import intrinsics_c2w_to_raymap_np
from occany.utils.transforms import SeqColorJitter, ImgNorm
import copy


class BaseSeqDatasetMultiView(BaseStereoViewDataset, EasyDataset_MUSt3R):

    def __init__(self,
                 transform=ImgNorm,
                 num_views_per_timestep=1,
                 fixed_cams=None,
                 *args, ROOT, seq_pkl_name, num_timesteps,
                 distill_model_name=None,
                 select_scenes=None, exclude_scenes=None,
                 **kwargs):
        # Timesteps in the consecutive window every item returns. Keyword-only and
        # undefaulted: item shape is fixed, so each arm must state it.
        self.num_timesteps = num_timesteps
        # Physical cameras per timestep (sequence layout constant).
        self.num_views_per_timestep = num_views_per_timestep
        # Cameras are named, never sampled; None = the whole rig.
        self.cams = list(fixed_cams) if fixed_cams is not None else list(range(num_views_per_timestep))
        assert all(0 <= c < num_views_per_timestep for c in self.cams), (
            f"fixed_cams indices must lie in [0, num_views_per_timestep={num_views_per_timestep}); "
            f"got {self.cams}"
        )

        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        # distill_model_name=None disables distill-image generation (flow training
        # doesn't consume view['distill_img']); skips the extra per-view tensor.
        if distill_model_name is None or str(distill_model_name).lower() == "none":
            self.distill_img_transform = None
        elif distill_model_name == "SAM3":
            self.distill_img_transform = get_SAM3_transforms(resolution=518)
        else:
            raise ValueError(f"Unsupported distill_model_name: {distill_model_name}")
        self.seq_pkl_name = seq_pkl_name
        self._load_data()
        # Config-local scene filtering (default None = no-op), applied before any
        # subclass split filter so both compose. Carves a held-out val set without
        # touching the subclass split boundary: exclude_scenes on train, select_scenes on val.
        if select_scenes is not None:
            self.select_scene(select_scenes)
        if exclude_scenes is not None:
            self.select_scene(exclude_scenes, opposite=True)
        self.is_metric_scale = True
        print(f"{self.__class__.__name__}: num_timesteps={self.num_timesteps}, cams={self.cams}")

        # transform only selects whether a per-item color jitter runs; DA3 normalization
        # is unconditional afterwards.
        if isinstance(transform, str):
            transform = eval(transform)
        self.transform = transform  # inherited dust3r __repr__ reads it
        self.is_seq_color_jitter = transform == SeqColorJitter

    def __len__(self):
        return len(self.seqs)

    def get_stats(self):
        return f'{len(self)} seqs from {len(self.scenes)} scenes'

    def _resize_image_and_sparse_depthmap(self, image, depthmap, intrinsics, resolution, rng=None, info=None,
                                          pseudo_depthmap=None):
        image, _, intrinsics2 = self._crop_resize_if_necessary(
            image, depthmap, intrinsics, resolution, rng, info)

        def reproject(d):
            pts3d_cam, valid = depthmap_to_camera_coordinates(d, intrinsics)
            reproj, _, _, _ = project_lidar_world2camera(
                pts3d_cam[valid],
                img_h=image.height, img_w=image.width,
                camera_pose=np.eye(4),
                cam_K=intrinsics2,
            )
            return reproj.astype(np.float32)

        depthmap2 = reproject(depthmap)
        # Dense pseudo depth uses the same sparse-style reprojection because
        # depthmap_to_camera_coordinates filters with (depth > 0), so the 0
        # sentinel (sky / invalid) is dropped naturally.
        pseudo_depthmap2 = reproject(pseudo_depthmap) if pseudo_depthmap is not None else None
        return image, depthmap2, intrinsics2.astype(np.float32), pseudo_depthmap2

    def _load_data(self):
        assert self.seq_pkl_name is not None, "seq_pkl_name must be provided"

        with open(osp.join(self.ROOT, self.seq_pkl_name), 'rb') as f:
            data = pickle.load(f)
            self.scenes = data['scenes']
            self.frames = data['frames']
            self.seqs = data['seqs']
            # Drop records too short for the fixed window (else _get_views asserts mid-run)
            min_len = self.num_timesteps * self.num_views_per_timestep
            self.seqs = [seq for seq in self.seqs if len(seq[1]) >= min_len]

        print(f'Loaded {self.get_stats()}')

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = set()
        for s in scenes:
            if s in self.scenes:
                scene_id.add(self.scenes.index(s))
        if not scene_id and not opposite:
            raise AssertionError('no scene found')

        if opposite:
            valid_seqs = [seq for seq in self.seqs if seq[0] not in scene_id]
        else:
            valid_seqs = [seq for seq in self.seqs if seq[0] in scene_id]

        print(f"Selected {len(valid_seqs)} seqs from {len(self.seqs)}")
        self.seqs = valid_seqs



    def _get_views(self, seq_idx, resolution, rng):
        scene_idx, seq, _ = self.seqs[seq_idx]  # pkl stride offsets unused: labels are dense
        scene_name = self.scenes[scene_idx]
        preprocessed_scene_dir = osp.join(self.ROOT, scene_name)

        max_vpt = self.num_views_per_timestep
        avail = len(seq) // max_vpt  # timesteps physically in the record
        assert avail >= self.num_timesteps, (
            f"{scene_name}: record has {avail} timesteps < num_timesteps={self.num_timesteps}"
        )

        # One ordered, consecutive run of timesteps at a random offset in the record.
        start = int(rng.integers(0, avail - self.num_timesteps + 1))
        # Records are timestep-major, camera-minor: index = t * max_vpt + cam.
        memory_view_indices = [t * max_vpt + c
                               for t in range(start, start + self.num_timesteps)
                               for c in self.cams]

        frames = [seq[i] for i in memory_view_indices]
        # Dense labels: the window offset is a sampling detail, not a timestamp.
        times = [t for t in range(self.num_timesteps) for _ in self.cams]

        views = []
        for frame_index, t in zip(frames, times):
            frame_id = self.frames[frame_index]

            npz_path = osp.join(preprocessed_scene_dir, f"{frame_id}.npz")
            try:
                data = np.load(npz_path)
            except Exception:
                raise RuntimeError(f"Failed to load dataset sample: {npz_path}")


            image = data['image']          # The image array
            depthmap = data['depthmap']    # The depth map
            intrinsics = np.float32(data['intrinsics']) # Camera intrinsics matrix
            camera_pose = np.float32(data['cam2world'])  # Camera-to-world transformation matrix


            # Set skew term to 0
            intrinsics[0, 1] = 0.0
            intrinsics[1, 0] = 0.0

            image, depthmap, intrinsics, _ = self._resize_image_and_sparse_depthmap(
                image, depthmap, intrinsics, resolution, rng,
                info=(scene_name, frame_id),
            )

            view = dict(
                img=image,
                timestep=t,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset=self.__class__.__name__,
                scene_name=scene_name,
                frame_id=frame_id,
                label=f"{scene_name}_{frame_id}",
                instance=frame_id)
            views.append(view)

        return views

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, resolution_idx = idx  # the batch sampler pins the aspect ratio
        else:
            # This is used by test data as we don't implement the BatchSampler in test
            assert len(self._resolutions) == 1
            resolution_idx = 0
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded codez_far
        resolution = self._resolutions[resolution_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)

        # Build a PIL→PIL color jitter once so all views in this sample share the same params.
        # DA3 normalization is applied after it.
        pil_jitter = SeqColorJitter(normalize=False) if self.is_seq_color_jitter else None

        in_camera0 = affine_inverse_np(views[0]['camera_pose'])
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, resolution_idx, v)
            view['is_metric_scale'] = self.is_metric_scale
            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            if self.distill_img_transform is not None:
                view['distill_img'] = self.distill_img_transform(view['img'])
            img_pil = pil_jitter(view['img']) if pil_jitter is not None else view['img']
            view['img'] = InputProcessor.NORMALIZE(to_tensor(img_pil))

            # convert all camera poses to the coordinate frame of camera 0
            view['camera_pose'] = in_camera0 @ view['camera_pose']

            assert 'camera_intrinsics' in view
            assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = self.z_far

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # Generate GT raymap that matches pts3d computation
            gt_raymap = intrinsics_c2w_to_raymap_np(
                view['camera_intrinsics'],  # (3, 3)
                view['camera_pose'],        # (4, 4) - camera-to-world
                height,
                width,
            )  # (H, W, 6)
            view['gt_raymap'] = gt_raymap
            pts3d_from_raymap = view['depthmap'][..., None] * gt_raymap[..., :3] + gt_raymap[..., 3:]
            if valid_mask.any():
                max_diff = np.abs((pts3d - pts3d_from_raymap)[valid_mask]).max()
                assert max_diff < 1e-3, f"pts3d mismatch: max_diff={max_diff:.6e} for view {view_name(view)}"

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

        ret_views = [copy.deepcopy(view) for view in views]

        # last thing done!
        for view in ret_views:
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
     
        
        return ret_views

