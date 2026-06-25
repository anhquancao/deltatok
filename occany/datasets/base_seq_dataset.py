# Based on the code from MUSt3R (https://github.com/naver/must3r)
import torch
import os.path as osp
import numpy as np
import cv2
from dust3r.utils.image import imread_cv2
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, transpose_to_landscape, is_good_type, view_name
from occany.loss.infinidepth_pseudo import INFINIDEPTH_PSEUDO_SCALE, INFINIDEPTH_PSEUDO_SUFFIX
from occany.utils.image_util import get_SAM3_transforms
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from occany.utils.helpers import project_lidar_world2camera, get_ray_map_lsvm
from dust3r.utils.geometry import depthmap_to_camera_coordinates
import pickle
from dust3r.datasets.base.easy_dataset import EasyDataset, CatDataset_MUSt3R, MulDataset_MUSt3R, ResizedDataset_MUSt3R
from dust3r.datasets.base.batched_sampler import DatasetAwareBatchSamplerOccAny
from torchvision.transforms.functional import to_tensor
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.geometry import affine_inverse_np
from occany.utils.helpers import intrinsics_c2w_to_raymap_np
from occany.utils.transforms import SeqColorJitter, ImgNorm
import copy
import math
import warnings


class EasyDataset_OccAny(EasyDataset):
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
        ray_map_prob = self.ray_map_prob
        ray_map_idx = self.ray_map_idx
        return BatchedRandomSampleOccAny(self, batch_size,
            num_of_aspect_ratios=num_of_aspect_ratios,
            min_memory_num_views=min_memory_num_views,
            max_memory_num_views=max_memory_num_views,
            ray_map_prob=ray_map_prob,
            ray_map_idx=ray_map_idx,
            world_size=world_size, rank=rank, drop_last=drop_last)




class BaseSeqDataset (BaseStereoViewDataset):

    def __init__(self, *args, ROOT, seq_pkl_name,
                 distill_model_name="SAM2", distill_img_size=None, img_size=512,
                 base_model="must3r",
                 max_seqs=None,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.base_model = base_model
        self.distill_model_name = distill_model_name
        # resolve distillation image size based on model type
        if distill_img_size is None:
            if distill_model_name == "SAM2":
                distill_img_size = img_size
            elif distill_model_name == "SAM3":
                distill_img_size = 518
        self.distill_img_size = distill_img_size
        # distill_model_name=None disables distill-image generation (flow training
        # doesn't consume view['distill_img']); skips the extra per-view tensor.
        if distill_model_name is None or str(distill_model_name).lower() == "none":
            self.distill_img_transform = None
        elif distill_model_name == "SAM3":
            self.distill_img_transform = get_SAM3_transforms(resolution=self.distill_img_size)
        else:
            raise ValueError(f"Unsupported distill_model_name: {distill_model_name}")
        self.seq_pkl_name = seq_pkl_name
        self.img_ext = ".jpg"
        self.num_views = 3
        self._load_data()
        # `max_seqs` is applied later via `_truncate_to_max_seqs()` so it runs
        # AFTER any split-based scene selection a subclass performs in its own
        # __init__ (e.g. Occ3dNuscenesSeqMultiView). Truncating here would keep
        # the first sequences across all scenes, which the split filter then
        # drops, yielding an empty dataset.
        self.max_seqs = max_seqs
        self.is_metric_scale = True

    def _truncate_to_max_seqs(self):
        # Overfit/smoke knob: keep only the first `max_seqs` sequences so the
        # dataset can be pinned to a tiny fixed subset. Call this AFTER all
        # scene/split selection so the kept sequences come from the final set.
        if self.max_seqs is not None:
            self.seqs = self.seqs[: self.max_seqs]
            print(f"Truncated to {len(self.seqs)} seqs (max_seqs={self.max_seqs})")
    
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
            # Filter sequences with length less than 3
            self.seqs = [seq for seq in self.seqs if len(seq[1]) >= 3]

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

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            # view['idx'] = (idx, ar_idx, v)
            view['is_metric_scale'] = self.is_metric_scale
            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            if self.distill_img_transform is not None:
                view['distill_img'] = self.distill_img_transform(view['img'])
            if self.base_model == 'da3':
                
                view['img'] = InputProcessor.NORMALIZE(to_tensor(view['img']))
            else:
                view['img'] = self.transform(view['img'])


            assert 'camera_intrinsics' in view
            assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = self.z_far
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"


        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views

class BaseSeqDatasetMultiView(BaseSeqDataset, EasyDataset_OccAny):
    def __init__(self,
        ray_map_prob=0.0,
        ray_map_idx=None,
        recon_view_idx=None,
        reverse_seq=False,
        shuffle_seq_prob=0.0,
        transform=ImgNorm,
        frame_interval=1,
        max_memory_num_views=10,
        min_memory_num_views=5,
        num_views_per_timestep=1,
        min_views_per_timestep=1,
        min_num_timesteps=1,
        anchor_cam=0,
        fixed_cams=None,
        no_partial_views=False,
        load_infinidepth_pseudo=False,
        *args, **kwargs):
        self.max_memory_num_views = max_memory_num_views
        self.min_memory_num_views = min_memory_num_views
        self.num_views_per_timestep = num_views_per_timestep
        self.min_views_per_timestep = min_views_per_timestep
        self.min_num_timesteps = min_num_timesteps
        self.anchor_cam = anchor_cam
        self.fixed_cams = list(fixed_cams) if fixed_cams is not None else None
        if self.fixed_cams is not None:
            assert all(0 <= c < num_views_per_timestep for c in self.fixed_cams), (
                f"fixed_cams indices must lie in [0, num_views_per_timestep={num_views_per_timestep}); "
                f"got {self.fixed_cams}"
            )
        self.no_partial_views = no_partial_views
        self.load_infinidepth_pseudo = load_infinidepth_pseudo

        super().__init__(*args, **kwargs)
        self.reverse_seq = reverse_seq
        self.shuffle_seq_prob = shuffle_seq_prob
        self.recon_view_idx = recon_view_idx
        self.ray_map_prob = ray_map_prob
        self.ray_map_idx = [] if ray_map_idx is None else list(ray_map_idx)
        self.frame_interval = frame_interval
        print(f"{self.__class__.__name__}: reverse_seq={self.reverse_seq}, shuffle_seq_prob={self.shuffle_seq_prob}, anchor_cam={self.anchor_cam}")

        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            transform = eval(transform)
        if transform == SeqColorJitter:
            self.is_seq_color_jitter = True
        self.transform = transform



    def _select_cameras(self, max_vpt, actual_vpt, rng):
        anchor = self.anchor_cam
        if actual_vpt >= max_vpt:
            others = np.arange(max_vpt)
            others = others[others != anchor]
            rng.shuffle(others)
            return np.concatenate([[anchor], others])
        if actual_vpt == 1:
            return np.array([anchor])
        others = np.arange(max_vpt)
        others = others[others != anchor]
        rest = rng.choice(others, size=actual_vpt - 1, replace=False)
        return np.concatenate([[anchor], rest])

    def _build_view_indices(self, chosen_timesteps, selected_cams, max_vpt, partial, rng):
        if partial:
            partial_idx = int(rng.integers(1, len(chosen_timesteps))) if len(chosen_timesteps) > 1 else 0
        else:
            partial_idx = -1
        indices = []
        for i, t in enumerate(chosen_timesteps):
            base = t * max_vpt
            if i == partial_idx:
                if i == 0:
                    rest = rng.choice(selected_cams[1:], size=partial - 1, replace=False) if partial > 1 else np.array([], dtype=int)
                    cams = np.concatenate([selected_cams[:1], rest])
                else:
                    cams = rng.choice(selected_cams, size=partial, replace=False)
            else:
                cams = selected_cams
            indices.extend(int(base + c) for c in cams)
        return indices

    def _get_views(self, seq_idx, resolution, memory_num_views, rng, views_per_timestep=None):
        scene_idx, seq, ts = self.seqs[seq_idx]
        scene_name = self.scenes[scene_idx]
        preprocessed_scene_dir = osp.join(self.ROOT, scene_name)

        seq_len = len(seq)
        max_vpt = self.num_views_per_timestep
        if self.fixed_cams is not None:
            selected_cams = np.array(self.fixed_cams, dtype=int)
            actual_vpt = len(selected_cams)
        elif views_per_timestep is not None:
            # Per-batch pinned count (clamped to the configured range): drawing it
            # per item here is what made batch items differ in size and broke
            # default_collate for batch_size > 1. The chosen cameras still vary.
            actual_vpt = int(np.clip(views_per_timestep, self.min_views_per_timestep, max_vpt))
            selected_cams = self._select_cameras(max_vpt, actual_vpt, rng)
        else:
            actual_vpt = rng.integers(self.min_views_per_timestep, max_vpt + 1)
            selected_cams = self._select_cameras(max_vpt, actual_vpt, rng)
        memory_num_views = max(memory_num_views, actual_vpt * self.min_num_timesteps)
        num_timesteps = seq_len // max_vpt
        assert num_timesteps > 0, (
            f"seq_len ({seq_len}) must be >= num_views_per_timestep ({max_vpt})"
        )

        do_reverse = self.reverse_seq and rng.random() < 0.5

        if self.no_partial_views:
            # Truncate memory_num_views down to a clean multiple of actual_vpt so
            # every timestep is full (no partial-camera timestep).
            effective_memory = (memory_num_views // actual_vpt) * actual_vpt
            assert effective_memory > 0, (
                f"no_partial_views=True requires memory_num_views ({memory_num_views}) "
                f">= actual_vpt ({actual_vpt})"
            )
            partial = 0
            timesteps_needed = min(effective_memory // actual_vpt, num_timesteps)
            effective_memory = timesteps_needed * actual_vpt
        else:
            effective_memory = memory_num_views
            partial = memory_num_views % actual_vpt
            timesteps_needed = min(math.ceil(memory_num_views / actual_vpt), num_timesteps)

        anchor = num_timesteps - 1 if do_reverse else 0
        candidates = np.delete(np.arange(num_timesteps), anchor)

        chosen_timesteps = [anchor]
        if timesteps_needed > 1:
            chosen_timesteps += list(rng.choice(candidates, size=timesteps_needed - 1, replace=False))
        chosen_timesteps = sorted((int(t) for t in chosen_timesteps), reverse=do_reverse)

        memory_view_indices = self._build_view_indices(chosen_timesteps, selected_cams, max_vpt, partial, rng)
        memory_view_indices = memory_view_indices[:effective_memory]

        frames = [seq[i] for i in memory_view_indices]
        times = [ts[i] for i in memory_view_indices]

        views = []
        for i, (frame_index, t) in enumerate(zip(frames, times)):
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

            pseudo_depthmap = None
            if self.load_infinidepth_pseudo:
                pseudo_path = osp.join(
                    preprocessed_scene_dir, f"{frame_id}{INFINIDEPTH_PSEUDO_SUFFIX}"
                )
                pseudo_u16 = imread_cv2(pseudo_path, cv2.IMREAD_UNCHANGED)
                if pseudo_u16.dtype != np.uint16:
                    raise RuntimeError(
                        f"Expected uint16 InfiniDepth pseudo-depth, got dtype={pseudo_u16.dtype} "
                        f"at {pseudo_path}"
                    )
                if pseudo_u16.shape[:2] != (image.shape[0], image.shape[1]):
                    raise RuntimeError(
                        f"InfiniDepth pseudo-depth shape {pseudo_u16.shape[:2]} does not match "
                        f"image shape {(image.shape[0], image.shape[1])} at {pseudo_path}"
                    )
                pseudo_depthmap = pseudo_u16.astype(np.float32) / INFINIDEPTH_PSEUDO_SCALE

            image, depthmap, intrinsics, pseudo_depthmap = self._resize_image_and_sparse_depthmap(
                image, depthmap, intrinsics, resolution, rng,
                info=(scene_name, frame_id), pseudo_depthmap=pseudo_depthmap,
            )

            view = dict(
                img=image,
                timestep=t,
                views_per_timestep=int(actual_vpt),  # sampler-pinned vpt; for num_cameras verification
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset=self.__class__.__name__,
                scene_name=scene_name,
                frame_id=frame_id,
                label=f"{scene_name}_{frame_id}",
                instance=frame_id)
            if pseudo_depthmap is not None:
                view['pseudo_depthmap'] = pseudo_depthmap
            views.append(view)
        
        return views

    def __getitem__(self, idx):
        # views_per_timestep is pinned per batch by DatasetAwareBatchSamplerOccAny
        # (5-tuple) so all items in a batch return the same view count; None means
        # the legacy 4-tuple / eval int path, where _get_views draws it per item.
        views_per_timestep = None
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio (+ per-batch view budget)
            if len(idx) == 5:
                idx, resolution_idx, memory_num_views, views_per_timestep, ray_map_idx = idx
            else:
                idx, resolution_idx, memory_num_views, ray_map_idx = idx
        else:
            # This is used by test data as we don't implement the BatchSampler in test
            assert len(self._resolutions) == 1
            resolution_idx = 0
            assert self.min_memory_num_views == self.max_memory_num_views, "Evaluation needs to be done with a fixed number of views, which is equal to min_memory_num_views and  min_memory_num_views must equal max_memory_num_views"
            memory_num_views = self.min_memory_num_views
            # assert len(self.ray_map_idx) != 0, "Evaluation needs to be done with fixed ray_map_idx"
            ray_map_idx = self.ray_map_idx
        
        assert all(ray_map_id < memory_num_views for ray_map_id in ray_map_idx), f"ray_map_idx should be smaller than memory_num_views ray_map_idx={ray_map_idx}, memory_num_views={memory_num_views}"
        # idx, ar_idx, memory_num_views = 290, 0, 10 # TODO: remove later as this is only for overfitting 1 example
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded codez_far
        resolution = self._resolutions[resolution_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, memory_num_views, self._rng, views_per_timestep=views_per_timestep)
        # _get_views may bump memory_num_views up (min_num_timesteps) or return
        # fewer views when the source sequence is shorter than the requested
        # budget. Sync to the actual returned count so view['idx'] and
        # view['memory_num_views'] reflect reality, and drop ray_map indices
        # that fall past it so gen_view_idx lookups stay in range.
        memory_num_views = len(views)
        ray_map_idx = [i for i in ray_map_idx if i < len(views)]

        # Build a PIL→PIL color jitter once so all views in this sample share the same params.
        # Normalization is then handled by the base_model branch below.
        pil_jitter = SeqColorJitter(normalize=False) if self.is_seq_color_jitter else None

        in_camera0 = affine_inverse_np(views[0]['camera_pose'])
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, resolution_idx, v, memory_num_views, ray_map_idx)
            view['memory_num_views'] = memory_num_views
            view['is_metric_scale'] = self.is_metric_scale
            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            if self.distill_img_transform is not None:
                view['distill_img'] = self.distill_img_transform(view['img'])
            img_pil = pil_jitter(view['img']) if pil_jitter is not None else view['img']
            if self.base_model == 'da3':
                view['img'] = InputProcessor.NORMALIZE(to_tensor(img_pil))
            elif pil_jitter is not None:
                view['img'] = ImgNorm(img_pil)
            else:
                view['img'] = self.transform(img_pil)

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

        if self.recon_view_idx is None:
            recon_view_idx = [i for i in range(len(views)) if i not in ray_map_idx]
            # use the dataset's RNG for reproducibility across workers
            gen_view_idx = ray_map_idx
        else:
            # only during test that we explicit pass recon_view_idx
            gen_view_idx = ray_map_idx
            recon_view_idx = self.recon_view_idx

        camera_poses = np.stack([view['camera_pose'] for view in views], axis=0)
        ret_views = []
        for v in recon_view_idx:
            view = copy.deepcopy(views[v])
            view['is_raymap'] = False
            ret_views.append(view)

        for v in gen_view_idx:
            view = copy.deepcopy(views[v])
            # get ray map — use this view's own intrinsics (multi-cam K can vary)
            K = views[v]['camera_intrinsics']
            camera_pose = torch.from_numpy(camera_poses[v])[None, None, :, :]
            fxfycxcy = torch.zeros((1, 1, 4), dtype=torch.float32)
            K_torch = torch.from_numpy(K)
            fxfycxcy[:, :, 0] = K_torch[0, 0]
            fxfycxcy[:, :, 1] = K_torch[1, 1]
            fxfycxcy[:, :, 2] = K_torch[0, 2]
            fxfycxcy[:, :, 3] = K_torch[1, 2]
            
            ray_map_lsvm = get_ray_map_lsvm(camera_pose, fxfycxcy, h=height, w=width)
            ray_map_lsvm = ray_map_lsvm.squeeze()
          
            ray_map_mask = np.array([1.0], dtype=np.float32)
            view['ray_map'] = ray_map_lsvm
            # view['ray_map_mask'] = ray_map_mask
            view['is_raymap'] = True
            ret_views.append(view)
        
        # last thing done!
        for view in ret_views:
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
     
        
        return ret_views

