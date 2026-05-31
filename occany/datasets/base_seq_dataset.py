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
from dust3r.datasets.base.batched_sampler import DatasetAwareBatchSamplerOccAny, DatasetAwareBatchSamplerFixedViews
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

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True,
                     per_dataset_sampling=False, fixed_views_per_batch=False):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet

        if fixed_views_per_batch and hasattr(self, 'dataset_configs'):
            return DatasetAwareBatchSamplerFixedViews(self, batch_size,
                                                     dataset_configs=self.dataset_configs,
                                                     world_size=world_size, rank=rank, drop_last=drop_last)

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
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.base_model = base_model
        self.distill_model_name = distill_model_name
        # SAM distillation is only constructed for models that consume `view['distill_img']`.
        # The DINOv3 wrapper has no SAM head, so we skip the transform + per-view distill_img.
        if base_model == 'dinov3':
            self.distill_img_size = None
            self.distill_img_transform = None
        else:
            if distill_img_size is None:
                if distill_model_name == "SAM2":
                    distill_img_size = img_size
                elif distill_model_name == "SAM3":
                    distill_img_size = 518
            self.distill_img_size = distill_img_size
            if distill_model_name == "SAM3":
                self.distill_img_transform = get_SAM3_transforms(resolution=self.distill_img_size)
            else:
                raise ValueError(f"Unsupported distill_model_name: {distill_model_name}")
        self.seq_pkl_name = seq_pkl_name
        self.img_ext = ".jpg"
        self.num_views = 3
        self._load_data()
        self.is_metric_scale = True
    
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
            if self.base_model in ('da3', 'dinov3'):
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
        min_memory_num_views=2,
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
        # min_num_timesteps caps actual_vpt in _get_views so every sample yields
        # at least this many timesteps (T = memory_num_views // actual_vpt). Set
        # to 2 for pair trainers (DeltaTok); short scenes that can't satisfy it
        # raise. Requires min_memory_num_views >= min_num_timesteps.
        self.min_num_timesteps = min_num_timesteps
        # DEPRECATED: legacy knob from seq_surround; ignored by unified _get_views.
        self.min_views_per_timestep = min_views_per_timestep
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
        # Always pick a random anchor per call; anchor occupies index 0.
        anchor = int(rng.integers(0, max_vpt))
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

    def _get_views(self, seq_idx, resolution, memory_num_views, rng, is_eval=False,
                   views_per_timestep=None):
        # is_eval is preserved in the signature for caller compat but unused —
        # the unified algorithm behaves identically at train and eval time;
        # eval reproducibility comes from `seed=42` plus optional `fixed_cams`.
        scene_idx, seq, ts = self.seqs[seq_idx]
        scene_name = self.scenes[scene_idx]
        preprocessed_scene_dir = osp.join(self.ROOT, scene_name)

        seq_len = len(seq)
        max_vpt = self.num_views_per_timestep
        # Camera pool: a malformed/short sequence (seq_len < max_vpt) is treated
        # as one short timestep with fewer physical cameras.
        camera_pool = min(max_vpt, seq_len)
        # At least one available timestep, even when seq_len < max_vpt.
        num_avail_T = max(1, seq_len // max_vpt)

        # 1. Choose cameras for this item. actual_vpt is capped by the camera
        # pool AND by the budget — no point sampling more cameras per timestep
        # than the budget allows. It is also capped at
        # ``memory_num_views // min_num_timesteps`` so the resulting layout
        # has T >= min_num_timesteps timesteps (pair trainers need T >= 2).
        min_T = max(1, int(self.min_num_timesteps))
        vpt_cap = memory_num_views // min_T
        assert vpt_cap >= 1, (
            f"memory_num_views ({memory_num_views}) < min_num_timesteps "
            f"({min_T}); raise min_memory_num_views or lower min_num_timesteps "
            f"in the dataset config (scene={scene_name})"
        )
        if self.fixed_cams is not None:
            selected_cams = np.array(self.fixed_cams, dtype=int)
            actual_vpt = len(selected_cams)
            assert actual_vpt <= vpt_cap, (
                f"fixed_cams has {actual_vpt} cameras but memory_num_views="
                f"{memory_num_views} only allows actual_vpt<={vpt_cap} to keep "
                f"T >= min_num_timesteps={min_T}"
            )
        elif views_per_timestep is not None:
            # Fixed vpt from sampler — use directly. The sampler guarantees
            # vpt <= dataset's num_views_per_timestep; if camera_pool is smaller
            # (malformed short sequence), fail loudly rather than silently
            # producing fewer views than the batch expects.
            actual_vpt = int(views_per_timestep)
            assert actual_vpt <= camera_pool, (
                f"Sampler requested vpt={actual_vpt} but sequence only has "
                f"camera_pool={camera_pool} (scene={scene_name}, seq_len={seq_len}). "
                f"Filter short sequences from the dataset."
            )
            selected_cams = self._select_cameras(camera_pool, actual_vpt, rng)
        else:
            upper = min(camera_pool, memory_num_views, vpt_cap)
            actual_vpt = int(rng.integers(1, upper + 1))
            # _select_cameras' first arg is the camera pool size (the parameter
            # is called max_vpt inside the function; we pass camera_pool because
            # short sequences may have fewer cameras available than max_vpt).
            selected_cams = self._select_cameras(camera_pool, actual_vpt, rng)

        # 2. Decompose mem_views into full timesteps + optional partial last.
        # When ``no_partial_views`` is set, drop the partial last timestep
        # (and any "pad to memory_num_views" repetition that would otherwise
        # land on a fractional timestep) so the returned views always satisfy
        # ``V == T * actual_vpt``. Trainers that reshape to (B, T, num_cameras,
        # ...) — DeltaTok, OccRAE-seq — rely on this invariant; without it a
        # partial last timestep silently breaks the reshape in
        # ``_extract_pair_feats``.
        if self.no_partial_views:
            memory_num_views = (memory_num_views // actual_vpt) * actual_vpt
            if memory_num_views == 0:
                memory_num_views = actual_vpt
        num_full_T = memory_num_views // actual_vpt
        partial = memory_num_views % actual_vpt
        timesteps_needed = num_full_T + (1 if partial else 0)

        # 3. Pick a contiguous window of timesteps.
        if num_avail_T >= timesteps_needed:
            start_T = int(rng.integers(0, num_avail_T - timesteps_needed + 1))
            chosen_T = list(range(start_T, start_T + timesteps_needed))
        else:
            chosen_T = list(range(num_avail_T))
            if self.no_partial_views:
                # Shrink memory_num_views so the uniform-layout invariant holds
                # even when the sequence couldn't supply enough timesteps —
                # padding (step 6) would otherwise add a fractional timestep.
                assert num_avail_T >= min_T, (
                    f"scene {scene_name} has num_avail_T={num_avail_T} but "
                    f"min_num_timesteps={min_T}; filter this scene out of the "
                    f"sequence list or lower min_num_timesteps"
                )
                num_full_T = num_avail_T
                partial = 0
                timesteps_needed = num_full_T
                memory_num_views = num_full_T * actual_vpt

        # 4. Optional reversal (same coin per item).
        if self.reverse_seq and rng.random() < 0.5:
            chosen_T = list(reversed(chosen_T))

        # 5. Build view indices. Same selected_cams (anchor first) across every
        # full timestep; partial last timestep takes selected_cams[:partial],
        # which always includes the anchor at index 0.
        memory_view_indices = []
        for i, t in enumerate(chosen_T):
            base = t * max_vpt
            cams = selected_cams if i < num_full_T else selected_cams[:partial]
            memory_view_indices.extend(int(base + c) for c in cams)

        # 6. Pad with random repetition if the sequence couldn't supply
        # `timesteps_needed` timesteps (preserves uniform batch tensor shape).
        if len(memory_view_indices) < memory_num_views:
            num_repeats = memory_num_views - len(memory_view_indices)
            repeated = rng.choice(memory_view_indices, size=num_repeats, replace=True)
            memory_view_indices = list(memory_view_indices) + [int(i) for i in repeated]

        assert len(memory_view_indices) == memory_num_views, (
            f"expected exactly {memory_num_views} indices, got {len(memory_view_indices)}"
        )

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
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            if len(idx) == 5:
                idx, resolution_idx, memory_num_views, ray_map_idx, views_per_timestep = idx
            else:
                idx, resolution_idx, memory_num_views, ray_map_idx = idx
                views_per_timestep = None
            is_eval = False
        else:
            # This is used by test data as we don't implement the BatchSampler in test
            assert len(self._resolutions) == 1
            resolution_idx = 0
            assert self.min_memory_num_views == self.max_memory_num_views, "Evaluation needs to be done with a fixed number of views, which is equal to min_memory_num_views and  min_memory_num_views must equal max_memory_num_views"
            memory_num_views = self.min_memory_num_views
            # assert len(self.ray_map_idx) != 0, "Evaluation needs to be done with fixed ray_map_idx"
            ray_map_idx = self.ray_map_idx
            views_per_timestep = None
            is_eval = True

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
        views = self._get_views(idx, resolution, memory_num_views, self._rng, is_eval=is_eval,
                                views_per_timestep=views_per_timestep)
        # Sync to the actual returned count so view['idx'] and
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
            if self.base_model in ('da3', 'dinov3'):
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

