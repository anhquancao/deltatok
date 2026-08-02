# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np
import itertools
import roma
from occany.loss.losses_da3 import build_aux_pseudo_depth
from occany.model.must3r_blocks.head import ActivationType, apply_activation

from dust3r.post_process import estimate_focal_knowing_depth
from occany.utils.image_util import quaternion_to_matrix, camera_to_pose_encoding
from occany.utils.helpers import convert_depth_to_point_cloud
from dust3r.utils.geometry import geotrf
from depth_anything_3.utils.geometry import affine_inverse
from occany.utils.helpers import depth2rgb

from torch_scatter import scatter_min


def _ensure_outputs_on_device(output, expected_device):
    expected_device = torch.device(expected_device)

    def _device_matches(actual_device, target_device):
        if actual_device.type != target_device.type:
            return False
        if target_device.index is None:
            return True
        return actual_device.index == target_device.index

    mismatches = []

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            if not _device_matches(value.device, expected_device):
                mismatches.append(f"{key}={value.device}")
        elif isinstance(value, (tuple, list)):
            for idx, item in enumerate(value):
                if isinstance(item, torch.Tensor) and not _device_matches(item.device, expected_device):
                    mismatches.append(f"{key}[{idx}]={item.device}")

    if mismatches:
        mismatch_str = ", ".join(mismatches)
        raise RuntimeError(
            f"inference_occany_da3() returned tensors off the expected device {expected_device}: "
            f"{mismatch_str}"
        )


@torch.autocast("cuda", dtype=torch.float32)
def postprocess(pointmaps, pose_out=None, pointmaps_activation=ActivationType.NORM_EXP, 
                compute_cam=False, pose_type="lvsm"):
    out = {}
    channels = pointmaps.shape[-1]
    out['pts3d'] = pointmaps[..., :3]
    out['pts3d'] = apply_activation(out['pts3d'], activation=pointmaps_activation)
    if channels >= 6:
        out['pts3d_local'] = pointmaps[..., 3:6]
        out['pts3d_local'] = apply_activation(out['pts3d_local'], activation=pointmaps_activation)
    if channels == 4 or channels >= 7:
        out['conf'] = 1.0 + pointmaps[..., 6].exp()
    if channels == 10:
        eps = 1e-6
        out['rgb'] = pointmaps[..., 7:].sigmoid() * (1 - 2 * eps) + eps
        out['rgb'] = (out['rgb'] - 0.5) * 2
      
    if compute_cam:
        H, W = out['conf'].shape[-2:]
        pp = torch.tensor((W / 2, H / 2), device=out['pts3d'].device)
        focal = estimate_focal_knowing_depth(out['pts3d_local'][:, 0], pp, focal_mode='weiszfeld')        
        out['focal'] = focal[:, None].expand(-1, out['pts3d_local'].shape[1])

        batch_dims = out['pts3d'].shape[:-3]
        num_batch_dims = len(batch_dims)
        R, T = roma.rigid_points_registration(
            out['pts3d_local'].reshape(*batch_dims, -1, 3),
            out['pts3d'].reshape(*batch_dims, -1, 3),
            weights=out['conf'].reshape(*batch_dims, -1) - 1.0, compute_scaling=False)

        c2w = torch.eye(4, device=out['pts3d'].device)
        c2w = c2w.view(*([1] * num_batch_dims), 4, 4).repeat(*batch_dims, 1, 1)
        c2w[..., :3, :3] = R
        c2w[..., :3, 3] = T.view(*batch_dims, 3)
        out['c2w'] = c2w

        out['pose_trans_registered'] = c2w[..., :3, 3]
        out['pose_rotmat_registered'] = c2w[..., :3, :3]
        out['pts3d_from_local_and_pose_registered'] = torch.einsum("bnij, bnhwj -> bnhwi", out['pose_rotmat_registered'], out['pts3d_local']) + out['pose_trans_registered'][:, :, None, None, :]


    if pose_out is not None:
        if pose_out.dim() == 3:
            B, N, _ = pose_out.shape
            out['pose_trans'] = pose_out[..., :3] # bs, n_imgs, 3
            out['pose_rotmat'] = quaternion_to_matrix(pose_out[..., 3:]) # bs, n_imgs, 3, 3
            c2w_pose = torch.eye(4, device=pose_out.device).expand(B, N, 4, 4).clone()
            c2w_pose[..., :3, :3] = out['pose_rotmat']
            c2w_pose[..., :3, 3] = out['pose_trans']
            out['pose_absT_quaR'] = pose_out
        else:
            B, N, _, _ = pose_out.shape
            c2w_pose = pose_out
            out['pose_rotmat'] = c2w_pose[..., :3, :3]
            out['pose_trans'] = c2w_pose[..., :3, 3]
            out['pose_absT_quaR'] = camera_to_pose_encoding(pose_out)
        
        out['c2w_pose'] = c2w_pose
       
        out['pts3d_from_local_and_pose'] = torch.einsum("bnij, bnhwj -> bnhwi", out['pose_rotmat'], out['pts3d_local']) + out['pose_trans'][:, :, None, None, :]
    
        
    return out


def split_list(lst, split_size):
    return [lst[i:i + split_size] for i in range(0, len(lst), split_size)]


def split_list_of_tensors(tensor, max_bs):
    tensor_splits = []
    for s in tensor:
        if isinstance(s, list):
            tensor_splits.extend(split_list(s, max_bs))
        else:
            tensor_splits.extend(torch.split(s, max_bs))
    return tensor_splits


def stack_views(true_shape, values, max_bs=None):
    # first figure out what the unique aspect ratios are
    unique_true_shape, inverse_indices = torch.unique(true_shape, dim=0, return_inverse=True)

    # we group the values that share the same AR
    true_shape_stacks = [[] for _ in range(unique_true_shape.shape[0])]
    index_stacks = [[] for _ in range(unique_true_shape.shape[0])]
    value_stacks = [
        [[] for _ in range(unique_true_shape.shape[0])]
        for _ in range(len(values))
    ]

    for i in range(true_shape.shape[0]):
        true_shape_stacks[inverse_indices[i]].append(true_shape[i])
        index_stacks[inverse_indices[i]].append(i)

        for j in range(len(values)):
            value_stacks[j][inverse_indices[i]].append(values[j][i])

    # regroup all None values together (these typically are missing encoder features that'll be recomputed later)
    for i in range(len(true_shape_stacks)):
        # get a mask for each type of value
        none_mask = [[vl is None for vl in v[i]]
                     for v in value_stacks
                     ]
        # apply "or" on all the different types of values
        none_mask = [any([v[j] for v in none_mask]) for j in range(len(true_shape_stacks[i]))]
        if not any(none_mask) or all(none_mask):
            # there was no None or all were None skip
            continue
        not_none_mask = [not x for x in none_mask]

        def get_filtered_list(lst, local_mask):
            return [v for v, m in zip(lst, local_mask) if m]
        true_shape_stacks.append(get_filtered_list(true_shape_stacks[i], none_mask))
        true_shape_stacks[i] = get_filtered_list(true_shape_stacks[i], not_none_mask)

        index_stacks.append(get_filtered_list(index_stacks[i], none_mask))
        index_stacks[i] = get_filtered_list(index_stacks[i], not_none_mask)

        for j in range(len(value_stacks)):
            value_stacks[j].append(get_filtered_list(value_stacks[j][i], none_mask))
            value_stacks[j][i] = get_filtered_list(value_stacks[j][i], not_none_mask)

    # stack tensors
    true_shape_stacks = [torch.stack(true_shape_stack, dim=0) for true_shape_stack in true_shape_stacks]
    value_stacks = [
        [torch.stack(v, dim=0) if None not in v else v for v in value_stack]
        for value_stack in value_stacks
    ]

    # split all sub-tensors in blocks of max_size = max_bs
    if max_bs is not None:
        true_shape_stacks = split_list_of_tensors(true_shape_stacks, max_bs)

        index_stacks = [torch.tensor(s) for s in index_stacks]
        index_stacks = split_list_of_tensors(index_stacks, max_bs)
        index_stacks = [s.tolist() for s in index_stacks]

        value_stacks = [
            split_list_of_tensors(value_stack, max_bs)
            for value_stack in value_stacks
        ]

    # some cleaning, replace list of None by a single None
    for value_stack in value_stacks:
        for j in range(len(value_stack)):
            if isinstance(value_stack[j], list):
                if None in value_stack[j]:
                    value_stack[j] = None

    return true_shape_stacks, index_stacks, *value_stacks


def prepare_imgs_and_true_shape_mem_batches(views, device):

    
    imgs = [b['img'] for b in views]
    imgs = torch.stack(imgs, dim=1).to(device)
    B, nimgs, C, H, W, = imgs.shape
    true_shape = [torch.as_tensor(b['true_shape']) for b in views]
    true_shape = torch.stack(true_shape, dim=1).to(device)
    mem_batches = [2]
    while sum(mem_batches) < nimgs:
        mem_batches.append(1)


    timesteps = [b['timestep'] for b in views]
    timesteps = torch.stack(timesteps, dim=1).to(device).type_as(imgs)

    
    
    return imgs, true_shape, mem_batches, timesteps #, distill_imgs

    

def inference_occany_da3(img_views, model,
                     device,
                     dtype=torch.float32,
                     sam_model="SAM2",
                     pose_from_depth_ray=False,
                     point_from_depth_and_pose=False,
                     **kwargs):
    with torch.autocast("cuda", dtype=dtype):
       
        imgs, true_shape_img, mem_batches, img_timesteps = prepare_imgs_and_true_shape_mem_batches(img_views, device)

        output = model(
            imgs,
            pose_from_depth_ray=pose_from_depth_ray,
            point_from_depth_and_pose=point_from_depth_and_pose,
            **kwargs,
        )

    _ensure_outputs_on_device(output, device)
    return output

def loss_of_one_batch_occany_da3(views, model, 
                             device, 
                             dtype=torch.float32,
                             distill_criterion=None, 
                             distill_model=None, 
                             is_distill=False,
                             use_ray_pose=False,
                             sam_model="SAM2",
                             pointmap_criterion=None,
                             depth_criterion=None,
                             raymap_criterion=None,
                             lambda_depth=1.0,
                             lambda_raymap=1.0,
                             lambda_pointmap=1.0,
                             pose_from_depth_ray=False,
                             scale_inv_depth_criterion=None,
                             lambda_scale_inv_depth=1.0,
                             aux_metric_pseudo_supervision=False,
                             da3_metric_model=None,
                             sky_mask_threshold=0.3,
                             infinidepth_pseudo_supervision=False,
                             infinidepth_depth_min=1e-3,
                             infinidepth_depth_max=150.0,
                             lambda_pointmap_lidar=1.0,
                             lambda_pointmap_pseudo=1.0,
                             lambda_depth_lidar=1.0,
                             lambda_depth_pseudo=1.0):
    """
    Compute loss for one batch with DA3 model.
    
    Args:
        views: List of view dictionaries.
        model: DA3 model.
        criterion: Pointmap loss criterion.
        device: Device to use.
        dtype: Data type for autocast.
        distill_criterion: Distillation criterion (optional).
        distill_model: Distillation model (optional).
        is_distill: Whether to use distillation.
        use_ray_pose: If True, model computes and returns pointmap (for testing/evaluation).
        sam_model: SAM model type.
        depth_criterion: DepthLosses criterion for depth supervision (optional).
        raymap_criterion: RaymapLoss criterion for raymap supervision (optional).
        lambda_depth: Weight for depth loss.
        lambda_raymap: Weight for raymap loss.
        lambda_pointmap: Weight for pointmap loss.
        pose_from_depth_ray: Whether to estimate pose from depth and raymap.
        scale_inv_depth_criterion: Scale-invariant depth loss criterion (optional).
        lambda_scale_inv_depth: Weight for scale-invariant depth loss.
        aux_metric_pseudo_supervision: If True, train against aux-derived pseudo targets.
        infinidepth_pseudo_supervision: If True, train against precomputed InfiniDepth
            pseudo targets (each view must carry `pseudo_depthmap`; mutually exclusive
            with aux_metric_pseudo_supervision and scale_inv_depth_loss).
        infinidepth_depth_min / infinidepth_depth_max: valid range used to build the
            supervision mask on the precomputed pseudo depth.
    """

    # with torch.cuda.amp.autocast(enabled=bool(use_amp)):
    img_views = views

    B = img_views[0]['img'].shape[0]
    nimgs = len(img_views)
    
    # use_ray_pose determines whether to compute pointmap (for testing) or keep raw output (for training)
    # For training, we need raw depth/raymap for loss computation
    # For testing, we compute pointmap for evaluation metrics
    
    output = inference_occany_da3(
                    img_views, model,
                     device,
                     dtype=dtype,
                     sam_model=sam_model,
                     pose_from_depth_ray=pose_from_depth_ray)
    
    with torch.autocast("cuda", dtype=torch.float32):
        depth = output.get('depth')
        depth_conf = output.get('depth_conf')
        ray = output.get('ray')  # (B, T, H, W, 6) when return_depth_and_raymap=True
        ray_conf = output.get('ray_conf')  # (B, T, H, W) when return_depth_and_raymap=True
        
        if depth is not None:
            depth = depth.to(device)
        if depth_conf is not None:
            depth_conf = depth_conf.to(device)
        if ray is not None:
            ray = ray.to(device)
        if ray_conf is not None:
            ray_conf = ray_conf.to(device)

        valid_mask = None
        if 'valid_mask' in img_views[0]:
            valid_mask = torch.stack([b['valid_mask'] for b in img_views], dim=1).to(device)

        details = {}
        total_loss = 0.0

        if aux_metric_pseudo_supervision and scale_inv_depth_criterion is not None and lambda_scale_inv_depth > 0:
            raise ValueError('aux_metric_pseudo_supervision and scale_inv_depth_loss are mutually exclusive')
        if infinidepth_pseudo_supervision and aux_metric_pseudo_supervision:
            raise ValueError('infinidepth_pseudo_supervision and aux_metric_pseudo_supervision are mutually exclusive')
        if infinidepth_pseudo_supervision and scale_inv_depth_criterion is not None and lambda_scale_inv_depth > 0:
            raise ValueError('infinidepth_pseudo_supervision and scale_inv_depth_loss are mutually exclusive')

        pseudo_depth = None
        pseudo_pointmap = None
        pseudo_supervision_mask = None
        if aux_metric_pseudo_supervision:
            if 'depthmap' not in img_views[0]:
                raise ValueError('aux_metric_pseudo_supervision requires depthmap in training views for lidar-based scale alignment')
            if 'camera_pose' not in img_views[0] or 'camera_intrinsics' not in img_views[0]:
                raise ValueError('aux_metric_pseudo_supervision requires camera_pose and camera_intrinsics to build pseudo pointmaps from pseudo depth')
            if da3_metric_model is None:
                raise ValueError('aux_metric_pseudo_supervision requires da3_metric_model for sky masking')

            # Per-image teacher pass: run the frozen shared backbone prefix
            # followed by the frozen copied tail + head to produce pseudo depth.
            metric_imgs = torch.stack([b['img'] for b in img_views], dim=1).to(device)
            teacher_model = model.module if hasattr(model, 'module') else model
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                aux_depth = teacher_model.inference_batch_individual(metric_imgs)
           
            gt_depth_for_scale = torch.stack([b['depthmap'] for b in img_views], dim=1).to(device)
            gt_c2w = torch.stack([b['camera_pose'] for b in img_views], dim=1).to(device)
            gt_intrinsics = torch.stack([b['camera_intrinsics'] for b in img_views], dim=1).to(device)
            pseudo_depth, aux_scales, scale_valid_mask = build_aux_pseudo_depth(
                aux_depth.float(),
                gt_depth_for_scale.float(),
                mask=valid_mask,
                max_gt_range=50.0,
            )
            pseudo_pointmap = convert_depth_to_point_cloud(
                pseudo_depth.float(),
                gt_intrinsics.float(),
                gt_c2w.float(),
            )

            # Compute non-sky mask from frozen DA3 metric model so we can supervise
            # everywhere except the sky regions.
            from depth_anything_3.utils.alignment import compute_sky_mask
            metric_imgs = torch.stack([b['img'] for b in img_views], dim=1).to(device)
            with torch.no_grad():
                metric_output = da3_metric_model(metric_imgs, export_feat_layers=[])
            non_sky_mask = compute_sky_mask(metric_output.sky, threshold=sky_mask_threshold)
            non_sky_mask = non_sky_mask.to(device=device).reshape_as(pseudo_depth)

            pseudo_supervision_mask = non_sky_mask & torch.isfinite(pseudo_depth) & (pseudo_depth > 0)
            pseudo_supervision_mask = pseudo_supervision_mask & torch.isfinite(pseudo_pointmap).all(dim=-1)
            details['avg_aux_scale'] = float(aux_scales.mean())
            details['aux_scale_valid_ratio'] = float(scale_valid_mask.float().mean())
            details['non_sky_ratio'] = float(non_sky_mask.float().mean())
            details['pseudo_supervision_ratio'] = float(pseudo_supervision_mask.float().mean())

        if infinidepth_pseudo_supervision:
            # Pseudo depth is precomputed offline by extract_infinidepth_pseudo.py
            # and loaded by the dataset (load_infinidepth_pseudo=True). Sky is
            # already masked at extract time (sky pixels = 0), so no DA3 metric
            # forward is needed here.
            if 'pseudo_depthmap' not in img_views[0]:
                raise ValueError(
                    'infinidepth_pseudo_supervision requires `pseudo_depthmap` in each view; '
                    'set `load_infinidepth_pseudo=True` on the dataset constructors and ensure '
                    '<stem>.infinidepth.png files exist next to each <stem>.npz.'
                )
            if 'camera_pose' not in img_views[0] or 'camera_intrinsics' not in img_views[0]:
                raise ValueError('infinidepth_pseudo_supervision requires camera_pose and camera_intrinsics to build pseudo pointmaps from pseudo depth')

            pseudo_depth = torch.stack([b['pseudo_depthmap'] for b in img_views], dim=1).to(device).float()
            gt_c2w_p = torch.stack([b['camera_pose'] for b in img_views], dim=1).to(device).float()
            gt_intr_p = torch.stack([b['camera_intrinsics'] for b in img_views], dim=1).to(device).float()
            pseudo_pointmap = convert_depth_to_point_cloud(pseudo_depth, gt_intr_p, gt_c2w_p)
            depth_range_mask = (
                torch.isfinite(pseudo_depth)
                & (pseudo_depth > infinidepth_depth_min)
                & (pseudo_depth < infinidepth_depth_max)
            )
            pseudo_supervision_mask = depth_range_mask & torch.isfinite(pseudo_pointmap).all(dim=-1)
            details.update({
                'pseudo_supervision_ratio': float(pseudo_supervision_mask.float().mean()),
                'infinidepth_valid_ratio': float(depth_range_mask.float().mean()),
                'infinidepth_mean_depth': (
                    float(pseudo_depth[depth_range_mask].mean()) if depth_range_mask.any() else 0.0
                ),
            })

        # Build (label, pointmap_gt, pointmap_mask, depth_gt, depth_mask, lambda_pointmap_eff, lambda_depth_eff)
        # for each GT source. Dual mode (lidar + pseudo) is active only when
        # infinidepth_pseudo_supervision is on; aux_metric_pseudo_supervision keeps
        # its original "pseudo replaces lidar" behavior.
        gt_branches = []
        if infinidepth_pseudo_supervision:
            lidar_pointmap_gt = torch.stack([b['pts3d'] for b in img_views], dim=1).to(device)
            lidar_depth_gt = (
                torch.stack([b['depthmap'] for b in img_views], dim=1).to(device)
                if 'depthmap' in img_views[0] else None
            )
            gt_branches.append((
                'lidar',
                lidar_pointmap_gt, valid_mask,
                lidar_depth_gt, valid_mask,
                lambda_pointmap * lambda_pointmap_lidar,
                lambda_depth * lambda_depth_lidar,
            ))
            gt_branches.append((
                'pseudo',
                pseudo_pointmap, pseudo_supervision_mask,
                pseudo_depth, pseudo_supervision_mask,
                lambda_pointmap * lambda_pointmap_pseudo,
                lambda_depth * lambda_depth_pseudo,
            ))
        elif aux_metric_pseudo_supervision:
            gt_branches.append((
                None,
                pseudo_pointmap, pseudo_supervision_mask,
                pseudo_depth, pseudo_supervision_mask,
                lambda_pointmap, lambda_depth,
            ))
        else:
            lidar_pointmap_gt = (
                torch.stack([b['pts3d'] for b in img_views], dim=1).to(device)
                if 'pts3d' in img_views[0] else None
            )
            lidar_depth_gt = (
                torch.stack([b['depthmap'] for b in img_views], dim=1).to(device)
                if 'depthmap' in img_views[0] else None
            )
            gt_branches.append((
                None,
                lidar_pointmap_gt, valid_mask,
                lidar_depth_gt, valid_mask,
                lambda_pointmap, lambda_depth,
            ))

        # Pointmap loss(es)
        if pointmap_criterion is not None and lambda_pointmap > 0:
            pointmap = output.get('pointmap', None)
            assert pointmap is not None, "DA3 output does not contain 'pointmap' or 'point_map'"
            pointmap = pointmap.to(device)
            # Prepare confidence for pointmap loss (use depth_conf if available)
            # depth_conf shape is (B, T, H, W), pointmap shape is (B, T, H, W, 3)
            # We pass depth_conf directly - PointmapLoss handles shape matching
            pointmap_conf = depth_conf  # Can be None if not available
            for label, pm_gt, pm_mask, _, _, lam_pm_eff, _ in gt_branches:
                if pm_gt is None or lam_pm_eff <= 0:
                    continue
                loss_pointmap, loss_details = pointmap_criterion(
                    pointmap.float(), pm_gt.float(), pm_mask, confidence=pointmap_conf
                )
                if label is None:
                    details.update(loss_details)
                else:
                    details.update({f"{k}_{label}": v for k, v in loss_details.items()})
                total_loss = total_loss + lam_pm_eff * loss_pointmap

        # Depth loss(es)
        if depth_criterion is not None and depth is not None and lambda_depth > 0:
            # Reshape predicted depth + confidence once; reused for every GT branch.
            if depth.ndim == 4:  # (B, T, H, W)
                depth_for_loss = depth.unsqueeze(2)  # (B, T, 1, H, W)
            else:
                depth_for_loss = depth
            if depth_conf.ndim == 4:  # (B, T, H, W)
                depth_conf_for_loss = depth_conf.unsqueeze(2)  # (B, T, 1, H, W)
            else:
                depth_conf_for_loss = depth_conf
            B_T = depth_for_loss.shape[0] * depth_for_loss.shape[1]
            H, W = depth_for_loss.shape[-2:]
            depth_for_loss = depth_for_loss.reshape(B_T, 1, H, W)
            depth_conf_for_loss = depth_conf_for_loss.reshape(B_T, 1, H, W)

            for label, _, _, dp_gt, dp_mask, _, lam_dp_eff in gt_branches:
                if dp_gt is None or lam_dp_eff <= 0:
                    continue
                gt_depth = dp_gt
                if gt_depth.ndim == 4:  # (B, T, H, W)
                    gt_depth = gt_depth.unsqueeze(2)  # (B, T, 1, H, W)
                gt_depth = gt_depth.reshape(B_T, 1, H, W)
                depth_mask = None
                if dp_mask is not None:
                    depth_mask = dp_mask.reshape(B_T, 1, H, W)
                depth_loss, depth_loss_details = depth_criterion(
                    depth_for_loss.float(), gt_depth.float(), depth_conf_for_loss.float(), depth_mask
                )
                total_loss = total_loss + lam_dp_eff * depth_loss
                if label is None:
                    details.update({f"depth_{k}": v for k, v in depth_loss_details.items()})
                else:
                    details.update({f"depth_{k}_{label}": v for k, v in depth_loss_details.items()})
        
        # Raymap loss
        if raymap_criterion is not None and lambda_raymap > 0:
            # Get GT camera poses (c2w) and intrinsics from views
            assert 'camera_pose' in img_views[0], "camera_pose must be present in views for raymap loss"
            assert 'camera_intrinsics' in img_views[0], "camera_intrinsics must be present in views for raymap loss"
            gt_c2w = torch.stack([b['camera_pose'] for b in img_views], dim=1).to(device)  # (B, T, 4, 4) camera-to-world
            gt_intrinsics = torch.stack([b['camera_intrinsics'] for b in img_views], dim=1).to(device)  # (B, T, 3, 3)
            
            # Use pre-computed gt_raymap from dataset if available
            gt_raymap = None
            if 'gt_raymap' in img_views[0]:
                gt_raymap = torch.stack([b['gt_raymap'] for b in img_views], dim=1).to(device)  # (B, T, H, W, 6)
            
            raymap_loss, raymap_loss_details = raymap_criterion(
                ray.float(), ray_conf.float(), gt_c2w.float(), gt_intrinsics.float(), gt_raymap=gt_raymap
            )
            total_loss = total_loss + lambda_raymap * raymap_loss
            details.update(raymap_loss_details)
        
        # SAM3 distillation loss
        if distill_criterion is not None and distill_model is not None:
            # Use pre-computed sam_feats from output (computed inside model forward for DDP compatibility)
            sam_feats = output.get('sam_feats')
            
            if sam_feats is not None:
                # Get teacher features from SAM3
                # SAM3 expects images normalized differently (not ImageNet normalized)
                # First undo ImageNet normalization, then apply SAM3 preprocessing
                distill_imgs = torch.stack([b['distill_img'] for b in img_views], dim=1).to(device)
                B_distill, T_distill = distill_imgs.shape[:2]
                
                with torch.no_grad():
                    # distill_model.forward_distill returns (feat_s0, feat_s1, feat_s2, pre_neck_feat)
                    distill_feats = distill_model.forward_distill(
                        distill_imgs.reshape(B_distill * T_distill, 3, *distill_imgs.shape[-2:])
                    )
                # Reshape back to (B, T, ...)
                distill_feats = [f.view(B_distill, T_distill, *f.shape[1:]).detach() for f in distill_feats]

                # Compute distillation loss
                if distill_criterion.use_conf:
                    loss_distill, distill_details = distill_criterion(
                        sam_feats, distill_feats, depth_conf.detach()
                    )
                else:
                    loss_distill, distill_details = distill_criterion(sam_feats, distill_feats)
                
                total_loss = total_loss + loss_distill
                details.update({f"distill_{k}": v for k, v in distill_details.items()})

        # Legacy scale-invariant depth loss: also driven by the per-image
        # teacher pass now that the multi-view aux path has been removed.
        if scale_inv_depth_criterion is not None and lambda_scale_inv_depth > 0:
            if depth is not None and 'depthmap' in img_views[0]:
                metric_imgs = torch.stack([b['img'] for b in img_views], dim=1).to(device)
                teacher_model = model.module if hasattr(model, 'module') else model
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                    aux_depth = teacher_model.inference_batch_individual(metric_imgs)
                gt_depth_for_scale = torch.stack([b['depthmap'] for b in img_views], dim=1).to(device)

                scale_inv_loss, scale_inv_details = scale_inv_depth_criterion(
                    depth.float(),
                    aux_depth.float(),
                    gt_depth_for_scale.float(),
                    valid_mask,
                )
                total_loss = total_loss + lambda_scale_inv_depth * scale_inv_loss
                details.update(scale_inv_details)

        loss = (total_loss, details)

        # Use GT images as placeholder for visualization
        rgb = output.get('rgb')  # (B, T, H, W, 3) if available
        if rgb is None:
            rgb = torch.stack([b['img'] for b in img_views], dim=1).to(device)  # (B, T, C, H, W)
            rgb = rgb.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        
        combined_preds = {
            'rgb': rgb,  # (B, T, H, W, 3)
            'depth': depth,  # (B, T, H, W) - predicted depth values
        }
        
        # combined_gt is the list of views for visualization
        combined_gt = img_views
        
        # # Debug: Check if view 0's pts3d is perpendicular to Z
        # # Print c2w for view 0
        # # Optionally save ground-truth data for debugging/visualization
        # save_gt_render_data(img_views, "debug_output/da3_gt/00000", batch_idx=2, imagenet_normalize=True)
        
        result = dict(
            loss=loss,
            combined_preds=combined_preds,
            combined_gt=combined_gt,
            img_preds=combined_preds,
            gt_img=img_views,
            pseudo_depth=pseudo_depth,
            pseudo_supervision_mask=pseudo_supervision_mask,
        )
        
    return result
