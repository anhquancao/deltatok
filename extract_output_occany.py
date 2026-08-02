# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import copy
import pickle
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
VENDORED_IMPORT_PATHS = [
    REPO_ROOT / "third_party",
    REPO_ROOT / "third_party" / "dust3r",
    REPO_ROOT / "third_party" / "croco" / "models" / "curope",
    REPO_ROOT / "third_party" / "Grounded-SAM-2",
    REPO_ROOT / "third_party" / "Grounded-SAM-2" / "grounding_dino",
    REPO_ROOT / "third_party" / "sam3",
    REPO_ROOT / "third_party" / "Depth-Anything-3" / "src",
]
for vendored_path in reversed(VENDORED_IMPORT_PATHS):
    vendored_path_str = str(vendored_path)
    if vendored_path.exists() and vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)

from occany.datasets.eval_helper import build_nuscenes_vis_time_index_map, prepare_eval_setting

import matplotlib.pyplot as pl
import torch.nn.functional as F
from occany.utils.helpers import (
    build_fine_prompt_metadata,
    create_voxel_prediction,
    generate_intermediate_poses,
    transform_points_torch,
    save_semantic_2d_images,
)
from occany.da3_inference import inference_occany_da3

from occany.semantic_inference import (
    ModelManager,
    infer_sam3_feats,
    infer_semantic_from_classname_and_sam3_inference_state,
    build_sam3_inference_state,
    get_box_dict_for_view,
    select_sam_feature_views,
)
from occany.model.sam3_model import Sam3ModelManager
from dust3r.depth_eval import compute_gt_depth_scale
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.inference_helper import (
    build_intrinsics_from_focal,
    convert_da3_output_to_occany_format,
    denormalize_da3_imgs_to_minus1_1,
    get_pts3d_from_voxel,
    is_distill_source,
    parse_semantic_mode,
    uses_sam3_projection_features,
    count_module_parameters,
    count_unique_parameters,
    get_pretrained_semantic_encoder_for_count,
)
from occany.utils.io_da3 import load_da3_model_from_checkpoint
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from PIL import Image
from sam3.model.position_encoding import PositionEmbeddingSine


pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", "-sz", type=int, default=512, choices=[512, 224, 768], help="image size")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")

    # My arguments
    parser.add_argument('--frame_interval', type=int, default=5, help='Frame interval for video processing')
    parser.add_argument('--dataset', '-ds', type=str, default='nuscenes', choices=['kitti', 'nuscenes'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'vis'], help='Dataset split to use')
    parser.add_argument('--kitti_root', type=str, default=str(REPO_ROOT / 'data' / 'kitti'), help='Root for the KITTI dataset (SemanticKITTI voxels and odometry images)')
    parser.add_argument('--nuscenes_root', type=str, default=str(REPO_ROOT / 'data' / 'nuscenes'), help='Root for NuScenes dataset')
    parser.add_argument('--exp_name', type=str, default='', help='exp_name for the output directory')
    parser.add_argument('--compute_pca', action='store_true', default=False, help='Enable PCA computation')
    parser.add_argument(
        '--model',
        type=str,
        default='occany_da3',
        choices=['occany_da3'],
        help='Model to use',
    )
    parser.add_argument('--setting', type=str, default='5frames', choices=['10frames', '5frames', '1frame', 'surround'], help='Setting for the output directory')
    
    parser.add_argument('--vis_interval', type=int, default=100,
                        help='Interval for saving visualization outputs (e.g., save every 50 items)')
    parser.add_argument(
        '--semantic', "-sem",
        type=str,
        choices=['pretrained@SAM3',
                 'distill@SAM3'],
        default=None,
        help='Semantic processing option. Choices: pretrained@SAM_small, distill@SAM_tiny, distill@SAM_small.'
    )
    parser.add_argument('--compute_segmentation_masks', action='store_true', default=False,
                        help='Compute segmentation masks')
    parser.add_argument('--sam3_conf_th', type=float, default=0.5,
                        help='Confidence threshold for SAM3 semantic inference')
    parser.add_argument('--sam3_resolution', type=int, default=1008,
                        help='Resolution for SAM3 model')
    parser.add_argument('--sam3_view_batch_size', type=int, default=1,
                        help='Number of views per SAM3 inference chunk (lower uses less GPU memory)')
    parser.add_argument('--world', type=int, default=1, help='Number of worlds for distributed processing')
    parser.add_argument('--pid', type=int, default=0, help='Process ID for distributed processing')
    parser.add_argument('--scale_by_gt_depth', action='store_true', default=False,
                        help='Scale reconstructed point cloud using ground-truth depth')
    parser.add_argument('--use_render_output', action='store_true', default=False,
                        help='Use render output instead of online output')
    parser.add_argument('--key_to_get_pts3d', type=str, default='pts3d',
                        help='Key to get pts3d from the output')
    parser.add_argument('--views_per_interval', '-vpi', type=int, default=2,
                        help='Number of views per interval for inference')
    parser.add_argument('--gen_rotate_novel_poses_angle', '-rot', type=int, default=0,
                        help='Angle to rotate novel poses')
    parser.add_argument('--gen_forward_novel_poses_dist', '-fwd', type=int, default=1,
                        help='Distance to move forward for novel poses (in meters)')
    parser.add_argument('--num_seed_rotations', '-nseed', type=int, default=0,
                        help='Number of seed rotations to generate (e.g., 5 for [-10, -5, 0, 5, 10]). If 0, uses standard mode.')
    parser.add_argument('--seed_rotation_angle', '-seed_rot', type=int, default=None,
                        help='Angle in degrees between seed rotations. If None, defaults to 15.0 degrees.')
    parser.add_argument('--seed_translation_distance', '-seed_trans', type=int, default=None,
                        help='Distance in meters to translate seed poses laterally. Positive rotations translate right, negative translate left.')
    parser.add_argument('--vis_output_dir', type=str, default=None,
                        help='Directory to save visualization results. If None, saves to output_dir/vis')
    parser.add_argument('--boxes_folder', type=str, default=None,
                        help='Folder name for bounding boxes (e.g., resized_512, resized_224)')
    parser.add_argument('--batch_gen_view', '-bs_gen', type=int, default=4,
                        help='Number of generated views per batch')
    parser.add_argument('--no_semantic_from_rotated_views', "-nsr", action='store_true', default=False,
                        help='Disable using semantics from rotated views (only use semantics from straight/forward views)')
    parser.add_argument('--use_visibility_mask', "-uvm", action='store_true', default=False,
                        help='Use visibility mask to filter generated view semantics based on visibility in reconstruction view')
    parser.add_argument('--box_conf_thres', type=float, default=0.05,
                        help='Confidence threshold for bounding box filtering')
    parser.add_argument('--merge_masks', action='store_true', default=False,
                        help='Merge masks by label and binned confidence (0.01 bins)')
    parser.add_argument('--only_semantic_from_recon_view', "-osr", action='store_true', default=False,
                        help='Use only semantic information from the reconstruction view (exclude all generated views)')
    parser.add_argument(
        '--gen_semantic_from_distill_sam3',
        action='store_true',
        default=False,
        help='For pretrained@SAM3, infer generated-view semantics from distilled SAM3 features when available',
    )
    
    parser.add_argument('--pose_from_depth_ray', action='store_true', default=False,
                        help='Use ray pose estimation (set to True for trained models that use ray pose)')
    parser.add_argument('--point_from_depth_and_pose', action='store_true', default=False,
                        help='Compute pointmap from depth, intrinsics and c2w')
    parser.add_argument('--novel_view_rgb_path', "-novel_view_rgb_path", type=str, default=None,
                        help='Path to the novel view rgb image')
    parser.add_argument('--recon_conf_thres', type=float, required=True,
                        help='Reconstruction confidence threshold.')
    parser.add_argument('--gen_conf_thres', type=float, required=True,
                        help='Generation confidence threshold.')
    return parser


def convert_images_to_uint8_hwc(images: torch.Tensor) -> np.ndarray:
    """Convert [-1, 1] CHW images to uint8 HWC numpy arrays."""
    images_hwc = images.permute(0, 2, 3, 1).clamp(-1.0, 1.0)
    images_uint8 = ((images_hwc + 1.0) * 127.5).round().to(torch.uint8)
    return images_uint8.cpu().numpy()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    semantic_feat_src, semantic_model_type, semantic_family = parse_semantic_mode(args.semantic)
    model_family = "da3"

    if model_family == "da3":
        if args.dataset == 'kitti':
            if args.image_size == 224:
                output_resolution = (224, 84)
            elif args.image_size == 768:
                output_resolution = (770, 238)
            else:
                output_resolution = (518, 168)
        elif args.dataset == 'nuscenes':
            if args.image_size == 224:
                output_resolution = (224, 140)
            elif args.image_size == 768:
                output_resolution = (770, 434)
            else:
                output_resolution = (518, 294)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    else:
        if args.dataset == 'kitti':
            if args.image_size == 224:
                output_resolution = (224, 80)
            elif args.image_size == 768:
                output_resolution = (768, 240)
            else:
                output_resolution = (512, 160)
        elif args.dataset == 'nuscenes':
            if args.image_size == 224:
                output_resolution = (224, 144)
            elif args.image_size == 768:
                output_resolution = (768, 432)
            else:
                output_resolution = (512, 288)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    toggle_memory_efficient_attention(enabled=True)


    save_dir = f"{args.exp_name}_{args.setting}_{args.dataset}{args.image_size}"
    if args.boxes_folder is not None:
        save_dir += f"_{args.boxes_folder}_boxth{int(args.box_conf_thres*100)}"
    if semantic_family == "SAM3":
        save_dir += f"_sam3th{int(args.sam3_conf_th * 100)}_res{args.sam3_resolution}"
    if args.seed_translation_distance is not None:
        save_dir += f"_sTrans{args.seed_translation_distance}"
    if args.no_semantic_from_rotated_views:
        save_dir += "_nsr"
    if args.merge_masks:
        save_dir += "_mm"
    if args.use_visibility_mask:
        save_dir += "_uvm"
    if args.only_semantic_from_recon_view:
        save_dir += "_osfr"
    if args.split == "vis":
        save_dir += "_vis"
    args.output_dir = f"{args.output_dir}/{save_dir}"
    if args.vis_output_dir is not None:
        args.vis_output_dir = f"{args.vis_output_dir}/{save_dir}"
    print(f"Output directory: {args.output_dir}")
    print(f"Vis output directory: {args.vis_output_dir}")

    checkpoint_args = None

    recon_weights = REPO_ROOT / "checkpoints" / "occany_plus_recon.pth"
    print("[INFO] Preparing DA3 model")
    da3_model_recon, checkpoint_args = load_da3_model_from_checkpoint(
        weights_path=recon_weights,
        output_resolution=output_resolution,
        semantic_feat_src=semantic_feat_src,
        semantic_family=semantic_family,
        device=args.device,
    )

    sam_model_for_inference = "SAM3"
        if isinstance(checkpoint_sam_model, str) and checkpoint_sam_model.upper() in ["SAM3"]:
            sam_model_for_inference = checkpoint_sam_model.upper()
    
    pretrained_semantic_encoder = None
    pretrained_semantic_encoder_name = None
    semantic_encoder_total_params = 0
    semantic_encoder_trainable_params = 0
    if semantic_feat_src == "pretrained" and semantic_family in ["SAM3"]:
        pretrained_semantic_encoder, pretrained_semantic_encoder_name = get_pretrained_semantic_encoder_for_count(
            semantic_feat_src=semantic_feat_src,
            semantic_family=semantic_family,
            semantic_model_type=semantic_model_type,
            device=args.device,
            image_size=args.image_size,
            sam3_resolution=args.sam3_resolution,
            sam3_conf_th=args.sam3_conf_th,
        )
        semantic_encoder_total_params, semantic_encoder_trainable_params = count_module_parameters(
            pretrained_semantic_encoder
        )
        if semantic_encoder_total_params > 0:
            semantic_encoder_label = pretrained_semantic_encoder_name or semantic_family
            print(
                f"Pretrained semantic encoder '{semantic_encoder_label}' - "
                f"total parameters: {semantic_encoder_total_params:,}, "
                f"trainable parameters: {semantic_encoder_trainable_params:,}"
            )

    # Print model parameter counts
    base_total_params, base_trainable_params = count_module_parameters(da3_model_recon)
    total_params = base_total_params + semantic_encoder_total_params
    trainable_params = base_trainable_params + semantic_encoder_trainable_params
    print(
        f"Model 'occany_plus_recon' - total parameters: {total_params:,}, "
        f"trainable parameters: {trainable_params:,}"
    )
    if semantic_encoder_total_params > 0:
        print(
            f"[INFO] Includes pretrained {semantic_family} encoder parameters: "
            f"{semantic_encoder_total_params:,}"
        )

    recon_conf_thres = args.recon_conf_thres
    gen_conf_thres = args.gen_conf_thres
    print(f"recon_conf_thres: {recon_conf_thres}")
    print(f"gen_conf_thres:   {gen_conf_thres}")



     # Distribute work based on world and pid
    num_worlds = args.world
    process_id = args.pid
    
    # For KITTI and NuScenes, use prepare_eval_setting
    base_model = 'da3'
    dataset, collate_fn, recon_view_idx = prepare_eval_setting(
        dataset=args.dataset, setting=args.setting,
        boxes_folder=args.boxes_folder,
        image_size=args.image_size,
        novel_view_rgb_path=args.novel_view_rgb_path,
        process_id=args.pid, num_worlds=args.world,
        split=args.split,
        sam3_resolution=args.sam3_resolution,
        base_model=base_model,
        kitti_root=args.kitti_root,
        nuscenes_root=args.nuscenes_root)

    print("dataset.other_class:", dataset.other_class)
    print("dataset.n_classes:", dataset.n_classes)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print("recon_view_idx:", recon_view_idx)

    nuscenes_vis_time_index_map = None
    if args.dataset == 'nuscenes' and args.split == 'vis':
        nuscenes_vis_time_index_map = build_nuscenes_vis_time_index_map(dataset)
        print(
            "[INFO] Using nuScenes VIS time-ordered frame ids for saving "
            f"({len(nuscenes_vis_time_index_map)} scenes)"
        )
    
    
    print("output_resolution:", output_resolution)


    # Process data using data_loader (dataset already filtered by pid/world)
    item_count = 0  # Track total items processed
    for i, data in enumerate(tqdm(data_loader, desc=f"Processing dataset (PID {process_id})")):
        imgs = data["imgs"].to(args.device)        sam3_imgs = None
        if args.semantic is not None:
            
                if "sam2_imgs" not in data:
                    print("[WARNING] SAM2 semantic mode requested but sam2_imgs is missing from dataset batch")
                else:
                    sam2_imgs = data["sam2_imgs"].to(args.device)
            elif semantic_family == "SAM3":
                if "sam3_imgs" not in data:
                    print("[WARNING] SAM3 semantic mode requested but sam3_imgs is missing from dataset batch")
                else:
                    sam3_imgs = data["sam3_imgs"].to(args.device)
        
        # Handle different field names for KITTI vs NuScenes
        if args.dataset == 'kitti':
            frame_id = data['begin_frame_id']
            # KITTI: T_velo_2_cam transforms velo→cam, we need cam→voxel (which is cam→velo)
            T_velo_2_cam = data['T_velo_2_cam'].to(args.device)
            T_cam_to_voxel = torch.inverse(T_velo_2_cam)  # cam→velo
        else:  # nuscenes
            frame_id = data['begin_frame_token']
            # NuScenes: cam0_to_ego already transforms cam0→ego (voxel frame)
            T_cam_to_voxel = data['cam0_to_ego'].to(args.device)
        
        gt_depths = data['gt_depths'].to(args.device)
        camera_poses = data['cam_poses_in_cam0'].to(args.device)
        K = data['cam_k_resized'].to(args.device)
        
        # Load camera masks if available (for NuScenes)
        if 'voxel_mask_camera' in data:
            voxel_mask_camera = data['voxel_mask_camera'].to(args.device)
        else:
            voxel_mask_camera = None

        B, nimgs, C, H, W = imgs.shape
        views = []
        for view_idx in range(nimgs):
            view = {
                "img": imgs[:, view_idx],
                "timestep": torch.tensor(view_idx * args.frame_interval, dtype=torch.float32).view(1).expand(B).to(args.device),
                "true_shape": torch.tensor(imgs.shape[-2:]).view(2).expand(B, 2).to(args.device),
                "gt_depth": gt_depths[:, view_idx],
                "camera_pose": camera_poses[:, view_idx],
                "box_dict": [get_box_dict_for_view(data, i, view_idx) for i in range(B)],
            }
            
            if semantic_family == "SAM3" and sam3_imgs is not None:
                view["sam3_img"] = sam3_imgs[:, view_idx]
            views.append(view)
        recon_views = []
        for v in recon_view_idx:
            view = copy.deepcopy(views[v])
            recon_views.append(view)
        
        
        ret_views = recon_views
        
        # Stack gt_depths if available
        if all('gt_depth' in view for view in ret_views):
            gt_depths = torch.stack([view['gt_depth'] for view in ret_views], dim=1)
        else:
            gt_depths = None

        # Precompute voxel parameters
        if args.dataset == 'kitti':
            voxel_origin = torch.from_numpy(dataset.voxel_origin).float().to(args.device)
            voxel_label = data['voxel_label']
        elif args.dataset == 'nuscenes':
            voxel_origin = torch.tensor([dataset.pc_range[0], dataset.pc_range[1], dataset.pc_range[2]]).float().to(args.device)
            voxel_label = data['voxel_label']
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        # Compute camera poses for recon views (common for all dataset types now)
        recon_camera_poses = torch.stack([view['camera_pose'] for view in recon_views], dim=1)
        
        voxel_label_visible_only = None
        
        
       
        
        with torch.inference_mode():
            x_ray = None
            sam_feats = None

            if model_family == "da3":
                recon_model_to_use = da3_model_recon
                pose_from_depth_ray_for_da3 = args.pose_from_depth_ray
                if force_pose_from_depth_ray_for_da3_gen and not args.pose_from_depth_ray and not getattr(args, "_logged_da3_pose_override", False):
                    print(
                        "[INFO] DA3 generation enabled: forcing pose_from_depth_ray=True "
                        "to generate novel views from predicted reconstruction poses."
                    )
                    args._logged_da3_pose_override = True
                recon_output = inference_occany_da3(
                    recon_views,
                    recon_model_to_use,
                    args.device,
                    dtype=torch.float32,
                    sam_model=sam_model_for_inference,
                    pose_from_depth_ray=pose_from_depth_ray_for_da3,
                    point_from_depth_and_pose=args.point_from_depth_and_pose,
                )
                recon_output.pop('aux_feats', None)
                recon_output.pop('aux_outputs', None)

            sam_feats_recon = sam_feats
            sam3_recon_distill_feats = sam_feats[:3] if sam_feats is not None else None
            sam3_gen_distill_feats = None
            n_gen_views = 0

            n_recon_and_gen_views = n_recon_views + n_gen_views

            if hasattr(dataset, "empty_class"):
                semantic_fill_value = dataset.empty_class
                other_class = getattr(dataset, "other_class", dataset.empty_class)
            else:
                semantic_fill_value = 0
                other_class = 0

            semantic_2ds = torch.full(
                (B, n_recon_and_gen_views, H, W),
                semantic_fill_value,
                dtype=torch.uint8,
            )

            
                if feat_src == 'pretrained':
                    if all('sam2_img' in view for view in recon_views):
                        sam2_imgs_recon = torch.stack([view['sam2_img'] for view in recon_views], dim=1)
                    else:
                        print("[WARNING] SAM2 pretrained mode requested but recon views do not contain sam2_img")

                if args.compute_segmentation_masks:
                    class_names = dataset.CLASS_NAMES
                    class2idx = {name: idx for idx, name in enumerate(class_names)}
                    ignore_ids = {dataset.empty_class, dataset.other_class, 255}

                    for batch_i in range(B):
                        if feat_src == 'pretrained':
                            if sam2_imgs_recon is None:
                                continue
                            sam2_feats = infer_sam2_feats(
                                sam2_model_type,
                                sam2_imgs_recon[batch_i],
                                args.device,
                                max_bs=args.batch_gen_view,
                            )
                        elif is_distill_source(feat_src):
                            if sam_feats_recon is None or len(sam_feats_recon) < 3:
                                print(
                                    "[WARNING] SAM2 distill mode requested but distilled SAM features are unavailable"
                                )
                                continue
                            sam2_feats = {
                                "image_embed": sam_feats_recon[0][batch_i],
                                "high_res_feats": [
                                    sam_feats_recon[2][batch_i],
                                    sam_feats_recon[1][batch_i],
                                ],
                            }
                        else:
                            raise ValueError(f"Unknown SAM2 feature source: {feat_src}")

                        sam2_feats_batch.append(sam2_feats)

                        for recon_view_i in range(n_recon_views):
                            box_dict = recon_views[recon_view_i]['box_dict'][batch_i]
                            boxes = box_dict['boxes']
                            confidences = box_dict['confidences']
                            labels = box_dict['labels']

                            valid_indices = [idx for idx, label in enumerate(labels) if label in class2idx]
                            if len(valid_indices) == 0:
                                continue

                            boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
                            conf_np = (
                                confidences.detach().cpu().numpy()
                                if torch.is_tensor(confidences)
                                else np.asarray(confidences)
                            )
                            if boxes_np.size == 0:
                                continue
                            boxes_np = boxes_np.reshape(-1, 4)[valid_indices]
                            conf_np = conf_np.reshape(-1)[valid_indices]
                            label_ids = [class2idx[labels[idx]] for idx in valid_indices]

                            corresponding_gen_view_ids = []

                            for gen_view_i in range(
                                0,
                                max(1, len(corresponding_gen_view_ids)),
                                args.batch_gen_view,
                            ):
                                recon_and_gen_ids = [recon_view_i] + corresponding_gen_view_ids[
                                    gen_view_i:gen_view_i + args.batch_gen_view
                                ]

                                sam2_feat_list = []
                                for view_id in recon_and_gen_ids:
                                    sam2_feat_list.append(
                                        {
                                            "high_res_feats": [
                                                sam2_feats['high_res_feats'][0][view_id:view_id + 1],
                                                sam2_feats['high_res_feats'][1][view_id:view_id + 1],
                                            ],
                                            "image_embed": sam2_feats['image_embed'][view_id:view_id + 1],
                                        }
                                    )

                                sem2d = infer_semantic_from_boxes_and_sam2_feat_list(
                                    sam2_model_type,
                                    H,
                                    W,
                                    label_ids,
                                    ignore_ids,
                                    boxes_np,
                                    conf_np,
                                    other_class=dataset.other_class,
                                    empty_class=dataset.empty_class,
                                    use_sam_video=True,
                                    sam2_feats_list=sam2_feat_list,
                                    poses=None,
                                    focals=None,
                                    depth_maps=None,
                                    device=args.device,
                                    box_conf_thres=args.box_conf_thres,
                                    merge_masks=args.merge_masks,
                                )

                                for local_idx, view_i in enumerate(recon_and_gen_ids):
                                    semantic_2ds[batch_i, view_i] = torch.from_numpy(sem2d[local_idx])

                                del sam2_feat_list, sem2d
                                torch.cuda.empty_cache()

                recon_semantic_2ds = semantic_2ds[:, :n_recon_views]
                gen_semantic_2ds = semantic_2ds[:, n_recon_views:] if n_gen_views > 0 else None
            elif semantic_family == "SAM3":
                recon_semantic_2ds = semantic_2ds[:, :n_recon_views]
                gen_semantic_2ds = semantic_2ds[:, n_recon_views:] if n_gen_views > 0 else None

                if not args.compute_segmentation_masks:
                    pass
                elif not hasattr(dataset, "PROMPT"):
                    print("[WARNING] Dataset PROMPT metadata is missing; skipping SAM3 semantic inference")
                else:
                    prompts, prompt_to_class_mapping = build_fine_prompt_metadata(dataset.PROMPT)
                    ignore_ids = {dataset.empty_class, other_class, 255}
                    allowed_gen_view_ids = []

                    recon_distill_feats = sam3_recon_distill_feats
                    gen_distill_feats = sam3_gen_distill_feats

                    sam3_imgs_recon = None
                    if feat_src == 'pretrained':
                        if all('sam3_img' in view for view in recon_views):
                            sam3_imgs_recon = torch.stack([view['sam3_img'] for view in recon_views], dim=1)
                        else:
                            print("[WARNING] SAM3 pretrained mode requested but recon views do not contain sam3_img")
                    elif not is_distill_source(feat_src):
                        raise ValueError(f"Unknown SAM3 feature source: {feat_src}")

                    if is_distill_source(feat_src) and recon_distill_feats is None:
                        print("[WARNING] SAM3 distill mode requested but reconstruction distilled features are unavailable")

                    allow_pretrained_gen_sam3_from_distill = (
                        feat_src == 'pretrained' and args.gen_semantic_from_distill_sam3
                    )

                    if (
                        feat_src == 'pretrained'
                        and n_gen_views > 0
                        and len(allowed_gen_view_ids) > 0
                        and not args.only_semantic_from_recon_view
                        and allow_pretrained_gen_sam3_from_distill
                        and gen_distill_feats is None
                    ):
                        print(
                            "[WARNING] pretrained@SAM3 cannot infer generated-view semantics because "
                            "distilled generated SAM3 features are unavailable"
                        )

                    if n_gen_views > 0 and len(allowed_gen_view_ids) == 0 and not args.only_semantic_from_recon_view:
                        print("[WARNING] No generated views selected for SAM3 semantics after view filtering")

                    can_infer_gen_sam3 = (
                        gen_semantic_2ds is not None
                        and not args.only_semantic_from_recon_view
                        and len(allowed_gen_view_ids) > 0
                        and gen_distill_feats is not None
                        and (
                            is_distill_source(feat_src)
                            or allow_pretrained_gen_sam3_from_distill
                        )
                    )

                    selected_gen_view_ids = allowed_gen_view_ids
                    n_gen_views_for_sam3 = n_gen_views
                    if can_infer_gen_sam3 and len(selected_gen_view_ids) < n_gen_views:
                        selected_gen_distill_feats = select_sam_feature_views(
                            gen_distill_feats,
                            selected_gen_view_ids,
                            n_gen_views,
                            context="gen_sam3_distill_subset",
                        )
                        if selected_gen_distill_feats is None:
                            print("[WARNING] Failed to build selected generated-view SAM3 features")
                            can_infer_gen_sam3 = False
                        else:
                            gen_distill_feats = selected_gen_distill_feats
                            n_gen_views_for_sam3 = len(selected_gen_view_ids)
                    elif can_infer_gen_sam3:
                        n_gen_views_for_sam3 = len(selected_gen_view_ids)

                    pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True)
                    view_batch_size = max(1, int(args.sam3_view_batch_size))
                    for batch_i in range(B):
                        if feat_src == 'pretrained':
                            if sam3_imgs_recon is None:
                                continue
                            recon_state = infer_sam3_feats(
                                sam3_imgs_recon[batch_i],
                                H,
                                W,
                                args.device,
                                args.sam3_resolution,
                            )
                        else:
                            recon_state = build_sam3_inference_state(
                                recon_distill_feats,
                                batch_i,
                                n_recon_views,
                                H,
                                W,
                                pos_enc,
                                context="recon_sam3_distill",
                            )

                        if recon_state is not None:
                            recon_semantic_2ds[batch_i] = infer_semantic_from_classname_and_sam3_inference_state(
                                prompts,
                                prompt_to_class_mapping,
                                recon_state,
                                ignore_ids,
                                dataset.empty_class,
                                args.device,
                                args.sam3_conf_th,
                                args.sam3_resolution,
                                view_batch_size,
                            )

                        if not can_infer_gen_sam3:
                            continue

                        gen_state_context = "gen_sam3_distill"
                        if feat_src == 'pretrained':
                            gen_state_context = "gen_sam3_distill_for_pretrained"

                        gen_state = build_sam3_inference_state(
                            gen_distill_feats,
                            batch_i,
                            n_gen_views_for_sam3,
                            H,
                            W,
                            pos_enc,
                            context=gen_state_context,
                        )
                        if gen_state is None:
                            continue

                        gen_semantics = infer_semantic_from_classname_and_sam3_inference_state(
                            prompts,
                            prompt_to_class_mapping,
                            gen_state,
                            ignore_ids,
                            dataset.empty_class,
                            args.device,
                            args.sam3_conf_th,
                            args.sam3_resolution,
                            view_batch_size,
                        )
                        if gen_semantics.shape[0] != len(selected_gen_view_ids):
                            print(
                                "[WARNING] Generated SAM3 semantics shape mismatch: "
                                f"expected {len(selected_gen_view_ids)} views, got {gen_semantics.shape[0]}"
                            )
                            continue
                        for local_view_idx, view_idx in enumerate(selected_gen_view_ids):
                            gen_semantic_2ds[batch_i, view_idx] = gen_semantics[local_view_idx]
            else:
                raise ValueError(f"Unknown semantic family: {semantic_family}")
       
        
        outputs = {}
    

        pts3d_render = res[args.key_to_get_pts3d]
        pts3d_local_render = res['pts3d_local']
        conf_render = res['conf']
        outputs["render"] = {
            "pts3d": pts3d_render,
            "pts3d_local": pts3d_local_render,
            "conf": conf_render,
            "colors": imgs,
            "gt_depths": gt_depths,
            "focal": res['focal'],
            "c2w": res['c2w'],
            "estimated_camera_poses": res['c2w_pose'] if 'c2w_pose' in res else res['c2w'],
            "semantic_2ds": recon_semantic_2ds,
            "is_recon": torch.ones(B, pts3d_render.shape[1], dtype=torch.bool, device=pts3d_render.device),
            # "c2w_pose": res['c2w_pose']
        }
            

            
            
        
        
        for j in tqdm(range(B), leave=False):

            if args.dataset == 'kitti':
                seq_name = f"{data['sequence'][j]}"
                frame_str = f"{data['begin_frame_id'][j]:06d}"
            elif args.dataset == 'nuscenes':
                seq_name = data['scene_name'][j]
                frame_token = data['begin_frame_token'][j]
                frame_str = frame_token
                if nuscenes_vis_time_index_map is not None:
                    scene_token_to_time_index = nuscenes_vis_time_index_map.get(seq_name)
                    if scene_token_to_time_index is not None and frame_token in scene_token_to_time_index:
                        frame_str = f"{scene_token_to_time_index[frame_token]:06d}"
            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")
            print("item_count", item_count)
            
            
            if item_count % args.vis_interval == 0 and args.vis_output_dir is not None:
                for name, output in outputs.items():
                    
                    save_dir = os.path.join(args.vis_output_dir, f"{seq_name}_{frame_str}")
                    pts3d_j = output['pts3d'][j].reshape(-1, 3)
                    colors_j = output['colors'][j].permute(0, 2, 3, 1).reshape(-1, 3)
                    colors_j = ((colors_j + 1.0) * 127.5).to(torch.uint8)
                    conf_j = output['conf'][j].reshape(-1)

                    if args.semantic is not None and "semantic_2ds" in output:
                        semantic_2ds_j = output["semantic_2ds"][j].to(conf_j.device).reshape(-1)
            
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_dict = {
                        "pts3d": output['pts3d'][j].cpu().numpy(),
                        "pts3d_local": output['pts3d_local'][j].cpu().numpy(),
                        "colors": output['colors'][j].permute(0, 2, 3, 1).cpu().numpy(),
                        "conf": output['conf'][j].cpu().numpy(),
                        "focal": output['focal'][j].cpu().numpy(),
                        "c2w": output['c2w'][j].cpu().numpy(),
                        # "c2w_pose": output['c2w_pose'][j].cpu().numpy()
                        "seq_name": seq_name,
                        "frame_str": frame_str,
                    }
                    if args.semantic is not None and "semantic_2ds" in output:
                        save_dict["semantic_2ds"] = output["semantic_2ds"][j].cpu().numpy()
                    
                    
                    save_path = os.path.join(save_dir, f"pts3d_{name}.npy")
                    np.save(save_path, save_dict)
                    
                    if args.semantic is not None and "semantic_2ds" in output:
                        save_dict["semantic_2ds"] = output["semantic_2ds"][j].cpu().numpy()
                        semantic_save_dir = os.path.join(save_dir, f"semantic_2ds_{feat_src}")
                        os.makedirs(semantic_save_dir, exist_ok=True)
                        save_semantic_2d_images(save_dict["semantic_2ds"], semantic_save_dir, dataset.COLORS, verbose=not args.silent)
                    
                    # Save PCA color visualizations as images for all feature types
                    if args.compute_pca:
                        pca_feature_types = ["pca_image_embed", "pca_high_res_0", "pca_high_res_1"]
                        for pca_type in pca_feature_types:
                            if pca_type in save_dict:
                                pca_save_dir = os.path.join(save_dir, f"{pca_type}_{feat_src}")
                                os.makedirs(pca_save_dir, exist_ok=True)
                                for view_idx, pca_img in enumerate(save_dict[pca_type]):
                                    pca_img_path = os.path.join(pca_save_dir, f"{view_idx:04d}.png")
                                    Image.fromarray(pca_img).save(pca_img_path)
                                if not args.silent:
                                    print(f"Saved {len(save_dict[pca_type])} {pca_type} visualizations to {pca_save_dir}")
                    
                    # Save RGB color visualizations as images
                    if "colors" in save_dict:
                        rgb_save_dir = os.path.join(save_dir, "rgb_colors")
                        os.makedirs(rgb_save_dir, exist_ok=True)
                        for view_idx, rgb_img in enumerate(save_dict["colors"]):
                            rgb_uint8 = np.clip((rgb_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
                            rgb_img_path = os.path.join(rgb_save_dir, f"{view_idx:04d}.png")
                            Image.fromarray(rgb_uint8).save(rgb_img_path)
                        if not args.silent:
                            print(f"Saved {len(save_dict['colors'])} RGB visualizations to {rgb_save_dir}")
                
            voxel_label_j = voxel_label[j]
            grid_size = voxel_label_j.shape

            
            if args.split == "vis":
                voxel_pred_save_dir = os.path.join(args.output_dir, seq_name, frame_str)
            else:
                voxel_pred_save_dir = os.path.join(args.output_dir, f"{seq_name}_{frame_str}")
            os.makedirs(voxel_pred_save_dir, exist_ok=True)
           
            gt_input_intrinsics = None
            if K is not None:
                if K.dim() == 4:
                    gt_input_intrinsics = K[j, recon_view_idx]
                elif K.dim() == 3:
                    n_recon_views = outputs['render']['focal'][j].shape[0]
                    gt_input_intrinsics = K[j].unsqueeze(0).expand(n_recon_views, -1, -1)
                else:
                    raise ValueError(f"Unsupported K rank: {K.dim()}")

            voxel_predictions_dict = {
                "voxel_label": voxel_label_j.cpu().numpy(),
                "T_cam_to_voxel": T_cam_to_voxel[j].cpu().numpy(),
                "estimated_input_camera_poses": outputs['render']['estimated_camera_poses'][j].cpu().numpy(),
                "gt_input_camera_poses": recon_camera_poses[j].cpu().numpy(),
                "estimated_input_intrinsics": build_intrinsics_from_focal(
                    outputs['render']['focal'][j],
                    H,
                    W,
                ).cpu().numpy(),
                "gt_input_intrinsics": gt_input_intrinsics.cpu().numpy() if gt_input_intrinsics is not None else None,
                "estimated_input_images": convert_images_to_uint8_hwc(outputs['render']['colors'][j]),
            }

            # Iterate over reconstruction thresholds
           
            recon_output = outputs['render']
            
            # Process render (reconstruction) output
            render_conf_mask = recon_output['conf'][j] > recon_conf_thres
            render_pts3d_th = recon_output['pts3d'][j][render_conf_mask]
            render_conf_th = recon_output['conf'][j][render_conf_mask]
            
            if args.semantic is not None:
                render_semantic_2ds_th = recon_output.get('semantic_2ds', [None])[j]
                if render_semantic_2ds_th is not None:
                    render_semantic_2ds_th = render_semantic_2ds_th.to(render_conf_mask.device)
                    render_semantic_2ds_th = render_semantic_2ds_th[render_conf_mask]
                    render_has_semantic = True
                else:
                    render_has_semantic = False
                    render_semantic_2ds_th = None
            else:
                render_has_semantic = False
                render_semantic_2ds_th = None
            
            # Create and save render voxel prediction
            render_pts3d_in_velo = transform_points_torch(T=T_cam_to_voxel[j].float(), points=render_pts3d_th)
            voxel_size = dataset.voxel_size
            render_voxel_pred = create_voxel_prediction(
                render_pts3d_in_velo, render_has_semantic, render_semantic_2ds_th, render_conf_th,
                grid_size, voxel_origin, voxel_size, 
                dataset.n_classes, dataset.other_class, dataset.empty_class
            )
            render_voxel_pred_np = render_voxel_pred.cpu().numpy().astype(np.uint8)
            print("Number of occupied voxels in render:", np.sum(render_voxel_pred_np != dataset.empty_class))
            
            voxel_predictions_dict[f"render_th{recon_conf_thres}"] = render_voxel_pred_np
            print(f"Added render voxel prediction: render_th{recon_conf_thres}")
            
            save_path = os.path.join(voxel_pred_save_dir, "voxel_predictions.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(voxel_predictions_dict, f)
            print(f"Saved voxel predictions dictionary: {save_path}")
        
            item_count += 1  # Increment item counter after processing each item in batch
            
            # Clean up tensors from current iteration
            del outputs, data, imgs            if 'sam3_imgs' in locals() and sam3_imgs is not None:
                del sam3_imgs
            if 'recon_semantic_2ds' in locals() and recon_semantic_2ds is not None:
                del recon_semantic_2ds
            if 'gen_semantic_2ds' in locals() and gen_semantic_2ds is not None:
                del gen_semantic_2ds
            if 'semantic_2ds' in locals():
                del semantic_2ds
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    print("=" * 50)
    print(f"Total items processed by PID {process_id}: {item_count}")
    print("Use 'compute_metrics_from_saved_voxels.py' to compute metrics.")
    print("=" * 50)
