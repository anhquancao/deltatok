# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import os
import sys
import types
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

import matplotlib.pyplot as pl
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from occany.utils.helpers import (
    apply_majority_pooling,
    build_fine_prompt_metadata,
    create_voxel_prediction,
    generate_intermediate_poses,
    transform_points_torch,
    save_semantic_2d_images,
)
from occany.utils.image_util import ImgNorm, get_SAM3_transforms, convert_images_to_uint8_hwc
from PIL import Image
from occany.utils.inference_helper import (
    build_demo_reconstruction_views,
    build_intrinsics_from_focal,
    convert_da3_output_to_occany_format,
    denormalize_da3_imgs_to_minus1_1,
    extract_demo_rgb_images,
    get_pts3d_from_voxel,
    is_distill_source,
    normalize_demo_rgb_image,
    parse_semantic_mode,
    uses_sam3_projection_features,
)
from occany.utils.resolution import get_output_resolution

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
BILINEAR_RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def get_output_resolution_from_image(image_path: str, model_family: str) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return get_output_resolution(image.size, model_family=model_family)


COLORS = np.array([
            [0, 0, 0, 255],
            [112, 128, 144, 255],
            [220, 20, 60, 255],
            [255, 127, 80, 255],
            [255, 158, 0, 255],
            [233, 150, 70, 255],
            [255, 61, 99, 255],
            [0, 0, 230, 255],
            [47, 79, 79, 255],
            [255, 140, 0, 255],
            [255, 98, 70, 255],
            [0, 207, 191, 255],
            [175, 0, 75, 255],
            [75, 0, 75, 255],
            [112, 180, 60, 255],
            [222, 184, 135, 255],
            [0, 175, 0, 255],
            [135, 206, 235, 255], # sky, empty
        ])
CLASS_NAMES = [
            "other",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
            "free",
        ]
empty_class = 17
other_class = 0
n_classes = 18
# GaussTR
OCC3D_CATEGORIES = (
    ['other'],
    ['barrier', 'concrete barrier', 'metal barrier', 'water barrier'],
    ['bicycle', 'bicyclist'],
    ['bus'],
    ['car'],
    ['crane'],
    ['motorcycle', 'motorcyclist'],
    ['pedestrian', 'adult', 'child'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['traffic island', 'rail track', 'lake', 'river'],
    ['sidewalk'],
    ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
    ['building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'street sign', 'traffic light'],
    ['tree', 'bush'],
    ['sky', 'empty'],
)
        
# PROMPT attribute for SAM-based semantic segmentation
PROMPT = list(OCC3D_CATEGORIES)












def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./demo_data/output',
        help="Directory where inference outputs and visualizations are written",
    )
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the RGB demo directory containing frame folders",
        default='./demo_data/input',
    )
    parser.add_argument('--frame_interval', type=int, default=5, help='Frame interval for video processing')
    parser.add_argument(
        '--model',
        type=str,
        default='occany_da3',
        choices=['occany_da3'],
        help='Model to use',
    )
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
    parser.add_argument('--sam3_conf_th', type=float, default=0.15,
                        help='Confidence threshold for SAM3 semantic inference')
    parser.add_argument('--sam3_resolution', type=int, default=1008,
                        help='Resolution for SAM3 model')
    parser.add_argument('--view_batch_size', type=int, default=4,
                        help='Number of views per SAM3 inference chunk (lower uses less GPU memory)')
    parser.add_argument('--key_to_get_pts3d', type=str, default='pts3d',
                        help='Key to get pts3d from the output')
    parser.add_argument('--views_per_interval', '-vpi', type=int, default=2,
                        help='Number of views per interval for inference')
    parser.add_argument('--seed_rotation_angle', '-seed_rot', type=int, default=None,
                        help='Angle in degrees between seed rotations. If None, defaults to 15.0 degrees.')
    parser.add_argument('--seed_translation_distance', '-seed_trans', type=int, default=None,
                        help='Distance in meters to translate seed poses laterally. Positive rotations translate right, negative translate left.')
    parser.add_argument('--batch_gen_view', '-bs_gen', type=int, default=4,
                        help='Number of generated views per batch')
    parser.add_argument('--no_semantic_from_rotated_views', "-nsr", action='store_true', default=False,
                        help='Disable using semantics from rotated views (only use semantics from straight/forward views)')
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
    parser.add_argument('--apply_majority_pooling', action='store_true', default=False,
                        help='Apply majority pooling to voxel predictions (3x3x3 neighborhood)')
    
    parser.add_argument('--pose_from_depth_ray', action='store_true', default=False,
                        help='Use ray pose estimation (set to True for trained models that use ray pose)')
    parser.add_argument('--point_from_depth_and_pose', action='store_true', default=False,
                        help='Compute pointmap from depth, intrinsics and c2w')
    parser.add_argument('--recon_conf_thres', type=float, default=2.0,
                        help='Reconstruction confidence threshold.')

    return parser










if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    def maybe_apply_pooling(voxel_pred_np: np.ndarray) -> np.ndarray:
        if not args.apply_majority_pooling:
            return voxel_pred_np

        return apply_majority_pooling(
            voxel_pred_np,
            n_classes=n_classes,
            other_class=other_class,
            empty_class=empty_class,
            is_geometry_only=False,
        )

    semantic_feat_src, semantic_model_type, semantic_family = parse_semantic_mode(args.semantic)
    model_family = "da3"

    if semantic_family is not None:
        if model_family == "da3":
            assert semantic_family == "SAM3", f"da3 model requires SAM3 semantic family, but got {semantic_family}"
    from occany.model.attention import toggle_memory_efficient_attention

    toggle_memory_efficient_attention(enabled=True)

    infer_sam2_feats = None
    infer_sam3_feats = None
    infer_semantic_from_boxes_and_sam2_feat_list = None
    infer_semantic_from_classname_and_sam3_inference_state = None
    build_sam3_inference_state = None
    select_sam_feature_views = None
    if semantic_feat_src is not None:
        from occany.semantic_inference import (
            build_sam3_inference_state,
            infer_sam3_feats,
            infer_semantic_from_classname_and_sam3_inference_state,
            select_sam_feature_views,
        )

    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    print(f"RGB demo input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"RGB demo input directory not found: {args.input_dir}")
    frame_dirs = sorted(
        frame_dir for frame_dir in Path(args.input_dir).iterdir() if frame_dir.is_dir()
    )
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {args.input_dir}")

    sample_image_paths, _ = extract_demo_rgb_images(str(frame_dirs[0]))
    output_resolution = get_output_resolution_from_image(
        sample_image_paths[0],
        model_family=model_family,
    )


    checkpoint_args = None

    from occany.da3_inference import inference_occany_da3
    from occany.utils.io_da3 import load_da3_model_from_checkpoint

    recon_weights = REPO_ROOT / "checkpoints" / "occany_plus_recon.pth"
    print("[INFO] Preparing DA3 model")
    da3_model_input_size = max(output_resolution)
    da3_model_recon, checkpoint_args = load_da3_model_from_checkpoint(
        weights_path=recon_weights,
        output_resolution=(da3_model_input_size, da3_model_input_size),
        semantic_feat_src=semantic_feat_src,
        semantic_family=semantic_family,
        device=args.device,
    )

    sam_model_for_inference = "SAM3"
        if isinstance(checkpoint_sam_model, str) and checkpoint_sam_model.upper() in ["SAM3"]:
            sam_model_for_inference = checkpoint_sam_model.upper()
    

    total_params = sum(p.numel() for p in da3_model_recon.parameters())
    trainable_params = sum(p.numel() for p in da3_model_recon.parameters() if p.requires_grad)
    print(
        f"Model 'occany_plus_recon' - total parameters: {total_params:,}, "
        f"trainable parameters: {trainable_params:,}"
    )

    recon_conf_thres = args.recon_conf_thres
    print(f"recon_conf_thres: {recon_conf_thres}")

    voxel_size = 0.4
    occ_size = [200, 200, 24]
    voxel_origin = torch.tensor([-40.0, -40.0, -3.6], device=args.device, dtype=torch.float32)
    
    print("voxel_size:", voxel_size)
    print("occ_size:", occ_size)
    print("voxel_origin:", voxel_origin)
    print(f"Found {len(frame_dirs)} RGB demo frame directories in {args.input_dir}")
    print("output_resolution (first frame):", output_resolution)

    item_count = 0
    T_cam_to_voxel = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=args.device,
    )

    for frame_dir in tqdm(frame_dirs, desc=f"Processing RGB demo frames"):
        demo_image_paths, demo_frame_id = extract_demo_rgb_images(str(frame_dir))
        frame_output_resolution = get_output_resolution_from_image(
            demo_image_paths[0],
            model_family=model_family,
        )
        recon_views = build_demo_reconstruction_views(
            image_paths=demo_image_paths,
            output_resolution=frame_output_resolution,
            model_family=model_family,
            semantic_family=semantic_family,
            frame_interval=args.frame_interval,
            sam3_resolution=args.sam3_resolution,
            device=args.device,
        )
        print(f"Loaded RGB demo frame '{demo_frame_id}' with {len(recon_views)} views from {frame_dir}")
        data = {
            "frame_id": [demo_frame_id],
        }
        B = recon_views[0]["img"].shape[0]
        _, C, H, W = recon_views[0]["img"].shape

        
            box_summary = populate_demo_sam2_box_dicts(
                recon_views=recon_views,
                class_names=CLASS_NAMES,
                device=args.device,
            )
            if box_summary["total_boxes"] == 0:
                print(
                    "[WARNING] GroundingDINO produced no SAM2 demo boxes. "
                    "Semantic outputs will remain empty unless detections are found."
                )
            else:
                print(
                    f"[INFO] Prepared {box_summary['total_boxes']} SAM2 demo boxes "
                    f"across {len(recon_views)} reconstruction views"
                )
            if box_summary["views_without_boxes"]:
                print(
                    "[WARNING] No SAM2 demo boxes detected for reconstruction views: "
                    f"{box_summary['views_without_boxes']}"
                )

        with torch.inference_mode():
            x_ray = None
            sam_feats = None

            if model_family == "da3":
                recon_output = inference_occany_da3(
                    recon_views,
                    da3_model_recon,
                    args.device,
                    dtype=torch.float32,
                    sam_model=sam_model_for_inference,
                    pose_from_depth_ray=args.pose_from_depth_ray,
                    point_from_depth_and_pose=args.point_from_depth_and_pose,
                )
                recon_output.pop('aux_feats', None)
                recon_output.pop('aux_outputs', None)

                img_out = convert_da3_output_to_occany_format(recon_output)
                if args.key_to_get_pts3d not in img_out:
                    img_out[args.key_to_get_pts3d] = img_out['pts3d']
                sam_feats = recon_output.get('sam_feats')
            sam_feats_recon = sam_feats
            sam3_recon_distill_feats = sam_feats[:3] if sam_feats is not None else None
            sam3_gen_distill_feats = None


        res = img_out
        
        imgs = [v['img'] for v in recon_views]
        imgs = torch.stack(imgs, dim=1)
        if model_family == "da3":
            imgs = denormalize_da3_imgs_to_minus1_1(imgs)

        
        recon_semantic_2ds = None
        gen_semantic_2ds = None        if args.semantic is not None:
            feat_src = semantic_feat_src
            n_recon_views = len(recon_views)
            n_gen_views = 0
            n_recon_and_gen_views = n_recon_views + n_gen_views

            semantic_fill_value = empty_class
            other_class =  other_class


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
                    class_names = CLASS_NAMES
                    class2idx = {name: idx for idx, name in enumerate(class_names)}
                    ignore_ids = {empty_class, other_class, 255}

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
                                    other_class=other_class,
                                    empty_class=empty_class,
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
                if args.compute_segmentation_masks and recon_semantic_2ds is not None:
                    if bool((recon_semantic_2ds == empty_class).all().item()):
                        print(
                            "[WARNING] SAM2 reconstruction semantics remained entirely empty "
                            f"(class {empty_class}) after inference"
                        )
                    if gen_semantic_2ds is not None and bool((gen_semantic_2ds == empty_class).all().item()):
                        print(
                            "[WARNING] SAM2 generated-view semantics remained entirely empty "
                            f"(class {empty_class}) after inference"
                        )
            elif semantic_family == "SAM3":
                recon_semantic_2ds = semantic_2ds[:, :n_recon_views]
                gen_semantic_2ds = semantic_2ds[:, n_recon_views:] if n_gen_views > 0 else None

                if not args.compute_segmentation_masks:
                    pass
                else:
                    prompts, prompt_to_class_mapping = build_fine_prompt_metadata(PROMPT)
                    ignore_ids = {empty_class, other_class, 255}
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

                    from sam3.model.position_encoding import PositionEmbeddingSine

                    pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True)
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
                                empty_class,
                                args.device,
                                args.sam3_conf_th,
                                args.sam3_resolution,
                                args.view_batch_size,
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
                            empty_class,
                            args.device,
                            args.sam3_conf_th,
                            args.sam3_resolution,
                            args.view_batch_size,
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
            "focal": res['focal'],
            "c2w": res['c2w'],
            "estimated_camera_poses": res['c2w_pose'] if 'c2w_pose' in res else res['c2w'],
            "semantic_2ds": recon_semantic_2ds,
            "is_recon": torch.ones(B, pts3d_render.shape[1], dtype=torch.bool, device=pts3d_render.device),
            # "c2w_pose": res['c2w_pose']
        }
            

            
            
        
        
        for j in tqdm(range(B), leave=False):

            frame_id = data['frame_id'][j]
            voxel_pred_save_dir = os.path.join(args.output_dir, f"{frame_id}_{args.model}")
            os.makedirs(voxel_pred_save_dir, exist_ok=True)

            for name, output in outputs.items():
                has_semantic_output = args.semantic is not None and output.get("semantic_2ds") is not None
                colors_hwc = output['colors'][j].permute(0, 2, 3, 1).cpu().numpy()
                save_dict = {
                    "pts3d": output['pts3d'][j].cpu().numpy(),
                    "pts3d_local": output['pts3d_local'][j].cpu().numpy(),
                    "colors": colors_hwc,
                    "conf": output['conf'][j].cpu().numpy(),
                    "focal": output['focal'][j].cpu().numpy(),
                    "c2w": output['c2w'][j].cpu().numpy(),
                }
                if has_semantic_output:
                    save_dict["semantic_2ds"] = output["semantic_2ds"][j].cpu().numpy()

                save_path = os.path.join(voxel_pred_save_dir, f"pts3d_{name}.npy")
                np.save(save_path, save_dict)

            grid_size = tuple(occ_size)
            voxel_predictions_dict = {
                "estimated_input_camera_poses": outputs['render']['estimated_camera_poses'][j].cpu().numpy(),
                "estimated_input_intrinsics": build_intrinsics_from_focal(
                    outputs['render']['focal'][j],
                    H,
                    W,
                ).cpu().numpy(),
                "estimated_input_images": convert_images_to_uint8_hwc(outputs['render']['colors'][j]),
                "voxel_size": voxel_size,
                "voxel_origin": voxel_origin.cpu().numpy(),
            }

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
            render_pts3d_in_velo = transform_points_torch(T=T_cam_to_voxel.float(), points=render_pts3d_th)

            render_voxel_pred = create_voxel_prediction(
                render_pts3d_in_velo, render_has_semantic, render_semantic_2ds_th, render_conf_th,
                grid_size, voxel_origin, voxel_size,
                n_classes, other_class, empty_class
            )
            render_voxel_pred_np = render_voxel_pred.cpu().numpy().astype(np.uint8)
            
            voxel_predictions_dict[f"render_th{recon_conf_thres}"] = render_voxel_pred_np
            
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
    print(f"Total items processed: {item_count}")
