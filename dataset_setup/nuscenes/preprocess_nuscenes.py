# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import os
import torch
import numpy as np

from dust3r.utils.image import imread_cv2

import matplotlib.pyplot as pl
from occany.datasets.nuscenes import NuScenesDataset
# from viser.extras import Record3dLoader_Customized, Record3dLoaderOccany
from tqdm import tqdm
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

# import open3d as o3d
import cv2
from torch.utils.data import DataLoader



# NuScenes camera list: {frame_idx}_{cam_id}.npz
NUSCENES_CAM_LIST = [          
    "CAM_FRONT",           # "xxx_0.npz"
    "CAM_FRONT_LEFT",      # "xxx_1.npz"
    "CAM_FRONT_RIGHT",     # "xxx_2.npz"
    "CAM_BACK",            # "xxx_3.npz"
    "CAM_BACK_LEFT",       # "xxx_4.npz"
    "CAM_BACK_RIGHT",      # "xxx_5.npz"
]


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf_threshold", type=float, default=1.0, help="Confidence threshold for the background")
    parser.add_argument("--foreground_conf_threshold", type=float, default=0.1, help="Confidence threshold for the foreground")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--mask_type", choices=["fg", "bg", "all"], default="fg", help="Type of mask to use")

    parser.add_argument('--frame_interval', type=int, default=1, help='Frame interval for video processing')
    parser.add_argument('--video_length', type=int, default=1, help='Video length for video processing')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--nuscenes_root', type=str, default=None, help='Path to Occ3D-nuScenes dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save preprocessed data')
    parser.add_argument('--output_resolution', type=int, nargs=2, default=[512, 288], help='Output resolution (W, H)')
  
    parser.add_argument('--n_workers', type=int, default=0)
    return parser




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # Set default paths based on environment
    if 'DSDIR' in os.environ:
        # running in Jeanzay
        DSDIR = os.environ['DSDIR']
        SCRATCH = os.environ['SCRATCH']
        nuscenes_root = args.nuscenes_root or os.path.join(DSDIR, "Occ3D-nuScenes")
        save_root = args.save_dir or os.path.join(SCRATCH, "data/occ3d_nuscenes_processed")
    else:
        # running in Karolina
        nuscenes_root = args.nuscenes_root or "/mnt/proj1/eu-25-92/data/nuscenes"
        save_root = args.save_dir or "/scratch/project/eu-25-92/data/occ3d_nuscenes_processed"

    output_resolution = tuple(args.output_resolution)
    
    print(f"NuScenes root: {nuscenes_root}")
    print(f"Save directory: {save_root}")
    print(f"Output resolution: {output_resolution}")
    print(f"Cameras: {NUSCENES_CAM_LIST}")
    print(f"Split: {args.split}")

    # Create NuScenes dataset
    nuscenes_dataset = NuScenesDataset(
        split=args.split,
        root=nuscenes_root,
        video_length=args.video_length,
        frame_interval=args.frame_interval,
        output_resolution=output_resolution,
        camera_names=NUSCENES_CAM_LIST,
        boxes_dir=None,
        apply_camera_mask=False,
        apply_lidar_mask=False,
    )
    
    print(f"Dataset loaded: {len(nuscenes_dataset)} samples")

    dataloader = DataLoader(
        nuscenes_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=lambda x: x[0]
    )

    os.makedirs(save_root, exist_ok=True)

    for sample_idx, item in enumerate(tqdm(dataloader, desc="Processing nuScenes frames")):
        scene_name = item["scene_name"]
        begin_frame_token = item["begin_frame_token"]
        camera_names = item["camera_names"]
        
        # Create scene directory
        scene_dir = os.path.join(save_root, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Get data arrays
        imgs = item["imgs"]  # [T*C, 3, H, W] tensor (normalized)
        gt_depths = item["gt_depths"]  # [T*C, H, W] numpy array
        cam_k_resized = item["cam_k_resized"]  # [T*C, 3, 3] numpy array
        cam_poses = item["cam_poses"]  # [T*C, 4, 4] numpy array (global camera-to-world)
        image_paths = item["image_paths"]  # List of image paths
        
        num_frames = len(camera_names)

        for i in range(num_frames):
            cam_name = camera_names[i]
            
            # Get camera index from the camera list
            cam_idx = NUSCENES_CAM_LIST.index(cam_name) if cam_name in NUSCENES_CAM_LIST else i
            
            # Load original image (before normalization) for saving
            img_path = image_paths[i]
            image = imread_cv2(img_path)
            H_orig, W_orig = image.shape[:2]
            
            # Resize image to output resolution
            image_resized = cv2.resize(image, output_resolution)
            # image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Get depth, intrinsics, and cam2world
            depthmap = gt_depths[i]  # [H, W]
            intrinsics = cam_k_resized[i]  # [3, 3]
            cam2world = cam_poses[i]  # [4, 4] - relative to first camera
            
            # Create filename: {frame_idx:06d}_{cam_idx}.npz
            frame_id = f"{sample_idx:06d}_{cam_idx}"
            save_path = os.path.join(scene_dir, f"{frame_id}.npz")
            
            # Save data
            np.savez_compressed(
                save_path,
                image=image_resized,
                depthmap=depthmap,
                intrinsics=intrinsics,
                cam2world=cam2world
            )
            
            if args.debug:
                tqdm.write(f"Saved: {save_path}")
                tqdm.write(f"  Image shape: {image_resized.shape}")
                tqdm.write(f"  Depth shape: {depthmap.shape}, range: [{depthmap.min():.2f}, {depthmap.max():.2f}]")
                tqdm.write(f"  Intrinsics shape: {intrinsics.shape}")
                tqdm.write(f"  Cam2world shape: {cam2world.shape}")
    
    print(f"\nPreprocessing complete. Data saved to: {save_root}")

