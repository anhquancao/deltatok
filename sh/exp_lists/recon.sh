common_args="--exp_name occany_plus_recon_1B --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B.pth"
lidar_scaled_depth_args="--exp_name occany_plus_recon_1B_sup_lidar_scaled_da3 --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B_lidar_scaled_depth_sup.pth"

exp_extra_args=(
    # 0: KITTI 5frames — OccAny+ 1B
    "$common_args --model occany_da3 --dataset kitti --setting 5frames"

    # 1: nuScenes surround — OccAny+ 1B
    "$common_args --model occany_da3 --dataset nuscenes --setting surround"

    # 2: KITTI 5frames — plain DA3 Giant
    "--exp_name da3_recon --model da3 --dataset kitti --setting 5frames"

    # 3: nuScenes surround — plain DA3 Giant
    "--exp_name da3_recon --model da3 --dataset nuscenes --setting surround"

    # 4: KITTI 5frames — OccAny+ 1B supervised by lidar scaled DA3 depth
    "$lidar_scaled_depth_args --model occany_da3 --dataset kitti --setting 5frames"

    # 5: nuScenes surround — OccAny+ 1B supervised by lidar scaled DA3 depth
    "$lidar_scaled_depth_args --model occany_da3 --dataset nuscenes --setting surround"
)
