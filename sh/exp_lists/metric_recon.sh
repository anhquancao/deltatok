RECON_OUTPUT="${RECON_OUTPUT:-./outputs}"

exp_extra_args=(
    # 0: KITTI 5frames — OccAny+ 1B
    "--exp_dir $RECON_OUTPUT/occany_plus_recon_1B_occany_da3_kitti_5frames_img512"

    # 1: nuScenes surround — OccAny+ 1B
    "--exp_dir $RECON_OUTPUT/occany_plus_recon_1B_occany_da3_nuscenes_surround_img512"

    # 2: KITTI 5frames — plain DA3 Giant
    "--exp_dir $RECON_OUTPUT/da3_recon_da3_kitti_5frames_img512"

    # 3: nuScenes surround — plain DA3 Giant
    "--exp_dir $RECON_OUTPUT/da3_recon_da3_nuscenes_surround_img512"

    # 4: KITTI 5frames — OccAny+ 1B supervised by lidar scaled DA3 depth
    "--exp_dir $RECON_OUTPUT/occany_plus_recon_1B_sup_lidar_scaled_da3_occany_da3_kitti_5frames_img512"

    # 5: nuScenes surround — OccAny+ 1B supervised by lidar scaled DA3 depth
    "--exp_dir $RECON_OUTPUT/occany_plus_recon_1B_sup_lidar_scaled_da3_occany_da3_nuscenes_surround_img512"
)
