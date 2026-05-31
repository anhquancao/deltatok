#!/bin/bash
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

: ${EXP_NAME:=occany_plus_recon_dinov3_vith16plus}
export EXP_NAME


: ${BATCH_SIZE:=1}
: ${EFFECTIVE_BATCH_SIZE:=64}
: ${N_WORKERS:=12}

: ${EPOCHS:=100}
export EPOCHS

# Default values for multi-node setup
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

export ACCUM_ITER
ACCUM_ITER="$(occany_compute_accum_iter "$EFFECTIVE_BATCH_SIZE" "$NUM_GPU_PER_NODE" "$BATCH_SIZE" "$NUM_NODE")"
occany_log_train_config "$EXP_NAME"
CMD="$(occany_select_train_cmd 'launch_dinov3.py')"
occany_log_start_cmd "$CMD"

WIDTH=528
HEIGHT=176

BSC_DATA_ROOT=/gpfs/scratch/ehpc793/occany_data

RAY_MAP_PROB=-1

$CMD \
    --train_dataset="5000 @ WaymoSeqMultiView(ROOT='$BSC_DATA_ROOT/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(512, 288), (512, 272), (512, 256), (512, 208), (512, 176), (512, 160)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        2000 @ VKittiSeqMultiView(VKITTI_PROCESSED_ROOT='$BSC_DATA_ROOT/vkitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=10, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(512, 288), (512, 272), (512, 256), (512, 208), (512, 176), (512, 160)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        5000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$BSC_DATA_ROOT/ddad_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(512, 288), (512, 272), (512, 256), (512, 208), (512, 176), (512, 160)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        5000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$BSC_DATA_ROOT/pandaset_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(512, 288), (512, 272), (512, 256), (512, 208), (512, 176), (512, 160)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        5000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$BSC_DATA_ROOT/once_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(512, 288), (512, 272), (512, 256), (512, 208), (512, 176), (512, 160)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True)"  \
    --test_dataset="206 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$BSC_DATA_ROOT/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, reverse_seq=False, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, recon_view_idx=[0, 2, 4, 6, 8], ray_map_idx=[1, 3, 5, 7], \
        resolution=[(528, 176)], base_model='dinov3') + \
        206 @ Occ3dNuscenesSeqMultiView(NUSCENES_PREPROCESSED_ROOT='$BSC_DATA_ROOT/occ3d_nuscenes_processed', \
        seq_pkl_name='seq_surround_temporal_sub1_stride9_all.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, num_views_per_timestep=6, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, fixed_cams=[0,1], \
        resolution=[(528, 256)], base_model='dinov3')" \
    --lr=5e-5 --min_lr=1e-6 --warmup_epochs=3 --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE --accum_iter=$ACCUM_ITER \
    --save_freq=3 --keep_freq=5 --eval_freq=1  --num_workers=$N_WORKERS --multiview \
    --amp bf16 --fixed_eval_set \
    --output_dir="/gpfs/scratch/ehpc793/tb_log_occany/$EXP_NAME" \
    --training_objective pointmap_depth_ray --fine_tune_layers "${FINE_TUNE_LAYERS:-11-31}" \
    --dinov3_config "${DINOV3_CONFIG:-occany/configs/dinov3/occany_dinov3_vith16plus.yaml}" \
    --dinov3_weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
    --loss_type L1 --pointmap_lambda_c 1.0 --depth_lambda_c 0.0 --raymap_lambda_c 1.0 \
    --infinidepth_pseudo_supervision \
     --lambda_depth 1.0 --lambda_pointmap 1.0 \
     --lambda_pointmap_lidar 1.0 --lambda_pointmap_pseudo 1.0 \
     --lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0
