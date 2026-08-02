#!/bin/bash
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

export EXP_NAME="occany_plus_recon_1B_infinite_depth"


: ${BATCH_SIZE:=1}
: ${EFFECTIVE_BATCH_SIZE:=64}
: ${N_WORKERS:=12}

export EPOCHS=100

# Default values for multi-node setup
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

export ACCUM_ITER
ACCUM_ITER="$(occany_compute_accum_iter "$EFFECTIVE_BATCH_SIZE" "$NUM_GPU_PER_NODE" "$BATCH_SIZE" "$NUM_NODE")"
occany_log_train_config "$EXP_NAME"
CMD="$(occany_select_train_cmd 'launch_da3.py')"
occany_log_start_cmd "$CMD"

WIDTH=518
HEIGHT=168




$CMD \
    --train_dataset="5000 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, min_views_per_timestep=1, min_num_timesteps=2, \
        z_far=50, split='train', no_partial_views=True, \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        2000 @ VKittiSeqMultiView(VKITTI_PROCESSED_ROOT='$SCRATCH/data/vkitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=10, min_num_timesteps=2, \
        z_far=50, split='train', no_partial_views=True, \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$SCRATCH/data/ddad_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=6, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, min_views_per_timestep=1, min_num_timesteps=2, \
        z_far=50, split='train', no_partial_views=True, \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$SCRATCH/data/pandaset_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=6, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, min_views_per_timestep=1, min_num_timesteps=2, \
        z_far=50, split='train', no_partial_views=True, \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$SCRATCH/data/once_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, min_views_per_timestep=1, min_num_timesteps=2, \
        z_far=50, split='train', no_partial_views=True, \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True)"  \
    --test_dataset="200 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$SCRATCH/data/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=5, max_memory_num_views=5, reverse_seq=False, \
        z_far=50, split='val', seed=42, no_partial_views=True, \
        resolution=[(518, 168)], distill_model_name='SAM3', base_model='da3') + \
        200 @ Occ3dNuscenesSeqMultiView(NUSCENES_PREPROCESSED_ROOT='$SCRATCH/data/occ3d_nuscenes_processed', \
        seq_pkl_name='seq_surround_temporal_sub1_stride9_all.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, num_views_per_timestep=6, \
        z_far=50, split='val', seed=42, no_partial_views=True, fixed_cams=[0,1], \
        resolution=[(518, 266)], distill_model_name='SAM3', base_model='da3')" \
    --lr=5e-5 --min_lr=1e-6 --warmup_epochs=3 --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE --accum_iter=$ACCUM_ITER \
    --save_freq=3 --keep_freq=5 --eval_freq=1  --num_workers=$N_WORKERS --multiview \
    --amp bf16 --fixed_eval_set --loss_enc_feat  \
    --output_dir="$PROJECT/tb_log_occany/$EXP_NAME" \
    --training_objective pointmap_depth_ray --fine_tune_layers 34,35,36,37,38,39 \
    --sam3_proj_lr_mult 10.0 \
    --da3_model_name depth-anything/DA3-GIANT-1.1 \
    --loss_type L1 --pointmap_lambda_c 1.0 --depth_lambda_c 0.0 --raymap_lambda_c 1.0 \
    --infinidepth_pseudo_supervision \
     --lambda_depth 1.0 --lambda_pointmap 1.0 \
     --lambda_pointmap_lidar 1.0 --lambda_pointmap_pseudo 1.0 \
     --lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0 \
     --sam3_use_dpt_proj

# SAM3 distillation disabled for now -- re-add to enable: \
#   --distill_model SAM3 --distill_criterion "DistillLoss(nn.L1Loss(), use_conf=False)"` \

# InfiniDepth pseudo-supervision replaces the legacy scale_inv_depth_loss / aux teacher tail.
# No `--aux_branch_layers` is set, and no InfiniDepth / DA3 metric model is loaded at
# training time: pseudo depth is read from precomputed sibling `<stem>.infinidepth.png`
# files (produced by `extract_infinidepth_pseudo.py`, verified by `verify_infinidepth_pseudo.py`),
# which already have sky masked out. `load_infinidepth_pseudo=True` on each train
# dataset enables loading those files.
#
# Dual GT: under --infinidepth_pseudo_supervision, training computes BOTH lidar-GT and
# InfiniDepth pseudo-GT pointmap + depth losses every step. Each is logged with a
# `_lidar` / `_pseudo` suffix (e.g. loss_pointmap_lidar, loss_pointmap_pseudo,
# depth_loss_depth_lidar, depth_loss_depth_pseudo) and weighted by the four
# `--lambda_*_lidar` / `--lambda_*_pseudo` flags on top of --lambda_pointmap / --lambda_depth.
# Eval still uses lidar GT only.
