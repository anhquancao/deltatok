#!/bin/bash
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

export EXP_NAME="occany_plus_recon_1B_infinite_depth_sam3_distill_memory_budget_pseudo_depth_only"


: ${BATCH_SIZE:=1}
: ${EFFECTIVE_BATCH_SIZE:=64}
: ${N_WORKERS:=8}

# Data root for the processed datasets. Defaults to Karolina's $SCRATCH/data;
# other clusters (e.g. jean-zay's fsn1 occany_data) override DATA_ROOT in their
# slurm wrapper.
: ${DATA_ROOT:=$SCRATCH/data}

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



RAY_MAP_PROB=-1

$CMD \
    --train_dataset="5000 @ WaymoSeqMultiView(ROOT='$DATA_ROOT/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=28, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        2000 @ VKittiSeqMultiView(VKITTI_PROCESSED_ROOT='$DATA_ROOT/vkitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=10, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$DATA_ROOT/ddad_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=28, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$DATA_ROOT/pandaset_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=28, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$DATA_ROOT/once_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=28, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True)"  \
    --test_dataset="206 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$DATA_ROOT/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, reverse_seq=False, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, recon_view_idx=[0, 2, 4, 6, 8], ray_map_idx=[1, 3, 5, 7], \
        resolution=[(518, 168)], distill_model_name='SAM3', base_model='da3') + \
        206 @ Occ3dNuscenesSeqMultiView(NUSCENES_PREPROCESSED_ROOT='$DATA_ROOT/occ3d_nuscenes_processed', \
        seq_pkl_name='seq_surround_temporal_sub1_stride9_all.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, num_views_per_timestep=6, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, fixed_cams=[0,1], \
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
    --lambda_raymap 2.0 \
    --infinidepth_pseudo_supervision \
     --lambda_depth 1.0 --lambda_pointmap 1.0 \
     --lambda_pointmap_lidar 1.0 --lambda_pointmap_pseudo 0.0 \
     --lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0 \
     --sam3_use_dpt_proj \
     --distill_model SAM3 --distill_criterion "DistillLoss(nn.L1Loss(), use_conf=False)" \
     --distill_k_schedule memory_budget

# InfiniDepth pseudo-supervision replaces the legacy scale_inv_depth_loss / aux teacher tail.
# No `--aux_branch_layers` is set, and no InfiniDepth / DA3 metric model is loaded at
# training time: pseudo depth is read from precomputed sibling `<stem>.infinidepth.png`
# files (produced by `extract_infinidepth_pseudo.py`, verified by `verify_infinidepth_pseudo.py`),
# which already have sky masked out. `load_infinidepth_pseudo=True` on each train
# dataset enables loading those files.
#
# Dual GT: under --infinidepth_pseudo_supervision, training mixes two GT sources every step.
#   - lidar GT (capped at z_far=50 m): pointmap + depth losses, AND the raymap loss
#     (raymap is always supervised by the metric GT camera_pose / intrinsics).
#   - InfiniDepth pseudo GT (capped at infinidepth_depth_max=150 m): DEPTH loss ONLY.
#     The pseudo pointmap term is disabled here via --lambda_pointmap_pseudo 0.0, so far
#     (50-150 m) supervision comes through metric depth rather than the per-scene-normalized
#     pointmap, leaving the metric ray-origin / trajectory signal to the lidar+raymap path.
# Each branch is logged with a `_lidar` / `_pseudo` suffix (e.g. loss_pointmap_lidar,
# depth_loss_depth_lidar, depth_loss_depth_pseudo) and weighted by the four
# `--lambda_*_lidar` / `--lambda_*_pseudo` flags on top of --lambda_pointmap / --lambda_depth.
# Eval still uses lidar GT only.
