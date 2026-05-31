#!/bin/bash
# BSC smoke test for the DINOv3 cross-view backbone memory footprint.
#
# Goal: stress the largest training sample so OOMs surface on step 1, not later.
# Fixed resolution (largest from production: 512x288) and fixed memory view count
# T (min == max) so every iteration has the same input shape.
#
# Usage (override via env vars):
#   T=30 bash sh/smoke_train_occany_plus_recon_dinov3_vith16plus_bsc.sh
#   T=20 bash sh/smoke_train_occany_plus_recon_dinov3_vith16plus_bsc.sh
#   T=10 bash sh/smoke_train_occany_plus_recon_dinov3_vith16plus_bsc.sh
#
# With alt_start=11 in the YAML, global cross-view passes flatten to
# (B, T*N, D) where N = 1 (cls) + 4 (storage) + 32*18 (patches @ 512x288) = 581.
# T=30 -> 17,430 tokens per global pass; 11 global passes total (odd layers 11..31).
# Each fine-tunable block (21 of them: blocks 11..31) keeps activations for backward.

set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

: ${T:=30}
: ${N_SEQUENCES:=50}

export EXP_NAME="smoke_occany_plus_recon_dinov3_vith16plus_bsc_T${T}"

: ${BATCH_SIZE:=1}
: ${EFFECTIVE_BATCH_SIZE:=1}
: ${N_WORKERS:=2}

export EPOCHS=1
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

export ACCUM_ITER
ACCUM_ITER="$(occany_compute_accum_iter "$EFFECTIVE_BATCH_SIZE" "$NUM_GPU_PER_NODE" "$BATCH_SIZE" "$NUM_NODE")"
occany_log_train_config "$EXP_NAME"
CMD="$(occany_select_train_cmd 'launch_dinov3.py')"
occany_log_start_cmd "$CMD"

BSC_DATA_ROOT="${BSC_DATA_ROOT:-/gpfs/scratch/ehpc793/occany_data}"
BSC_TB_LOG_ROOT="${BSC_TB_LOG_ROOT:-/gpfs/scratch/ehpc793/tb_log_occany}"

echo "[SMOKE] T=$T  N_SEQUENCES=$N_SEQUENCES  RESOLUTION=(512, 288)"
echo "[SMOKE] BSC_DATA_ROOT=$BSC_DATA_ROOT  BSC_TB_LOG_ROOT=$BSC_TB_LOG_ROOT"

$CMD \
    --train_dataset="${N_SEQUENCES} @ WaymoSeqMultiView(ROOT='$BSC_DATA_ROOT/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=${T}, frame_interval=1, max_memory_num_views=${T}, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(512, 288)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True)" \
    --test_dataset="8 @ WaymoSeqMultiView(ROOT='$BSC_DATA_ROOT/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=${T}, frame_interval=1, max_memory_num_views=${T}, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, \
        resolution=[(512, 288)], base_model='dinov3')" \
    --lr=5e-5 --min_lr=1e-6 --warmup_epochs=0 --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE --accum_iter=$ACCUM_ITER \
    --save_freq=999 --keep_freq=999 --eval_freq=1 --print_freq=1 \
    --num_workers=$N_WORKERS --multiview \
    --amp bf16 \
    --output_dir="$BSC_TB_LOG_ROOT/$EXP_NAME" \
    --training_objective pointmap_depth_ray --fine_tune_layers 11-31 \
    --dinov3_config occany/configs/dinov3/occany_dinov3_vith16plus.yaml \
    --dinov3_weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
    --loss_type L1 --pointmap_lambda_c 1.0 --depth_lambda_c 0.0 --raymap_lambda_c 1.0 \
    --infinidepth_pseudo_supervision \
    --lambda_depth 1.0 --lambda_pointmap 1.0 \
    --lambda_pointmap_lidar 1.0 --lambda_pointmap_pseudo 1.0 \
    --lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0
