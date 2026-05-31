#!/bin/bash
# Tiny single-GPU smoke for the DINOv3 recon training pipeline.
#
# Tests the end-to-end path with real weights: dataset construction → model
# build → DDP wrap → forward + backward → finite loss + log keys.
#
# Eval is disabled (--eval_freq=0) to keep the smoke under a few minutes.
#
# Usage (on Karolina):
#   sbatch --partition=qgpu_exp --time=00:30:00 --nodes=1 --ntasks-per-node=1 \
#     --gres=gpu:1 -A eu-25-92 --cpus-per-task=16 --job-name=dinov3_smoke \
#     --output=slurm/output/dinov3_smoke_%j.out \
#     --error=slurm/output/dinov3_smoke_%j.err \
#     --wrap='eval "$(conda shell.bash hook)" && conda activate occany && bash sh/smoke_train_dinov3.sh'

set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

export EXP_NAME="dinov3_smoke"
: ${BATCH_SIZE:=1}
: ${N_WORKERS:=2}
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

CMD="$(occany_select_train_cmd 'launch_dinov3.py')"
echo "Smoke CMD: $CMD"

$CMD \
    --train_dataset="20 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$SCRATCH/data/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=2, max_memory_num_views=2, reverse_seq=False, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='train', seed=42, \
        resolution=[(528, 176)], base_model='dinov3', load_infinidepth_pseudo=True)" \
    --test_dataset="4 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$SCRATCH/data/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=2, max_memory_num_views=2, reverse_seq=False, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, recon_view_idx=[0], ray_map_idx=[1], \
        resolution=[(528, 176)], base_model='dinov3')" \
    --lr=5e-5 --min_lr=1e-6 --warmup_epochs=0 --epochs=1 \
    --batch_size=$BATCH_SIZE --accum_iter=1 \
    --save_freq=0 --keep_freq=0 --eval_freq=0 --num_workers=$N_WORKERS --multiview \
    --amp bf16 \
    --output_dir="$PROJECT/tb_log_occany/$EXP_NAME" \
    --training_objective pointmap_depth_ray --fine_tune_layers 26,27,28,29,30,31 \
    --dinov3_config occany/configs/dinov3/occany_dinov3_vith16plus.yaml \
    --dinov3_weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
    --loss_type L1 --pointmap_lambda_c 1.0 --depth_lambda_c 0.0 --raymap_lambda_c 1.0 \
    --infinidepth_pseudo_supervision \
    --lambda_depth 1.0 --lambda_pointmap 1.0 \
    --lambda_pointmap_lidar 1.0 --lambda_pointmap_pseudo 1.0 \
    --lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0 \
    --print_freq=1
