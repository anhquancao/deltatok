#!/bin/bash
# Smoke test for SAM3 distillation memory footprint.
#
# Goal: find which (T, K) combinations fit on a single GPU at resolution 518x294.
# - T = exact number of memory views per sequence (both min and max set to T)
# - K = --distill_max_frames; when T > K, K is the random subset SAM3 distillation
#       runs on. For T <= 10 set K=-1 so distillation runs on all T frames.
#
# Usage (override via env vars):
#   T=15 K=10 bash sh/smoke_train_sam3_distill.sh
#   T=20 K=8  bash sh/smoke_train_sam3_distill.sh
#   T=25 K=6  bash sh/smoke_train_sam3_distill.sh
#   T=30 K=4  bash sh/smoke_train_sam3_distill.sh
#
# Known-good combinations (fit on a single 40 GiB GPU at 518x294):
#   T=17 K=17  OK max mem: 34558
#   T=18 K=18  OK max mem: 35899 so T <=18, K=T
#   T=19 K=16  OK max mem: 36486
#   T=20 K=14  OK max mem: 36416
#   T=21 K=12  OK max mem: 
#   T=22 K=10  OK max mem: 
#   T=23 K=8   OK max mem: 
#  T=24 K=6   OK max mem: 
#   T=25 K=4   Ok max mem: 
#   T=26 K=2   Ok max mem:
#  T=26, 27, 28 K=0

# Single Waymo dataset, one resolution, no eval, ~50 sequences so it exits
# quickly after the first few train iterations (OOMs surface immediately).
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

: ${T:=28}
: ${K:=0}
: ${N_SEQUENCES:=100}

export EXP_NAME="smoke_sam3_distill_T${T}_K${K}"

: ${BATCH_SIZE:=1}
: ${EFFECTIVE_BATCH_SIZE:=1}
: ${N_WORKERS:=2}

export EPOCHS=1
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

export ACCUM_ITER
ACCUM_ITER="$(occany_compute_accum_iter "$EFFECTIVE_BATCH_SIZE" "$NUM_GPU_PER_NODE" "$BATCH_SIZE" "$NUM_NODE")"
occany_log_train_config "$EXP_NAME"
CMD="$(occany_select_train_cmd 'launch_da3.py')"
occany_log_start_cmd "$CMD"

echo "[SMOKE] T=$T  K=$K  N_SEQUENCES=$N_SEQUENCES"

$CMD \
    --train_dataset="${N_SEQUENCES} @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=${T}, frame_interval=1, max_memory_num_views=${T}, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True)" \
    --test_dataset="8 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=${T}, frame_interval=1, max_memory_num_views=${T}, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, \
        resolution=[(518, 294)], distill_model_name='SAM3', base_model='da3')" \
    --lr=5e-5 --min_lr=1e-6 --warmup_epochs=0 --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE --accum_iter=$ACCUM_ITER \
    --save_freq=999 --keep_freq=999 --eval_freq=1 --print_freq=1 \
    --num_workers=$N_WORKERS --multiview \
    --amp bf16 --loss_enc_feat \
    --output_dir="$PROJECT/tb_log_occany/$EXP_NAME" \
    --training_objective pointmap_depth_ray --fine_tune_layers 34,35,36,37,38,39 \
    --sam3_proj_lr_mult 10.0 \
    --da3_model_name depth-anything/DA3-GIANT-1.1 \
    --loss_type L1 --pointmap_lambda_c 1.0 --depth_lambda_c 0.0 --raymap_lambda_c 1.0 \
    --infinidepth_pseudo_supervision \
    --lambda_depth 1.0 --lambda_pointmap 1.0 \
    --lambda_pointmap_lidar 1.0 --lambda_pointmap_pseudo 1.0 \
    --lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0 \
    --sam3_use_dpt_proj \
    --distill_model SAM3 --distill_criterion "DistillLoss(nn.L1Loss(), use_conf=False)" \
    --distill_max_frames ${K}
