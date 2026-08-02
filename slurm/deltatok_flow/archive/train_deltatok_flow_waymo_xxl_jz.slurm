#!/bin/bash
#SBATCH --job-name=deltatok_flow_waymo_xxl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=cya@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=slurm/output/train_deltatok_flow_waymo_xxl_jz_%j.out
#SBATCH --error=slurm/output/train_deltatok_flow_waymo_xxl_jz_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# Default qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_jz.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Unique DDP port per job (default 29500 collides with a co-located/leftover job).
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))

# The module's site-packages bundles TensorFlow, which some import drags in;
# silence its startup spam (oneDNN notes, cuFFT/cuDNN/cuBLAS factory errors).
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# WIDTH-FIX of train_deltatok_flow_waymo_whiten_pos_jz.slurm (see
# docs/journal/deltatok_flow_whitenpos_failure_analysis.html): xlarge (1152) <
# C=1536 gave a compressive in_proj the sampler can't invert. Single change vs
# whitenpos: vit_size=xxlarge (width==C). out_proj zero-init is now unconditional
# in code. EMA and mu/sigma deliberately left off to isolate the width fix (EMA
# has never run in a flow job; the mu sweep + EMA are follow-ups). Everything
# else matches whitenpos.
export CONFIG_NAME=train_deltatok_flow_waymo_jeanzay
export RESULTS_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log"
# Distinct name so this run lands in its own subdir (the whitenpos run keeps its dir).
export RUN_NAME=deltatok_flow_waymo_dtok64_deltaCtxCross_global_fullT_wd05_raymapoff_50k_whitenpos_xxl
# Frozen tokenizer = dtok64 (K=64 delta tokens/camera, non-affine z-norm; deltatok_ckpt override below).
# delta_ctx_cross: first delta token (frame 0->1) is the clean in-seq context AND
# frame-0 patch feats condition every block via per-camera cross-attn.
# global: one attention over all delta tokens. camera_embed gives per-camera identity
# (native-K: a camera's K delta tokens share its embedding; K identity via spatial_pos).
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera (must match the frozen dtok64 ckpt)
    model.deltatok_ckpt=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_surround_layer12_bsize2_dtok64_cos1e3_1e5_b95_gradskip_noaff/ckpts/current.pth
    model.deltatok.norm_affine=false           # dtok64 tokenizer is non-affine z-norm (declared true in flow cfg)
    # Per-slot (K, C) whitening of the delta tokens (carried over from whitenpos).
    # Produced by STATS_MODE=position slurm/compute_deltatok_latent_stats_jz.slurm;
    # must match the tokenizer above.
    model.whiten_stats=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_surround_layer12_bsize2_dtok64_cos1e3_1e5_b95_gradskip_noaff/ckpts/latent_stats_waymo_pos.pth
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 tokens as cross-attn condition
    model.attn_mode=global                     # single global attention over all delta tokens
    model.vit_use_camera_embed=true            # absolute per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE = 1/(1-t)^2-weighted x-MSE (upweights near-clean t->1)
    model.vit_size=xxlarge                     # width 1536 == latent C (was xlarge 1152): full-rank in_proj
    # cache_frozen_encode left at config default (false): per-sample encode cache grows unbounded over the full dataset
    # weight_decay left at config default (0.05): restore regularization for the full-data run
    training.exit_before_time_limit=true       # clean exit near SLURM wall (chain-resume safe)
    training.epoch=50                          # 50-epoch schedule; chain-resume across SLURM walls
    training.eval_num_steps=50                 # multi-step ODE: integrate full trajectory from noise (real generative test, all t now supervised)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# Fresh start by default: xxlarge changes the architecture, so no prior ckpt can
# transfer (its own RUN_NAME dir differs anyway). Chained resume jobs of
# THIS run still need RESUME=1:
#   sbatch --dependency=afterany:<prev_jobid> --export=ALL,RESUME=1 <this script>
# RESUME=1 makes sh/train_deltatok_flow.sh pass --resume (loads ckpts/current.pth).
export RESUME="${RESUME:-0}"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
