#!/bin/bash
#SBATCH --job-name=deltatok_flow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=zuw@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/jz_train_deltatok_flow_%j.out
#SBATCH --error=slurm/output/jz_train_deltatok_flow_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# Default qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_train_deltatok_flow.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The module's site-packages bundles TensorFlow, which some import drags in;
# silence its startup spam (oneDNN notes, cuFFT/cuDNN/cuBLAS factory errors).
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# Overfit debug run (camera-embed variant): single dynamic nuScenes scene
# (scene-0625) repeated 256x, single GPU (config bsize=32, effective_bsize=32 ->
# grad_cum 1; needs world_size==1). Keeps the dtok16 tokenizer override below.
export CONFIG_NAME=train_deltatok_flow_overfit_jeanzay
export RESULTS_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log"
# Distinct name so this run lands in its own subdir.
export RUN_NAME=deltatok_flow_overfit_dtok16_deltaCtx_global_fullT_wd0
# Frozen tokenizer = dtok16 (K=16 delta tokens/camera; deltatok_ckpt override below).
# dtok32 (best, 0.36) blew up to an all-NaN ckpt (see docs/journal/deltatok_flow_dtok32_smoketest.html);
# dtok16 (0.47) is the finite second-best. Swap back to dtok32 once the guarded retrain lands.
# delta_ctx: first delta token (frame 0->1) is the clean in-seq context, no cross-attn.
# global: one attention over all delta tokens. camera_embed gives per-camera identity
# (native-K: a camera's K delta tokens share its embedding; K identity via spatial_pos).
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=16         # K=16 delta tokens/camera (finite ckpt; dtok32 is NaN)
    model.deltatok_ckpt=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_surround_layer12_bsize2_dtok16_cos1e3_1e5_b95/ckpts/current.pth
    model.cond_mode=delta_ctx                  # first delta token as clean in-seq context (no cross-attn)
    model.attn_mode=global                     # single global attention over all delta tokens
    model.vit_use_camera_embed=true            # absolute per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE = 1/(1-t)^2-weighted x-MSE (upweights near-clean t->1)
    training.cache_frozen_encode=true          # overfit: memoize frozen OccRAE+DeltaTok encode per sample -> 1B backbone runs once, not every step
    training.weight_decay=0                    # overfit: drop regularization to isolate optimization from WD
    training.exit_before_time_limit=true       # clean exit near SLURM wall (chain-resume safe)
    training.epoch=400                         # extend overfit past epoch 100 to watch MSEToken converge
    training.eval_num_steps=50                 # multi-step ODE: integrate full trajectory from noise (real generative test, all t now supervised)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# Fresh start each debug run by default; override for chained resume jobs via
#   sbatch --dependency=afterany:<prev_jobid> --export=ALL,RESUME=1 <this script>
# RESUME=1 makes sh/train_deltatok_flow.sh pass --resume (loads ckpts/current.pth).
export RESUME="${RESUME:-0}"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
