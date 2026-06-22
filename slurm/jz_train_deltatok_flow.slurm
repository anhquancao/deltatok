#!/bin/bash
#SBATCH --job-name=deltatok_flow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=lwy@h100
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

# Overfit debug run (camera-rope variant): single dynamic nuScenes scene
# (scene-0625) repeated 256x, single GPU (config bsize=32, effective_bsize=32 ->
# grad_cum 1; needs world_size==1).
export CONFIG_NAME=train_deltatok_flow_overfit_jeanzay
export RESULTS_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log"
# Distinct run name so the camera-rope overfit ckpts/viz land in their own subdir
# and don't clobber the const-global deltatok_flow_overfit_v3 run.
export RUN_NAME=deltatok_flow_overfit_cameraRope32
# Camera rope on for BOTH the tokenizer and the flow ViT. The frozen tokenizer is
# repointed at the cameraRope32 run (the overfit_jeanzay default ckpt is
# const-global-rope, which would mismatch the rope mode); it's the live current.pth,
# so the flow job snapshots whatever weights exist at start.
DELTATOK_CKPT="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_surround_layer12_bsize2_cameraRope32/ckpts/current.pth"
EXTRA_CFG_ARGS=(
    model.deltatok.use_camera_rope=true        # tokenizer camera rope
    model.vit_use_camera_rope=true             # flow ViT camera rope
    model.deltatok_ckpt="$DELTATOK_CKPT"       # frozen camera-rope tokenizer
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# Fresh start each debug run by default; override for chained resume jobs via
#   sbatch --dependency=afterany:<prev_jobid> --export=ALL,RESUME=1 <this script>
# RESUME=1 makes sh/train_deltatok_flow.sh pass --resume (loads ckpts/current.pth).
export RESUME="${RESUME:-0}"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
