#!/bin/bash
#SBATCH --job-name=deltatok_jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=lwy@h100
##SBATCH --qos=qos_gpu_h100-t3
##SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/jz_train_deltatok_%j.out
#SBATCH --error=slurm/output/jz_train_deltatok_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_train_deltatok.slurm'"

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

# DeltaTok tokenizer training on Jean Zay (camera-rope layer-12 run), mirroring
# slurm/karolina_train_deltatok_layer12.slurm with JZ dataset roots
# (configs/train_deltatok_jeanzay.yaml) and the JZ log/ckpt tier.
export CONFIG_NAME=train_deltatok_jeanzay
export LOG_AND_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export RUN_NAME=deltatok_surround_layer12_bsize1_cameraRope32
export RESUME=1
export EXTRA_CFG="model.deltatok.use_camera_rope=true training.bsize=1"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
