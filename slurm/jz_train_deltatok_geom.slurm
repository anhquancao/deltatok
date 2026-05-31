#!/bin/bash
#SBATCH --job-name=deltatok_geom
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=lwy@h100
#SBATCH --time=20:00:00
#SBATCH --output=slurm/output/deltatok_geom_%j.out
#SBATCH --error=slurm/output/deltatok_geom_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# Default qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).

mkdir -p slurm/output

source ~/.bashrc

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/OccAny

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# jean-zay config + output dirs (consumed by sh/train_deltatok_geom.sh).
# Logs/ckpts go to the persistent fswork tier (fsn1 purges after 30 days).
export CONFIG_NAME="${CONFIG_NAME:-train_deltatok_geom_jeanzay}"
export LOG_AND_CKPT_DIR="${LOG_AND_CKPT_DIR:-$TRG_WORK/deltatok_log}"
export RESULTS_DIR="${RESULTS_DIR:-$TRG_WORK/code/OccAny/results}"

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/OccAny

srun --cpu-bind=none bash sh/train_deltatok_geom.sh
