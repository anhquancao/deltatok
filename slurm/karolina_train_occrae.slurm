#!/bin/bash
#SBATCH --job-name=occrae
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --partition=qgpu
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/karolina_train_occrae_%j.out
#SBATCH --error=slurm/output/karolina_train_occrae_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

cd "$HOME/code/OccAny"

export CONFIG_NAME=train_occrae_fm_karolina
export RESULTS_DIR="/mnt/proj1/eu-25-92/occrae_log"

srun --cpu-bind=none bash sh/train_occrae.sh