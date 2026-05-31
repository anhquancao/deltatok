#!/bin/bash
#SBATCH --job-name=deltatok_geom_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --partition=qgpu_exp
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/train_deltatok_geom_smoke_%j.out
#SBATCH --error=slurm/output/train_deltatok_geom_smoke_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

cd "$HOME/OccAny"

srun --cpu-bind=none bash sh/train_deltatok_geom_smoke.sh
