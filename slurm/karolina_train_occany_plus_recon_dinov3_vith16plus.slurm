#!/bin/bash
#SBATCH --job-name=train_occany_plus_recon_dinov3_vith16plus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --partition=qgpu
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/train_occany_plus_recon_dinov3_vith16plus_%j.out
#SBATCH --error=slurm/output/train_occany_plus_recon_dinov3_vith16plus_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-4}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"

NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" \
NUM_NODE="$NUM_NODE" \
N_WORKERS="${N_WORKERS:-10}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
bash sh/train_occany_plus_recon_dinov3_vith16plus.sh
