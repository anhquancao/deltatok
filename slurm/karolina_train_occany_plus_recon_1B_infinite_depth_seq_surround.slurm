#!/bin/bash
#SBATCH --job-name=train_recon_1B_seq_surround
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --partition=qgpu
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/train_recon_1B_seq_surround_%j.out
#SBATCH --error=slurm/output/train_recon_1B_seq_surround_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-2}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"

# N_WORKERS lowered from 10 to 6 to reduce /dev/shm pressure: the surround
# blocks pin num_views_per_timestep=5/6 per sample, which inflates collate
# tensors and previously timed out torch_shm_manager (see runs 4319847 and
# 4322955 — `RuntimeError: Shared memory manager connection has timed out`).
NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" \
NUM_NODE="$NUM_NODE" \
N_WORKERS="${N_WORKERS:-6}" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
bash sh/train_occany_plus_recon_1B_infinite_depth_seq_surround.sh
