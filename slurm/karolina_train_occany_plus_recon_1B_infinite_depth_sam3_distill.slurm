#!/bin/bash
#SBATCH --job-name=train_occany_plus_recon_1B_sam3_distill
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --partition=qgpu
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/train_occany_plus_recon_1B_sam3_distill_%j.out
#SBATCH --error=slurm/output/train_occany_plus_recon_1B_sam3_distill_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-4}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"

# 4-GPU DDP smoke on qgpu_exp (max mem MiB, +~2-3 GiB vs single-GPU baseline):
#   T=19 K=18  37450
#   T=20 K=16  37381
#   T=21 K=14  37312
#   T=22 K=12  37242
#   T=23 K=10  37173
#   T=24 K=8   37104
#   T=25 K=6   37035
#   T=26 K=4   36966
#   T=27 K=2   37283
#   T=28 K=0   37385
#
# At 32 GPUs (4 nodes x 8) expect another ~0.5-1.5 GiB of NCCL/IB overhead on
# top of the 4-GPU numbers -> peak ~38-39 GiB. Tight but should fit; consider a
# full-scale smoke before launching the 48 h job.

NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" \
NUM_NODE="$NUM_NODE" \
N_WORKERS="${N_WORKERS:-8}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
bash sh/train_occany_plus_recon_1B_infinite_depth_sam3_distill.sh
