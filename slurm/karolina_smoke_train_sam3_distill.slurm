#!/bin/bash
#SBATCH --job-name=smoke_train_sam3_distill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --partition=qgpu_exp
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/smoke_train_sam3_distill_%j.out
#SBATCH --error=slurm/output/smoke_train_sam3_distill_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-1}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-4}"

NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" \
NUM_NODE="$NUM_NODE" \
N_WORKERS="${N_WORKERS:-10}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
EFFECTIVE_BATCH_SIZE=4 \
T="${T:-28}" \
K="${K:-0}" \
bash sh/smoke_train_sam3_distill.sh
