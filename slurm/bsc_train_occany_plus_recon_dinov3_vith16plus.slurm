#!/bin/bash
#SBATCH --job-name=dv3_vith16p
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc793
#SBATCH --partition=acc
#SBATCH --output=slurm/output/dv3_vith16p_%j.out
#SBATCH --error=slurm/output/dv3_vith16p_%j.err
#SBATCH --exclusive

##SBATCH --time=2:00:00
##SBATCH --qos=acc_debug
##SBATCH --nodes=4

#SBATCH --time=48:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=4

#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4


source ~/.bashrc
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"


cd "$HOME/code/OccAny"
source env_bsc.sh

export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-4}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-4}"

NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" \
NUM_NODE="$NUM_NODE" \
N_WORKERS="${N_WORKERS:-12}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
bash sh/train_occany_plus_recon_dinov3_vith16plus_bsc.sh
