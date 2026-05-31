#!/bin/bash
#SBATCH --job-name=eval_deltatok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --partition=qgpu
#SBATCH -A eu-25-92
#SBATCH --array=0-2
#SBATCH --output=slurm/output/karolina_eval_deltatok_%A_%a.out
#SBATCH --error=slurm/output/karolina_eval_deltatok_%A_%a.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

cd "$HOME/OccAny"

RUN_NAMES=(
    deltatok
    deltatok_surround
    deltatok_surround_constGlobalRope
)

export CONFIG_NAME=train_deltatok_karolina
export LOG_AND_CKPT_DIR="${LOG_AND_CKPT_DIR:-/mnt/proj1/eu-25-92/deltatok_log}"
export RESULTS_DIR="${RESULTS_DIR:-$HOME/OccAny/results}"
export RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"
export EVAL_ONLY=1

echo "[karolina_eval_deltatok] task=$SLURM_ARRAY_TASK_ID RUN_NAME=$RUN_NAME"
echo "[karolina_eval_deltatok] outputs -> $RESULTS_DIR/$RUN_NAME/eval_only/eval_depth/"

srun --cpu-bind=none bash sh/train_deltatok.sh
