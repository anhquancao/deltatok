#!/bin/bash
#SBATCH --job-name=deltatok_l12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --partition=qgpu_exp
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/karolina_train_deltatok_layer12_%j.out
#SBATCH --error=slurm/output/karolina_train_deltatok_layer12_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

cd "$HOME/deltatok"

export CONFIG_NAME=train_deltatok_karolina
export RESULTS_DIR="/mnt/proj1/eu-25-92/deltatok_log"
export RUN_NAME=deltatok_surround_constGlobalRope_layer12
export RESUME=1
export ENCODE_LAYER=12

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "ENCODE_LAYER=$ENCODE_LAYER"

srun --cpu-bind=none bash sh/train_deltatok.sh
