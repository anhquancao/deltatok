#!/bin/bash

#SBATCH --job-name=extract_feat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=qgpu

#SBATCH --output=slurm/output/extract_feat_%A_%a.out
#SBATCH --error=slurm/output/extract_feat_%A_%a.err

# 9 datasets x 20 shards = 180 tasks
#SBATCH --array=0-179

#SBATCH -A eu-25-92

source ~/.bashrc

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 2D array: dataset (0-8) x shard (0-19) ----
SHARDS_PER_DATASET=20

DATASETS=(
    waymo
    once
    ddad
    pandaset
    vkitti
    waymo_surround
    once_surround
    ddad_surround
    pandaset_surround
)

DATASET_INDEX=$(( SLURM_ARRAY_TASK_ID / SHARDS_PER_DATASET ))
SHARD_INDEX=$(( SLURM_ARRAY_TASK_ID % SHARDS_PER_DATASET ))

if [ "$DATASET_INDEX" -ge "${#DATASETS[@]}" ]; then
    echo "Invalid DATASET_INDEX=$DATASET_INDEX (max ${#DATASETS[@]}-1)"
    exit 1
fi

export DATASET_NAME="${DATASETS[$DATASET_INDEX]}"
export PID="$SHARD_INDEX"
export WORLD="$SHARDS_PER_DATASET"

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET_NAME, shard=$PID/$WORLD"

bash sh/extract_occany_features.sh
