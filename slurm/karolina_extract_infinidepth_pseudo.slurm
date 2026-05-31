#!/bin/bash

#SBATCH --job-name=extract_infinidepth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=qgpu

#SBATCH --output=slurm/output/extract_infinidepth_%A_%a.out
#SBATCH --error=slurm/output/extract_infinidepth_%A_%a.err

# Global slice: each task processes 1/WORLD of every processed root.
#SBATCH --array=0-1

#SBATCH -A eu-25-92

source ~/.bashrc

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PID="$SLURM_ARRAY_TASK_ID"
export WORLD=100

echo "Task $SLURM_ARRAY_TASK_ID: global shard=$PID/$WORLD (all processed roots)"

bash sh/extract_infinidepth_pseudo.sh
