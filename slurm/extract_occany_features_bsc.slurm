#!/bin/bash
#SBATCH --job-name=extract_occany_feat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --hint=nomultithread
#SBATCH --account=ehpc551
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --output=slurm/output/extract_occany_feat/%A_%a.out
#SBATCH --error=slurm/output/extract_occany_feat/%A_%a.err

source ~/.bashrc

cd "$HOME/code/OccAny"

source env_bsc.sh

PID="${SLURM_ARRAY_TASK_ID:-0}" WORLD="${SLURM_ARRAY_TASK_COUNT:-20}" \
    bash sh/extract_occany_features.sh
