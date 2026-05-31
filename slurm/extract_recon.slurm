#!/bin/bash
#SBATCH --job-name=extract_recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --hint=nomultithread
#SBATCH -A eu-25-92
#SBATCH --partition=qgpu_exp
#SBATCH --output=slurm/output/extract_recon_%A_%a.out
#SBATCH --error=slurm/output/extract_recon_%A_%a.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export EXP_LIST="${EXP_LIST:-recon}"
# Synced e55 pseudo-depth-only checkpoint. One submission per setting (this is a
# 4-job array over WORLD shards): EXP_ID=8 (KITTI, default), EXP_ID=9 (nuScenes).
export EXP_ID="${EXP_ID:-8}"
export WORLD="${WORLD:-4}"

bash sh/extract_recon.sh
