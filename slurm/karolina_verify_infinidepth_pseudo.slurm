#!/bin/bash

#SBATCH --job-name=verify_infinidepth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=qgpu

#SBATCH --output=slurm/output/verify_infinidepth_%j.out
#SBATCH --error=slurm/output/verify_infinidepth_%j.err

#SBATCH -A eu-25-92

source ~/.bashrc

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

python verify_infinidepth_pseudo.py --check_readable
