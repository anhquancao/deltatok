#!/bin/bash
#SBATCH --job-name=occrae_overfit
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc551
#SBATCH --partition=acc
#SBATCH --output=slurm/output/bsc_train_occrae_overfit_%j.out
#SBATCH --error=slurm/output/bsc_train_occrae_overfit_%j.err

#SBATCH --time=2:00:00
#SBATCH --qos=acc_debug

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1


source ~/.bashrc
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$HOME/code/OccAny"
source env_bsc.sh

CONFIG_NAME=train_occrae_fm_overfit RUN_NAME=overfit srun --cpu-bind=none bash sh/train_occrae.sh
