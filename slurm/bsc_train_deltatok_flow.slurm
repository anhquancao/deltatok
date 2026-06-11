#!/bin/bash
#SBATCH --job-name=bsc_deltatok_flow
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc551
#SBATCH --partition=acc
#SBATCH --output=slurm/output/bsc_train_deltatok_flow_%j.out
#SBATCH --error=slurm/output/bsc_train_deltatok_flow_%j.err


##SBATCH --time=2:00:00
##SBATCH --qos=acc_debug

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

# srun --cpu-bind=none bash sh/train_deltatok_flow.sh
# Overfit command:
# CONFIG_NAME=train_deltatok_flow_overfit RUN_NAME=overfit srun --cpu-bind=none bash sh/train_deltatok_flow.sh
srun --cpu-bind=none bash sh/train_deltatok_flow.sh
