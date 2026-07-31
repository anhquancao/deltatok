#!/bin/bash
#SBATCH --job-name=deltatok_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=48:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/bsc_train_deltatok_%j.out
#SBATCH --error=slurm/output/bsc_train_deltatok_%j.err

# BSC acc partition: 4 H100/node, 80 cores -> 20 cpus/task.
# acc_ehpc caps walltime at 72h; acc_debug is 2h for smoke runs.
#
#   ssh bsc "cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/bsc_train_deltatok.slurm"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# DeltaTok tokenizer training on BSC (no-camera-rope layer-12 run), mirroring
# slurm/jz_train_deltatok.slurm with ehpc1001 dataset roots
# (configs/train_deltatok_bsc.yaml) and the ehpc1001 scratch log/ckpt tier.
export CONFIG_NAME=train_deltatok_bsc
export LOG_AND_CKPT_DIR="$SCRATCH/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export RUN_NAME=deltatok_surround_layer12_bsize2_featloss
export RESUME=${RESUME:-0}   # override with --export=ALL,RESUME=1 to resume current.pth
EXTRA_CFG_ARGS=(
    model.deltatok.use_camera_rope=false       # no-camera-rope layer-12 run
    training.exit_before_time_limit=true       # stop after a ckpt when SLURM wall is near (chain-resume safe)
    training.feature_loss_weight=0.1           # downstream DA3 [19,27,33,39] feature loss
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
