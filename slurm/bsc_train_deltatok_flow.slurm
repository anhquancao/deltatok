#!/bin/bash
#SBATCH --job-name=deltatok_flow_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=48:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/bsc_train_deltatok_flow_%j.out
#SBATCH --error=slurm/output/bsc_train_deltatok_flow_%j.err

# BSC acc partition: 4 H100/node, 80 cores -> 20 cpus/task.
# acc_ehpc caps walltime at 48h; acc_debug is 2h for smoke runs.
#
#   ssh bsc "cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/bsc_train_deltatok_flow.slurm"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# Full-dataset run mirroring slurm/jz_train_deltatok_flow.slurm with ehpc1001
# dataset roots (configs/train_deltatok_flow_bsc.yaml). Single GPU (config
# bsize=16, effective_bsize=32 -> grad_cum 2).
export CONFIG_NAME=train_deltatok_flow_bsc
export RESULTS_DIR="$SCRATCH/deltatok_flow_log"
export RUN_NAME=deltatok_flow_full_layer12
# Frozen tokenizer: the BSC tokenizer run's checkpoint. Requires
# slurm/bsc_train_deltatok.slurm to have produced ckpts/current.pth first.
DELTATOK_CKPT="$SCRATCH/deltatok_log/deltatok_surround_layer12_bsize2_featloss/ckpts/current.pth"
EXTRA_CFG_ARGS=(
    model.deltatok_ckpt="$DELTATOK_CKPT"       # frozen tokenizer from the BSC tokenizer run
    model.deltatok.use_camera_rope=false       # must match the tokenizer run's arch (strict=True load)
    training.exit_before_time_limit=true       # clean exit near SLURM wall (chain-resume safe)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# Fresh start each run by default; override for chained resume jobs via
#   sbatch --dependency=afterany:<prev_jobid> --export=ALL,RESUME=1 <this script>
export RESUME="${RESUME:-0}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
