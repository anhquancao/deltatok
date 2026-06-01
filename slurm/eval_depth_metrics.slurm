#!/bin/bash
#SBATCH --job-name=eval_depth_metrics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0,1
#SBATCH --hint=nomultithread
#SBATCH -A eu-25-92
#SBATCH --partition=qgpu
#SBATCH --output=slurm/output/eval_depth_metrics_%A_%a.out
#SBATCH --error=slurm/output/eval_depth_metrics_%A_%a.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PID="${PID:-0}"
WORLD="${WORLD:-1}"

PROJECT="${PROJECT:-$PWD}"
nuscenes_root="${NUSCENES_ROOT:-$PROJECT/data/nuscenes}"

# Array task -> model: 0=da3, 1=infinidepth_depth_sensor.
# An explicit MODEL env var still wins.
if [ -z "${MODEL:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
        MODEL="da3"
    elif [ "${SLURM_ARRAY_TASK_ID}" = "1" ]; then
        MODEL="infinidepth_depth_sensor"
    else
        echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
    fi
fi

EXP_NAME="${EXP_NAME:-da3_gtscale_depth}"
DA3_MODEL_NAME="${DA3_MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
INFINIDEPTH_CKPT="${INFINIDEPTH_CKPT:-/home/it4i-anhquan/deltatok/checkpoints/infinidepth_depthsensor.ckpt}"

LITE_FLAG=""
if [ "${LITE:-0}" = "1" ]; then
    LITE_FLAG="--lite"
fi

python eval_depth_metrics.py \
    --dataset nuscenes \
    --setting surround \
    --split val \
    --model "$MODEL" \
    --exp_name "$EXP_NAME" \
    --da3_model_name "$DA3_MODEL_NAME" \
    --infinidepth_ckpt "$INFINIDEPTH_CKPT" \
    --nuscenes_root "$nuscenes_root" \
    --output_dir "${DEPTH_OUTPUT:-./results}" \
    --world="$WORLD" --pid="$PID" \
    $LITE_FLAG
