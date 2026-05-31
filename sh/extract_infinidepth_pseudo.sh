#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Extract InfiniDepth pseudo-depth for preprocessed temporal datasets.
#
# Writes a sibling <stem>.infinidepth.png (uint16, KITTI scale=256, 0=invalid)
# next to each source <stem>.npz. The first --num_vis samples also get a
# colorized side-by-side PNG under $VIS_DIR for verification.
#
# Environment variables:
#   PREPROCESS_DATA_ROOT  – root containing processed dataset folders
#                           (default: /scratch/project/eu-25-92/data)
#   OUTPUT_DIR            – pseudo-depth output root (default: same as
#                           PREPROCESS_DATA_ROOT → sibling files)
#   INFINIDEPTH_CKPT      – checkpoint path
#                           (default: ./checkpoints/infinidepth_depthsensor.ckpt)
#   BATCH_SIZE            – views per InfiniDepth forward (default: 16, tuned for 40 GB)
#   CHUNK_SIZE            – inner re-chunk size (default: equal to BATCH_SIZE)
#   DEPTH_MIN / DEPTH_MAX – LiDAR conditioning filter (default: 1e-3 / 80.0)
#   NUM_VIS               – number of side-by-side vis PNGs (default: 5; 0 disables)
#   VIS_DIR               – vis output dir (default: ./results/infinidepth_pseudo_vis)
#   PID / WORLD           – sharding (defaults: SLURM_ARRAY_TASK_ID / 1)
#   MAX_SAMPLES           – per-shard cap for smoke-testing (default: -1 = all).
#                           Not needed for resumption — existing sibling PNGs
#                           are skipped when OVERWRITE=0.
#   OVERWRITE             – set to 1 to recompute even if output exists
#   DA3_METRIC_MODEL_NAME – DA3 metric model for sky masking
#                           (default: depth-anything/DA3METRIC-LARGE)
#   SKY_MASK_THRESHOLD    – sky-score threshold; pixels >= are masked (default: 0.3)
#   SKIP_SKY_MASK         – set to 1 to disable DA3 sky masking entirely
# ---------------------------------------------------------------------------

PREPROCESS_DATA_ROOT="${PREPROCESS_DATA_ROOT:-/scratch/project/eu-25-92/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PREPROCESS_DATA_ROOT}"
INFINIDEPTH_CKPT="${INFINIDEPTH_CKPT:-./checkpoints/infinidepth_depthsensor.ckpt}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-$BATCH_SIZE}"
DEPTH_MIN="${DEPTH_MIN:-0.001}"
DEPTH_MAX="${DEPTH_MAX:-80.0}"
NUM_VIS="${NUM_VIS:-5}"
VIS_DIR="${VIS_DIR:-./results/infinidepth_pseudo_vis}"
PID="${PID:-${SLURM_ARRAY_TASK_ID:-0}}"
WORLD="${WORLD:-1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
OVERWRITE="${OVERWRITE:-0}"
DA3_METRIC_MODEL_NAME="${DA3_METRIC_MODEL_NAME:-depth-anything/DA3METRIC-LARGE}"
SKY_MASK_THRESHOLD="${SKY_MASK_THRESHOLD:-0.3}"
SKIP_SKY_MASK="${SKIP_SKY_MASK:-0}"

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi
if [ "$SKIP_SKY_MASK" = "1" ]; then
    EXTRA_ARGS+=(--skip_sky_mask)
    SKY_STATUS="off"
else
    SKY_STATUS="on@thr=$SKY_MASK_THRESHOLD"
fi

echo "[extract_infinidepth_pseudo] root=$PREPROCESS_DATA_ROOT pid=$PID/$WORLD batch=$BATCH_SIZE lidar_input_depth_range=[$DEPTH_MIN,$DEPTH_MAX]m sky_mask=$SKY_STATUS vis=$NUM_VIS"

python extract_infinidepth_pseudo.py \
    --preprocess_data_root "$PREPROCESS_DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --infinidepth_ckpt "$INFINIDEPTH_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --chunk_size "$CHUNK_SIZE" \
    --depth_min "$DEPTH_MIN" \
    --depth_max "$DEPTH_MAX" \
    --num_vis "$NUM_VIS" \
    --vis_dir "$VIS_DIR" \
    --pid "$PID" \
    --world "$WORLD" \
    --max_samples "$MAX_SAMPLES" \
    --da3_metric_model_name "$DA3_METRIC_MODEL_NAME" \
    --sky_mask_threshold "$SKY_MASK_THRESHOLD" \
    "${EXTRA_ARGS[@]}"
