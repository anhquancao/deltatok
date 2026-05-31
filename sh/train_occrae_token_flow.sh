#!/bin/bash
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

: "${CONFIG_PATH:=configs/occrae_token_flow.yaml}"
: "${RESULTS_DIR:=results}"
: "${TRAIN_FEATURE_ROOT:=./outputs/occany_features_train}"
: "${VAL_FEATURE_ROOT:=}"
: "${PRECISION:=bf16}"

if [ "${NUM_GPU_PER_NODE:-1}" -gt 1 ]; then
    CMD="torchrun --standalone --nproc_per_node=${NUM_GPU_PER_NODE} train_occany_tokens.py"
else
    CMD="python train_occany_tokens.py"
fi

occany_log_start_cmd "$CMD"

if [ -n "$VAL_FEATURE_ROOT" ]; then
    $CMD \
        --config "$CONFIG_PATH" \
        --results-dir "$RESULTS_DIR" \
        --precision "$PRECISION" \
        --run-name occrae_token_flow \
        --train-feature-root "$TRAIN_FEATURE_ROOT" \
        --val-feature-root "$VAL_FEATURE_ROOT"
else
    $CMD \
        --config "$CONFIG_PATH" \
        --results-dir "$RESULTS_DIR" \
        --precision "$PRECISION" \
        --run-name occrae_token_flow \
        --train-feature-root "$TRAIN_FEATURE_ROOT"
fi
