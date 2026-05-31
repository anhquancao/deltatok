#!/usr/bin/env bash
set -euo pipefail

# DeltaTok with train-time geometric supervision. Same trainer as
# sh/train_deltatok.sh; differs only in RUN_NAME so logs/ckpts land in
# a distinct directory. All loss / training params live in the Hydra
# config (configs/train_deltatok*.yaml).
#
# EVAL_ONLY=1 bash sh/train_deltatok_geom.sh

: "${CONFIG_DIR:=configs}"
: "${CONFIG_NAME:=train_deltatok_geom_karolina}"
: "${EVAL_ONLY:=0}"
: "${LOG_AND_CKPT_DIR:=/mnt/proj1/eu-25-92/deltatok_log}"
: "${RESULTS_DIR:=/home/it4i-anhquan/OccAny/results}"
: "${RUN_NAME:=deltatok_surround_constGlobalRope_geom_recon100_pm0p01_d0p01_ray0p01}"
: "${PRECISION:=bf16}"
ARGS=(
    --config-dir "$CONFIG_DIR"
    --config-name "$CONFIG_NAME"
    --log-and-ckpt-dir "$LOG_AND_CKPT_DIR"
    --precision "$PRECISION"
    --run-name "$RUN_NAME"
)

if [[ "$EVAL_ONLY" == "1" ]]; then
    ARGS+=(--eval-only --results-dir "$RESULTS_DIR")
fi

if [[ -n "${OCCANY_RECON_CKPT:-}" ]]; then
    ARGS+=(--occany_recon_ckpt "$OCCANY_RECON_CKPT")
fi

CMD=(python train_deltatok.py)

if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    printf 'Executing:'
    printf ' %q' "${CMD[@]}" "${ARGS[@]}"
    printf '\n'
fi

exec "${CMD[@]}" "${ARGS[@]}"
