#!/usr/bin/env bash
set -euo pipefail

# Smoke variant of sh/train_deltatok_geom.sh. Forces every step to use the
# worst-case shape (24 views @ 518x294) so a small max_iter is enough to
# verify the decode_grad depth_head OOM fix on the qgpu_exp express queue.

: "${CONFIG_DIR:=configs}"
: "${CONFIG_NAME:=train_deltatok_geom_smoke}"
: "${EVAL_ONLY:=0}"
: "${LOG_AND_CKPT_DIR:=/mnt/proj1/eu-25-92/deltatok_log}"
: "${RESULTS_DIR:=/home/it4i-anhquan/OccAny/results}"
: "${RUN_NAME:=deltatok_geom_oomfix_smoke}"
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
