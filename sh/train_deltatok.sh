#!/usr/bin/env bash
set -euo pipefail

# EVAL_ONLY=1 bash sh/train_deltatok.sh

: "${CONFIG_DIR:=configs}"
: "${CONFIG_NAME:=train_deltatok_karolina}"
: "${EVAL_ONLY:=0}"
# LOG_AND_CKPT_DIR is the training root: TB logs at <LOG_AND_CKPT_DIR>/<RUN_NAME>/
# tb_logs/, checkpoints at <LOG_AND_CKPT_DIR>/<RUN_NAME>/ckpts/. Eval-only loads
# from the same ckpts dir.
: "${LOG_AND_CKPT_DIR:=/mnt/proj1/eu-25-92/deltatok_log}"
# RESULTS_DIR is only used under EVAL_ONLY=1 — eval panels are written to
# <RESULTS_DIR>/<RUN_NAME>/eval_only/.
: "${RESULTS_DIR:=/home/it4i-anhquan/OccAny/results}"
: "${RUN_NAME:=deltatok_surround_constGlobalRope}"
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

if [[ "${RESUME:-0}" == "1" ]]; then
    ARGS+=(--resume)
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