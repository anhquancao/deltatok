#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_DIR:=configs}"
: "${CONFIG_NAME:=train_deltatok_flow}"
: "${RESULTS_DIR:=/mnt/proj1/eu-25-92/deltatok_flow_log}"
: "${RUN_NAME:=deltatok_flow}"
: "${PRECISION:=bf16}"
: "${NUM_GPU_PER_NODE:=${SLURM_GPUS_ON_NODE:-1}}"
: "${RESUME:=0}"

ARGS=(
    --config-dir "$CONFIG_DIR"
    --config-name "$CONFIG_NAME"
    --results-dir "$RESULTS_DIR"
    --precision "$PRECISION"
    --run-name "$RUN_NAME"
    # --ckpt "/gpfs/scratch/ehpc551/occrae_exps/overfit/ckpts/current.pth"
)

# RESUME=1 resumes from <RESULTS_DIR>/<RUN_NAME>/ckpts/current.pth (no-op if absent).
if [[ "$RESUME" == "1" ]]; then
    ARGS+=(--resume)
fi

# Forward extra Hydra overrides (space-separated, e.g.
# "model.deltatok.use_camera_rope=true"). Unquoted on purpose so each override
# becomes a separate arg to --cfg (nargs="*").
if [[ -n "${EXTRA_CFG:-}" ]]; then
    ARGS+=(--cfg $EXTRA_CFG)
fi

if [[ "${SLURM_NTASKS:-1}" -gt 1 ]]; then
    CMD=(python train_deltatok_flow.py)
    echo "Launching: local_rank=${SLURM_LOCALID:-0}, global_rank=${SLURM_PROCID:-0}, nodes=${SLURM_NNODES:-1}, world_size=${SLURM_NTASKS}, gpus_per_node=${SLURM_NTASKS_PER_NODE:-unknown}"
else
    echo "Launching standalone torchrun on local node"
    CMD=(
        torchrun
        --standalone
        --nnodes=1
        --nproc_per_node="$NUM_GPU_PER_NODE"
        train_deltatok_flow.py
    )
fi

if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    printf 'Executing:'
    printf ' %q' "${CMD[@]}" "${ARGS[@]}"
    printf '\n'
fi

exec "${CMD[@]}" "${ARGS[@]}"
