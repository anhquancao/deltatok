#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_NAME:=train_occrae_img_decoder_karolina}"
CMD=(python train_occrae_img_decoder.py --config-dir configs --config-name "$CONFIG_NAME")

if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    printf 'Executing:' ; printf ' %q' "${CMD[@]}" ; printf '\n'
fi

exec "${CMD[@]}"
