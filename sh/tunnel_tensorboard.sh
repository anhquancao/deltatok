#!/usr/bin/env bash
# Forward ports 6006 and 6007 from Karolina to localhost (e.g. for TensorBoard).
# Usage: bash sh/tunnel_tensorboard.sh
set -euo pipefail

HOST="${KAROLINA_HOST:-karolina}"
PORTS=(6006 6007)

FWD_ARGS=()
for p in "${PORTS[@]}"; do
    FWD_ARGS+=(-L "${p}:localhost:${p}")
done

echo "Forwarding ${PORTS[*]} from ${HOST} -> localhost. Ctrl+C to stop."
exec ssh -N "${FWD_ARGS[@]}" "${HOST}"
