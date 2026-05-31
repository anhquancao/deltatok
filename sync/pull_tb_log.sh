#!/usr/bin/env bash
set -euo pipefail

# Default configuration
REMOTE_HOST="bsc"
REMOTE_SRC="/gpfs/scratch/ehpc558/quan/occrae_log/"
LOCAL_DST="$HOME/tb_logs/occany/"

echo "Syncing TensorBoard logs: ${REMOTE_HOST}:${REMOTE_SRC} -> ${LOCAL_DST}"

# Ensure local directory exists
mkdir -p "${LOCAL_DST}"

# Sync only event files, pruning empty directories
rsync -az \
    --partial \
    --info=progress2 \
    --include='*/' \
    --include='events.out.tfevents*' \
    --exclude='*' \
    --prune-empty-dirs \
    "${REMOTE_HOST}:${REMOTE_SRC}" \
    "${LOCAL_DST}"
