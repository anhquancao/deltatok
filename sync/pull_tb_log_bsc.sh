#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="bsc"
REMOTE_SRC="/gpfs/scratch/ehpc793/tb_log_occany/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_DST="${REPO_ROOT}/tb_logs/"

echo "Syncing TensorBoard logs (excluding *.pth): ${REMOTE_HOST}:${REMOTE_SRC} -> ${LOCAL_DST}"

mkdir -p "${LOCAL_DST}"

rsync -az --delete \
    --partial \
    --info=stats2,progress2 \
    --human-readable \
    --exclude='*.pth' \
    "${REMOTE_HOST}:${REMOTE_SRC}" \
    "${LOCAL_DST}"
