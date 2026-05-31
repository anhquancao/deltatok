#!/usr/bin/env bash

REMOTE_HOST="karolina"
REMOTE_DIR="/home/it4i-anhquan"

usage() {
  cat <<'EOF'
Usage:
  push_code_karolina.sh [--dry-run] [--remote HOST] [--remote-dir DIR]

Examples:
  ./push_code_karolina.sh
  ./push_code_karolina.sh --dry-run
  ./push_code_karolina.sh --remote karolina --remote-dir /home/it4i-anhquancao/code

Notes:
  - Assumes SSH connectivity to HOST (e.g. via ~/.ssh/config alias "karolina").
  - Syncs the repository root (one level above this script directory).
EOF
}

DRY_RUN=${DRY_RUN:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    --remote)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found in PATH" >&2
  exit 1
fi

RSYNC_FLAGS=(
  -az --delete
  --partial
  --info=stats2,progress2
  --human-readable
  --prune-empty-dirs
  --exclude ".git/"
  --exclude "__pycache__/"
  --exclude "**/__pycache__/"
  --exclude "*.pyc"
  --exclude "results/**"
  --exclude "slurm/*.out"
  --exclude "slurm/*.err"
  --exclude "slurm/output/"
  --exclude "outputs/"
  --exclude "pretrained_ckpts/"
  --exclude "checkpoint/"
  --exclude "**/checkpoint/"
  --exclude "checkpoints/"
  --exclude "**/checkpoints/"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run -v)
  echo "Dry run enabled"
fi

SRC="${REPO_ROOT}/"
DST="${REMOTE_HOST}:${REMOTE_DIR}/OccAny/"
SYNC_INTERVAL=3

echo "Syncing every ${SYNC_INTERVAL}s: ${SRC} -> ${DST}"

while true; do
  rsync "${RSYNC_FLAGS[@]}" "${SRC}" "${DST}"
  sleep "${SYNC_INTERVAL}"
done