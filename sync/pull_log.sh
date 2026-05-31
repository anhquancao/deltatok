#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="bsc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_LOG_DIRS=(
  # "/home/vale/vale352205/code/OccAny/results"
  "/home/vale/vale352205/code/OccAny/slurm/output"
)
LOCAL_LOG_DIRS=(
  # "${REPO_ROOT}/results"
  "${REPO_ROOT}/slurm/output"
)

usage() {
  cat <<'EOF'
Usage:
  pull_log.sh [--dry-run] [--remote HOST] [--src DIR --dst DIR]...

Examples:
  ./pull_log.sh
  ./pull_log.sh --dry-run
  ./pull_log.sh --remote bsc
  ./pull_log.sh --src /remote/results --dst /local/results

Notes:
  - Pulls logs from remote to local using rsync.
  - By default, syncs results/ and slurm/output/.
  - --src/--dst can be repeated; they are matched by position.
EOF
}

DRY_RUN=0
CUSTOM_PATHS=0

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
    --src)
      if [[ "${CUSTOM_PATHS}" -eq 0 ]]; then
        REMOTE_LOG_DIRS=()
        LOCAL_LOG_DIRS=()
        CUSTOM_PATHS=1
      fi
      REMOTE_LOG_DIRS+=("$2")
      shift 2
      ;;
    --dst)
      if [[ "${CUSTOM_PATHS}" -eq 0 ]]; then
        REMOTE_LOG_DIRS=()
        LOCAL_LOG_DIRS=()
        CUSTOM_PATHS=1
      fi
      LOCAL_LOG_DIRS+=("$2")
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

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found in PATH" >&2
  exit 1
fi

if [[ "${#REMOTE_LOG_DIRS[@]}" -ne "${#LOCAL_LOG_DIRS[@]}" ]]; then
  echo "Mismatched sync pairs: got ${#REMOTE_LOG_DIRS[@]} --src values and ${#LOCAL_LOG_DIRS[@]} --dst values" >&2
  exit 2
fi

if [[ "${#REMOTE_LOG_DIRS[@]}" -eq 0 ]]; then
  echo "No sync paths configured" >&2
  exit 2
fi

RSYNC_FLAGS=(
  -az --delete
  --partial
  --info=stats2,progress2
  --human-readable
  --exclude "checkpoint/"
  --exclude "**/checkpoint/"
  --exclude "checkpoints/"
  --exclude "**/checkpoints/"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run)
  echo "Dry run enabled"
fi

TOTAL="${#REMOTE_LOG_DIRS[@]}"

for idx in "${!REMOTE_LOG_DIRS[@]}"; do
  REMOTE_LOG_DIR="${REMOTE_LOG_DIRS[$idx]%/}"
  LOCAL_LOG_DIR="${LOCAL_LOG_DIRS[$idx]%/}"

  mkdir -p "${LOCAL_LOG_DIR}"

  SRC="${REMOTE_HOST}:${REMOTE_LOG_DIR}/"
  DST="${LOCAL_LOG_DIR}/"

  echo "Pulling logs [$((idx + 1))/${TOTAL}]: ${SRC} -> ${DST}"
  rsync "${RSYNC_FLAGS[@]}" "${SRC}" "${DST}"
done
