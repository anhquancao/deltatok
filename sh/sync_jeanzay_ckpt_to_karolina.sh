#!/usr/bin/env bash
# Stream a single checkpoint from Jean Zay -> Karolina, routing through the
# LOCAL machine without ever writing to local disk (a pure SSH pipe).
#
# Topology: from here we can reach both `ssh jean-zay` (reverse tunnel via
# Karolina) and `ssh karolina`. Bytes flow:
#     jean-zay (cat)  ->  local ssh pipe  ->  karolina (cat > file)
# so nothing is staged on local disk. Writes to a .partial and only renames to
# the final name after size + (optional) sha256 match on both ends.
#
# Usage:
#   sh/sync_jeanzay_ckpt_to_karolina.sh [SRC_NAME] [DST_NAME]
#     SRC_NAME  checkpoint file in the exp dir on jean-zay (default checkpoint-55.pth)
#     DST_NAME  destination filename on karolina (default derived from exp+epoch)
#
# Env overrides:
#   JZ_HOST=jean-zay  KAR_HOST=karolina
#   KAR_DST_DIR=/home/it4i-anhquan/OccAny/checkpoints
#   VERIFY_SHA=1      set 0 to skip sha256 (size check is always done)

set -euo pipefail

# --- config -----------------------------------------------------------------
JZ_HOST="${JZ_HOST:-jean-zay}"
KAR_HOST="${KAR_HOST:-karolina}"

EXP_NAME="occany_plus_recon_1B_infinite_depth_sam3_distill_memory_budget_pseudo_depth_only"
JZ_EXP_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/tb_log_occany/${EXP_NAME}"

SRC_NAME="${1:-checkpoint-55.pth}"
SRC_FILE="${JZ_EXP_DIR}/${SRC_NAME}"

EPOCH="${SRC_NAME#checkpoint-}"; EPOCH="${EPOCH%.pth}"
KAR_DST_DIR="${KAR_DST_DIR:-/home/it4i-anhquan/OccAny/checkpoints}"
DST_NAME="${2:-${EXP_NAME}_e${EPOCH}.pth}"
DST_FILE="${KAR_DST_DIR}/${DST_NAME}"
PART_FILE="${DST_FILE}.partial"

VERIFY_SHA="${VERIFY_SHA:-1}"
# ----------------------------------------------------------------------------

echo "Source : ${JZ_HOST}:${SRC_FILE}"
echo "Dest   : ${KAR_HOST}:${DST_FILE}"
echo

# --- pre-flight -------------------------------------------------------------
echo "[1/5] Pre-flight checks ..."
SRC_SIZE="$(ssh "$JZ_HOST" "stat -c %s '$SRC_FILE'" 2>/dev/null)" || {
  echo "ERROR: source not found on ${JZ_HOST}: ${SRC_FILE}" >&2; exit 1; }
echo "       source size: ${SRC_SIZE} bytes ($(( SRC_SIZE / 1024 / 1024 )) MiB)"

# Skip if final dest already exists with matching size.
EXIST_SIZE="$(ssh "$KAR_HOST" "stat -c %s '$DST_FILE' 2>/dev/null || echo 0")"
if [ "$EXIST_SIZE" = "$SRC_SIZE" ]; then
  echo "       dest already present with matching size — nothing to do."
  exit 0
fi

ssh "$KAR_HOST" "mkdir -p '$KAR_DST_DIR'"
AVAIL_KB="$(ssh "$KAR_HOST" "df -P '$KAR_DST_DIR' | awk 'NR==2{print \$4}'")"
echo "       karolina free: $(( AVAIL_KB / 1024 / 1024 )) GiB"
if [ "$(( AVAIL_KB * 1024 ))" -lt "$SRC_SIZE" ]; then
  echo "ERROR: not enough space on ${KAR_HOST}" >&2; exit 1
fi

# --- stream -----------------------------------------------------------------
echo "[2/5] Streaming (jean-zay -> local pipe -> karolina, no local disk) ..."
if command -v pv >/dev/null 2>&1; then
  ssh "$JZ_HOST" "cat '$SRC_FILE'" | pv -s "$SRC_SIZE" | ssh "$KAR_HOST" "cat > '$PART_FILE'"
else
  ssh "$JZ_HOST" "cat '$SRC_FILE'" | ssh "$KAR_HOST" "cat > '$PART_FILE'"
fi

# --- verify size ------------------------------------------------------------
echo "[3/5] Verifying size ..."
DST_SIZE="$(ssh "$KAR_HOST" "stat -c %s '$PART_FILE'")"
if [ "$DST_SIZE" != "$SRC_SIZE" ]; then
  echo "ERROR: size mismatch (src=${SRC_SIZE} dst=${DST_SIZE}); leaving ${PART_FILE} for inspection." >&2
  exit 1
fi
echo "       size OK (${DST_SIZE} bytes)"

# --- verify sha256 ----------------------------------------------------------
if [ "$VERIFY_SHA" = "1" ]; then
  echo "[4/5] Verifying sha256 (reads ~$(( SRC_SIZE / 1024 / 1024 / 1024 )) GiB on each side) ..."
  SRC_SHA="$(ssh "$JZ_HOST" "sha256sum '$SRC_FILE' | cut -d' ' -f1")"
  DST_SHA="$(ssh "$KAR_HOST" "sha256sum '$PART_FILE' | cut -d' ' -f1")"
  if [ "$SRC_SHA" != "$DST_SHA" ]; then
    echo "ERROR: sha256 mismatch; leaving ${PART_FILE} for inspection." >&2
    echo "       src=${SRC_SHA}" >&2
    echo "       dst=${DST_SHA}" >&2
    exit 1
  fi
  echo "       sha256 OK (${SRC_SHA})"
else
  echo "[4/5] Skipping sha256 (VERIFY_SHA=0)"
fi

# --- finalize ---------------------------------------------------------------
echo "[5/5] Renaming .partial -> final ..."
ssh "$KAR_HOST" "mv -f '$PART_FILE' '$DST_FILE'"
echo
echo "Done: ${KAR_HOST}:${DST_FILE}"
