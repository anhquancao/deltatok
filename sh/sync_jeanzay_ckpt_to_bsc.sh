#!/usr/bin/env bash
# Stream a single DeltaTok checkpoint from Jean Zay -> BSC, routing through the
# LOCAL machine without ever writing to local disk (a pure SSH pipe).
#
# Topology: from here we can reach both `ssh jean-zay` (reverse tunnel via
# Karolina) and `ssh bsc`; the two sites cannot reach each other. Bytes flow:
#     jean-zay (cat)  ->  local ssh pipe  ->  bsc (cat > file)
# so nothing is staged on local disk. Writes to a .partial and only renames to
# the final name after size + (optional) sha256 match on both ends.
#
# Usage:
#   sh/sync_jeanzay_ckpt_to_bsc.sh [ARM] [CKPT]
#     ARM   run dir under deltatok_log on both sides (default: the stage-2 bottleneck arm)
#     CKPT  checkpoint file inside <ARM>/ckpts (default epoch_20.pth)
#
# Env overrides:
#   JZ_HOST=jean-zay  BSC_HOST=bsc          (bsc-transfer routes via transfer1.bsc.es)
#   JZ_LOG_DIR / BSC_LOG_DIR                deltatok_log roots on each side
#   VERIFY_SHA=1      set 0 to skip sha256 (size check is always done)

set -euo pipefail

# --- config -----------------------------------------------------------------
JZ_HOST="${JZ_HOST:-jean-zay}"
BSC_HOST="${BSC_HOST:-bsc}"

ARM="${1:-deltatok_l12_dtok64_tc768_nozn_bneckrec1.0_rawtgt_stage2bneck_ep40}"
CKPT="${2:-epoch_20.pth}"

JZ_LOG_DIR="${JZ_LOG_DIR:-/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log}"
SRC_FILE="${JZ_LOG_DIR}/${ARM}/ckpts/${CKPT}"

# $SCRATCH is only defined in BSC's own login profile, so resolve it there.
BSC_LOG_DIR="${BSC_LOG_DIR:-$(ssh "$BSC_HOST" 'echo $SCRATCH/deltatok_log')}"
DST_DIR="${BSC_LOG_DIR}/${ARM}/ckpts"
DST_FILE="${DST_DIR}/${CKPT}"
PART_FILE="${DST_FILE}.partial"

VERIFY_SHA="${VERIFY_SHA:-1}"
# ----------------------------------------------------------------------------

echo "Source : ${JZ_HOST}:${SRC_FILE}"
echo "Dest   : ${BSC_HOST}:${DST_FILE}"
echo

# --- pre-flight -------------------------------------------------------------
echo "[1/5] Pre-flight checks ..."
SRC_SIZE="$(ssh "$JZ_HOST" "stat -c %s '$SRC_FILE'" 2>/dev/null)" || {
  echo "ERROR: source not found on ${JZ_HOST}: ${SRC_FILE}" >&2; exit 1; }
echo "       source size: ${SRC_SIZE} bytes ($(( SRC_SIZE / 1024 / 1024 )) MiB)"

# Skip if final dest already exists with matching size.
EXIST_SIZE="$(ssh "$BSC_HOST" "stat -c %s '$DST_FILE' 2>/dev/null || echo 0")"
if [ "$EXIST_SIZE" = "$SRC_SIZE" ]; then
  echo "       dest already present with matching size — nothing to do."
  exit 0
fi

ssh "$BSC_HOST" "mkdir -p '$DST_DIR'"
AVAIL_KB="$(ssh "$BSC_HOST" "df -P '$DST_DIR' | awk 'NR==2{print \$4}'")"
echo "       bsc free: $(( AVAIL_KB / 1024 / 1024 )) GiB"
if [ "$(( AVAIL_KB * 1024 ))" -lt "$SRC_SIZE" ]; then
  echo "ERROR: not enough space on ${BSC_HOST}" >&2; exit 1
fi

# --- stream -----------------------------------------------------------------
echo "[2/5] Streaming (jean-zay -> local pipe -> bsc, no local disk) ..."
if command -v pv >/dev/null 2>&1; then
  ssh "$JZ_HOST" "cat '$SRC_FILE'" | pv -s "$SRC_SIZE" | ssh "$BSC_HOST" "cat > '$PART_FILE'"
else
  ssh "$JZ_HOST" "cat '$SRC_FILE'" | ssh "$BSC_HOST" "cat > '$PART_FILE'"
fi

# --- verify size ------------------------------------------------------------
echo "[3/5] Verifying size ..."
DST_SIZE="$(ssh "$BSC_HOST" "stat -c %s '$PART_FILE'")"
if [ "$DST_SIZE" != "$SRC_SIZE" ]; then
  echo "ERROR: size mismatch (src=${SRC_SIZE} dst=${DST_SIZE}); leaving ${PART_FILE} for inspection." >&2
  exit 1
fi
echo "       size OK (${DST_SIZE} bytes)"

# --- verify sha256 ----------------------------------------------------------
if [ "$VERIFY_SHA" = "1" ]; then
  echo "[4/5] Verifying sha256 (reads ~$(( SRC_SIZE / 1024 / 1024 / 1024 )) GiB on each side) ..."
  SRC_SHA="$(ssh "$JZ_HOST" "sha256sum '$SRC_FILE' | cut -d' ' -f1")"
  DST_SHA="$(ssh "$BSC_HOST" "sha256sum '$PART_FILE' | cut -d' ' -f1")"
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
ssh "$BSC_HOST" "mv -f '$PART_FILE' '$DST_FILE'"
echo
echo "Done: ${BSC_HOST}:${DST_FILE}"
