#!/bin/bash
# rsync from Karolina -> BSC: transfer missing files AND re-transfer files
# whose size differs (catches truncated / partial previous transfers).
# Files that exist on BSC with matching size are left untouched.
#
# Run on Karolina. Requires the `bsc` host alias in ~/.ssh/config.
#
# Usage:
#   bash sh/verify_karolina_to_bsc.sh                  # all datasets
#   bash sh/verify_karolina_to_bsc.sh ddad_processed   # one or more datasets
#   DRY_RUN=1 bash sh/verify_karolina_to_bsc.sh        # preview only
#
# Reading the output:
#   --itemize-changes prints one line per file. With --size-only, files whose
#   sizes match are silent; only missing or size-mismatched files are sent.
#   Directory lines like
#       .d..t...... Scene20-rain/
#   just note that the directory's mtime differs (harmless -- dir mtimes
#   change whenever files inside are added/removed). The --stats block at
#   the end shows the total bytes actually sent.
#
# Note: --size-only does NOT catch silent bit-rot (same size, different
# bytes). If you suspect that, run with rsync --checksum manually.
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/scratch/project/eu-25-92/data}"
DST_ROOT="${DST_ROOT:-bsc:/gpfs/scratch/ehpc793/occany_data}"

DATASETS=(
    waymo_processed
    vkitti_processed
    ddad_processed
    pandaset_processed
    once_processed
    kitti_processed
    occ3d_nuscenes_processed
)
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
fi

RSYNC_OPTS=(-aH -h --stats --itemize-changes --size-only
            --exclude='tmp/' --exclude='.tmp/')
if [ "${DRY_RUN:-0}" = "1" ]; then
    RSYNC_OPTS+=(-n)
fi

for d in "${DATASETS[@]}"; do
    echo "==> $d"
    rsync "${RSYNC_OPTS[@]}" \
        -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3" \
        "$SRC_ROOT/$d/" "$DST_ROOT/$d/"
done
