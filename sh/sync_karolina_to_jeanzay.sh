#!/bin/bash
# Sync the datasets used by sh/train_occany_plus_recon_1B_infinite_depth.sh
# from Karolina $SCRATCH/data/ to Jean Zay datasets_preprocess_backup/.
#
# Run this ON KAROLINA. The reverse SSH tunnel (Karolina:2222 -> jean-zay3:22)
# must be open and Karolina's it4i-anhquan key must be authorized for
# uyl37fq@jean-zay. See docs/ssh_tunnel_jz.md.
#
# Usage:
#   ssh karolina
#   bash ~/OccAny/sh/sync_karolina_to_jeanzay.sh           # sync all
#   bash ~/OccAny/sh/sync_karolina_to_jeanzay.sh waymo_processed   # one dir
#   DRY_RUN=1 bash ~/OccAny/sh/sync_karolina_to_jeanzay.sh         # dry run
set -euo pipefail

: "${SCRATCH:?SCRATCH must be set (run on Karolina with conda env active)}"

SRC_ROOT="$SCRATCH/data"
DST_USER="uyl37fq"
DST_HOST="localhost"
DST_PORT="2222"
DST_ROOT="/lustre/fsstor/projects/rech/trg/uyl37fq/datasets_preprocess_backup"

DIRS=(
    waymo_processed
    vkitti_processed
    ddad_processed
    pandaset_processed
    once_processed
    kitti_processed
    occ3d_nuscenes_processed
)

if [ "$#" -gt 0 ]; then
    DIRS=("$@")
fi

RSYNC_OPTS=(-aH --info=progress2,stats2 --delete --partial --inplace
            --exclude='tmp/' --exclude='tmp/**'
            --exclude='*.tmp' --exclude='.tmp/' --exclude='.tmp/**')
if [ "${DRY_RUN:-0}" = "1" ]; then
    RSYNC_OPTS+=(--dry-run)
fi

SSH_CMD="ssh -p $DST_PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

# Verify tunnel reachability before transferring anything.
echo "==> verifying tunnel: $DST_USER@$DST_HOST:$DST_PORT"
$SSH_CMD "$DST_USER@$DST_HOST" "hostname && mkdir -p '$DST_ROOT'"

for d in "${DIRS[@]}"; do
    src="$SRC_ROOT/$d/"
    dst="$DST_USER@$DST_HOST:$DST_ROOT/$d/"
    if [ ! -d "$SRC_ROOT/$d" ]; then
        echo "!! skip: $SRC_ROOT/$d does not exist on Karolina"
        continue
    fi
    echo "==> rsync $src -> $dst"
    rsync "${RSYNC_OPTS[@]}" -e "$SSH_CMD" "$src" "$dst"
done

echo "==> done"
