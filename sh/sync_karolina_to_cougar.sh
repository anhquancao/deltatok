#!/bin/bash
# Pull the datasets used by sh/train_occany_plus_recon_1B_infinite_depth.sh
# from Karolina $SCRATCH/data/ to cougar's local /home/acao/daf/...
#
# Run this ON COUGAR. Cougar must have an SSH alias `karolina` configured.
#
# Strategy: build ONE global queue of (dataset, scene) work units across all
# requested datasets, then run `PARALLEL` rsyncs concurrently over the whole
# queue with a fast cipher and --whole-file. After the parallel pass each
# dataset gets a single `rsync -d --delete` cleanup over its top level.
#
# Usage:
#   bash sh/sync_karolina_to_cougar.sh                  # all 7 dirs
#   bash sh/sync_karolina_to_cougar.sh waymo_processed  # one dir
#   PARALLEL=8 bash sh/sync_karolina_to_cougar.sh       # 8 streams
#   DRY_RUN=1 bash sh/sync_karolina_to_cougar.sh        # preview
set -euo pipefail

SRC_HOST="karolina"
DST_ROOT="/home/acao/daf/datasets_daf/quan/occany_preprocess_data"
PARALLEL="${PARALLEL:-4}"

DIRS=(
    waymo_processed
    vkitti_processed
    ddad_processed
    pandaset_processed
    once_processed
    kitti_processed
    occ3d_nuscenes_processed
)
[ "$#" -gt 0 ] && DIRS=("$@")

# Karolina sshd only offers aes256-ctr / aes256-gcm. aes256-gcm uses AES-NI
# on modern CPUs and is the fastest of the two.
SSH_OPTS="-c aes256-gcm@openssh.com,aes256-ctr \
          -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
SSH_CMD="ssh $SSH_OPTS"

# -W skips the rolling-checksum delta algorithm (pointless on initial bulk pull).
# No -z: dataset content is already compressed images/depths.
RSYNC_FLAGS="-aHW --info=stats1 --partial --inplace \
--exclude=tmp/ --exclude=tmp/** \
--exclude=*.tmp --exclude=.tmp/ --exclude=.tmp/**"

DRY=""
[ "${DRY_RUN:-0}" = "1" ] && DRY="--dry-run"

mkdir -p "$DST_ROOT"
SRC_ROOT="$($SSH_CMD "$SRC_HOST" 'echo $SCRATCH/data')"
echo "==> source: $SRC_HOST:$SRC_ROOT"
echo "==> dest:   $DST_ROOT"
echo "==> parallel streams: $PARALLEL"

queue_file="$(mktemp)"
trap 'rm -f "$queue_file"' EXIT

# Build a flat queue of "<dataset>\t<entry>" lines for every first-level entry
# (scene dir or top-level file) across all requested datasets.
for d in "${DIRS[@]}"; do
    src_dir="$SRC_ROOT/$d"
    dst_dir="$DST_ROOT/$d"
    mkdir -p "$dst_dir"

    echo "==> enumerating $d on Karolina"
    entries="$($SSH_CMD "$SRC_HOST" \
        "find '$src_dir' -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null || true")"
    if [ -z "$entries" ]; then
        echo "!! $d: empty or missing on source, skipping"
        continue
    fi
    printf '%s\n' "$entries" | sed "s|^|$d\t|" >> "$queue_file"
done

n_total=$(wc -l < "$queue_file")
echo "==> $n_total scenes/files queued across ${#DIRS[@]} datasets"

# Export what each parallel worker needs.
export SRC_HOST SRC_ROOT DST_ROOT SSH_CMD RSYNC_FLAGS DRY

xargs -a "$queue_file" -d '\n' -P "$PARALLEL" -I{} bash -c '
    line="$1"
    d="${line%%	*}"
    entry="${line#*	}"
    # shellcheck disable=SC2086
    rsync $RSYNC_FLAGS $DRY -e "$SSH_CMD" \
        "$SRC_HOST:$SRC_ROOT/$d/$entry" "$DST_ROOT/$d/"
' _ {}

# Top-level cleanup per dataset: drop dest entries that no longer exist on src.
# -d = no recursion, so this only touches the top level of each dataset.
for d in "${DIRS[@]}"; do
    src_dir="$SRC_ROOT/$d"
    dst_dir="$DST_ROOT/$d"
    [ -d "$dst_dir" ] || continue
    echo "==> $d: top-level cleanup (--delete extraneous)"
    # shellcheck disable=SC2086
    rsync -d --delete $DRY -e "$SSH_CMD" \
        "$SRC_HOST:$src_dir/" "$dst_dir/" >/dev/null
done

echo "==> done"
