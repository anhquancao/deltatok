#!/bin/bash
# Push InfiniDepth pseudo-label PNGs (+ per-dataset seq pkl files) from Karolina
# to BSC bsc-transfer, preserving the parent directory structure under each
# *_processed dataset root.
#
# Datasets covered match
# sh/train_occany_plus_recon_1B_infinite_depth_sam3_distill.sh
# (5 train datasets + 2 test datasets).
#
# Run this ON KAROLINA. Requires an `bsc-transfer` host alias in
# ~/.ssh/config that resolves to vale352205@... at BSC.
#
# Usage:
#   ssh karolina
#   bash ~/OccAny/sh/sync_karolina_to_bsc.sh                # sync all
#   bash ~/OccAny/sh/sync_karolina_to_bsc.sh waymo_processed pandaset_processed
#   DRY_RUN=1 bash ~/OccAny/sh/sync_karolina_to_bsc.sh      # dry run
set -euo pipefail

SRC_ROOT="/scratch/project/eu-25-92/data"
DST_HOST="bsc"
DST_ROOT="/gpfs/scratch/ehpc793/occany_data"

# dataset_dir : seq_pkl_filename (matches the train script)
DATASETS=(
    "waymo_processed:seq_surround_temporal_sub5_stride9_all.pkl"
    "vkitti_processed:seq_exact_len_sub5_stride9.pkl"
    "ddad_processed:seq_surround_temporal_sub5_stride9_all.pkl"
    "pandaset_processed:seq_surround_temporal_sub5_stride9_all.pkl"
    "once_processed:seq_surround_temporal_sub5_stride9_all.pkl"
    "kitti_processed:seq_exact_len_sub5_stride9.pkl"
    "occ3d_nuscenes_processed:seq_surround_temporal_sub1_stride9_all.pkl"
)

# Optional positional filter: only sync the named dataset dirs.
if [ "$#" -gt 0 ]; then
    FILTER=("$@")
    FILTERED=()
    for entry in "${DATASETS[@]}"; do
        d="${entry%%:*}"
        for want in "${FILTER[@]}"; do
            if [ "$d" = "$want" ]; then
                FILTERED+=("$entry")
                break
            fi
        done
    done
    DATASETS=("${FILTERED[@]}")
fi

: "${CHUNK_SIZE:=20000}"

RSYNC_OPTS=(-aH -h -v
            --progress
            --info=flist2,stats2
            --partial)
if [ "${DRY_RUN:-0}" = "1" ]; then
    RSYNC_OPTS+=(--dry-run)
fi

SSH_CMD="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

echo "==> verifying $DST_HOST reachability"
$SSH_CMD "$DST_HOST" "hostname && mkdir -p '$DST_ROOT'"

for entry in "${DATASETS[@]}"; do
    d="${entry%%:*}"
    pkl="${entry##*:}"
    src="$SRC_ROOT/$d/"
    dst="$DST_HOST:$DST_ROOT/$d/"
    if [ ! -d "$SRC_ROOT/$d" ]; then
        echo "!! skip: $SRC_ROOT/$d does not exist on Karolina"
        continue
    fi
    echo "==> enumerating $d (pseudo PNGs + $pkl)"
    # Build the file list locally with find — avoids rsync's slow remote
    # directory walk that was timing out on bsc-transfer. Filenames in these
    # datasets never contain newlines, so plain -print is safe.
    work_dir=$(mktemp -d)
    trap 'rm -rf "$work_dir"' EXIT
    file_list="$work_dir/all.list"
    find "$src" \
        \( -type d \( -name tmp -o -name .tmp \) -prune \) -o \
        \( -type f \( -name '*.infinidepth.png' -o -name "$pkl" \) -print \) \
        | sed "s|^$src||" > "$file_list"
    count=$(wc -l < "$file_list")
    echo "    -> $count files"
    if [ "$count" -eq 0 ]; then
        rm -rf "$work_dir"
        trap - EXIT
        continue
    fi
    # Chunk the list so each rsync builds a bounded in-memory file list and
    # the bsc-transfer SSH connection sees progress between chunks.
    split -d -a 4 -l "$CHUNK_SIZE" --additional-suffix=.list \
          "$file_list" "$work_dir/chunk-"
    chunks=("$work_dir"/chunk-*.list)
    total=${#chunks[@]}
    i=0
    for chunk in "${chunks[@]}"; do
        i=$((i+1))
        n=$(wc -l < "$chunk")
        echo "==> rsync $d chunk $i/$total ($n files) -> $dst"
        rsync "${RSYNC_OPTS[@]}" -e "$SSH_CMD" \
              --files-from="$chunk" "$src" "$dst"
    done
    rm -rf "$work_dir"
    trap - EXIT
done

echo "==> done"
