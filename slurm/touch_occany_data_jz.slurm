#!/bin/bash

#SBATCH --job-name=touch_occany
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --array=0-9


#SBATCH --output=slurm/output/touch_occany_%A_%a.out
#SBATCH --error=slurm/output/touch_occany_%A_%a.err
#SBATCH --hint=nomultithread

#SBATCH --partition=archive

#SBATCH -A trg@cpu

#SBATCH --cpus-per-task=8

# fsn1 purges files unread for 30 days -> refresh atime AND mtime on the whole working tier.
# Sharded two ways: scene dirs split round-robin across the array, each task touching its
# share with 8 parallel xargs streams. Throttle with --array=0-9%N if the MDS gets hammered.
#
# Submit from a LOGIN shell:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/touch_occany_data_jz.slurm'"

source ~/.bashrc

cd $TRG_WORK/code/deltatok

mkdir -p slurm/output

ROOT=$SCRATCH/occany_dataset

echo "start task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT: $(date)"

# task 0 also covers dataset-root files (seq pkl indexes) that the scene shards miss
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    find "$ROOT" -maxdepth 2 -type f -print0 | xargs -0 -r -P 8 -n 1000 touch
fi

# scene dirs (depth 2 = <dataset>/<scene>), round-robin across the array
find "$ROOT" -mindepth 2 -maxdepth 2 -type d | sort | \
    awk -v pid="$SLURM_ARRAY_TASK_ID" -v world="$SLURM_ARRAY_TASK_COUNT" '(NR - 1) % world == pid' | \
    while IFS= read -r d; do
        find "$d" -print0 | xargs -0 -r -P 8 -n 1000 touch
    done

echo "done task $SLURM_ARRAY_TASK_ID: $(date)"
