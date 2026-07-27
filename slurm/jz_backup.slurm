#!/bin/bash

#SBATCH --job-name=backup_occany
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --array=0-9


#SBATCH --output=slurm/output/backup_occany_%A_%a.out
#SBATCH --error=slurm/output/backup_occany_%A_%a.err
#SBATCH --hint=nomultithread

##SBATCH --partition=visu
#SBATCH --partition=archive

#SBATCH -A trg@cpu

#SBATCH --cpus-per-task=10

# Tar $SCRATCH/occany_dataset scene dirs into the fsstor backup tier (one archive per
# scene, uncompressed). Reverse of jz_extract_backup.slurm.
#
# Submit from a LOGIN shell:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_backup.slurm'"

source ~/.bashrc

module purge

module load pytorch-gpu/py3/2.5.0


cd $TRG_WORK/code/deltatok

mkdir -p slurm/output

# fresh dir: the 2025 archive set is stale, and argoverse2/kitti_pair/nuscenes live only there
export BACKUP_DST_NAME=${BACKUP_DST_NAME:-datasets_preprocess_backup_2026-07}

python dataset_setup/backup.py --pid $SLURM_ARRAY_TASK_ID --world $SLURM_ARRAY_TASK_COUNT --num_workers 10 --dst_name $BACKUP_DST_NAME
