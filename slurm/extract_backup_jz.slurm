#!/bin/bash

#SBATCH --job-name=extract_backup
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --array=0-9


#SBATCH --output=slurm/output/extract_backup_%j.out
#SBATCH --error=slurm/output/extract_backup_%j.err
#SBATCH --hint=nomultithread

##SBATCH --partition=visu
#SBATCH --partition=archive

#SBATCH -A trg@cpu

#SBATCH --cpus-per-task=10


source ~/.bashrc

module purge

module load pytorch-gpu/py3/2.5.0


cd $TRG_WORK/code/deltatok

python dataset_setup/backup.py --pid $SLURM_ARRAY_TASK_ID --world $SLURM_ARRAY_TASK_COUNT --extract --num_workers 10