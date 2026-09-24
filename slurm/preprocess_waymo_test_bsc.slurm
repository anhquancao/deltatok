#!/bin/bash
#SBATCH --job-name=preprocess_waymo_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --output=slurm/output/preprocess_waymo_test_bsc_%j.out
#SBATCH --error=slurm/output/preprocess_waymo_test_bsc_%j.err

# Waymo val tfrecords -> 1024 px npz. CPU work; ehpc1001 has no gpp QoS.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/preprocess_waymo_test_bsc.slurm'"
# Then: DATA_ROOT=/gpfs/scratch/ehpc1001/quan/data python dataset_setup/make_surround_temporal_seq.py --dataset waymo_test

source ~/.bashrc

cd /gpfs/projects/ehpc1001/code/deltatok

source env_waymo_bsc.sh

# One shard per node (acc_debug: 1 job/user). 40 workers: each holds a decoded segment (~7 GB).
# srun does not inherit --cpus-per-task from sbatch.
srun --cpus-per-task=80 bash -c 'python dataset_setup/waymo/preprocess_waymo.py \
    --waymo_dir /gpfs/scratch/ehpc1001/quan/data/waymo_val_raw \
    --output_dir /gpfs/scratch/ehpc1001/quan/data/waymo_test_processed \
    --workers 40 --pid "$SLURM_PROCID" --world "$SLURM_NTASKS"'
