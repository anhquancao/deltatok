#!/bin/bash

#SBATCH --job-name=extract_ddad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00


#SBATCH --output=slurm/output/extract_ddad_%j.out
#SBATCH --error=slurm/output/extract_ddad_%j.err


#SBATCH -A eu-25-92
#SBATCH --partition=qgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16



source ~/.bashrc

module purge

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PYTHONUSERBASE=/mnt/proj1/eu-25-92/envs/dgp

python dataset_setup/ddad/preprocess.py \
    --ddad_root /mnt/proj1/eu-25-92/data/DDAD \
    --preprocessed_root $SCRATCH/data/ddad_processed --n_workers 16