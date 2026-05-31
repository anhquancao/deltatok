#!/bin/bash

#SBATCH --job-name=eval_traj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=qgpu


#SBATCH --output=slurm/output/eval_traj_%A_%a.out
#SBATCH --error=slurm/output/eval_traj_%A_%a.err

#SBATCH --array=4

#SBATCH -A eu-25-92

source ~/.bashrc

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany


if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon.pth \
        bash sh/infer_occany_nuscenes_traj.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon_1B.pth \
        bash sh/infer_occany_nuscenes_traj.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon_1B_lidar_scaled_depth_sup.pth \
        bash sh/infer_occany_nuscenes_traj.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    OCCANY_PLUS_RECON_CKPT="$PROJECT/tb_log_occany/occany_plus_recon_1B_infinite_depth_sam3_distill_memory_budget/checkpoint-last.pth" \
        OUTPUT_DIR=./outputs/occany_plus_recon_1B_infinite_depth_sam3_distill_memory_budget_nuscenes_traj \
        bash sh/infer_occany_nuscenes_traj.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon_1B_infinite_depth_sam3_distill_memory_budget_pseudo_depth_only_e20.path \
        OUTPUT_DIR=./outputs/occany_plus_recon_1B_infinite_depth_sam3_distill_memory_budget_pseudo_depth_only_e20_nuscenes_traj \
        bash sh/infer_occany_nuscenes_traj.sh
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi
