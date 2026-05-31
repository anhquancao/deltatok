#!/bin/bash
#SBATCH --job-name=train_occany_plus_recon_1B_sam3_distill_cvpr_sampling
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=lwy@h100
#SBATCH --time=20:00:00
#SBATCH --output=slurm/output/train_occany_plus_recon_1B_sam3_distill_cvpr_sampling_%j.out
#SBATCH --error=slurm/output/train_occany_plus_recon_1B_sam3_distill_cvpr_sampling_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# This job: 4 nodes x 4 GPUs = 16 GPUs, BATCH_SIZE=2 -> 32 samples/step.
# EFFECTIVE_BATCH_SIZE=64 => accum_iter = 64 / (4 * 2 * 4) = 2.
#
# Default qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# CVPR-style view sampling variant: same loss/distill recipe as
# jz_train_occany_plus_recon_1B_infinite_depth_sam3_distill.slurm, only the
# train dataset mix changes (single-camera temporal + pure surround).

mkdir -p slurm/output

source ~/.bashrc

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/OccAny

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# Processed datasets on jean-zay live on the fsn1 working tier (not $SCRATCH/data).
export DATA_ROOT="${DATA_ROOT:-/lustre/fsn1/projects/rech/trg/uyl37fq/occany_data}"
# Logs/ckpts go to the persistent fswork tier ($PROJECT/tb_log_occany/<EXP_NAME>);
# fsn1 purges files untouched for 30 days.
export PROJECT="${PROJECT:-$TRG_WORK}"

export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-4}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-4}"



NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" \
NUM_NODE="$NUM_NODE" \
N_WORKERS="${N_WORKERS:-8}" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
bash sh/train_occany_plus_recon_1B_infinite_depth_sam3_distill_cvpr_sampling.sh
