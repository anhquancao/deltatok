#!/bin/bash
#SBATCH --job-name=dflow_waymo_randcam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc880
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=48:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/dflow_waymo_randcam_bsc_%j.out
#SBATCH --error=slurm/output/dflow_waymo_randcam_bsc_%j.err

# Random-camera variant of the front-cam-only waymo xxl sigreg arm: each train
# sample draws 1 of 5 Waymo cameras uniformly; val stays on cam 0.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_sigreg_randcam_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_flow_waymo_randcam_bsc
export RESULTS_DIR="$SCRATCH/deltatok_flow_log"
export RUN_NAME=dflow_waymo_xxl_dit_randcam
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc768_nozn_bneckrec1.0_rawtgt_stage2bneck_ep40/ckpts/epoch_20.pth"
    model.deltatok.target_channels=768
    model.deltatok.z_norm=false
    model.deltatok.norm_affine=false
    model.deltatok.use_camera_rope=false
    model.cond_mode=delta_ctx_cross
    model.attn_mode=global
    model.vit_use_camera_embed=true
    model.loss_mode=v
    model.vit_size=xxlarge_d20
    model.vit_dit_adaln=true
    training.exit_before_time_limit=true
    training.epoch=100
    training.max_iter=200000
    training.lr_schedule=constant
    model.t_dist=uniform
    model.t_shared=true
    model.sampler_scheduler_mode=linear
    model.sampler_alpha=0
    training.eval_num_steps=20
    training.val_bsize=4
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
