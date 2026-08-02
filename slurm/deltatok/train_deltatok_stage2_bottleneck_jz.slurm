#!/bin/bash
#SBATCH --job-name=deltatok_jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
##SBATCH --qos=qos_gpu_h100-t3
##SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/train_deltatok_stage2_bottleneck_jz_%j.out
#SBATCH --error=slurm/output/train_deltatok_stage2_bottleneck_jz_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/deltatok/train_deltatok_stage2_bottleneck_jz.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The module's site-packages bundles TensorFlow, which some import drags in;
# silence its startup spam (oneDNN notes, cuFFT/cuDNN/cuBLAS factory errors).
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# STAGE 2 over the frozen tc1536 arm: load its epoch_40 weights, add the 1536->768 bottleneck,
# and train ONLY z_proj_down/up + pre_bottleneck_norm. The joint tc768 arms plateau ~7x above the
# uncompressed ceiling although a plain PCA-768 reaches 0.023 -- an optimization failure, so the
# projection gets a stationary code plus a direct round-trip loss instead of decoder grads alone.
# current.pth overrides INIT_CKPT (fit() prefers it), which is what chained jobs want.
# EXPECTED in the log: "[warn] ckpt has no z bottleneck projections" + trainable ~2.4M of 460M.
#   sbatch --export=ALL,BNECK_RECON_WEIGHT=0.3 slurm/deltatok/train_deltatok_stage2_bottleneck_jz.slurm
export CONFIG_NAME=train_deltatok_jeanzay
export LOG_AND_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
# Stage-1 source: epoch_40.pth not current.pth -- immutable if that arm ever resumes.
export INIT_CKPT="$LOG_AND_CKPT_DIR/deltatok_surround_layer12_bsize2_dtok64_tc1536_cos1e3_1e5_b95_gradskip_noaff_prenorm_sigreg0.05_ns3072_pool8192_nozn/ckpts/epoch_40.pth"
# The one tuned knob: weight of the round-trip loss on z_proj_up(z_proj_down(z)) vs the 1536-d
# code. Starts ~40x the decoder recon (log-cosh of a unit-var residual), so it leads early by
# design; lower it to rebalance. In the run name, so arms never share a dir.
export BNECK_RECON_WEIGHT=${BNECK_RECON_WEIGHT:-1.0}
export RUN_NAME=deltatok_l12_dtok64_tc768_nozn_bneckrec${BNECK_RECON_WEIGHT}_rawtgt_stage2bneck_ep40
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera; must match stage 1
    model.deltatok.target_channels=768         # builds z_proj_down/up + pre_bottleneck_norm
    model.deltatok.bottleneck_mlp=false        # linear down/up; true swaps in the 2-layer SiLU MLP
    training.freeze_except_bottleneck=true     # stage 2: only those three modules train
    training.bottleneck_recon_weight=${BNECK_RECON_WEIGHT}  # see BNECK_RECON_WEIGHT
    training.sigreg_weight=0                   # OFF here: isolate freeze + round-trip loss first
    # 1125 iters/epoch on JZ. max_iter is both the hard stop and the LR-cosine horizon, and
    # warm_up 5000 would burn a quarter of a 20-epoch run.
    training.epoch=20
    training.max_iter=22500
    training.warm_up=500
    training.save_ckpt_every_n_epochs=5        # snapshots at 5,10,15,20
    training.lr=1e-3                           # cosine-decayed to 1e-5; the head is small, 1e-3 is safe
    model.deltatok.use_camera_rope=false       # no-camera-rope layer-12 run, as in stage 1
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for stage-1 parity
    model.deltatok.z_norm=false                # no z-norm; decoder sees raw z, as in stage 1
    training.bsize=4                           # 4 GPU x 4 x pairs 1 -> grad_cum 1 at effective_bsize 16;
                                               # same math as bsize 2, one forward instead of two (~29/81 GB)
    training.grad_clip=0.1                     # tighter than base 1.0 (upstream DeltaTok uses 1e-2)
    training.exit_before_time_limit=true       # stop after a ckpt when SLURM wall is near (chain-resume safe)
    # feature_loss_weight omitted -> base default 0.0 (no downstream DA3 feature loss)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
