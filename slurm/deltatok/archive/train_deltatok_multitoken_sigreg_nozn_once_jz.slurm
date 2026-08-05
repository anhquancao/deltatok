#!/bin/bash
#SBATCH --job-name=deltatok_once_jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
##SBATCH --qos=qos_gpu_h100-dev
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/train_deltatok_multitoken_sigreg_nozn_once_jz_%j.out
#SBATCH --error=slurm/output/train_deltatok_multitoken_sigreg_nozn_once_jz_%j.err

# JZ gpu_p6: 4 GPUs/node, 96 cores -> 24 cpus/task. t3 caps at 20h; -t4 gives 100h, fewer GPUs.
# Submit from a LOGIN shell so `module` (login-only) reaches the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/deltatok/train_deltatok_multitoken_sigreg_nozn_once_jz.slurm'"
#
# DATA swap of train_deltatok_multitoken_sigreg_nozn_jz: ONCE alone (843k seqs) instead of
# the 5-dataset mix at a single (518,266), no time flips, 16 held-out scenes for the flow arm.
# Same 18000 items -> same schedule.
# Monocular cam03; train and eval share the 6cam pkl. NOT comparable to the mix arm.
#
# DeltaTok tokenizer, layer-12, no camera rope. tc1536 == hidden_size -> no channel bottleneck,
# and z_norm=false -> no z LayerNorm: SIGReg alone shapes a raw 1536-d z. K is the only compression.
# SIGReg matches the mix arm (w=0.05, pool=8192): pool=32768 is not weight-neutral -- it jumps
# scale=n_pooled/n_live 11.7 -> 43.7 and delays feedback 4x, and Eval never left 0.126 (2026-07-31).
# WATCH: LossSIGReg must descend past iter 2000.
#   sbatch --export=ALL,SIGREG_WEIGHT=0.05 slurm/deltatok/train_deltatok_multitoken_sigreg_nozn_once_jz.slurm

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The module's site-packages bundles TensorFlow; silence its startup spam.
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_once_jeanzay
export LOG_AND_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"

# The only three knobs; each changes the estimator or the pressure, so all three are in RUN_NAME.
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.05}          # LeJEPA lamb; same pressure as the mix arm
export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-3072}  # 2*Cz; measured sufficient (signal is K-invariant)
export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-8192}  # GLOBAL rows; rescaled by pool/live

# Only the architecture + the three swept knobs; the rest is fixed by this script.
export RUN_NAME=deltatok_once_l12_dtok64_tc1536_nozn_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_cam03

EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera (~12:1 compression at full res)
    model.deltatok.target_channels=1536        # == hidden_size -> no z_proj_down/up built
    model.deltatok.z_norm=false                # decoder + SIGReg both see raw z
    model.deltatok.use_camera_rope=false       # layer-12 no-rope run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for name parity
    model.deltatok.bottleneck_mlp=false        # inert at tc1536 (no bottleneck to make nonlinear)
    training.lr=1e-3                           # peak lr, cosine-decayed to 1e-5
    training.grad_clip=0.1                     # tighter than base 1.0; tames early grads
    training.grad_skip_thresh=0.0               # >0 also skips finite grad_norm spikes
    training.sigreg_weight=${SIGREG_WEIGHT}
    training.sigreg_num_slices=${SIGREG_NUM_SLICES}
    training.sigreg_pool_samples=${SIGREG_POOL_SAMPLES}
    training.sigreg_warmup=2000                 # linear weight ramp (iters)
    training.exit_before_time_limit=true        # stop after a ckpt near the wall (chain-resume safe)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
