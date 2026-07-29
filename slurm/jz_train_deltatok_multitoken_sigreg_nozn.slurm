#!/bin/bash
#SBATCH --job-name=deltatok_jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=zuw@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
##SBATCH --qos=qos_gpu_h100-dev
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/jz_train_deltatok_multitoken_sigreg_nozn_%j.out
#SBATCH --error=slurm/output/jz_train_deltatok_multitoken_sigreg_nozn_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_train_deltatok_multitoken_sigreg_nozn.slurm'"

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

# JZ port of slurm/bsc_train_deltatok_multitoken_sigreg.slurm with Z_NORM=false.
# No channel bottleneck (tc1536 == hidden_size -> no z_proj_down/up, so BOTTLENECK_MLP and
# bottleneck_pre_norm are inert) and now no z LayerNorm either: SIGReg shapes a raw 1536-d z
# instead of one pinned to |z|=sqrt(Cz), as in LeJEPA. K is the only compression axis.
# WATCH: LossSIGReg should DESCEND; flat = under-resolved estimator. ZTail is free without
# z-norm, so a growing z scale lets the MSE term dominate.
#   sbatch --export=ALL,SIGREG_WEIGHT=0.3 slurm/jz_train_deltatok_multitoken_sigreg_nozn.slurm
export CONFIG_NAME=train_deltatok_jeanzay
export LOG_AND_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export NUM_DELTA_TOKENS=${NUM_DELTA_TOKENS:-64}   # delta tokens/camera; override via --export=ALL,NUM_DELTA_TOKENS=K
# 1536 == hidden_size -> no bottleneck (see header); set e.g. 768 to re-enable compression.
export TARGET_CHANNELS=${TARGET_CHANNELS:-1536}
export BOTTLENECK_MLP=${BOTTLENECK_MLP:-false}    # inert at TARGET_CHANNELS=1536 (no bottleneck to make nonlinear)
export Z_NORM=${Z_NORM:-false}                    # this arm: no z LayerNorm at all (true restores the BSC arm)
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.05}       # SIGReg anti-collapse weight on z, LeJEPA's lamb (0=off)
export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-3072}  # 1D projections/step; >= 2*Cz, and Cz=1536 here
export SIGREG_WARMUP=${SIGREG_WARMUP:-2000}       # linear ramp of the weight over N iters
# Rolling FIFO pool (GLOBAL rows) the estimator fires over every micro-batch; clears the
# finite-sample floor at Cz=1536. Rescaled by pool/live, so not a weight knob.
export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-8192}
export PEAK_LR=${PEAK_LR:-1e-3}                   # peak lr (cosine-decayed); lower for high-K stability
export GRAD_CLIP=${GRAD_CLIP:-0.1}               # tighter than base 1.0 (upstream DeltaTok uses 1e-2); tames early grads
export GRAD_SKIP_THRESH=${GRAD_SKIP_THRESH:-0.0} # >0 also skips finite grad_norm spikes; set ~3-5x steady-state after a peek
export RUN_SUFFIX=${RUN_SUFFIX:-}                 # appended to RUN_NAME for a distinct dir (e.g. _lr5e4_gradskip)
# _cos1e3_1e5_b95: anti blow-up recipe (cosine peak_lr->1e-5 + AdamW beta2=0.95, eps 1e-8 default).
# _gradskip: trainer skips non-finite-grad steps + won't overwrite a good ckpt with NaN.
# _noaff/_prenorm: kept for name parity with the BSC arm; both inert here (no z-norm, no bottleneck).
TC_TAG=""; [ "${TARGET_CHANNELS}" -gt 0 ] && TC_TAG="_tc${TARGET_CHANNELS}"   # _tc1536 marks the uncompressed-z arm
MLP_TAG=""; [ "${BOTTLENECK_MLP}" = "true" ] && MLP_TAG="_mlp"                 # nonlinear bottleneck -> distinct dir
# weight + slices + pool all in the name: each changes the pressure, so arms never share a dir.
case "${SIGREG_WEIGHT}" in 0|0.0|0.00|"") SIGREG_TAG="" ;; *) SIGREG_TAG="_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}" ;; esac
ZN_TAG=""; [ "${Z_NORM}" = "false" ] && ZN_TAG="_nozn"                        # no z-norm -> distinct arch, fresh dir
export RUN_NAME=deltatok_surround_layer12_bsize2_dtok${NUM_DELTA_TOKENS}${TC_TAG}_cos1e3_1e5_b95_gradskip_noaff_prenorm${MLP_TAG}${SIGREG_TAG}${ZN_TAG}${RUN_SUFFIX}
export RESUME=${RESUME:-0}   # override with --export=ALL,RESUME=1 to resume current.pth
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=${NUM_DELTA_TOKENS}   # K delta tokens/camera (K=64 ~12:1 compression at full res)
    model.deltatok.target_channels=${TARGET_CHANNELS}     # 1536 == hidden_size -> no channel bottleneck built
    training.lr=${PEAK_LR}                      # peak lr override (default 1e-3; lower for high-K stability)
    model.deltatok.use_camera_rope=false       # no-camera-rope layer-12 run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept so a Z_NORM=true override stays non-affine
    model.deltatok.z_norm=${Z_NORM}            # false here: no z-norm; decoder + SIGReg see raw z (see Z_NORM)
    model.deltatok.bottleneck_pre_norm=true    # inert at tc1536; kept so a TARGET_CHANNELS override still gets the fix
    model.deltatok.bottleneck_mlp=${BOTTLENECK_MLP}  # inert at tc1536 (see BOTTLENECK_MLP)
    training.sigreg_weight=${SIGREG_WEIGHT}       # SIGReg anti-collapse loss weight on z (0 = off)
    training.sigreg_num_slices=${SIGREG_NUM_SLICES}  # sliced-Gaussian test directions/step
    training.sigreg_warmup=${SIGREG_WARMUP}       # linear weight ramp (iters)
    training.sigreg_pool_samples=${SIGREG_POOL_SAMPLES}  # rolling FIFO pool size (global rows)
    training.grad_clip=${GRAD_CLIP}            # tighter clip than base 1.0 (see GRAD_CLIP)
    training.grad_skip_thresh=${GRAD_SKIP_THRESH}  # finite grad_norm spike guard (0 = off)
    training.exit_before_time_limit=true       # stop after a ckpt when SLURM wall is near (chain-resume safe)
    # feature_loss_weight omitted -> base default 0.0 (no downstream DA3 feature loss)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
