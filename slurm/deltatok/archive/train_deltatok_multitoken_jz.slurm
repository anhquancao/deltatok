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
#SBATCH --output=slurm/output/train_deltatok_jz_%j.out
#SBATCH --error=slurm/output/train_deltatok_jz_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/deltatok/train_deltatok_multitoken_jz.slurm'"

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

# DeltaTok tokenizer training on Jean Zay (no-camera-rope layer-12 run), K delta
# tokens/camera variant of slurm/deltatok/archive/train_deltatok_jz.slurm (1-token baseline preserved).
# Token count is NUM_DELTA_TOKENS (default 64); sweep via e.g.
#   sbatch --export=ALL,NUM_DELTA_TOKENS=16 slurm/deltatok/train_deltatok_multitoken_jz.slurm
# Each K gets its own RUN_NAME dir and starts fresh (new arch, no baseline resume).
export CONFIG_NAME=train_deltatok_jeanzay
export LOG_AND_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export NUM_DELTA_TOKENS=${NUM_DELTA_TOKENS:-64}   # delta tokens/camera; override via --export=ALL,NUM_DELTA_TOKENS=K
# z channel bottleneck: default 768 compresses each delta token 1536->768 (fresh _tc768
# dir). Pass TARGET_CHANNELS=0 to resume/reproduce the uncompressed dtok runs — a plain
# --export=ALL,RESUME=1 relaunch now targets the _tc dir, NOT the old baseline.
export TARGET_CHANNELS=${TARGET_CHANNELS:-768}
export PEAK_LR=${PEAK_LR:-1e-3}                   # peak lr (cosine-decayed); lower for high-K stability
export RUN_SUFFIX=${RUN_SUFFIX:-}                 # appended to RUN_NAME for a distinct dir (e.g. _lr5e4_gradskip)
# _cos1e3_1e5_b95: anti blow-up recipe (cosine peak_lr->1e-5 + AdamW beta2=0.95, eps 1e-8 default).
# _gradskip: trainer now skips non-finite-grad steps + won't overwrite a good ckpt with NaN (dtok32
# blew up to an all-NaN ckpt). For a high-K retrain pass PEAK_LR=5e-4 RUN_SUFFIX=_lr5e4.
# _noaff: z-norm affine off (fixed unit-var/zero-mean latent); new arch, cannot load affine ckpts.
# fresh dirs — old _dtok*/_lr1e4/_eps1e6 dirs hold abandoned ckpts, do not resume them.
TC_TAG=""; [ "${TARGET_CHANNELS}" -gt 0 ] && TC_TAG="_tc${TARGET_CHANNELS}"   # empty at 0 -> RUN_NAME unchanged
export RUN_NAME=deltatok_surround_layer12_bsize2_dtok${NUM_DELTA_TOKENS}${TC_TAG}_cos1e3_1e5_b95_gradskip_noaff${RUN_SUFFIX}
export RESUME=${RESUME:-0}   # override with --export=ALL,RESUME=1 to resume current.pth
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=${NUM_DELTA_TOKENS}   # K delta tokens/camera (K=64 ~12:1 compression at full res)
    model.deltatok.target_channels=${TARGET_CHANNELS}     # z channel bottleneck (0 = off)
    training.lr=${PEAK_LR}                      # peak lr override (default 1e-3; lower for high-K stability)
    model.deltatok.use_camera_rope=false       # no-camera-rope layer-12 run
    model.deltatok.norm_affine=false           # non-learnable z-norm -> fixed unit-var/zero-mean latent for flow matching
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
