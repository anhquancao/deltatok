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
#SBATCH --output=slurm/output/jz_train_deltatok_%j.out
#SBATCH --error=slurm/output/jz_train_deltatok_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_train_deltatok.slurm'"

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
# tokens/camera. _mlp_sigreg variant of slurm/jz_train_deltatok_multitoken_mlp.slurm: keeps the
# MLP bottleneck (capacity fix) and ADDS SIGReg (LeJEPA) on the z bottleneck (anti-collapse /
# optimization fix). Diagnostic (job 160304): tc768 is BOTH optimization-limited (trained 0.065 vs
# PCA@768 0.023) AND linear-capacity-limited; the MLP targets capacity, SIGReg targets the rank-290
# collapse. Job 168363 (first sigreg arm) FAILED TO LEARN: LossTot pinned ~0.115 for 21k steps
# while the plain _mlp arm reached 0.078 by 8.5k, and nuScenes eval drifted UP. Cause was the
# estimator, not the weight: SIGReg saw only S = M*N*K = 128..256 samples of dim 768, where its
# finite-sample floor (~0.0085 at S=128) buries the rank-290 signal (~0.0002) -- the observed
# 0.0088 plateau was 1.04x the floor, i.e. pure estimator bias. _pool fixes it the way the
# reference does (lejepa/univariate/epps_pulley.py): shared slice directions + a differentiable
# cross-rank CF all-reduce, plus pooling over the grad_cum micro-batches -> 8x the samples, and
# num_slices >= 2*Cz now that directions no longer vary per rank.
# WATCH: LossSIGReg should now sit well above its (much lower) floor and actually DESCEND; if it
# is flat again the estimator is still under-resolved. Reference weight is ~200x ours in raw-
# statistic terms (lamb 0.05 * N 4096 vs our 1.0 with *N dropped), so SIGREG_WEIGHT is a knob to
# RAISE, not lower, once the estimator is trustworthy.
# Token count is NUM_DELTA_TOKENS (default 64); sweep K or the SIGReg weight via e.g.
#   sbatch --export=ALL,SIGREG_WEIGHT=0.3 slurm/jz_train_deltatok_multitoken_mlp_sigreg.slurm
# New arch (MLP bottleneck params) -> its own RUN_NAME dir, starts fresh (cannot load the
# old _tc768/_prenorm ckpts). Each K gets its own dir.
export CONFIG_NAME=train_deltatok_jeanzay
export LOG_AND_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export NUM_DELTA_TOKENS=${NUM_DELTA_TOKENS:-64}   # delta tokens/camera; override via --export=ALL,NUM_DELTA_TOKENS=K
# z channel bottleneck: default 768 compresses each delta token 1536->768 (fresh _tc768
# dir). Pass TARGET_CHANNELS=0 to resume/reproduce the uncompressed dtok runs — a plain
# --export=ALL,RESUME=1 relaunch now targets the _tc dir, NOT the old baseline.
export TARGET_CHANNELS=${TARGET_CHANNELS:-768}
export BOTTLENECK_MLP=${BOTTLENECK_MLP:-true}     # nonlinear 2-layer SiLU MLP down/up (vs single Linear); beats the linear-768 84%-var cap
export Z_NORM=${Z_NORM:-true}                     # false drops the z LayerNorm entirely -> SIGReg regularizes a free z (LeJEPA applies it to a bare Linear head, not a normed one)
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-1.0}        # SIGReg anti-collapse weight on z (0=off); unchanged from 168363 so the estimator fix is the only variable
export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-2048}  # 1D projections/step; >= 2*Cz since ranks now share directions (ref: 1000-4096 at Cz=512)
export SIGREG_WARMUP=${SIGREG_WARMUP:-2000}       # linear ramp of the weight over N iters -> full pressure entering the 3-5k breakthrough window
export PEAK_LR=${PEAK_LR:-1e-3}                   # peak lr (cosine-decayed); lower for high-K stability
export GRAD_CLIP=${GRAD_CLIP:-0.1}               # tighter than base 1.0 (upstream DeltaTok uses 1e-2); tames early grads
export GRAD_SKIP_THRESH=${GRAD_SKIP_THRESH:-0.0} # >0 also skips finite grad_norm spikes; set ~3-5x steady-state after a peek
export RUN_SUFFIX=${RUN_SUFFIX:-}                 # appended to RUN_NAME for a distinct dir (e.g. _lr5e4_gradskip)
# _cos1e3_1e5_b95: anti blow-up recipe (cosine peak_lr->1e-5 + AdamW beta2=0.95, eps 1e-8 default).
# _gradskip: trainer now skips non-finite-grad steps + won't overwrite a good ckpt with NaN (dtok32
# blew up to an all-NaN ckpt). For a high-K retrain pass PEAK_LR=5e-4 RUN_SUFFIX=_lr5e4.
# _noaff: z-norm affine off (fixed unit-var/zero-mean latent); new arch, cannot load affine ckpts.
# fresh dirs — old _dtok*/_lr1e4/_eps1e6 dirs hold abandoned ckpts, do not resume them.
TC_TAG=""; [ "${TARGET_CHANNELS}" -gt 0 ] && TC_TAG="_tc${TARGET_CHANNELS}"   # empty at 0 -> RUN_NAME unchanged
MLP_TAG=""; [ "${BOTTLENECK_MLP}" = "true" ] && MLP_TAG="_mlp"                 # nonlinear bottleneck -> distinct dir
# weight in name -> per-weight sweep dir; off spellings keep the name == the plain _mlp arm.
# _pool marks the fixed estimator -> fresh dir, must NOT resume 168363's stuck ckpt.
case "${SIGREG_WEIGHT}" in 0|0.0|0.00|"") SIGREG_TAG="" ;; *) SIGREG_TAG="_sigreg${SIGREG_WEIGHT}_pool" ;; esac
ZN_TAG=""; [ "${Z_NORM}" = "false" ] && ZN_TAG="_nozn"                        # no z-norm -> distinct arch, fresh dir
export RUN_NAME=deltatok_surround_layer12_bsize2_dtok${NUM_DELTA_TOKENS}${TC_TAG}_cos1e3_1e5_b95_gradskip_noaff_prenorm${MLP_TAG}${SIGREG_TAG}${ZN_TAG}${RUN_SUFFIX}
export RESUME=${RESUME:-0}   # override with --export=ALL,RESUME=1 to resume current.pth
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=${NUM_DELTA_TOKENS}   # K delta tokens/camera (K=64 ~12:1 compression at full res)
    model.deltatok.target_channels=${TARGET_CHANNELS}     # z channel bottleneck (0 = off)
    training.lr=${PEAK_LR}                      # peak lr override (default 1e-3; lower for high-K stability)
    model.deltatok.use_camera_rope=false       # no-camera-rope layer-12 run
    model.deltatok.norm_affine=false           # final z-norm stays non-affine -> unit-var/zero-mean latent for flow matching
    model.deltatok.z_norm=${Z_NORM}            # false: no z-norm at all; decoder + SIGReg both see raw z_proj_down (see Z_NORM)
    model.deltatok.bottleneck_pre_norm=true    # affine LN before z_proj_down: kills the ~60x post-bottleneck grad amplification
    model.deltatok.bottleneck_mlp=${BOTTLENECK_MLP}  # nonlinear MLP down/up bottleneck (see BOTTLENECK_MLP)
    training.sigreg_weight=${SIGREG_WEIGHT}       # SIGReg anti-collapse loss weight on z (0 = off)
    training.sigreg_num_slices=${SIGREG_NUM_SLICES}  # sliced-Gaussian test directions/step
    training.sigreg_warmup=${SIGREG_WARMUP}       # linear weight ramp (iters)
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
