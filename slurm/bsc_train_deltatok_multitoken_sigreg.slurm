#!/bin/bash
#SBATCH --job-name=deltatok_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=48:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/bsc_train_deltatok_multitoken_sigreg_%j.out
#SBATCH --error=slurm/output/bsc_train_deltatok_multitoken_sigreg_%j.err

# BSC acc partition: 4 H100/node, 80 cores -> 20 cpus/task.
# acc_ehpc caps walltime at 48h; acc_debug is 2h for smoke runs.
#
#   ssh bsc "cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/bsc_train_deltatok_multitoken_sigreg.slurm"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# DeltaTok tokenizer training on BSC (no-camera-rope layer-12 run), K delta
# tokens/camera. BSC port of slurm/jz_train_deltatok_multitoken_mlp_sigreg.slurm with the
# channel bottleneck REMOVED: TARGET_CHANNELS=1536 == hidden_size, so deltatok_trainer.py:150
# ("if self.z_dim != cfg.hidden_size") builds NO z_proj_down/up and no pre-bottleneck LN at all.
# z stays at the full 1536 dims -- K is then the only compression axis. Consequently
# BOTTLENECK_MLP and bottleneck_pre_norm are inert here; BOTTLENECK_MLP defaults to false so the
# name and the arch agree.
# WHY this arm: the measured deficit on the tc768 arms was the 2:1 Cz ratio, not utilisation --
# raw participation ratio was 173.1/768 for tc768 vs 332.2/1536 for uncompressed dtok64, i.e. BOTH
# ~22% of their channel budget. Dropping the bottleneck while KEEPING SIGReg tests whether SIGReg's
# anti-collapse pressure buys real capacity once the 768-channel cap is gone.
# SIGReg (LeJEPA) is retained with the fixed _pool estimator (shared slice directions +
# differentiable cross-rank CF all-reduce + pooling over grad_cum micro-batches). Its finite-sample
# floor is what sank the first sigreg arm (job 168363), so:
# WATCH: LossSIGReg should sit well above its floor and actually DESCEND; if it is flat the
# estimator is under-resolved. Reference weight is ~200x ours in raw-statistic terms, so
# SIGREG_WEIGHT is a knob to RAISE, not lower, once the estimator is trustworthy.
# Token count is NUM_DELTA_TOKENS (default 64); sweep K or the SIGReg weight via e.g.
#   sbatch --export=ALL,SIGREG_WEIGHT=0.3 slurm/bsc_train_deltatok_multitoken_sigreg.slurm
export CONFIG_NAME=train_deltatok_bsc
export LOG_AND_CKPT_DIR="$SCRATCH/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export NUM_DELTA_TOKENS=${NUM_DELTA_TOKENS:-64}   # delta tokens/camera; override via --export=ALL,NUM_DELTA_TOKENS=K
# z channel bottleneck: 1536 == hidden_size -> bottleneck absent (see header). Set a smaller
# value (e.g. 768) to re-enable channel compression; 0 also means "off" but tags the dir differently.
export TARGET_CHANNELS=${TARGET_CHANNELS:-1536}
export BOTTLENECK_MLP=${BOTTLENECK_MLP:-false}    # inert at TARGET_CHANNELS=1536 (no bottleneck to make nonlinear)
export Z_NORM=${Z_NORM:-true}                     # false drops the z LayerNorm entirely -> SIGReg regularizes a free z
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-1.0}        # SIGReg anti-collapse weight on z (0=off)
export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-3072}  # 1D projections/step; >= 2*Cz and Cz is now 1536, not 768
export SIGREG_WARMUP=${SIGREG_WARMUP:-2000}       # linear ramp of the weight over N iters
# Rolling FIFO sample pool (GLOBAL rows) the estimator runs over every micro-batch. The
# finite-sample floor is what sank job 168363, and Cz=1536 needs ~2x the samples of 768.
# 8192 ~ S/C 5-6 here; kept modest so queued rows stay only a few steps stale. Pressure is
# rescaled by pool/live, so this does not double as a weight knob.
export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-8192}
export PEAK_LR=${PEAK_LR:-1e-3}                   # peak lr (cosine-decayed); lower for high-K stability
export GRAD_CLIP=${GRAD_CLIP:-0.1}               # tighter than base 1.0 (upstream DeltaTok uses 1e-2); tames early grads
export GRAD_SKIP_THRESH=${GRAD_SKIP_THRESH:-0.0} # >0 also skips finite grad_norm spikes; set ~3-5x steady-state after a peek
export RUN_SUFFIX=${RUN_SUFFIX:-}                 # appended to RUN_NAME for a distinct dir (e.g. _lr5e4_gradskip)
# _cos1e3_1e5_b95: anti blow-up recipe (cosine peak_lr->1e-5 + AdamW beta2=0.95, eps 1e-8 default).
# _gradskip: trainer skips non-finite-grad steps + won't overwrite a good ckpt with NaN.
# _noaff: z-norm affine off (fixed unit-var/zero-mean latent); new arch, cannot load affine ckpts.
TC_TAG=""; [ "${TARGET_CHANNELS}" -gt 0 ] && TC_TAG="_tc${TARGET_CHANNELS}"   # _tc1536 marks the uncompressed-z arm
MLP_TAG=""; [ "${BOTTLENECK_MLP}" = "true" ] && MLP_TAG="_mlp"                 # nonlinear bottleneck -> distinct dir
# weight + slices + pool size all in the name: each changes the estimator or the pressure,
# so two arms differing in any of them must not share a ckpt dir.
case "${SIGREG_WEIGHT}" in 0|0.0|0.00|"") SIGREG_TAG="" ;; *) SIGREG_TAG="_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}" ;; esac
ZN_TAG=""; [ "${Z_NORM}" = "false" ] && ZN_TAG="_nozn"                        # no z-norm -> distinct arch, fresh dir
export RUN_NAME=deltatok_surround_layer12_bsize2_dtok${NUM_DELTA_TOKENS}${TC_TAG}_cos1e3_1e5_b95_gradskip_noaff_prenorm${MLP_TAG}${SIGREG_TAG}${ZN_TAG}${RUN_SUFFIX}
export RESUME=${RESUME:-0}   # override with --export=ALL,RESUME=1 to resume current.pth
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=${NUM_DELTA_TOKENS}   # K delta tokens/camera (K=64 ~12:1 compression at full res)
    model.deltatok.target_channels=${TARGET_CHANNELS}     # 1536 == hidden_size -> no channel bottleneck built
    training.lr=${PEAK_LR}                      # peak lr override (default 1e-3; lower for high-K stability)
    model.deltatok.use_camera_rope=false       # no-camera-rope layer-12 run
    model.deltatok.norm_affine=false           # final z-norm stays non-affine -> unit-var/zero-mean latent for flow matching
    model.deltatok.z_norm=${Z_NORM}            # false: no z-norm at all; decoder + SIGReg both see raw z (see Z_NORM)
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
