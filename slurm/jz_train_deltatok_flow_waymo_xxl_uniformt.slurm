#!/bin/bash
#SBATCH --job-name=deltatok_flow_waymo_xxl_uniformt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=zuw@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=slurm/output/jz_train_deltatok_flow_waymo_xxl_uniformt_%j.out
#SBATCH --error=slurm/output/jz_train_deltatok_flow_waymo_xxl_uniformt_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# Default qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_train_deltatok_flow_waymo_xxl_uniformt.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Unique DDP port per job (default 29500 collides with a co-located/leftover job).
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))

# The module's site-packages bundles TensorFlow, which some import drags in;
# silence its startup spam (oneDNN notes, cuFFT/cuDNN/cuBLAS factory errors).
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# UNIFORM-t variant of jz_train_deltatok_flow_waymo_xxl_incontext.slurm: same model, only
# the t schedule changes (train + sampler together). Tests the untested half of VGGT-World's
# escape (docs/journal/deltatok_flow_wall_diagnosis.html): uniform t puts 5.0% of draws
# above t=0.95 vs 0.46%. Judge on LossDepth vs the *_tok ceilings, not MSEToken.
export CONFIG_NAME=train_deltatok_flow_jeanzay_waymo
export RESULTS_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log"
# Distinct name so this run lands in its own subdir (the incontext run keeps its dir).
export RUN_NAME=deltatok_flow_waymo_dtok64_deltaCtxIncontext_uniformTshared_lin20a0_wd05_raymapoff_50k_whitenpos_xxl_dit
# Frozen tokenizer = dtok64 (K=64 delta tokens/camera, non-affine z-norm; deltatok_ckpt override below).
# delta_ctx_incontext: first delta token (frame 0->1) is the clean in-seq context AND
# frame-0 patch grid concatenated in-sequence (clean t=1, no loss), no cross-attn.
# global: one attention over all delta + frame-0 tokens. camera_embed gives per-camera
# identity, shared across delta and frame-0 tokens (context-cam n aligns with delta-cam n).
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera (must match the frozen dtok64 ckpt)
    model.deltatok_ckpt=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_surround_layer12_bsize2_dtok64_cos1e3_1e5_b95_gradskip_noaff/ckpts/current.pth
    model.deltatok.norm_affine=false           # dtok64 tokenizer is non-affine z-norm (declared true in flow cfg)
    # Per-slot (K, C) whitening of the delta tokens (carried over from whitenpos).
    # Produced by STATS_MODE=position slurm/jz_compute_deltatok_latent_stats.slurm;
    # must match the tokenizer above.
    model.whiten_stats=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_surround_layer12_bsize2_dtok64_cos1e3_1e5_b95_gradskip_noaff/ckpts/latent_stats_waymo_pos.pth
    model.cond_mode=delta_ctx_incontext        # clean in-seq context + frame-0 patch grid CONCATENATED in-sequence (no cross-attn)
    model.attn_mode=global                     # single global attention over all delta + frame-0 tokens
    model.vit_use_camera_embed=true            # absolute per-camera identity in the flow ViT (shared by delta + frame-0 tokens)
    training.bsize=8                           # frame-0 concat makes the seq ~4x longer; bsize 16->8 (grad_cum auto-derives 2, effective_bsize 32 unchanged)
    model.loss_mode=v                          # v-MSE = 1/(1-t)^2-weighted x-MSE (upweights near-clean t->1)
    model.vit_size=xxlarge_d20                 # width 1536 == latent C (full-rank in_proj), depth 20 (was 28): trims the depth-costly per-block adaLN
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed (RAE/DiT t-conditioning); ~1.2B total (vs xxl 1.17B, depth-28 dit would be ~1.7B)
    # cache_frozen_encode left at config default (false): per-sample encode cache grows unbounded over the full dataset
    # weight_decay left at config default (0.05): restore regularization for the full-data run
    training.exit_before_time_limit=true       # clean exit near SLURM wall (chain-resume safe)
    training.epoch=50                          # 50-epoch schedule; chain-resume across SLURM walls
    model.t_dist=uniform                       # train t ~ U(0,1): 5.0% of draws above t=0.95 vs 0.46% for logit-normal
    model.t_shared=true                        # one t per sample, all slots — matches sampler_alpha=0
    model.sampler_scheduler_mode=linear        # uniform-in-t grid, matching the train prior
    model.sampler_alpha=0                      # no per-frame stagger (no slot parked at t=1 holding its own x_hat)
    training.eval_num_steps=20                 # at 20 steps clamp(1-t,0.05) never binds: final blend ratio 1.0, no residual noise
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# Fresh start: architecture is identical to incontext so its ckpt WOULD load, but it has
# already adapted to the logit-normal prior — warm-starting would confound the A/B.
# Chained resume jobs of THIS run still need RESUME=1:
#   sbatch --dependency=afterany:<prev_jobid> --export=ALL,RESUME=1 <this script>
# RESUME=1 makes sh/train_deltatok_flow.sh pass --resume (loads ckpts/current.pth).
export RESUME="${RESUME:-0}"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
