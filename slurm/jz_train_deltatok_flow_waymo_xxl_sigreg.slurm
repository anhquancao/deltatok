#!/bin/bash
#SBATCH --job-name=deltatok_flow_waymo_xxl_sigreg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=zuw@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=slurm/output/jz_train_deltatok_flow_waymo_xxl_sigreg_%j.out
#SBATCH --error=slurm/output/jz_train_deltatok_flow_waymo_xxl_sigreg_%j.err

# Jean Zay H100 partition (gpu_p6): 4 GPUs/node, 96 cores -> 24 cpus/task.
# Default qos_gpu_h100-t3 caps walltime at 20h; for longer use
#   #SBATCH --qos=qos_gpu_h100-t4   (max 100h, fewer GPUs).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_train_deltatok_flow_waymo_xxl_sigreg.slurm'"

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

# TOKENIZER swap of jz_train_deltatok_flow_waymo_xxl_uniformt_cross.slurm: the flow recipe is
# byte-identical, only the frozen tokenizer changes — uncompressed dtok64 (Cz=1536) -> the
# tc768 SIGReg arm (Cz=768). Tests whether SIGReg's spread buys downstream flow quality:
# measured at ep40, sigreg0.05 uses 173/768 dims (participation ratio) vs 17.6 for its
# no-sigreg twin, and lands z at N(0,I) scale (row mean-square 1.008, max channel mean 0.040).
# See docs/reference/deltatok_z_spread_analysis.html. Judge on LossDepth against the
# uniformt_cross run, which differs ONLY in the tokenizer.
export CONFIG_NAME=train_deltatok_flow_jeanzay_waymo
export RESULTS_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log"
# Tokenizer arm. The no-sigreg twin (..._prenorm_nozn) has the IDENTICAL architecture, so it
# runs through this same script for a paired A/B — but its z has row mean-square 108, so it
# needs WHITEN_STATS set or the flow's isotropic noise is wildly mis-scaled against it.
TOK_ROOT="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log"
TOK_ARM=${TOK_ARM:-deltatok_surround_layer12_bsize2_dtok64_tc768_cos1e3_1e5_b95_gradskip_noaff_prenorm_sigreg0.05_pool_nozn}
# epoch_40, not current.pth: immutable, and the exact ckpt the z-spread numbers above were
# measured on. current.pth (~ep45) is marginally better recon but moves if the arm resumes.
TOK_EPOCH=${TOK_EPOCH:-epoch_40.pth}
TOK_TAG=${TOK_TAG:-tc768sigreg005ep40}       # RUN_NAME fragment; change with TOK_ARM
WHITEN_STATS=${WHITEN_STATS:-}               # empty -> config default null (no whitening)
# Distinct name so this run lands in its own subdir (the uniformt_cross run keeps its dir).
WHITEN_TAG="nowhiten"; [ -n "${WHITEN_STATS}" ] && WHITEN_TAG="whitenpos"
export RUN_NAME=deltatok_flow_waymo_${TOK_TAG}_deltaCtxCross_uniformTshared_lin20a0_wd05_raymapoff_sub1_100k_constlr_${WHITEN_TAG}_xxl_dit
# delta_ctx_cross: first delta token (frame 0->1) is the clean in-seq context AND
# frame-0 patch feats condition every block via per-camera cross-attn.
# global: one attention over all delta tokens. camera_embed gives per-camera identity
# (native-K: a camera's K delta tokens share its embedding; K identity via spatial_pos).
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera (must match the frozen ckpt)
    model.deltatok_ckpt=${TOK_ROOT}/${TOK_ARM}/ckpts/${TOK_EPOCH}
    # --- tokenizer arch: must match the frozen ckpt exactly. The load is strict=True, so a
    # parametered mismatch (tc768/mlp) aborts at startup — but use_camera_rope adds NO
    # parameters, so a mismatch there would silently produce the wrong z. Declared explicitly.
    model.deltatok.target_channels=768         # _tc768 channel bottleneck (flow cfg defaults to 0 = off)
    model.deltatok.z_norm=false                # _nozn: no final z LayerNorm; SIGReg alone pins z's scale
    model.deltatok.norm_affine=false           # _noaff (declared true in flow cfg)
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm; parameterless, so strict= cannot catch it
    # bottleneck_mlp omitted: absent from the flow cfg and _make_deltatok_module defaults it
    # to False, which is what this arm trained with (no _mlp tag in TOK_ARM).
    # No whitening by default — SIGReg already put z at N(0,I) scale, which is the claim under
    # test. The uniformt_cross stats file is Cz=1536 and CANNOT be reused (the shape assert in
    # _load_whiten_stats rejects it); regenerate per-arm via
    # STATS_MODE=position slurm/jz_compute_deltatok_latent_stats.slurm before setting WHITEN_STATS.
    ${WHITEN_STATS:+model.whiten_stats=${WHITEN_STATS}}
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 tokens as cross-attn condition
    model.attn_mode=global                     # single global attention over all delta tokens
    model.vit_use_camera_embed=true            # absolute per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE = 1/(1-t)^2-weighted x-MSE (upweights near-clean t->1)
    model.vit_size=xxlarge_d20                 # width 1536, now 2x the latent Cz=768 (was ==1536); in_proj stays full-rank. Held fixed so the tokenizer is the only variable
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed (RAE/DiT t-conditioning); ~1.2B total (vs xxl 1.17B, depth-28 dit would be ~1.7B)
    # cache_frozen_encode left at config default (false): per-sample encode cache grows unbounded over the full dataset
    # weight_decay left at config default (0.05): restore regularization for the full-data run
    training.exit_before_time_limit=true       # clean exit near SLURM wall (chain-resume safe)
    training.epoch=100                         # 100-epoch schedule (100k updates); chain-resume across SLURM walls
    training.max_iter=200000                   # lift the iter cap (was exactly 100k): epoch=100 is the sole stop
    training.lr_schedule=constant              # constant 1e-4 after the 1000-iter warmup (no cosine decay)
    model.t_dist=uniform                       # train t ~ U(0,1): 5.0% of draws above t=0.95 vs 0.46% for logit-normal
    model.t_shared=true                        # one t per sample, all slots — matches sampler_alpha=0
    model.sampler_scheduler_mode=linear        # uniform-in-t grid, matching the train prior
    model.sampler_alpha=0                      # no per-frame stagger (no slot parked at t=1 holding its own x_hat)
    training.eval_num_steps=20                 # at 20 steps clamp(1-t,0.05) never binds: final blend ratio 1.0, no residual noise
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# Fresh start is FORCED here, not a choice: the flow's in_proj/out_proj are sized by the
# latent Cz, which drops 1536 -> 768, so no existing flow ckpt can load into this arm.
# Paired no-sigreg twin (same arch, needs whitening — see TOK_ARM above):
#   sbatch --export=ALL,TOK_ARM=deltatok_surround_layer12_bsize2_dtok64_tc768_cos1e3_1e5_b95_gradskip_noaff_prenorm_nozn,TOK_TAG=tc768nosigregep40,WHITEN_STATS=<...>/latent_stats_waymo_pos.pth <this script>
# Chained resume jobs of THIS run still need RESUME=1:
#   sbatch --dependency=afterany:<prev_jobid> --export=ALL,RESUME=1 <this script>
# RESUME=1 makes sh/train_deltatok_flow.sh pass --resume (loads ckpts/current.pth).
export RESUME="${RESUME:-0}"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
