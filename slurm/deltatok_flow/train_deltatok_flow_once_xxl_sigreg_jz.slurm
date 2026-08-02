#!/bin/bash
#SBATCH --job-name=deltatok_flow_once_xxl_sigreg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=slurm/output/train_deltatok_flow_once_xxl_sigreg_jz_%j.out
#SBATCH --error=slurm/output/train_deltatok_flow_once_xxl_sigreg_jz_%j.err

# JZ gpu_p6: 4 GPUs/node, 24 cpus/task. t3 caps at 20h; -t4 gives 100h, fewer GPUs.
# Submit from a LOGIN shell so `module` reaches the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_once_xxl_sigreg_jz.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Unique DDP port per job (29500 collides with leftover jobs).
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))

# Silence the bundled TensorFlow's startup spam.
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# DATA swap of train_deltatok_flow_waymo_xxl_sigreg_jz.slurm: same tokenizer and flow
# recipe on ONCE instead of Waymo — 843k vs 122k train seqs (~6.9x), 16 held-out scenes.
# Eval is ONCE val, so NOT comparable to the waymo arm's numbers.
export CONFIG_NAME=train_deltatok_flow_once_jeanzay
export RESULTS_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log"
# Only the data differs from the waymo arm; the shared recipe tags live in this script, not the name.
export RUN_NAME=deltatok_flow_once_tc768s2bneck_xxl_dit
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    # epoch_20 (stage 2's last) not current.pth: immutable if that arm resumes.
    model.deltatok_ckpt=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_l12_dtok64_tc768_nozn_bneckrec1.0_rawtgt_stage2bneck_ep40/ckpts/epoch_20.pth
    # Tokenizer arch must match the ckpt. strict=True catches parametered mismatches, but
    # use_camera_rope is parameterless — a wrong value silently yields the wrong z.
    model.deltatok.target_channels=768         # channel bottleneck (flow cfg defaults to 0 = off)
    model.deltatok.z_norm=false                # no final z LayerNorm
    model.deltatok.norm_affine=false           # flow cfg declares true
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm
    # bottleneck_mlp omitted -> False, as this arm trained (linear down/up).
    # whiten_stats omitted -> null. Stage 2 ran sigreg off at z_norm=false, so nothing pins z's
    # scale: revisit first if the flow looks mis-scaled. The uniformt_cross file is Cz=1536 and
    # is rejected on shape; regenerate via STATS_MODE=position slurm/compute_deltatok_latent_stats_jz.slurm.
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 tokens as cross-attn condition
    model.attn_mode=global                     # one attention over all delta tokens
    model.vit_use_camera_embed=true            # per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE: 1/(1-t)^2-weighted x-MSE, upweights t->1
    model.vit_size=xxlarge_d20                 # width 1536, 2x the latent Cz; held fixed so the tokenizer is the only variable
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed; ~1.2B total
    # cache_frozen_encode default false: the per-sample cache grows unbounded over the full dataset
    # weight_decay default 0.05: regularization for the full-data run
    training.exit_before_time_limit=true       # clean exit near the SLURM wall (chain-resume safe)
    training.epoch=100                         # 32000 items / eff 32 = 1000 it/epoch
    training.max_iter=100000                   # 100 x 1000: same stop as epoch=100
    training.lr_schedule=constant              # constant 1e-4 after warmup
    model.t_dist=uniform                       # t ~ U(0,1): 5.0% of draws above t=0.95 vs 0.46% logit-normal
    model.t_shared=true                        # one t per sample, all slots — matches sampler_alpha=0
    model.sampler_scheduler_mode=linear        # uniform-in-t grid, matching the train prior
    model.sampler_alpha=0                      # no per-frame stagger
    training.eval_num_steps=20                 # clamp(1-t,0.05) never binds: no residual noise
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# --resume is a no-op without current.pth, so job 1 starts fresh and chained jobs continue it:
#   sbatch --dependency=afterany:<prev_jobid> <this script>
export RESUME=1

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
