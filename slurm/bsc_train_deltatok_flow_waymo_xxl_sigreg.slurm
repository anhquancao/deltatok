#!/bin/bash
#SBATCH --job-name=deltatok_flow_waymo_xxl_sigreg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=40:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/bsc_train_deltatok_flow_waymo_xxl_sigreg_%j.out
#SBATCH --error=slurm/output/bsc_train_deltatok_flow_waymo_xxl_sigreg_%j.err

# BSC acc: 4 H100/node, 80 cores -> 20 cpus/task. acc_ehpc caps 72h; acc_debug is 2h.
# 40h not 72h: shorter backfills better (p90 4min@12h vs 8.6h@48h); chain the rest.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/bsc_train_deltatok_flow_waymo_xxl_sigreg.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# BSC port of slurm/jz_train_deltatok_flow_waymo_xxl_sigreg.slurm: same recipe, ehpc1001 paths.
# Frozen tokenizer is the stage-2 bottleneck arm (Cz 1536 -> 768: tc1536 sigreg0.05 frozen, only
# the projections trained on a raw-code round-trip). Judge on LossDepth vs the JZ twin.
export CONFIG_NAME=train_deltatok_flow_bsc_waymo
export RESULTS_DIR="$SCRATCH/deltatok_flow_log"
export RUN_NAME=deltatok_flow_waymo_tc768stage2bneckrawep20_deltaCtxCross_uniformTshared_lin20a0_wd05_raymapoff_sub1_100k_constlr_nowhiten_xxl_dit
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    # epoch_20 (stage 2's last) not current.pth: immutable if that arm resumes.
    # Trained on JZ: copy the arm dir to BSC before the first submit.
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc768_nozn_bneckrec1.0_rawtgt_stage2bneck_ep40/ckpts/epoch_20.pth"
    # Tokenizer arch must match the ckpt. strict=True catches parametered mismatches, but
    # use_camera_rope is parameterless — a wrong value silently yields the wrong z.
    model.deltatok.target_channels=768         # channel bottleneck (flow cfg defaults to 0 = off)
    model.deltatok.z_norm=false                # no final z LayerNorm
    model.deltatok.norm_affine=false           # flow cfg declares true
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm
    # bottleneck_mlp omitted -> False, as this arm trained (linear down/up).
    # whiten_stats omitted -> null. Stage 2 ran sigreg off at z_norm=false, so nothing pins z's
    # scale: revisit first if the flow looks mis-scaled. Stats are per-arm and shape-checked on
    # Cz; regenerate with STATS_MODE=position (see slurm/jz_compute_deltatok_latent_stats.slurm).
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 tokens as cross-attn condition
    model.attn_mode=global                     # one attention over all delta tokens
    model.vit_use_camera_embed=true            # per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE: 1/(1-t)^2-weighted x-MSE, upweights t->1
    model.vit_size=xxlarge_d20                 # width 1536, 2x the latent Cz; held fixed so the tokenizer is the only variable
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed; ~1.2B total
    training.bsize=8                           # 4 ranks x 8 = effective_bsize 32 -> grad_cum 1, as JZ's 2x16
    training.exit_before_time_limit=true       # clean exit near the SLURM wall (chain-resume safe)
    training.epoch=100                         # 100k updates, chained across SLURM walls
    training.max_iter=200000                   # lifted so epoch=100 is the sole stop
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

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
