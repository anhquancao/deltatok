#!/bin/bash
#SBATCH --job-name=df_compose_tindep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc880
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=40:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/df_compose_tindep_bsc_%j.out
#SBATCH --error=slurm/output/df_compose_tindep_bsc_%j.err

# Ablation of the compose arm: t_shared=false (base uses true=shared t).
# Isolates per-slot independent t sampling. All other settings identical.
# Charged to ehpc880; $SCRATCH/$PROJECT stay ehpc1001.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_tindep_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# Unique DDP port per job (default 29500 collides with a co-scheduled arm).
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_flow_waymo_bsc   # same data as the tc1536mg9 config (comments aside)
export RESULTS_DIR="$SCRATCH/deltatok_flow_log"
export RUN_NAME=df_ctx3fwd2_tc128mg9s005compose_tindep_xxl
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    # epoch_40: matches the non-compose twin's pin. compose_weight is train-only, so arch is identical.
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0/ckpts/epoch_40.pth"
    # Tokenizer arch must match the ckpt — parameterless flags silently yield wrong z.
    model.deltatok.target_channels=128         # linear z_proj_down/up bottleneck (flow cfg defaults 0 = off)
    model.deltatok.z_norm=false                # no final z LayerNorm
    model.deltatok.norm_affine=false           # flow cfg declares true
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm
    # bottleneck_mlp omitted -> False, as this arm trained (linear down/up).
    # whiten_stats omitted -> null. SIGReg already delivers unit-scale z.
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 tokens as cross-attn condition
    model.num_ctx_deltas=2                     # 2 clean deltas = frames 0-2 given, 3-4 forecast
    model.attn_mode=global                     # one attention over all delta tokens
    model.vit_use_camera_embed=true            # per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE: 1/(1-t)^2-weighted x-MSE, upweights t->1
    model.vit_size=xxlarge_d20                 # width 1536, 12x the latent Cz=128
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed; ~1.2B total
    training.exit_before_time_limit=true       # clean exit near the SLURM wall (chain-resume safe)
    training.epoch=100                         # 100k updates, chained across SLURM walls
    training.max_iter=200000                   # lifted so epoch=100 is the sole stop
    training.lr_schedule=constant              # constant 1e-4 after warmup
    model.t_dist=uniform                       # t ~ U(0,1)
    model.t_shared=false                       # ABLATED: independent t per slot
    model.sampler_scheduler_mode=linear        # uniform-in-t grid, matching the train prior
    model.sampler_alpha=0                      # no per-frame stagger
    training.eval_num_steps=20                 # clamp(1-t,0.05) never binds: no residual noise
    training.val_bsize=4                       # BSC eval OOM at bsize 16
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"
# Resume is automatic; job 1 starts fresh, chained jobs continue it:
#   sbatch --dependency=afterany:<prev_jobid> <this script>

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
