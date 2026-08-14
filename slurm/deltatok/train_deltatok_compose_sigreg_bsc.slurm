#!/bin/bash
#SBATCH --job-name=deltatok_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=24:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/train_deltatok_compose_sigreg_bsc_%j.out
#SBATCH --error=slurm/output/train_deltatok_compose_sigreg_bsc_%j.err

# Additive-composition arm WITH SIGReg. Set SIGREG_WEIGHT; COMPOSE_WEIGHT defaults 1.0:
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && SIGREG_WEIGHT=0.05 sbatch --export=ALL slurm/deltatok/train_deltatok_compose_sigreg_bsc.slurm'"
# w=0 -> use the nosigreg compose script. WATCH: LossSIGReg must descend past iter 2000.

: "${SIGREG_WEIGHT:?SIGREG_WEIGHT must be set}"
case "${SIGREG_WEIGHT}" in 0|0.0|0.00) echo "w=0 -> use the nosigreg compose script" >&2; exit 1 ;; esac
export COMPOSE_WEIGHT=${COMPOSE_WEIGHT:-1.0}

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_nt10_bsc   # num_timesteps=10 -> triplet spans up to max_gap=9
export LOG_AND_CKPT_DIR="$SCRATCH/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"

export RUN_NAME=deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg${SIGREG_WEIGHT}_compose${COMPOSE_WEIGHT}

EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera (~12:1 compression at full res)
    model.deltatok.target_channels=1536        # == hidden_size -> no z_proj_down/up built
    model.deltatok.z_norm=false                # compose sums two z; a sum leaves the LN sphere
    model.deltatok.use_camera_rope=false       # layer-12 no-rope run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for name parity
    model.deltatok.bottleneck_mlp=false        # inert at tc1536 (no bottleneck to make nonlinear)
    training.lr=1e-3                           # peak lr, cosine-decayed to 1e-5
    training.grad_clip=0.1                     # tighter than base 1.0; tames early grads
    training.grad_skip_thresh=0.0               # >0 also skips finite grad_norm spikes
    training.sigreg_weight=${SIGREG_WEIGHT}     # SIGReg on raw z (z_norm=false -> reg + decoder see raw)
    training.sigreg_num_slices=3072             # 2*Cz at tc1536
    training.sigreg_pool_samples=8192           # NOT weight-neutral; keep 8192 (2026-07-31)
    training.sigreg_warmup=2000                 # held constant across the sweep
    training.compose_weight=${COMPOSE_WEIGHT}   # d(t0->t1)+d(t1->t2) decodes f_t2 from f_t0
    training.max_gap=9                          # triplet spans cap at 9 (= nt10 window - 1)
    training.bsize=2                            # 2 encodes + 3 decodes/step ~2.5x single-pair; halve batch
    training.effective_bsize=16                 # 4 GPU x bsize 2 -> grad_cum 2
    training.exit_before_time_limit=true        # stop after a ckpt near the wall (chain-resume safe)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
