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
#SBATCH --output=slurm/output/train_deltatok_maxgap_sigreg_nozn_tc256_bsc_%j.out
#SBATCH --error=slurm/output/train_deltatok_maxgap_sigreg_nozn_tc256_bsc_%j.err

# SIGReg sweep on the tc256 linear-bottleneck maxgap arm (e2e). Set SIGREG_WEIGHT (enters RUN_NAME):
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && SIGREG_WEIGHT=0.01 sbatch --export=ALL slurm/deltatok/train_deltatok_maxgap_sigreg_nozn_tc256_bsc.slurm'"
# w=0 is the nosigreg script. WATCH: LossSIGReg must descend past iter 2000 (1/S floor 1.32e-4).

: "${SIGREG_WEIGHT:?SIGREG_WEIGHT must be set}"
case "${SIGREG_WEIGHT}" in 0|0.0|0.00) echo "w=0 -> use the nosigreg script" >&2; exit 1 ;; esac

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_nt10_bsc   # num_timesteps=10 -> supports max_gap up to 9
export LOG_AND_CKPT_DIR="$SCRATCH/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"

export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-512}      # 2*Cz at tc256
export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-8192}  # NOT weight-neutral; keep 8192 (2026-07-31)
export SIGREG_WARMUP=${SIGREG_WARMUP:-2000}              # held constant across the sweep

export RUN_NAME=deltatok_l12_dtok64_tc256_nozn_maxgap9_vpt1to2_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}

EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera (~12:1 compression at full res)
    model.deltatok.target_channels=256         # < hidden_size -> linear z_proj_down/up bottleneck
    model.deltatok.z_norm=false                # decoder + SIGReg both see raw z
    model.deltatok.use_camera_rope=false       # layer-12 no-rope run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for name parity
    model.deltatok.bottleneck_mlp=false        # linear bottleneck (no MLP nonlinearity)
    training.lr=1e-3                           # peak lr, cosine-decayed to 1e-5
    training.grad_clip=0.1                     # tighter than base 1.0; tames early grads
    training.grad_skip_thresh=0.0               # >0 also skips finite grad_norm spikes
    training.sigreg_weight=${SIGREG_WEIGHT}
    training.sigreg_num_slices=${SIGREG_NUM_SLICES}
    training.sigreg_pool_samples=${SIGREG_POOL_SAMPLES}
    training.sigreg_warmup=${SIGREG_WARMUP}
    training.max_gap=9
    training.exit_before_time_limit=true        # stop after a ckpt near the wall (chain-resume safe)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
