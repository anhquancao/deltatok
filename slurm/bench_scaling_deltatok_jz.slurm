#!/bin/bash
#SBATCH --job-name=deltatok_bench_tok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=00:45:00
#SBATCH --output=slurm/output/bench_scaling_deltatok_jz_%j.out
#SBATCH --error=slurm/output/bench_scaling_deltatok_jz_%j.err

# Strong/weak scaling point for the DeltaTok tokenizer (EuroHPC Regular Access, Section 2.6).
# Times 500 optimizer steps and trains nothing: no eval, no checkpoint, throwaway log root.
# --nodes/--ntasks-per-node/--qos come from the sbatch command line; see sh/bench_scaling_submit.sh.
#   MODE=weak NGPU=8 BSIZE=2 EFF_BSIZE=16 sbatch --nodes=2 --export=ALL,MODE,NGPU,BSIZE,EFF_BSIZE \
#     slurm/bench_scaling_deltatok_jz.slurm

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# Unique DDP port per job; the ladder runs points back to back and 29500 collides.
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export MODE="${MODE:-weak}"                       # weak | strong, tags the run name only
export BSIZE="${BSIZE:-2}"                        # micro-batch per GPU
export EFF_BSIZE="${EFF_BSIZE:-8}"                # global batch -> grad_cum = EFF/(world*BSIZE)
export NGPU="${NGPU:-${SLURM_NTASKS:-4}}"

export CONFIG_NAME=train_deltatok_bench_jeanzay   # single-shard, fixed-shape, 64k items/epoch
export LOG_AND_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_bench_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export RUN_NAME=bench_tok_${MODE}_g${NGPU}_b${BSIZE}_e${EFF_BSIZE}

# Arch flags copied verbatim from the production compose tc128 arm so the timing is that
# recipe's. Only bsize/effective_bsize move across the ladder.
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera
    model.deltatok.target_channels=128         # linear z_proj_down/up bottleneck
    model.deltatok.z_norm=false                # compose sums two z; a sum leaves the LN sphere
    model.deltatok.use_camera_rope=false       # layer-12 no-rope run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for parity
    model.deltatok.bottleneck_mlp=false        # linear bottleneck
    training.lr=1e-3
    training.grad_clip=0.1
    training.grad_skip_thresh=0.0
    training.sigreg_weight=0.005
    training.sigreg_num_slices=256
    training.sigreg_pool_samples=8192
    training.sigreg_warmup=2000
    training.compose_weight=1.0                # 2 encodes + 3 decodes/step
    training.max_gap=9                         # triplet spans cap at 9 (= nt10 window - 1)
    training.bsize=${BSIZE}
    training.effective_bsize=${EFF_BSIZE}
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "BENCH MODE=$MODE NGPU=$NGPU BSIZE=$BSIZE EFF_BSIZE=$EFF_BSIZE NODES=${SLURM_NNODES:-?}"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
