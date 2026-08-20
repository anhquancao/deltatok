#!/bin/bash
#SBATCH --job-name=deltatok_bench_flow
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=00:45:00
#SBATCH --output=slurm/output/bench_scaling_deltatok_flow_jz_%j.out
#SBATCH --error=slurm/output/bench_scaling_deltatok_flow_jz_%j.err

# Strong/weak scaling point for DeltaTok-flow (EuroHPC Regular Access, Section 2.6).
# Times 500 optimizer steps and trains nothing: no eval, no checkpoint, throwaway log root.
# 4 tasks/node on purpose — the production flow arms use 2, and a half-empty node is not a
# scaling baseline. --nodes/--ntasks-per-node/--qos come from the sbatch command line.
#   MODE=weak NGPU=8 BSIZE=16 EFF_BSIZE=128 sbatch --nodes=2 --export=ALL,MODE,NGPU,BSIZE,EFF_BSIZE \
#     slurm/bench_scaling_deltatok_flow_jz.slurm

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
export BSIZE="${BSIZE:-16}"                       # micro-batch per GPU
export EFF_BSIZE="${EFF_BSIZE:-64}"               # global batch -> grad_cum = EFF/(world*BSIZE)
export NGPU="${NGPU:-${SLURM_NTASKS:-4}}"

export CONFIG_NAME=train_deltatok_flow_bench_jeanzay   # 320k items/epoch, no eval set
export RESULTS_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_bench_log"
export RUN_NAME=bench_flow_${MODE}_g${NGPU}_b${BSIZE}_e${EFF_BSIZE}

# Frozen tokenizer: the JZ-resident tc768 stage-2 ckpt. Width is timing-neutral — BSC measured
# tc128/tc768/tc1536 flow steps within 1% at equal memory — so nothing is copied in from BSC.
# Arch flags must match the ckpt; parameterless ones fail silently, not loudly.
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the ckpt
    model.deltatok_ckpt=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_l12_dtok64_tc768_nozn_bneckrec1.0_rawtgt_stage2bneck_ep40/ckpts/epoch_20.pth
    model.deltatok.target_channels=768         # channel bottleneck (flow cfg defaults 0 = off)
    model.deltatok.z_norm=false                # no final z LayerNorm
    model.deltatok.norm_affine=false           # flow cfg declares true
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 cross-attn cond
    model.num_ctx_deltas=2                     # frames 0-2 given, 3-4 forecast
    model.attn_mode=global                     # one attention over all delta tokens
    model.vit_use_camera_embed=true
    model.loss_mode=v                          # v-MSE
    model.vit_size=xxlarge_d20                 # width 1536
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed; ~1.2B
    model.t_dist=uniform
    model.t_shared=true
    training.lr_schedule=constant
    training.bsize=${BSIZE}
    training.effective_bsize=${EFF_BSIZE}
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "BENCH MODE=$MODE NGPU=$NGPU BSIZE=$BSIZE EFF_BSIZE=$EFF_BSIZE NODES=${SLURM_NNODES:-?}"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok_flow.sh
