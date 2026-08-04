#!/bin/bash
#SBATCH --job-name=deltatok_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc880
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=40:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/train_deltatok_multitoken_sigreg_nozn_randint_revseq_bsc_%j.out
#SBATCH --error=slurm/output/train_deltatok_multitoken_sigreg_nozn_randint_revseq_bsc_%j.err

# BSC acc: 4 H100/node, 80 cores -> 20 cpus/task. acc_ehpc caps 72h; acc_debug is 2h.
# 40h not 72h: shorter backfills better (p90 4min@12h vs 8.6h@48h); ~53 ep/job, chained.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_multitoken_sigreg_nozn_randint_revseq_bsc.slurm'"
#
# Random-interval + 50% reverse: timesteps spread across the record with a coin-flip
# descending order (backward-motion pairs as data augmentation). SIGReg on (w=0.05,
# pool=8192). Pairs with the randint (no revseq) arm to isolate the reversal effect.

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_randint_revseq_bsc
export LOG_AND_CKPT_DIR="$SCRATCH/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"

# Architecture + window + the three SIGReg knobs; everything else is fixed by this script.
export RUN_NAME=deltatok_l12_dtok64_tc1536_nozn_randint5_revseq_vpt1to2_sigreg0.05_ns3072_pool8192

EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera (~12:1 compression at full res)
    model.deltatok.target_channels=1536        # == hidden_size -> no z_proj_down/up built
    model.deltatok.z_norm=false                # decoder + SIGReg both see raw z
    model.deltatok.use_camera_rope=false       # layer-12 no-rope run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for name parity
    model.deltatok.bottleneck_mlp=false        # inert at tc1536 (no bottleneck to make nonlinear)
    training.lr=1e-3                           # peak lr, cosine-decayed to 1e-5
    training.grad_clip=0.1                     # tighter than base 1.0; tames early grads
    training.grad_skip_thresh=0.0               # >0 also skips finite grad_norm spikes
    training.sigreg_weight=0.05                 # the pre-32768 setting; pairs with pool=8192
    training.sigreg_num_slices=3072              # 2*Cz; measured sufficient (signal is K-invariant)
    training.sigreg_pool_samples=8192            # GLOBAL rows, S/C ~5.3; rescaled by pool/live
    training.sigreg_warmup=2000                 # linear weight ramp (iters)
    training.exit_before_time_limit=true        # stop after a ckpt near the wall (chain-resume safe)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
