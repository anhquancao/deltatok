#!/bin/bash
#SBATCH --job-name=deltatok_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=48:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/bsc_train_deltatok_multitoken_sigreg_nozn_%j.out
#SBATCH --error=slurm/output/bsc_train_deltatok_multitoken_sigreg_nozn_%j.err

# BSC acc: 4 H100/node, 80 cores -> 20 cpus/task. acc_ehpc caps 48h; acc_debug is 2h.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/bsc_train_deltatok_multitoken_sigreg_nozn.slurm'"
#
# DeltaTok tokenizer, layer-12, no camera rope. tc1536 == hidden_size -> no channel bottleneck,
# and z_norm=false -> no z LayerNorm: SIGReg alone shapes a raw 1536-d z. K is the only compression.
# pool=32768 (was 8192): at 8192 the 1/S floor was 1.32e-4 == 80% of the 1.65e-4 anisotropy signal,
# an O(1/S) bias in the fixed point. Affordable now that SIGReg runs the detached pool under
# no_grad (occrae/sigreg.py). Cost: the pool spans ~16-32 optimizer steps, so rows are staler.
# WATCH: LossSIGReg must DESCEND and stay above its floor (~3.2e-5); flat = under-resolved.
# SIGREG_WEIGHT is a knob to RAISE -- we drop the reference's *N scaling, so 0.05 here is ~2 orders
# weaker than LeJEPA's lamb=0.05 and their 0.01-0.1 grid does not transfer. Sweep log-spaced:
#   for W in 0.05 0.5 5; do sbatch --export=ALL,SIGREG_WEIGHT=$W slurm/bsc_train_deltatok_multitoken_sigreg_nozn.slurm; done

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_bsc
export LOG_AND_CKPT_DIR="$SCRATCH/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"
export RESUME=${RESUME:-0}   # 0 restarts and clobbers current.pth; --export=ALL,RESUME=1 to continue

# The only three knobs; each changes the estimator or the pressure, so all three are in RUN_NAME.
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.05}          # matches the JZ ep25 twin -> pool is the only change
export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-3072}  # 2*Cz; measured sufficient (signal is K-invariant)
export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-32768}  # GLOBAL rows, S/C ~21; rescaled by pool/live

export RUN_NAME=deltatok_surround_layer12_bsize2_dtok64_tc1536_cos1e3_1e5_b95_gradskip_noaff_prenorm_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_nozn

EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera (~12:1 compression at full res)
    model.deltatok.target_channels=1536        # == hidden_size -> no z_proj_down/up built
    model.deltatok.z_norm=false                # decoder + SIGReg both see raw z
    model.deltatok.use_camera_rope=false       # layer-12 no-rope run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for name parity
    model.deltatok.bottleneck_pre_norm=true    # inert at tc1536; kept so a tc override still gets the fix
    model.deltatok.bottleneck_mlp=false        # inert at tc1536 (no bottleneck to make nonlinear)
    training.lr=1e-3                           # peak lr, cosine-decayed to 1e-5
    training.grad_clip=0.1                     # tighter than base 1.0; tames early grads
    training.grad_skip_thresh=0.0               # >0 also skips finite grad_norm spikes
    training.sigreg_weight=${SIGREG_WEIGHT}
    training.sigreg_num_slices=${SIGREG_NUM_SLICES}
    training.sigreg_pool_samples=${SIGREG_POOL_SAMPLES}
    training.sigreg_warmup=2000                 # linear weight ramp (iters)
    training.exit_before_time_limit=true        # stop after a ckpt near the wall (chain-resume safe)
)
export EXTRA_CFG="${EXTRA_CFG_ARGS[*]}"

echo "PWD=$(pwd)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "LOG_AND_CKPT_DIR=$LOG_AND_CKPT_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
