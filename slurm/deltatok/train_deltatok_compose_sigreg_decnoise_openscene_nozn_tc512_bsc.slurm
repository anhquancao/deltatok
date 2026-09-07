#!/bin/bash
#SBATCH --job-name=deltatok_decnoise_os_bsc
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
#SBATCH --output=slurm/output/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc_%j.out
#SBATCH --error=slurm/output/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc_%j.err

# End-to-end decoder-noise twin of BSC:45296347: encoder + decoder train through
# z + sigma*eps, sigma ~ U(0, tau). Everything else matches that arm.
# + OpenScene trainval (front cam, tar store) in train, openscene_test in eval
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_sigreg_decnoise_openscene_nozn_tc512_bsc.slurm'"
# Control ladder on the twin's ckpt (no training): prepend
#   EVAL_ONLY=1 DECNOISE_TAU=0 RUN_NAME=<fresh> INIT_CKPT=<twin>/ckpts/epoch_80.pth

export COMPOSE_WEIGHT=${COMPOSE_WEIGHT:-1.0}
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.02}    # matches the twin BSC:45296347
export DECNOISE_TAU=${DECNOISE_TAU:-0.8}       # sigma ~ U(0, tau) per sample, decoder input only
export DECNOISE_WARMUP=${DECNOISE_WARMUP:-2000}  # = SIGREG_WARMUP

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1000}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

export CONFIG_NAME=train_deltatok_nt10_openscene_bsc   # nt10 (max_gap=9) + OpenScene front cam
export LOG_AND_CKPT_DIR="$SCRATCH/deltatok_log"
export RESULTS_DIR="$LOG_AND_CKPT_DIR"

export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-1024}      # 2*Cz at tc512
export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-8192}  # NOT weight-neutral; keep 8192 (2026-07-31)
export SIGREG_WARMUP=${SIGREG_WARMUP:-2000}              # held constant across the sweep

# ${RUN_NAME:-} so the eval-only ladder can point at a fresh dir instead of resuming this arm.
export RUN_NAME=${RUN_NAME:-deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}_openscene}

EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera (~12:1 compression at full res)
    model.deltatok.target_channels=512         # < hidden_size -> linear z_proj_down/up bottleneck
    model.deltatok.z_norm=false                # compose sums two z; a sum leaves the LN sphere
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
    training.compose_weight=${COMPOSE_WEIGHT}   # d(t0->t1)+d(t1->t2) decodes f_t2 from f_t0
    model.deltatok.decode_noise_tau=${DECNOISE_TAU}    # train-mode only; z itself stays clean
    training.decode_noise_warmup=${DECNOISE_WARMUP}    # linear ramp of tau over N iters
    "training.eval_noise_sigmas=[0.32,0.55,0.82]"       # eval ladder, matches the 2026-09-04 probe grid
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
