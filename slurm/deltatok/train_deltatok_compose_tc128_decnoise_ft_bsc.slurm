#!/bin/bash
#SBATCH --job-name=deltatok_decnoise_ft_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=08:00:00
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --output=slurm/output/train_deltatok_compose_tc128_decnoise_ft_bsc_%j.out
#SBATCH --error=slurm/output/train_deltatok_compose_tc128_decnoise_ft_bsc_%j.err

# 10-epoch decoder-only noise finetune of the tc128 compose arm's epoch_100: the decoder learns
# to decode z + sigma*N(0,I), the encoder is frozen so z is unchanged and the flow stays valid.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_tc128_decnoise_ft_bsc.slurm'"

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

export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-256}      # 2*Cz at tc128
export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-8192}  # NOT weight-neutral; keep 8192 (2026-07-31)
export SIGREG_WARMUP=${SIGREG_WARMUP:-2000}              # held constant across the sweep

# Fresh name: no current.pth, so the INIT_CKPT warm start below is the one that loads.
export RUN_NAME=deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0_decnoise0.8_ft10

# Weights-only load (--ckpt): iter restarts at 0, so the short schedule below is the finetune's own.
export INIT_CKPT="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0/ckpts/epoch_100.pth"

EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K delta tokens/camera (~12:1 compression at full res)
    model.deltatok.target_channels=128         # < hidden_size -> linear z_proj_down/up bottleneck
    model.deltatok.z_norm=false                # compose sums two z; a sum leaves the LN sphere
    model.deltatok.use_camera_rope=false       # layer-12 no-rope run
    model.deltatok.norm_affine=false           # inert at z_norm=false; kept for name parity
    model.deltatok.bottleneck_mlp=false        # linear bottleneck (no MLP nonlinearity)
    model.deltatok.decode_noise_tau=0.8        # decoder input z + sigma*N(0,I), sigma ~ U(0,0.8) (RAE)
    training.freeze_except_decoder=true         # only decoder_blocks + z_proj_up move -> z unchanged
    training.lr=1e-4                           # finetune peak, cosine-decayed to 1e-5
    training.min_lr_ratio=0.1
    training.warm_up=500
    training.epoch=10
    training.max_iter=11250                     # 1125 updates/epoch at effective_bsize 16
    training.save_ckpt_every_n_epochs=5          # epoch_5 tells us if epoch_10 was still moving
    training.grad_clip=0.1                     # tighter than base 1.0; tames early grads
    training.grad_skip_thresh=0.0               # >0 also skips finite grad_norm spikes
    training.sigreg_weight=0                    # z is frozen; the pool would only cost
    training.sigreg_num_slices=${SIGREG_NUM_SLICES}
    training.sigreg_pool_samples=${SIGREG_POOL_SAMPLES}
    training.sigreg_warmup=${SIGREG_WARMUP}
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
echo "INIT_CKPT=$INIT_CKPT"
echo "EXTRA_CFG=$EXTRA_CFG"

srun --cpu-bind=none bash sh/train_deltatok.sh
