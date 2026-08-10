#!/bin/bash
#SBATCH --job-name=deltatok_latent_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/compute_deltatok_latent_stats_bsc_%j.out
#SBATCH --error=slurm/output/compute_deltatok_latent_stats_bsc_%j.err

# Whitening stats of the frozen tc1536 maxgap9 nosigreg DeltaTok -> latent_stats_<tag>.pth,
# beside epoch_49.pth. Feeds model.whiten_stats in the tc1536mg9 flow arm.
# STATS_MODE=channel: pooled (C,) | position: (K, C) per delta-slot. Read [GATE] in the log:
# std.max()/std.min() decides whether whitening is worth it. No flow ckpt; single GPU, no DDP.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/compute_deltatok_latent_stats_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

: "${NUM_BATCHES:=500}"       # ~2M token vectors, ~10 min. Smoke first: NUM_BATCHES=2 sbatch ...
: "${SPLIT:=train}"           # whitening is a train-set statistic
: "${STATS_MODE:=channel}"    # channel: pooled (C,) — start here (Fable); position: (K, C) per slot
: "${TAG:=mg9waymo}"          # -> latent_stats_mg9waymo.pth beside the tokenizer ckpt

# Tokenizer arch MUST match the frozen ckpt — same block as the tc1536mg9 flow slurm.
# target_channels/z_norm/use_camera_rope are NOT in the script's built-in defaults, so
# pass them here or the raw z is silently wrong. bottleneck_mlp is omitted: not in the
# flow config struct (strict compose rejects it), and inert at tc1536 anyway.
TOKENIZER_CFG=(
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_nosigreg/ckpts/epoch_49.pth"
    model.deltatok.num_delta_tokens=64
    model.deltatok.target_channels=1536
    model.deltatok.z_norm=false
    model.deltatok.norm_affine=false
    model.deltatok.use_camera_rope=false
)

echo "PWD=$(pwd)"
echo "STATS_MODE=$STATS_MODE  NUM_BATCHES=$NUM_BATCHES  TAG=$TAG"

srun python compute_deltatok_latent_stats.py \
    --config-dir configs/deltatok_flow \
    --config-name train_deltatok_flow_waymo_tc1536mg9_bsc \
    --num_batches "$NUM_BATCHES" \
    --split "$SPLIT" \
    --stats_mode "$STATS_MODE" \
    --tag "$TAG" \
    --cfg "${TOKENIZER_CFG[@]}" ${EXTRA_ARGS:-}
