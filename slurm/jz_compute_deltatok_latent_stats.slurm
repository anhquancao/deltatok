#!/bin/bash
#SBATCH --job-name=deltatok_latent_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=zuw@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/jz_compute_deltatok_latent_stats_%j.out
#SBATCH --error=slurm/output/jz_compute_deltatok_latent_stats_%j.err

# Per-channel mean/std of the frozen DeltaTok delta tokens -> latent_stats_<tag>.pth,
# written beside the tokenizer ckpt. Feeds model.whiten_stats (RAE-style whitening).
# No flow ckpt involved; single GPU, no DDP. The frozen OccRAE encode dominates.
# Read [GATE] in the log: std.max()/std.min() decides whether whitening is worth it.
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_compute_deltatok_latent_stats.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The module's site-packages bundles TensorFlow, which some import drags in;
# silence its startup spam (oneDNN notes, cuFFT/cuDNN/cuBLAS factory errors).
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# The script already defaults to the dtok64 non-affine tokenizer the waymo flow runs
# freeze. Override for another tokenizer, e.g.
#   EXTRA_ARGS="--cfg model.deltatok_ckpt=<other>/current.pth model.deltatok.num_delta_tokens=32" sbatch ...
# Smoke first: NUM_BATCHES=2 sbatch ...
: "${NUM_BATCHES:=500}"   # ~4.2M token vectors, ~12 min; 100 -> 500 moves std <1% (converged)
: "${SPLIT:=train}"
: "${TAG:=waymo}"

srun python compute_deltatok_latent_stats.py \
    --num_batches "$NUM_BATCHES" \
    --split "$SPLIT" \
    --tag "$TAG" \
    ${EXTRA_ARGS:-}
