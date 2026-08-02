#!/bin/bash
#SBATCH --job-name=deltatok_flow_t_bins
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=zuw@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/eval_deltatok_flow_t_bins_jz_%j.out
#SBATCH --error=slurm/output/eval_deltatok_flow_t_bins_jz_%j.err

# Per-noise-level (t-bin) error profile for a trained DeltaTok flow run: sweeps t over
# 50 uniform bins and reports teacher-forced error per bin on train + val.
# Eval-only, single GPU, no DDP. ~20-40 min (frozen OccRAE encode dominates).
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/eval_deltatok_flow_t_bins_jz.slurm'"

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

# Defaults in the script already target the waymo dtok64 run's ckpt + its EXTRA_CFG arch
# flags. Override from the CLI for another run, e.g.
#   CKPT=<other>/ckpts/current.pth EXTRA_ARGS="--cfg model.cond_mode=delta_ctx" sbatch ...
: "${CKPT:=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log/deltatok_flow_waymo_dtok64_deltaCtxCross_global_fullT_wd05_raymapoff_50k/ckpts/current.pth}"
: "${OUTPUT_DIR:=results/deltatok_flow_t_bins}"
: "${NUM_BINS:=50}"
: "${NUM_BATCHES:=8}"
: "${NOISE_REPEATS:=4}"
: "${SPLITS:=train,val}"

srun python eval_deltatok_flow_t_bins.py \
    --ckpt "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_bins "$NUM_BINS" \
    --num_batches "$NUM_BATCHES" \
    --noise_repeats "$NOISE_REPEATS" \
    --splits "$SPLITS" \
    ${EXTRA_ARGS:-}
