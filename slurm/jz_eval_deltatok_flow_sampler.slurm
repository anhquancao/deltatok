#!/bin/bash
#SBATCH --job-name=deltatok_flow_sampler_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=zuw@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/jz_eval_deltatok_flow_sampler_%j.out
#SBATCH --error=slurm/output/jz_eval_deltatok_flow_sampler_%j.err

# Trainer-faithful generative eval (eval_one_epoch) of a flow ckpt under both
# sampler step modes: ode (reproduces the run's logged numbers = load sanity)
# then damped (VGGT-World blend-to-x_hat; tests the sampler-amplification fix).
# Eval-only, single GPU, no DDP. Defaults target the xxl_dit run's current.pth.
#
# Submit from a LOGIN shell so `module` (login-only) is carried into the job:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_eval_deltatok_flow_sampler.slurm'"

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

# Defaults in the script already target the xxl_dit run's ckpt + its EXTRA_CFG arch
# flags. Override from the CLI for another run, e.g.
#   CKPT=<other>/ckpts/current.pth EXTRA_ARGS="--cfg model.vit_size=xxlarge" sbatch ...
: "${CKPT:=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_flow_log/deltatok_flow_waymo_dtok64_deltaCtxCross_global_fullT_wd05_raymapoff_50k_whitenpos_xxl_dit/ckpts/current.pth}"
: "${OUTPUT_DIR:=results/deltatok_flow_sampler_eval}"
: "${STEP_MODES:=ode,damped}"

srun python eval_deltatok_flow_sampler.py \
    --ckpt "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --step_modes "$STEP_MODES" \
    ${EXTRA_ARGS:-}
