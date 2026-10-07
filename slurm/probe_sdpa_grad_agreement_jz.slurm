#!/bin/bash
#SBATCH --job-name=probe_sdpa_grad_agreement
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=00:45:00
#SBATCH --output=slurm/output/probe_sdpa_grad_agreement_jz_%j.out
#SBATCH --error=slurm/output/probe_sdpa_grad_agreement_jz_%j.err

# cuDNN vs flash SDPA grads on the same step (flash twice = floor), 1 GPU, no optim step.
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/probe_sdpa_grad_agreement_jz.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

# The no-sum arm's overrides (seed_ls_skip_jz.slurm); bsize 2 as trained.
EXTRA_CFG_ARGS=(
    model.deltatok.num_delta_tokens=64
    model.deltatok.target_channels=1536
    model.deltatok.force_bottleneck=false
    model.deltatok.z_row_clip=0
    model.deltatok.layer_scale_init=0.1
    training.init_seed=2
    model.deltatok.z_norm=false
    model.deltatok.use_camera_rope=false
    model.deltatok.norm_affine=false
    model.deltatok.bottleneck_mlp=false
    training.lr=1e-3
    training.grad_clip=0.1
    training.grad_skip_thresh=0.0
    training.sigreg_weight=0.02
    training.sigreg_num_slices=3072
    training.sigreg_pool_samples=24576
    training.sigreg_warmup=0
    training.compose_weight=0
    model.deltatok.decode_noise_tau=0
    training.max_gap=9
    training.bsize=2
    training.effective_bsize=16
)
L=/lustre/fswork/projects/rech/trg/uyl37fq/deltatok_log/deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.02_ns3072_pool24576
R=${L}_compose0_sw0_seed2_ls0.1_3pairs_skip/ckpts   # no-sum, clean to ep 22
C=${L}_compose1.0_sw0_seed2_ls0.1_3pairs/ckpts       # compose, run here without its sum term

srun python probe_sdpa_grad_agreement.py \
    --config-dir configs/deltatok \
    --config-name train_deltatok_nt10_jeanzay \
    --cfg "${EXTRA_CFG_ARGS[@]}" \
    --ckpts "$R/epoch_10.pth" "$R/epoch_20.pth" "$C/epoch_20.pth" "$C/epoch_50.pth" \
    --num-batches "${NUM_BATCHES:-4}"
