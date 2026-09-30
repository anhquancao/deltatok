#!/bin/bash
#SBATCH --job-name=test_chamfer_parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/test_chamfer_parity_jz_%j.out
#SBATCH --error=slurm/output/test_chamfer_parity_jz_%j.err

# occrae/chamfer_metrics.py vs Gen3R's open3d + pytorch3d original. JZ: the env has both.
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/test_chamfer_parity_jz.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

srun python test_chamfer_parity.py ${EXTRA_ARGS:-}
