#!/bin/bash
#SBATCH --job-name=build_pytorch3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/build_pytorch3d_bsc_%j.out
#SBATCH --error=slurm/output/build_pytorch3d_bsc_%j.err

# Build pytorch3d 0.7.8 (CUDA, sm_90) into envs/maskgit for occrae/chamfer_metrics.py FPS.
# Offline: source fetched on the login node via the reverse SOCKS tunnel into $PROJECT/src.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/build_pytorch3d_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=20

pip install --no-build-isolation --no-deps --no-index -v /gpfs/projects/ehpc1001/src/pytorch3d-0.7.8 2>&1 | tail -20

python - <<'EOF'
import torch
from pytorch3d.ops import sample_farthest_points
x = torch.rand(2, 5000, 3, device="cuda", dtype=torch.float64)
pts, idx = sample_farthest_points(x, K=100)
print("pytorch3d FPS on cuda OK", tuple(pts.shape), idx.device)
EOF
