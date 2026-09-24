# Waymo preprocessing env. ACC nodes only (alogin*, acc compute): this module tree is ACC's.
# TF 2.16 + torch 2.4 come from the python module; the venv adds waymo, cv2 (see install_env_waymo_bsc.sh).
module purge
module load gcc impi mkl hdf5/1.14.1-2-gcc python/3.11.5-gcc cuda/12.3 cudnn/9.1.0-cuda12 nccl

export PYTHONNOUSERSITE=1

: "${PROJECT:?Error: PROJECT is not set}"
VENV_DIR="${PROJECT}/envs/waymo_tf"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Keep the module's PYTHONPATH: TF and torch live there.
