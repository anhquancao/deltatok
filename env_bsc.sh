module purge
module load EB/apps
module load EB/install
module load CUDA/12.1.1
module unload anaconda
module load intel mkl hdf5 python/3.12.1
module load FFmpeg/6.0


# Use an isolated venv (created by sync_bsc/install_offline_hpc.sh) so we do not
# pick up system site-packages (can include a fake `tensorflow` namespace that
# breaks TensorBoard / SummaryWriter imports).
export PYTHONNOUSERSITE=1

VENV_DIR="${HOME}/envs/maskgit"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Keep local third-party sources on PYTHONPATH.
# NOTE: it's important to set the PYTHONPATH to either empty or your package to shadow the default python local env
export PYTHONPATH="/home/vale/vale352205/code/VATIX/third_party/Depth-Anything-3/src"
