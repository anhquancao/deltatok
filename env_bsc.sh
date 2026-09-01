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

: "${PROJECT:?Error: PROJECT is not set}"
VENV_DIR="${PROJECT}/envs/maskgit"

# ~/.cache, ~/.local and ~/.conda symlink into the dead ehpc793 project (no
# longer readable), so every import of matplotlib stalls then falls back to
# /scratch/tmp. Keep caches on scratch instead.
: "${SCRATCH:?Error: SCRATCH is not set}"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Keep local third-party sources on PYTHONPATH. Resolve against this script's own
# repo so a deltatok checkout uses its vendored DA3, not another repo's copy.
# NOTE: it's important to set the PYTHONPATH to either empty or your package to shadow the default python local env
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_REPO_ROOT}/third_party/Depth-Anything-3/src"
