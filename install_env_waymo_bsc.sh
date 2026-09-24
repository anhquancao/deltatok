#!/usr/bin/env bash
# Builds $PROJECT/envs/waymo_tf for env_waymo_bsc.sh. Run from the repo root on alogin1 in a
# login shell, with the proxy up locally: ssh -N -R 15432 -o HostName=alogin1.bsc.es bsc
set -euo pipefail

proxy=socks5h://localhost:15432

: "${PROJECT:?Error: PROJECT is not set}"
venv_dir="${PROJECT}/envs/waymo_tf"
if [ ! -f "$venv_dir/bin/activate" ]; then
  module purge
  module load gcc impi mkl hdf5/1.14.1-2-gcc python/3.11.5-gcc cuda/12.3 cudnn/9.1.0-cuda12 nccl
  # System site-packages: the module exposes torch through .pth / egg-link files there.
  python3 -m venv --system-site-packages "$venv_dir"
fi

source env_waymo_bsc.sh

# pip needs PySocks to use a socks proxy, so fetch that one wheel with curl.
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
url=$(curl -sS --proxy "$proxy" https://pypi.org/pypi/PySocks/1.7.1/json \
  | python -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['filename'].endswith('py3-none-any.whl')][0])")
curl -sSL --proxy "$proxy" -o "$tmp_dir/PySocks-1.7.1-py3-none-any.whl" "$url"
pip install --no-index --no-cache-dir "$tmp_dir/PySocks-1.7.1-py3-none-any.whl"

# --no-deps: TF, numpy, protobuf come from the module. cv2 4.9 is a numpy-1.x build.
pip install --no-deps --no-cache-dir --proxy "$proxy" \
  waymo-open-dataset-tf-2-12-0==1.6.4 opencv-python-headless==4.9.0.80
