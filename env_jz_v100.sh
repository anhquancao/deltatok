module purge

unset PYTHONUSERBASE
module load pytorch-gpu/py3/2.5.0



export OMP_NUM_THREADS=3

export TORCH_CUDA_ARCH_LIST="7.0+PTX"