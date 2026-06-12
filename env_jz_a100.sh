module purge

module load arch/a100

module load pytorch-gpu/py3/2.5.0



export OMP_NUM_THREADS=3
export PYTHONUSERBASE=$TRG_WORK/python_envs/occany_a100

export TORCH_CUDA_ARCH_LIST="8.0+PTX"