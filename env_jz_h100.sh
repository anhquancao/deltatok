module purge

module load arch/h100
module load pytorch-gpu/py3/2.5.0

export OMP_NUM_THREADS=3
export PYTHONUSERBASE=$TRG_WORK/python_envs/occany
export TORCH_CUDA_ARCH_LIST="9.0+PTX"

echo "Environment for H100 set:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "PYTHONUSERBASE=$PYTHONUSERBASE"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"