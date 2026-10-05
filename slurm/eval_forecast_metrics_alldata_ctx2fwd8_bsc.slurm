#!/bin/bash
#SBATCH --job-name=eval_forecast_metrics_alldata_ctx2fwd8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=03:00:00
#SBATCH --output=slurm/output/eval_forecast_metrics_alldata_ctx2fwd8_bsc_%j.out
#SBATCH --error=slurm/output/eval_forecast_metrics_alldata_ctx2fwd8_bsc_%j.err

# Score one set of dumped forecasts (eval_vggt_world.py / eval_gen3r.py), 1 GPU.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch --export=ALL,SET=kitti,PRED_DIR=<run dir> slurm/eval_forecast_metrics_alldata_ctx2fwd8_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

case "${SET:-}" in
    kitti|nuscenes|waymotest) ;;
    *) echo "SET must be kitti | nuscenes | waymotest, got '${SET:-}'" >&2; exit 1 ;;
esac
: "${CONFIG_NAME:=eval_deltatok_flow_alldata_ctx2fwd8_${SET}_bsc}"
: "${NUM_ITEMS:=1000000}"  # above any split: window_stride sets each set's size
[ -d "${PRED_DIR:-}" ] || { echo "PRED_DIR must be a dump run dir, got '${PRED_DIR:-}'" >&2; exit 1; }

srun python eval_forecast_metrics.py \
    --config-dir configs/deltatok_flow \
    --config-name "$CONFIG_NAME" \
    --num_items "$NUM_ITEMS" \
    --pred_dir "$PRED_DIR" \
    ${EXTRA_ARGS:-}
