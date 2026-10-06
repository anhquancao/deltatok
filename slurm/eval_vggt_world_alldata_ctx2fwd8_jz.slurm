#!/bin/bash
#SBATCH --job-name=eval_vggt_world_alldata_ctx2fwd8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH -C h100
#SBATCH --account=trg@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=16:00:00
#SBATCH --output=slurm/output/eval_vggt_world_alldata_ctx2fwd8_jz_%j.out
#SBATCH --error=slurm/output/eval_vggt_world_alldata_ctx2fwd8_jz_%j.err

# Dump VGGT-World forecasts for one ctx2fwd8 eval set, 1 GPU.
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch --export=ALL,SET=kitti slurm/eval_vggt_world_alldata_ctx2fwd8_jz.slurm'"

mkdir -p slurm/output

cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok

source env_jz_h100.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export HF_HOME=/lustre/fswork/projects/rech/trg/uyl37fq/.cache/huggingface   # DA3METRIC-LARGE cached here
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

case "${SET:-}" in
    kitti|nuscenes) ;;  # no Waymo test data on JZ
    *) echo "SET must be kitti | nuscenes, got '${SET:-}'" >&2; exit 1 ;;
esac
: "${CONFIG_NAME:=eval_deltatok_flow_alldata_ctx2fwd8_${SET}_jeanzay}"
: "${CKPT:=/lustre/fsn1/projects/rech/trg/uyl37fq/vggt_world/kitti.pt}"
: "${ROLLOUT:=stride1}"
: "${RESOLUTION:=native}"
: "${NUM_ITEMS:=1000000}"  # above any split: window_stride sets each set's size
: "${OUTPUT_DIR:=/lustre/fsn1/projects/rech/trg/uyl37fq/forecast_preds/vggt_world_alldata_ctx2fwd8}"

srun python eval_vggt_world.py \
    --config-dir configs/deltatok_flow \
    --config-name "$CONFIG_NAME" \
    --num_items "$NUM_ITEMS" \
    --ckpt "$CKPT" \
    --rollout "$ROLLOUT" \
    --resolution "$RESOLUTION" \
    --output_dir "$OUTPUT_DIR" \
    ${EXTRA_ARGS:-}
