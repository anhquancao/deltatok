#!/bin/bash
#SBATCH --job-name=extract_deltatok_flow_zcache_tokcompose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm/output/extract_deltatok_flow_zcache_tokcompose_bsc_%A_%a.out
#SBATCH --error=slurm/output/extract_deltatok_flow_zcache_tokcompose_bsc_%A_%a.err

# Flow z-cache: GT delta tokens per frame pair (4 heights) + c2w per frame, all-data train set.
# Tokenizer flags mirror slurm/probe_flow_motion_bsc.slurm.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/extract_deltatok_flow_zcache_tokcompose_bsc.slurm'"
# Smoke: sbatch --array=0 --qos=acc_debug --time=02:00:00 --export=ALL,WORLD=200,VERIFY=32 <this>

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

: "${OUT_ROOT:=$SCRATCH/quan/deltatok_flow_zcache/l12_dtok64_tc1536_sigreg0.02_seed2_ls0.1_3pairs_compose_ep100}"
: "${WORLD:=20}"
: "${VERIFY:=0}"

CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    # epoch_100 of the compose triplet tokenizer (BSC:47053827).
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.02_ns3072_pool24576_compose1.0_sw0_seed2_ls0.1_3pairs/ckpts/epoch_100.pth"
    model.deltatok.target_channels=1536        # == hidden size, no z bottleneck
    model.deltatok.z_norm=false                # no final z LayerNorm
    model.deltatok.norm_affine=false           # flow cfg declares true
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm
    model.whiten_stats=null                    # SIGReg arm: z already unit-scale
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 cross-attn condition
    model.num_ctx_deltas=1                     # frames 0-1 given, 2-9 forecast
    model.attn_mode=global                     # one attention over all delta tokens
    model.vit_use_camera_embed=true            # per-camera identity in the flow ViT
    model.vit_size=xxlarge_d20                 # width 1536, depth 20
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed
)

srun python extract_deltatok_flow_zcache.py \
    --config-dir configs/deltatok_flow \
    --config-name train_deltatok_flow_alldata_ctx2fwd8_bsc \
    --out_root "$OUT_ROOT" \
    --heights 280,266,210,168 \
    --world "$WORLD" \
    --pid "${SLURM_ARRAY_TASK_ID:-0}" \
    --num_workers 16 \
    --verify "$VERIFY" \
    --cfg "${CFG_ARGS[@]}" \
    ${EXTRA_ARGS:-}
