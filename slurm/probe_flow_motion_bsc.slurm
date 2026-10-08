#!/bin/bash
#SBATCH --job-name=probe_flow_motion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc1001
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:30:00
#SBATCH --output=slurm/output/probe_flow_motion_bsc_%j.out
#SBATCH --error=slurm/output/probe_flow_motion_bsc_%j.err

# Motion probe, all-data ctx2fwd8 flow arm: decode forecast deltas 6 ways, fit cameras.
# Arch flags mirror slurm/deltatok_flow/train_deltatok_flow_alldata_xxl_tc1536mg9_sigreg002seed1_ctx2fwd8_bsc.slurm.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/probe_flow_motion_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

: "${CKPT:=$SCRATCH/quan/deltatok_flow_log/deltatok_flow_alldata_consec10cam0_ctx2fwd8_tc1536mg9sigreg002compose_seed1_ep70tok_xxl_dit/ckpts/iter_050000.pth}"
: "${OUTPUT_DIR:=results/probe_flow_motion}"

CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    # epoch_70: the tokenizer this arm trained against.
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.02_ns3072_pool24576_compose1.0_decnoise0_sw0_seed1_rclip0/ckpts/epoch_70.pth"
    model.deltatok.target_channels=1536        # == hidden size, no z bottleneck
    model.deltatok.z_norm=false                # no final z LayerNorm
    model.deltatok.norm_affine=false           # flow cfg declares true
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm
    model.whiten_stats=null                    # SIGReg arm: z already unit-scale
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 cross-attn condition
    model.num_ctx_deltas=1                     # frames 0-1 given, 2-9 forecast
    model.attn_mode=global                     # one attention over all delta tokens
    model.vit_use_camera_embed=true            # per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE, matching training
    model.vit_size=xxlarge_d20                 # width 1536, depth 20
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed
    model.t_dist=uniform                       # t ~ U(0,1)
    model.t_shared=true                        # one t per sample, all slots
    model.sampler_scheduler_mode=linear        # uniform-in-t grid, matching the train prior
    model.sampler_alpha=0                      # no per-frame stagger
    training.bsize=4                           # probe loaders use bsize for train and val
)
CFG_ARGS+=(${EXTRA_CFG:-})  # e.g. EXTRA_CFG="model.pose_cond=true" for the oracle arm

srun python probe_flow_motion.py \
    --config-dir configs/deltatok_flow \
    --config-name train_deltatok_flow_alldata_ctx2fwd8_bsc \
    --ckpt "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_steps 20 \
    --num_batches 16 \
    --splits train,val \
    --cfg "${CFG_ARGS[@]}" \
    ${EXTRA_ARGS:-}
