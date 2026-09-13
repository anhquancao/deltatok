#!/bin/bash
#SBATCH --job-name=deltatok_flow_zshaped
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc880
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/eval_deltatok_flow_zshaped_bsc_%j.out
#SBATCH --error=slurm/output/eval_deltatok_flow_zshaped_bsc_%j.err

# Σ_z-shaped noise ladder (matched MSEToken) + flow error spectrum per eigen-direction. --z_basis
# banks the Waymo GT-z covariance first. TC/CKPT/DELTATOK_CKPT select the arm; SHAPED=false = plain pass.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/eval_deltatok_flow_zshaped_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

: "${CKPT:=$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc512mg9sigreg002compose_dndetach_ep100tok_xxl_dit/ckpts/iter_100000.pth}"
: "${OUTPUT_DIR:=results/deltatok_flow_zshaped/tc512detach_ep100}"
: "${NUM_STEPS:=1}"
: "${STEP_MODES:=ode}"
# >0 writes decoded frame panels under $OUTPUT_DIR/eval_viz; pair with EXTRA_ARGS="--viz_rgb"
: "${NUM_VIZ:=0}"
: "${NOISE_SIGMAS:=0.32,0.55,0.82}"
: "${SHAPED:=true}"
: "${TC:=512}"
# Detached arm's epoch_100, the flow's tokenizer; override for epoch_80 or the clean twin.
: "${DELTATOK_CKPT:=$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8_detach/ckpts/epoch_100.pth}"

CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    model.deltatok_ckpt="$DELTATOK_CKPT"
    model.deltatok.target_channels=${TC}       # linear z_proj_down/up bottleneck
    model.deltatok.z_norm=false                # no final z LayerNorm
    model.deltatok.norm_affine=false           # flow cfg declares true
    model.deltatok.use_camera_rope=false       # layer-12 no-camera-rope arm
    model.whiten_stats=null                    # SIGReg arm: z already unit-scale
    model.cond_mode=delta_ctx_cross            # clean in-seq context + frame-0 cross-attn condition
    model.num_ctx_deltas=2                     # frames 0-2 given, 3-4 forecast
    model.attn_mode=global                     # one attention over all delta tokens
    model.vit_use_camera_embed=true            # per-camera identity in the flow ViT
    model.loss_mode=v                          # v-MSE, matching training
    model.vit_size=xxlarge_d20                 # width 1536, depth 20
    model.vit_dit_adaln=true                   # per-block AdaLN-Zero + Fourier t-embed
    model.t_dist=uniform                       # t ~ U(0,1)
    model.t_shared=true                        # one t per sample, all slots
    model.sampler_scheduler_mode=linear        # uniform-in-t grid, matching the train prior
    model.sampler_alpha=0                      # no per-frame stagger
    training.val_bsize=4                       # BSC eval OOM at bsize 16
    training.eval_noise_probe_shaped=${SHAPED}   # Σ_z-shaped probe noise
    training.eval_num_visualizations=${NUM_VIZ}  # 0 = scalars only; >0 also writes decoded panels
)

echo "CKPT=$CKPT"
echo "DELTATOK_CKPT=$DELTATOK_CKPT"

srun python eval_deltatok_flow_sampler.py \
    --config-dir configs/deltatok_flow \
    --config-name train_deltatok_flow_waymo_bsc \
    --ckpt "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_steps "$NUM_STEPS" \
    --step_modes "$STEP_MODES" \
    --z_basis \
    --noise_sigmas "$NOISE_SIGMAS" \
    --cfg "${CFG_ARGS[@]}" \
    ${EXTRA_ARGS:-}
