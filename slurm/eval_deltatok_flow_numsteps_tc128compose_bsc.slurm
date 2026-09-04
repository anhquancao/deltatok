#!/bin/bash
#SBATCH --job-name=deltatok_flow_numsteps_tc128compose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc880
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/eval_deltatok_flow_numsteps_tc128compose_bsc_%j.out
#SBATCH --error=slurm/output/eval_deltatok_flow_numsteps_tc128compose_bsc_%j.err

# Few-step sweep on the tc128mg9sigreg005compose flow arm (ep100, iter 200000):
# one full eval_one_epoch per NUM_STEPS entry, ode sampler only. Same 128 val
# seqs and same seed as the training-time eval, so steps=20 reproduces the run's
# logged LossRaymap/LossDepth/LossPointmap and anchors the 1-4 step numbers.
# steps=1 is the pure x_hat(noise, t=0) regression baseline, no ODE integration.
# Arch flags mirror train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_bsc.slurm --
# the .py's _RUN_DEFAULTS are Jean Zay paths and must all be overridden here.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

: "${CKPT:=$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit/ckpts/current.pth}"
: "${OUTPUT_DIR:=results/deltatok_flow_numsteps_tc128compose}"
: "${NUM_STEPS:=1,2,3,4,20}"
: "${STEP_MODES:=ode}"
# epoch_100 = iter 112500: the exact tokenizer the flow arm trained against. Override to read
# the same flow through a decoder-only finetune of it (encoder frozen -> same z).
: "${DELTATOK_CKPT:=$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0/ckpts/epoch_100.pth}"

CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    model.deltatok_ckpt="$DELTATOK_CKPT"
    model.deltatok.target_channels=128         # linear z_proj_down/up bottleneck
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
    training.eval_num_visualizations=0         # scalars only; 5 passes of images is wasted I/O
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
    --cfg "${CFG_ARGS[@]}" \
    ${EXTRA_ARGS:-}
