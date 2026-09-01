#!/bin/bash
#SBATCH --job-name=deltatok_flow_numsteps_tc128compose_viz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --account=ehpc880
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --output=slurm/output/eval_deltatok_flow_numsteps_tc128compose_viz_bsc_%j.out
#SBATCH --error=slurm/output/eval_deltatok_flow_numsteps_tc128compose_viz_bsc_%j.err

# Panels-only companion to eval_deltatok_flow_numsteps_tc128compose_bsc.slurm:
# same ckpt, same NUM_STEPS, but decodes depth+RGB jpgs instead of full metrics.
# Only the first batch is visualized, so eval_num_items is cut to 8 -- the printed
# scalars are NOT the sweep's numbers, use the scalar run for those. shuffle=False
# pins the same 4 val seqs in every pass, so the panels are directly comparable.
# --viz_rgb keeps the MAE image decoder loaded (the scalar run drops it).
# Arch flags mirror train_deltatok_flow_waymo_xxl_tc128mg9_sigreg005_compose_bsc.slurm --
# the .py's _RUN_DEFAULTS are Jean Zay paths and must all be overridden here.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/eval_deltatok_flow_numsteps_tc128compose_viz_bsc.slurm'"

mkdir -p slurm/output

cd /gpfs/projects/ehpc1001/code/deltatok

source env_bsc.sh

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# file_descriptor shm strategy holds an fd per shared tensor per worker
ulimit -n "$(ulimit -Hn)"

: "${CKPT:=$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit/ckpts/current.pth}"
: "${OUTPUT_DIR:=results/deltatok_flow_numsteps_tc128compose_viz}"
: "${NUM_STEPS:=1,2,3,4,20}"
: "${STEP_MODES:=ode}"

CFG_ARGS=(
    model.deltatok.num_delta_tokens=64         # K=64 delta tokens/camera; must match the frozen ckpt
    # epoch_100 = iter 112500: the exact tokenizer the flow arm trained against.
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0/ckpts/epoch_100.pth"
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
    training.eval_num_visualizations=4         # 4 panels = the whole first val batch
    training.eval_num_items=8                  # 2 batches; only the first one is drawn
)

srun python eval_deltatok_flow_sampler.py \
    --config-dir configs/deltatok_flow \
    --config-name train_deltatok_flow_waymo_bsc \
    --ckpt "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_steps "$NUM_STEPS" \
    --step_modes "$STEP_MODES" \
    --viz_rgb \
    --cfg "${CFG_ARGS[@]}" \
    ${EXTRA_ARGS:-}
