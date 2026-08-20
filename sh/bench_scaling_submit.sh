#!/usr/bin/env bash
# Submit the EuroHPC strong/weak scaling ladder on Jean Zay (4/8/16/32 GPUs = 1/2/4/8 nodes).
# 14 jobs cover 16 table rows: the tokenizer's 16-GPU point and the flow's 8-GPU point are
# identical in both ladders (MODE=both), so each is run once.
#
# Everything runs in qos_gpu_h100-dev, which caps a user at 32 GPUs and 10 submitted jobs. Each
# model's chain is 7 jobs, so submit ONE model per invocation:
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && DRY_RUN=1 MODELS=\"tok flow\" bash sh/bench_scaling_submit.sh'"
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && MODELS=tok bash sh/bench_scaling_submit.sh'"
#   ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && MODELS=flow bash sh/bench_scaling_submit.sh'"
#
# Points run in parallel; the dev QoS's own 32-GPU-per-user cap does the throttling. No
# --dependency=singleton: measured `data:` is 0.13% of a tokenizer step and IDRIS gives each job
# exclusive nodes, so co-scheduled points do not contend. Ascending GPU order, so the first 8-GPU
# point still lands early as the 2-node smoke test.
set -euo pipefail

: "${DRY_RUN:=0}"
: "${MODELS:=tok}"              # tok | flow | "tok flow" (the last exceeds dev's 10-job cap)

# mode:ngpu:bsize:eff_bsize   (mode "both" = the point shared by the strong and weak ladders)
TOK_POINTS=(
    weak:4:2:8
    strong:4:2:32
    weak:8:2:16
    strong:8:2:32
    both:16:2:32
    weak:32:2:64
    strong:32:1:32
)
FLOW_POINTS=(
    weak:4:16:64
    strong:4:16:128
    both:8:16:128
    weak:16:16:256
    strong:16:8:128
    weak:32:16:512
    strong:32:4:128
)

submit_one() {
    local model=$1 point=$2
    IFS=: read -r mode ngpu bsize eff <<<"$point"
    local nodes=$(( ngpu / 4 ))
    # dev: 2 h wall, 32 GPUs and 10 submitted jobs per user. Every point fits (longest is the
    # tokenizer's gc=4 job at ~22 min) and an 8-node point simply waits for the GPU cap to clear.
    local qos="qos_gpu_h100-dev"

    local script name
    if [[ $model == tok ]]; then
        script=slurm/bench_scaling_deltatok_jz.slurm; name=deltatok_bench_tok
    else
        script=slurm/bench_scaling_deltatok_flow_jz.slurm; name=deltatok_bench_flow
    fi

    local args=(
        --nodes="$nodes" --ntasks-per-node=4 --gres=gpu:4
        --qos="$qos" --job-name="$name"
        --export="ALL,MODE=$mode,NGPU=$ngpu,BSIZE=$bsize,EFF_BSIZE=$eff"
        "$script"
    )
    local gc=$(( eff / (ngpu * bsize) ))
    printf '%-4s %-6s %2d nodes %2d gpu  b%-3d global %-4d gc %-2d  %s\n' \
        "$model" "$mode" "$nodes" "$ngpu" "$bsize" "$eff" "$gc" "$qos"
    if [[ $DRY_RUN != 1 ]]; then
        sbatch "${args[@]}"
    fi
}

for model in $MODELS; do
    if [[ $model == tok ]]; then points=("${TOK_POINTS[@]}"); else points=("${FLOW_POINTS[@]}"); fi
    for p in "${points[@]}"; do
        submit_one "$model" "$p"
    done
done

[[ $DRY_RUN == 1 ]] && echo "(DRY_RUN=1: nothing submitted)"
exit 0
