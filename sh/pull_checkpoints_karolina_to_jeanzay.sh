#!/usr/bin/env bash
# Pull model checkpoints from Karolina to THIS machine, then push them on to
# jean-zay. Run LOCALLY. Needs ssh aliases `karolina` and `jean-zay`
# (jean-zay rides cougar's reverse tunnel through Karolina — see
# docs/ssh_tunnel_jz.md). fswork is persistent, so no 30-day-purge workaround.
#
# The jean-zay deltatok config resolves checkpoints relative to the checkout's
# checkpoints/ dir, so every entry in CHECKPOINTS must already live under
# checkpoints/ on Karolina. The img_decoder ckpt normally sits in
# occrae_output/...; copy it into checkpoints/ on Karolina FIRST:
#
#   ssh karolina 'cp /mnt/proj1/eu-25-92/occrae_output/occrae_img_decoder/ckpts/current.pt \
#                    /home/it4i-anhquan/OccAny/checkpoints/occrae_img_decoder.pt'
#
# Usage:
#   bash sh/pull_checkpoints_karolina_to_jeanzay.sh                          # all
#   bash sh/pull_checkpoints_karolina_to_jeanzay.sh occany_plus_recon_1B.pth # subset
#   DRY_RUN=1 bash sh/pull_checkpoints_karolina_to_jeanzay.sh                # preview
#   LOCAL_STAGE=/data/ckpt_stage bash sh/pull_checkpoints_karolina_to_jeanzay.sh
set -euo pipefail

KAROLINA_HOST="karolina"
JEANZAY_HOST="jean-zay"

KAROLINA_CKPT_DIR="/home/it4i-anhquan/OccAny/checkpoints"
JEANZAY_CKPT_DIR="/lustre/fswork/projects/rech/trg/uyl37fq/code/OccAny/checkpoints"
LOCAL_STAGE="${LOCAL_STAGE:-$HOME/occany_ckpt_stage}"

# --- checkpoints to transfer (filenames under checkpoints/ on BOTH ends) ---
CHECKPOINTS=(
    occany_plus_recon_1B.pth   # model.occany_recon_ckpt        (~7.5G)
    occrae_img_decoder.pt      # model.img_decoder.ckpt_path    (~6.5G, copied from occrae_output)
)
# Optional CLI override: pass a subset of the names above.
[ "$#" -gt 0 ] && CHECKPOINTS=("$@")

# Karolina sshd offers aes256-gcm (AES-NI accelerated) — fastest for bulk.
SSH_KAROLINA="ssh -c aes256-gcm@openssh.com,aes256-ctr -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
SSH_JEANZAY="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

RSYNC_OPTS=(-aH --partial --inplace --info=progress2,stats2)
[ "${DRY_RUN:-0}" = "1" ] && RSYNC_OPTS+=(--dry-run)

mkdir -p "$LOCAL_STAGE"
echo "==> staging dir: $LOCAL_STAGE"
echo "==> checkpoints: ${CHECKPOINTS[*]}"

# 1) Karolina -> local stage
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "==> pull  $KAROLINA_HOST:$KAROLINA_CKPT_DIR/$ckpt"
    rsync "${RSYNC_OPTS[@]}" -e "$SSH_KAROLINA" \
        "$KAROLINA_HOST:$KAROLINA_CKPT_DIR/$ckpt" "$LOCAL_STAGE/"
done

# 2) local stage -> jean-zay
# shellcheck disable=SC2086
$SSH_JEANZAY "$JEANZAY_HOST" "mkdir -p '$JEANZAY_CKPT_DIR'"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "==> push  $JEANZAY_HOST:$JEANZAY_CKPT_DIR/$ckpt"
    rsync "${RSYNC_OPTS[@]}" -e "$SSH_JEANZAY" \
        "$LOCAL_STAGE/$ckpt" "$JEANZAY_HOST:$JEANZAY_CKPT_DIR/"
done

echo "==> done — ${#CHECKPOINTS[@]} checkpoint(s) under $JEANZAY_HOST:$JEANZAY_CKPT_DIR"
