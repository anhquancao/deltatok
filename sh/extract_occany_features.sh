#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Extract OccAny backbone tokens on training datasets.
#
# Environment variables:
#   OCCANY_RECON_CKPT  – path to OccAny recon checkpoint
#   OUTPUT_DIR         – where .pth files are written
#   PID / WORLD        – optional sharding overrides
# ---------------------------------------------------------------------------

OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/ehpc558/quan/occrae_emb}"
PID="${SLURM_ARRAY_TASK_ID:-0}"
WORLD="${WORLD:-1}"


python extract_occany_features.py \
    --output_dir "$OUTPUT_DIR" \
    --pid "$PID" \
    --world "$WORLD"
