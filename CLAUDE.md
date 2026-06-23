# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git commits

Never add `Co-Authored-By` trailers to commit messages.

## Asking questions

Always use the `AskUserQuestion` tool when you need to ask the user clarifying questions or offer them choices. Do not ask questions inline in plain text.

## Task tracking

Use the `TaskCreate` tool to track progress whenever a task requires more than one step. Mark each task completed as soon as it is done; do not batch updates.

## Critical: Do not run code locally

This is a local environment — do not run any code or scripts directly here. To test or run commands, SSH to Karolina first:

```bash
ssh karolina "<command>"
```

Cluster context: project work runs on **Karolina** (account `eu-25-92`, partition `qgpu`). The previous BSC setup is deprecated.

## Environment bootstrap

Before any `python ...` command on Karolina, activate the conda env:

```bash
conda activate occany
```

Do NOT `source env_bsc.sh` on Karolina. It is the deprecated BSC script (it `module purge`es, activates a non-existent `maskgit` venv, and clobbers `PYTHONPATH`), which breaks the `occany` env (e.g. `transformers` import fails on a missing `httpx`). Clean `occany` already has everything (transformers, httpx, ...); vendored `third_party/` paths come from entrypoints / `occany.utils.runtime_paths.prepend_vendored_import_paths()`.

`PROJECT` and `SCRATCH` env vars are part of the repo contract — evaluation helpers default datasets under `$PROJECT/data/...` and processed artifacts under `$SCRATCH/...`.

## Common commands

**Smoke test (OccRAE roundtrip):**
```bash
conda activate occany && python test_occ_rae.py \
  --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
  --input_dir ./demo_data/input \
  --output_dir ./demo_data/output_occ_rae
```

Add `--compare` to compare roundtrip output against direct model inference.

**Demo inference:**
```bash
conda activate occany && python inference.py \
  --batch_gen_view 2 --view_batch_size 2 \
  --semantic distill@SAM3 --compute_segmentation_masks \
  --gen -rot 30 -vpi 2 -fwd 5 \
  --seed_translation_distance 2 \
  --recon_conf_thres 2.0 --gen_conf_thres 6.0 \
  --apply_majority_pooling \
  --model occany_da3
```

**Evaluation (use shell presets — not ad-hoc CLI):**
```bash
EXP_LIST=occany_plus EXP_ID=1 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany_plus EXP_ID=1 bash sh/compute_metric.sh
```

**Reconstruction metrics:**
```bash
conda activate occany && python extract_recon.py \
  --model occany_da3 --dataset nuscenes --setting surround \
  --exp_name occany_plus_recon_1B \
  --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B.pth

conda activate occany && python compute_recon_metrics.py \
  --exp_dir ./outputs/occany_plus_recon_1B_occany_da3_nuscenes_surround_img512
```

**Training:**
```bash
bash sh/train_occany_plus_recon.sh        # Stage 1: reconstruction
bash sh/train_occany_plus_gen.sh          # Stage 2: novel-view rendering
bash sh/train_occany_plus_recon_1B.sh     # 1.1B variant
bash sh/train_deltatok_flow.sh            # OccRAE flow matching
bash sh/train_occrae_img_decoder.sh       # OccRAE image decoder (MAE-style)
bash sh/train_deltatok.sh                 # DeltaTok tokenizer
```

**SLURM:**
```bash
sbatch slurm/train_occany_plus.slurm
sbatch slurm/eval_occany.slurm
```

When submitting chained training jobs (a dependency chain of resume jobs for a long run), use the `chain-slurm-jobs` skill rather than hand-rolling the submission.

## Critical: Verify cluster files before submitting SLURM jobs

Before submitting any SLURM job on a cluster, always verify that the script on the cluster matches the local version. The user syncs files manually — never assume a local edit has been pushed, and never sync files to the cluster yourself (no scp/rsync); always ask the user to sync. Check with e.g.:
```bash
ssh karolina "bash -lc 'grep -E \"partition|time\" ~/deltatok/slurm/<script>.slurm'"
```
If the file is stale, tell the user to sync before submitting.

## Architecture

The repo is **script-driven**. Root entrypoints (`inference.py`, `extract_output_occany.py`, `extract_recon.py`, `infer_trajectory.py`, `test_occ_rae.py`) orchestrate workflows; reusable logic lives under `occany/`.

### Primary model path: OccAny+

**OccAny+** (Depth Anything 3 + SAM3) is the primary model path:
- `occany/model/model_da3.py` — DA3Wrapper extending DepthAnything3
- `occany/da3_inference.py` — reconstruction inference
- `occany/training_da3.py` — training logic
- `occany/semantic_inference.py` — SAM2/SAM3 semantic segmentation

The secondary path is **OccAny** (Must3R + SAM2): `occany/model/model_must3r.py` and `occany/must3r_inference.py`.

### Inference flow

`inference.py` reads `demo_data/input/`, normalizes via `occany.utils.image_util.ImgNorm`, calls reconstruction (DA3 or Must3R), optionally renders novel views and runs semantic inference, then fuses into a voxel grid. Outputs two stable artifact families:
- `pts3d_*.npy` — point-cloud contract for `vis_viser.py`
- `voxel_predictions.pkl` — voxel contract for `vis_voxel.py` and `compute_metrics_from_saved_voxels.py`

### Evaluation flow (two stages)

1. `extract_output_occany.py` (wrapped by `sh/eval_occany.sh`) — runs inference, saves voxels; `prepare_eval_setting()` in `occany/datasets/eval_helper.py` builds dataset-specific loaders.
2. `compute_metrics_from_saved_voxels.py` — loads saved voxels, computes IoU/SSC metrics; `prepare_metric_eval_setting()` rebuilds matching datasets.

KITTI/nuScenes voxel grids differ: KITTI is 256×256×32 at 0.2 m; nuScenes is 200×200×16 at 0.4 m.

### OccRAE (token caching)

Caches DA3 intermediate tokens at **layer 18** (local-attention layer) to enable downstream flow-matching training and feature reuse. See `docs/occrae.md`. Components:
- `extract_occany_features.py` — dump tokens dataset-wide
- `occrae/deltatok_flow_trainer.py` — flow-matching trainer
- `occrae/img_decoder_trainer.py` — MAE-style image decoder trainer on frozen OccRAE 3-level features
- `occrae/deltatok_trainer.py` — DeltaTok tokenizer trainer over layer-18 frame features
- `occrae/dataset/occrae_tokens.py` — OccRAETokenDataset

Training entrypoints: `train_deltatok_flow.py` (flow matching), `train_occrae_img_decoder.py` (image decoder), `train_deltatok.py` (DeltaTok). All use Hydra configs under `configs/` and support SLURM distributed training.

### Vendored third-party dependencies

`third_party/` (dust3r, croco/curope, Depth-Anything-3, sam3, Grounded-SAM-2) must be importable from the checkout. Entrypoints prepend these paths inline or via `occany.utils.runtime_paths.prepend_vendored_import_paths()`. Shell wrappers do it via `occany_prepend_pythonpath`.

The `curope` C++ extension must be compiled once:
```bash
cd third_party/croco/models/curope && python setup.py install
```

## Tooling conventions

- **Clarifying questions:** when the request is ambiguous or has multiple reasonable interpretations, use the `AskUserQuestion` tool to confirm before acting.
- **Task tracking:** any task with more than one step must be tracked with the `TodoWrite` tool. Create todos up front and mark each one completed as soon as it's done.
- **Edits:** surgical and explainable — change only what the task requires (no drive-by refactors, renames, or reformatting), and explain each edit so a human can verify it easily.
- **Comments:** be terse. Prefer the fewest words that convey the *why*, not the *what*; default to a single short line. No long prose blocks or multi-line docstring essays — this applies to shell/slurm/config comments too (one short line beats a multi-line block). The inline tensor shape annotations below are the only expected multi-part comments.
- **New variant scripts:** when a new script is a variant of an existing one (e.g. a new `visualize_*`/`test_*` entrypoint), copy the closest existing file first (`cp old.py new.py`) and apply surgical edits to it — do not rewrite from scratch. This keeps the shared structure identical and makes the diff reviewable.
- **Tensor code comments:** comment each line of tensor-manipulation code, and always annotate the resulting tensor shape inline, e.g. `x = rearrange(x, 'b (t s) d -> (b s) t d', t=t, s=s)  # (B*S, T, D)`. Also state what each shape symbol means when it first appears.

## Key conventions

- **Code search:** prefer `grep` over `rg` in this repo.
- **Benchmark presets:** evaluation/training is selected via `EXP_LIST` + `EXP_ID`; new benchmark variants belong in `sh/exp_lists/*.sh`, not hardcoded branches in wrappers.
- **Training dataset strings** in `sh/train_occany_plus_*.sh` are passed to `occany.datasets.get_data_loader()` via `eval()` — do not simplify them; quoting and constructor names must stay valid Python expressions.
- **Semantic mode strings** follow `<source>@<model>` (e.g., `distill@SAM3`, `pretrained@SAM3`), parsed centrally by `occany.utils.inference_helper.parse_semantic_mode()`.
- **Output resolution** is dataset-specific on the DA3 path. Use `occany.utils.resolution.get_output_resolution()` or existing eval presets — common defaults are `518×294` for nuScenes and `518×168` for KITTI.
- **Sharded batch processing** uses `--world` and `--pid` consistently across extraction scripts and dataset preprocessing.
