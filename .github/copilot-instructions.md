# OccRAE repository instructions

**CRITICAL: Do not run any code or scripts in this repository. This is a local environment.**

### Environment bootstrap (required)

Before running any `python ...` command in this repository, source the environment first:

```bash
source env_bsc.sh
```

### Focused smoke test

Use the checked-in OccRAE roundtrip script as the smallest test-like entrypoint:

```bash
source env_bsc.sh && python test_occ_rae.py \
  --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
  --input_dir ./demo_data/input \
  --output_dir ./demo_data/output_occ_rae
```

To compare roundtrip output against direct model inference:

```bash
source env_bsc.sh && python test_occ_rae.py \
  --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
  --input_dir ./demo_data/input \
  --output_dir ./demo_data/output_occ_rae \
  --compare
```

### Main runnable workflows

Demo inference:

```bash
source env_bsc.sh && python inference.py \
  --batch_gen_view 2 \
  --view_batch_size 2 \
  --semantic distill@SAM3 \
  --compute_segmentation_masks \
  --gen \
  -rot 30 \
  -vpi 2 \
  -fwd 5 \
  --seed_translation_distance 2 \
  --recon_conf_thres 2.0 \
  --gen_conf_thres 6.0 \
  --apply_majority_pooling \
  --model occany_da3
```

Occupancy extraction + metrics use the shell wrappers and `EXP_LIST`/`EXP_ID` presets, not ad-hoc CLI combinations:

```bash
EXP_LIST=occany_plus EXP_ID=1 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany_plus EXP_ID=1 bash sh/compute_metric.sh
```

Reconstruction metrics:

```bash
source env_bsc.sh && python extract_recon.py \
  --model occany_da3 \
  --dataset nuscenes \
  --setting surround \
  --exp_name occany_plus_recon_1B \
  --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B.pth

source env_bsc.sh && python compute_recon_metrics.py \
  --exp_dir ./outputs/occany_plus_recon_1B_occany_da3_nuscenes_surround_img512
```

Training is normally launched through the checked-in OccAny+ wrappers:

```bash
bash sh/train_occany_plus_recon.sh
bash sh/train_occany_plus_gen.sh
```

## High-level architecture

- The repo is script-driven. Root entrypoints (`inference.py`, `extract_output_occany.py`, `extract_recon.py`, `infer_trajectory.py`, `test_occ_rae.py`) orchestrate the workflows; the reusable logic lives under `occany/`.
- The primary model path in this repo is **OccAny+**: Depth Anything 3 + SAM3 (`occany.training_da3`, `occany.da3_inference`, `occany/model/model_da3.py`).
- Demo inference (`inference.py`) reads scene folders under `demo_data/input`, normalizes images through `occany.utils.inference_helper`, optionally runs semantic inference through `occany.semantic_inference`, then writes two artifact families:
  - `pts3d_*.npy` for point-cloud style outputs consumed by `vis_viser.py`
  - `voxel_predictions.pkl` for voxel outputs consumed by `vis_voxel.py` and `compute_metrics_from_saved_voxels.py`
- KITTI/nuScenes evaluation goes through `extract_output_occany.py` plus `occany.datasets.eval_helper`. `prepare_eval_setting()` builds the dataset-specific loader/reconstruction-view layout for extraction, while `prepare_metric_eval_setting()` rebuilds matching datasets for metric computation.
- The `sh/exp_lists/*.sh` files encode the benchmark presets. The shell and SLURM wrappers are the authoritative place for reproducible evaluation/training launches.
- `docs/occrae.md`, `test_occ_rae.py`, and `extract_occany_features.py` cover the OccRAE path: cache DA3 intermediate tokens at layer 18, decode them back into normal OccAny+ reconstruction outputs, or dump features dataset-wide.

## Key conventions

- For command-line code search in this repository, prefer `grep` over `rg`.
- Vendored dependencies in `third_party/` are expected to be importable from the repo checkout. Runtime entrypoints either prepend those paths inline or call `occany.utils.runtime_paths.prepend_vendored_import_paths()`. Shell training wrappers do the same via `occany_prepend_pythonpath`.
- Do not simplify the training dataset strings in `sh/train_occany_plus_*.sh`. `occany.datasets.get_data_loader()` uses `eval()` on those strings, so quoting and constructor names must stay valid Python expressions.
- Keep using the wrapper preset system for benchmark changes. Evaluation/reconstruction workflows are selected with `EXP_LIST` + `EXP_ID`; new benchmark variants should usually be added in `sh/exp_lists/*.sh` instead of hard-coding another branch in the wrapper.
- Output resolution is dataset specific on the DA3 path. Use `occany.utils.resolution.get_output_resolution()` or the existing eval presets instead of assuming raw input size; common defaults are `518x294` for nuScenes and `518x168` for KITTI.
- Semantic mode strings follow the `<source>@<model>` convention (`distill@SAM3`, `pretrained@SAM3`) and are parsed centrally by `occany.utils.inference_helper.parse_semantic_mode()`.
- Artifact formats are stable contracts across scripts:
  - `pts3d_*.npy` is the viewer contract for `vis_viser.py`
  - `voxel_predictions.pkl` is the metric/render contract for `vis_voxel.py` and `compute_metrics_from_saved_voxels.py`
- Sharded batch processing consistently uses `--world` and `--pid` across extraction scripts and dataset preprocessing.
- `PROJECT` and `SCRATCH` environment variables are part of the repo contract. Evaluation helpers default datasets under `$PROJECT/data/...` and processed artifacts/output roots under `$SCRATCH/...`; the shell wrappers rely on that layout.
