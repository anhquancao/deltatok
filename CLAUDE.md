# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git commits

Never add `Co-Authored-By` trailers to commit messages.

## Asking questions

Always use the `AskUserQuestion` tool when you need to ask the user clarifying questions or offer them choices. Do not ask questions inline in plain text.

## Task tracking

Use the `TaskCreate` tool to track progress whenever a task requires more than one step. Mark each task completed as soon as it is done; do not batch updates.

## Critical: Do not run code locally

This is a local environment — do not run any code or scripts directly here. To test or run commands, SSH to a cluster first:

```bash
ssh jean-zay "<command>"     # or: ssh bsc "<command>"
```

Cluster context: project work runs on **Jean Zay** and **BSC (MareNostrum)**. Karolina is deprecated.

| | Jean Zay | BSC |
|---|---|---|
| SSH alias | `jean-zay` (reverse tunnel — see `jeanzay-karolina-tunnel` skill if refused) | `bsc` |
| Account | `trg@h100` | `ehpc1001` |
| Partition / QoS | `gpu_p6` via `-C h100`; `qos_gpu_h100-t3` (20 h) / `-t4` (100 h) / `-dev` (2 h) | `acc`; `acc_ehpc` (prod, 72 h) / `acc_debug` (2 h) |
| Node shape | 4 GPU, `--cpus-per-task=24` | 4 H100, `--cpus-per-task=20` |
| Repo path | `$TRG_WORK/code/deltatok` | `/gpfs/projects/ehpc1001/code/deltatok` |
| Env | `source env_jz_h100.sh` | `source env_bsc.sh` |
| Scripts | `slurm/jz_*.slurm` | `slurm/bsc_*.slurm` |

Many checked-in scripts carry **stale accounts** (`zuw@h100`, `cya@h100`, `lwy@h100`; `ehpc551`). New and edited scripts must use `trg@h100` / `ehpc1001`.

## Job monitor / RAG knowledge base

`../monitor_jobs` (`/home/acao/code/monitor_jobs`) tracks this repo's SLURM jobs and doubles as a structured knowledge base. **Read its `data/*.json` directly — do not start its server or call its HTTP API.** Full schemas: the "RAG knowledge base" section of `../monitor_jobs/CLAUDE.md`.

- `data/monitor_jobs.json`, `data/archived_jobs.json` — job records keyed `"<Cluster>:<JobID>"` (state, script, log, account, qos, tags, `depends_on`). Search both for full history.
- `data/logs/<Cluster>/<name>_<jobid>.{out,err}` — cached SLURM stdout/stderr; grep here for "why did job N fail".
- `data/research.json` — curated HTML journal entries (observation/finding/resolution) per research direction.
- `data/results.json` — curated benchmark/experiment tables.
- `data/projects.json` — the `Deltatok` entry owns the per-cluster deploy paths plus the rsync `excludes` that `syncer.py` actually uses (`sync/push_code_*` is the legacy equivalent). New local output dirs must be excluded here **and** in `.gitignore`.

## Environment bootstrap

Source the cluster's own env script before any `python ...` — there is no conda on either site:

```bash
ssh jean-zay "cd \$TRG_WORK/code/deltatok && source env_jz_h100.sh && <command>"
ssh bsc "cd /gpfs/projects/ehpc1001/code/deltatok && source env_bsc.sh && <command>"
```

Jean Zay has `env_jz_{h100,a100,v100}.sh`; default to H100 unless the job targets another partition. Never cross them — `env_bsc.sh` `module purge`es and activates a venv that does not exist on Jean Zay. `conda activate occany` belongs to the deprecated Karolina setup only.

Submit from a **login shell** (`ssh <host> "bash -lc '... sbatch ...'"`): `module` is defined only by the login profile and `--export=ALL` will not carry it into the job otherwise. Symptom is a ~1 s job death with `module: command not found` in `*.err` then `ModuleNotFoundError: No module named 'torch'`, while `*.out` still prints the env script's trailing echoes.

`PROJECT` and `SCRATCH` env vars are part of the repo contract — evaluation helpers default datasets under `$PROJECT/data/...` and processed artifacts under `$SCRATCH/...`.

## Common commands

**Smoke test (OccRAE roundtrip):**
```bash
source env_jz_h100.sh && python test_occ_rae.py \
  --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
  --input_dir ./demo_data/input \
  --output_dir ./demo_data/output_occ_rae
```

Add `--compare` to compare roundtrip output against direct model inference.

**Demo inference:**
```bash
source env_jz_h100.sh && python inference.py \
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
source env_jz_h100.sh && python extract_recon.py \
  --model occany_da3 --dataset nuscenes --setting surround \
  --exp_name occany_plus_recon_1B \
  --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B.pth

source env_jz_h100.sh && python compute_recon_metrics.py \
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

**SLURM:** (always via `bash -lc` — see Environment bootstrap)
```bash
ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/jz_<name>.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/bsc_<name>.slurm'"
```

`RESUME` defaults to `0` in the `jz_*`/`bsc_*` train scripts, so a plain relaunch starts fresh and clobbers `ckpts/current.pth` (only one backup deep). Pass `--export=ALL,RESUME=1` to continue a run. Un-prefixed `slurm/*.slurm` (e.g. `train_occany_plus.slurm`) are Karolina-era and still `conda activate`.

When submitting chained training jobs (a dependency chain of resume jobs for a long run), use the `chain-slurm-jobs` skill rather than hand-rolling the submission.

On BSC, prefer chained 24 h jobs over one long job: the scheduler is `sched/backfill` (`bf_window=4320`) and walltime does **not** enter the priority formula, so a shorter request only ever fits more gaps — measured p90 queue wait was 4 min at 12 h vs 8.6 h at 48 h. `training.exit_before_time_limit=true` makes the split checkpoint-safe.

## Critical: Verify cluster files before submitting SLURM jobs

Before submitting any SLURM job on a cluster, always verify that the script on the cluster matches the local version. The user syncs files manually — never assume a local edit has been pushed, and never sync files to the cluster yourself (no scp/rsync); always ask the user to sync. Check with e.g.:
```bash
ssh jean-zay "bash -lc 'grep -E \"account|partition|time\" \$TRG_WORK/code/deltatok/slurm/<script>.slurm'"
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
- `occrae/deltatok_shared.py`, `occrae/network/` — shared blocks + tokenizer/flow nets
- `occrae/sigreg.py`, `occrae/z_spread.py` — SIGReg latent regulariser + z-spread diagnostics

Training entrypoints: `train_deltatok_flow.py` (flow matching), `train_occrae_img_decoder.py` (image decoder), `train_deltatok.py` (DeltaTok). All use Hydra configs under `configs/` and support SLURM distributed training.

Configs are **per-cluster** — `configs/train_deltatok{,_flow}_{jeanzay,bsc,karolina}.yaml`, picked by `CONFIG_NAME` in the slurm script. Edit the one matching the cluster you submit to.

DeltaTok is the active research direction. Read before proposing changes: `docs/deltatok.md`, `docs/deltatok_flow_*.md`, `docs/sigreg_*.md`, dated findings in `docs/journal/*.html`, and design docs in `docs/plans/*.md`.

### Vendored third-party dependencies

`third_party/` (dust3r, croco/curope, Depth-Anything-3, sam3, Grounded-SAM-2) must be importable from the checkout. Entrypoints prepend these paths inline or via `occany.utils.runtime_paths.prepend_vendored_import_paths()`. Shell wrappers do it via `occany_prepend_pythonpath`.

The `curope` C++ extension must be compiled once:
```bash
cd third_party/croco/models/curope && python setup.py install
```

## Tooling conventions

- **Clarifying questions / task tracking:** see "Asking questions" and "Task tracking" above — `AskUserQuestion` for ambiguity, `TaskCreate` for anything multi-step.
- **Edits:** surgical and explainable — change only what the task requires (no drive-by refactors, renames, or reformatting), and explain each edit so a human can verify it easily.
- **Comments:** be terse. Prefer the fewest words that convey the *why*, not the *what*; default to a single short line. No long prose blocks or multi-line docstring essays — this applies to shell/slurm/config comments too (one short line beats a multi-line block). The inline tensor shape annotations below are the only expected multi-part comments.
  - **Editing an existing comment must not lengthen it.** When revising a comment block, the replacement must be no longer than what it replaces — if a new finding needs recording, cut a stale line to make room. Comment blocks accrete otherwise. Long-form rationale (measurements, dated findings, sweep history) belongs in `docs/journal/*.html` or `../monitor_jobs` `data/research.json`, not in a slurm/config header.
- **New variant scripts:** when a new script is a variant of an existing one (e.g. a new `visualize_*`/`test_*` entrypoint), copy the closest existing file first (`cp old.py new.py`) and apply surgical edits to it — do not rewrite from scratch. This keeps the shared structure identical and makes the diff reviewable.
- **Tensor code comments:** comment each line of tensor-manipulation code, and always annotate the resulting tensor shape inline, e.g. `x = rearrange(x, 'b (t s) d -> (b s) t d', t=t, s=s)  # (B*S, T, D)`. Also state what each shape symbol means when it first appears.

## Key conventions

- **Code search:** prefer `grep` over `rg` in this repo.
- **Benchmark presets:** evaluation/training is selected via `EXP_LIST` + `EXP_ID`; new benchmark variants belong in `sh/exp_lists/*.sh`, not hardcoded branches in wrappers.
- **Training dataset strings** in `sh/train_occany_plus_*.sh` are passed to `occany.datasets.get_data_loader()` via `eval()` — do not simplify them; quoting and constructor names must stay valid Python expressions.
- **Semantic mode strings** follow `<source>@<model>` (e.g., `distill@SAM3`, `pretrained@SAM3`), parsed centrally by `occany.utils.inference_helper.parse_semantic_mode()`.
- **Output resolution** is dataset-specific on the DA3 path. Use `occany.utils.resolution.get_output_resolution()` or existing eval presets — common defaults are `518×294` for nuScenes and `518×168` for KITTI.
- **Sharded batch processing** uses `--world` and `--pid` consistently across extraction scripts and dataset preprocessing.
