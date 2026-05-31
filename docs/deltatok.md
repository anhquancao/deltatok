# DeltaTok

`DeltaTok` is a tokenizer trained over per-frame DA3 features cached at layer 18 (the OccRAE caching layer — see [occrae.md](occrae.md)). It learns to predict the next frame's features from the previous frame's, with reconstruction supervised by log-cosh on the layer-18 features.

Trainer: [occrae/deltatok_trainer.py](../occrae/deltatok_trainer.py)
Entrypoint: [train_deltatok.py](../train_deltatok.py)
Shell wrapper: [sh/train_deltatok.sh](../sh/train_deltatok.sh)
Config: `configs/train_deltatok_karolina.yaml`

## Training

```bash
bash sh/train_deltatok.sh
```

Environment overrides (all have defaults in the script):

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIG_DIR` | `configs` | Hydra config directory |
| `CONFIG_NAME` | `train_deltatok_karolina` | Hydra config name |
| `LOG_AND_CKPT_DIR` | `/mnt/proj1/eu-25-92/deltatok_log` | Training root — TB logs at `<LOG_AND_CKPT_DIR>/<RUN_NAME>/tb_logs/`, checkpoints at `<LOG_AND_CKPT_DIR>/<RUN_NAME>/ckpts/` |
| `RUN_NAME` | `deltatok` | Run subdirectory under `LOG_AND_CKPT_DIR` and `RESULTS_DIR` |
| `PRECISION` | `bf16` | Mixed-precision mode |
| `OCCANY_RECON_CKPT` | unset | If set, passed as `--occany_recon_ckpt` |

Eval-only mode loads checkpoints from the same `<LOG_AND_CKPT_DIR>/<RUN_NAME>/ckpts/` directory used for training.

## Eval-only mode

```bash
EVAL_ONLY=1 bash sh/train_deltatok.sh
```

This adds `--eval-only --results-dir "$RESULTS_DIR"` to the Python call. `RESULTS_DIR` defaults to `/home/it4i-anhquan/OccAny/results`; override it (and `RUN_NAME`) to redirect outputs:

```bash
EVAL_ONLY=1 RESULTS_DIR=$(pwd)/results RUN_NAME=deltatok bash sh/train_deltatok.sh
```

### Evaluating the trained runs

Three trained runs live under `LOG_AND_CKPT_DIR=/mnt/proj1/eu-25-92/deltatok_log/`:

- `deltatok`
- `deltatok_surround`
- `deltatok_surround_constGlobalRope`

`RUN_NAME` selects which `ckpts/` directory is loaded *and* the output subdirectory under `RESULTS_DIR/<RUN_NAME>/eval_only/`. Run each from the repo root:

```bash
# deltatok
EVAL_ONLY=1 RUN_NAME=deltatok bash sh/train_deltatok.sh

# deltatok_surround
EVAL_ONLY=1 RUN_NAME=deltatok_surround bash sh/train_deltatok.sh

# deltatok_surround_constGlobalRope
EVAL_ONLY=1 RUN_NAME=deltatok_surround_constGlobalRope bash sh/train_deltatok.sh
```

Outputs land at:

```
$RESULTS_DIR/deltatok/eval_only/eval_depth/<test_name>/...
$RESULTS_DIR/deltatok_surround/eval_only/eval_depth/<test_name>/...
$RESULTS_DIR/deltatok_surround_constGlobalRope/eval_only/eval_depth/<test_name>/...
```

To redirect outputs into the repo's `results/` (matching the example path in this doc), override `RESULTS_DIR`:

```bash
EVAL_ONLY=1 RESULTS_DIR=$(pwd)/results RUN_NAME=deltatok bash sh/train_deltatok.sh
EVAL_ONLY=1 RESULTS_DIR=$(pwd)/results RUN_NAME=deltatok_surround bash sh/train_deltatok.sh
EVAL_ONLY=1 RESULTS_DIR=$(pwd)/results RUN_NAME=deltatok_surround_constGlobalRope bash sh/train_deltatok.sh
```

### SLURM (job array)

All three evals are bundled into a single Karolina job array at [slurm/karolina_eval_deltatok.slurm](../slurm/karolina_eval_deltatok.slurm) — `--array=0-2` indexes into a `RUN_NAMES` bash array, so each task evaluates exactly one run on its own node (8 GPUs).

```bash
sbatch slurm/karolina_eval_deltatok.slurm
```

Override defaults from the command line if needed (e.g. to point at a different ckpt root or output dir):

```bash
sbatch --export=ALL,LOG_AND_CKPT_DIR=/some/path,RESULTS_DIR=$(pwd)/results \
    slurm/karolina_eval_deltatok.slurm
```

Per-task logs land in `slurm/output/karolina_eval_deltatok_<JOBID>_<TASKID>.{out,err}`.

### Output layout

Eval-only visualizations are written to:

```
<RESULTS_DIR>/<RUN_NAME>/eval_only/eval_depth/<test_name>/<frame_id>_epoch<epoch>_concat.jpg
```

Example with the defaults:

```
results/deltatok/eval_only/eval_depth/206 @ Occ3dNuscenesSeqMultiView/scene-0270_001379_0_t0-8_epoch0_concat.jpg
```

Path assembly:

- `<RESULTS_DIR>/<RUN_NAME>/eval_only/` — set by [sh/train_deltatok.sh:13-15,26-28](../sh/train_deltatok.sh); becomes `eval_viz_dir` in the trainer.
- `eval_depth/<test_name>/` — appended by `_log_viz_sample` via `tb_prefix=f"eval_depth/{test_name}"` at [occrae/deltatok_trainer.py:1174](../occrae/deltatok_trainer.py).
- `<frame_id>_epoch<epoch>_concat.jpg` — final filename written at [occrae/visualization_helper.py:122-127](../occrae/visualization_helper.py).

`<test_name>` (e.g. `206 @ Occ3dNuscenesSeqMultiView`) comes from the test-loader entries in `configs/train_deltatok_karolina.yaml`. The number prefix is the per-loader item count.

Each JPEG is a multi-panel concat: ground-truth image, predicted depth (teacher-forced), and — when AR rollout is enabled — predicted RGB (TF), predicted RGB (AR), and the GT-token RGB. Column titles and panel order are set in the same call site in `deltatok_trainer.py`.
