# CLAUDE.md

Guidance for Claude Code in this repo.

> **Job/cluster/project RAG:** `../monitor_jobs/` is a read-only knowledge base for SLURM job history & states, cluster/GPU metadata, compute allocations/budgets, and curated experiment results. Read its committed `data/*.json` and `data/logs/<Cluster>/<name>_<jobid>.{out,err}` directly (grep / `python -c "import json"`) — don't start its server or call its HTTP API. See `../monitor_jobs/CLAUDE.md` for schemas and the retrieval recipe.

## Workflow rules

- **Git commits:** no `Co-Authored-By` trailers.
- **Clarifying questions:** use `AskUserQuestion`, not inline text.
- **Multi-step tasks:** track with `TaskCreate`; complete one at a time.
- **Code search:** prefer `grep` over `rg`.

## Skills (`Skill` tool)

For GPU jobs use the cluster skill matching the target — `karolina-job` (primary), `bsc-job` (secondary), `jeanzay-job` (H100) — don't hand-roll `sbatch`/`squeue`. User instructions here override skill defaults.

**Always `sbatch` from a login shell, on every cluster:** wrap remote submission in `bash -lc`, e.g. `ssh <cluster> "bash -lc 'cd <repo> && sbatch <script>'"`. A bare `ssh <cluster> "sbatch ..."` runs non-login/non-interactive, so env setup (`module` on Jean Zay, `conda` on Karolina) is never initialized and only reaches the job if it happens to be inherited — that's why a job "works sometimes." The login shell exports those functions and SLURM `--export=ALL` carries them into the job.

## Critical: GPU work runs on Karolina (primary), BSC (secondary)

No local GPU. CPU-only Python is fine here. GPU work (train/eval/inference/feature extraction) goes through Karolina by default:

```bash
ssh karolina "conda activate occany && <command>"
```

Covers `train_*.py`, `extract_*.py`, `eval_occ_rae.py`, `test_occ_rae.py`, `sh/{train,eval,extract}_*.sh`. Enforced by `.claude/hooks/block-claude-md-rules.py`.

BSC is the secondary cluster, used only for jobs that explicitly target it — `slurm/bsc_*.slurm` and `sh/*_bsc.sh`. Reach it directly with `ssh bsc` (alias is configured locally; the two-hop `ssh karolina "ssh bsc ..."` also works).

## Architecture

This repo implements **DeltaTok** and **OccRAE**, a tokenization and prediction pipeline built on top of the OccAny DA3 reconstruction backbone.

### Training pipeline (3 stages)

The full pipeline trains three models in sequence:

1. **Feature extraction** (`extract_occany_features.py` via `sh/extract_occany_features.sh`) — runs the frozen OccAny DA3-Giant backbone up to layer 18 and dumps cached tokens as `.pth` files per frame. This is a one-time batch job per dataset, not a training step.

2. **Image decoder** (`train_occrae_img_decoder.py` via `sh/train_occrae_img_decoder.sh`) — trains an MAE-style decoder (`occrae/img_decoder_trainer.py`) on frozen OccRAE 3-level decode features to reconstruct RGB images. The trained checkpoint is used by DeltaTok eval to visualize predicted tokens as images.

3. **DeltaTok tokenizer** (`train_deltatok.py` via `sh/train_deltatok.sh` or `sh/train_deltatok_geom.sh`) — trains a transformer tokenizer (`occrae/deltatok_trainer.py`) over the DA3 layer-18 features to predict next-frame tokens from previous-frame tokens (log-cosh loss on layer-18 features). Optionally adds geometric supervision losses (pointmap, depth, raymap) as regularizers.

### OccRAE core (`occany/model/occ_rae.py`)

Splits the DA3-Giant backbone at layer 18 into `encode()` (layers 0–18 → cached tokens) and `decode()` (layers 19+ → depth/pointmap/ray output). Layer 18 is a local-attention layer where `x == local_x`, so only one tensor is cached (halves storage vs. global-attention layers). See `docs/occrae.md`.

### Key source files

| Component | Files |
|-----------|-------|
| **DeltaTok trainer** | `occrae/deltatok_trainer.py` |
| **OccRAE flow-matching trainer** | `occrae/occrae_trainer.py` |
| **Image decoder trainer** | `occrae/img_decoder_trainer.py` |
| **Base trainer** | `occrae/abstract_trainer.py` → `occrae/trainer.py` |
| **DeltaTok network** | `occrae/network/efficient_transformer.py`, `occrae/network/transformer_block.py` |
| **OccRAE model** | `occany/model/occ_rae.py` |
| **Dataset loaders** | `occrae/dataset/preprocessed_sequence.py`, `occrae/dataset/occrae_tokens.py` |
| **Loss / metrics** | `occrae/loss.py`, `occrae/metric.py` |
| **Visualization** | `occrae/visualization_helper.py`, `occrae/evaluation_helper.py` |
| **Feature extraction** | `extract_occany_features.py` |
| **OccRAE eval** | `eval_occ_rae.py` |
| **Roundtrip test** | `test_occ_rae.py` |
| **OccAny DA3 backbone** | `occany/model/model_da3.py`, `occany/training_da3.py` |

### Entrypoints

| Script | Launcher | Purpose |
|--------|----------|---------|
| `train_deltatok.py` | `sh/train_deltatok.sh` | DeltaTok training (token-only loss) |
| `train_deltatok.py` | `sh/train_deltatok_geom.sh` | DeltaTok + geometric supervision |
| `train_occrae.py` | `sh/train_occrae.sh` | OccRAE flow-matching training |
| `train_occrae_img_decoder.py` | `sh/train_occrae_img_decoder.sh` | Image decoder training |
| `extract_occany_features.py` | `sh/extract_occany_features.sh` | Dataset-wide layer-18 token extraction |
| `eval_occ_rae.py` | — | Evaluate trained OccRAE flow-matching model |
| `test_occ_rae.py` | — | OccRAE encode/decode roundtrip test |

### Hydra config hierarchy

Configs use Hydra `defaults` layering. Base configs define the model/training; cluster variants override only dataset roots.

```
configs/train_deltatok.yaml           ← base (model arch, training params, BSC dataset roots)
├── train_deltatok_karolina.yaml      ← Karolina dataset roots (inherits train_deltatok)
├── train_deltatok_jeanzay.yaml       ← Jean Zay dataset roots (inherits train_deltatok)
└── train_deltatok_geom.yaml          ← adds geometric loss weights (inherits train_deltatok)
    ├── train_deltatok_geom_karolina.yaml   ← Karolina roots + geom (inherits both)
    ├── train_deltatok_geom_jeanzay.yaml    ← Jean Zay roots + geom (H100: disables decode grad ckpt)
    └── train_deltatok_geom_smoke.yaml      ← smoke test (inherits geom_karolina)

configs/train_occrae.yaml             ← OccRAE flow-matching base
├── train_occrae_fm.yaml              ← flow-matching variant
│   ├── train_occrae_fm_karolina.yaml
│   └── train_occrae_fm_overfit.yaml
└── train_occrae_img_decoder.yaml     ← image decoder base
    └── train_occrae_img_decoder_karolina.yaml
```

## Training commands

### DeltaTok (token-only)

```bash
# Karolina (default)
bash sh/train_deltatok.sh

# Key env overrides:
CONFIG_NAME=train_deltatok_karolina   # Hydra config (default)
LOG_AND_CKPT_DIR=/mnt/proj1/eu-25-92/deltatok_log   # TB logs + checkpoints
RUN_NAME=deltatok_surround_constGlobalRope           # subdirectory under LOG_AND_CKPT_DIR
RESUME=1                              # resume from current.pth
```

### DeltaTok with geometric supervision

```bash
bash sh/train_deltatok_geom.sh
# Default RUN_NAME: deltatok_surround_constGlobalRope_geom_recon100_pm0p01_d0p01_ray0p01
```

Geom supervision needs `training.use_decode_grad_checkpoint: true` on A100-40GB (Karolina) but disables it on H100-80GB (Jean Zay) for speed.

### DeltaTok eval-only

```bash
EVAL_ONLY=1 RUN_NAME=deltatok bash sh/train_deltatok.sh
EVAL_ONLY=1 RUN_NAME=deltatok_surround bash sh/train_deltatok.sh
EVAL_ONLY=1 RUN_NAME=deltatok_surround_constGlobalRope bash sh/train_deltatok.sh
```

Eval outputs: `<RESULTS_DIR>/<RUN_NAME>/eval_only/eval_depth/<test_name>/<frame_id>_epoch<epoch>_concat.jpg`. SLURM job array for all 3 runs: `slurm/karolina_eval_deltatok.slurm`.

### OccRAE flow-matching

```bash
bash sh/train_occrae.sh
# CONFIG_NAME=train_occrae_fm, RESULTS_DIR=/gpfs/scratch/ehpc558/quan/occrae_output
```

### Image decoder

```bash
bash sh/train_occrae_img_decoder.sh
# CONFIG_NAME=train_occrae_img_decoder_karolina
```

### Feature extraction

```bash
bash sh/extract_occany_features.sh
# OUTPUT_DIR, PID/WORLD for sharding
```

## Checkpoint dependencies

`train_deltatok.py` resolves `model.occany_recon_ckpt` and `model.img_decoder.ckpt_path` **relative to the repo root**, so both must live under `checkpoints/` on each cluster:

- `checkpoints/occany_plus_recon_1B.pth` — frozen OccAny DA3 backbone
- `checkpoints/occrae_img_decoder.pt` — trained image decoder (copy from `occrae_output/.../current.pt`)

Set `model.img_decoder.ckpt_path: null` in config to skip RGB decode during eval.

## SLURM scripts

| Cluster | DeltaTok | DeltaTok+Geom | OccRAE | Img Decoder | Eval |
|---------|----------|---------------|--------|-------------|------|
| Karolina | `karolina_train_deltatok.slurm` | `karolina_train_deltatok_geom.slurm` | `karolina_train_occrae.slurm` | `karolina_train_occrae_img_decoder.slurm` | `karolina_eval_deltatok.slurm` |
| Jean Zay | `jz_train_deltatok.slurm` | `jz_train_deltatok_geom.slurm` | — | — | — |
| BSC | — | — | `bsc_train_occrae.slurm` | — | — |

All under `slurm/`. Smoke test: `karolina_train_deltatok_geom_smoke.slurm`.

## Cluster paths

|                       | Karolina (primary)                                                  | BSC (secondary)                                                |
| --------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| SSH (from local)      | `ssh karolina`                                                      | `ssh bsc`                                                      |
| Account / partition   | `eu-25-92` / `qgpu`                                                 | `ehpc793` / `acc` (QoS `acc_ehpc`, debug `acc_debug`)          |
| Repo checkout         | `/home/it4i-anhquan/OccAny`                                         | `/home/vale/vale352205/code/OccAny` (`$HOME/code/OccAny`)      |
| Data root             | `/scratch/project/eu-25-92/data` (= `$SCRATCH/data`)                | `/gpfs/scratch/ehpc793/occany_data`                            |
| Train logs / ckpts    | `$PROJECT/tb_log_occany/<EXP_NAME>` or `$PROJECT/deltatok_log/<RUN_NAME>` | `/gpfs/scratch/ehpc793/tb_log_occany/<EXP_NAME>`               |
| SLURM stdout / stderr | `slurm/output/*.{out,err}` in the repo                              | `slurm/output/*.{out,err}` in the repo                         |
| Env activation        | `conda activate occany`                                             | `source env_bsc.sh` (loads venv `~/envs/maskgit`)              |
| SLURM scripts         | `slurm/karolina_*.slurm`                                            | `slurm/bsc_*.slurm`                                            |

Karolina → BSC data sync: `sh/verify_karolina_to_bsc.sh` (run **on Karolina**; uses the `bsc` ssh alias on Karolina to push to `/gpfs/scratch/ehpc793/occany_data`).

Submission/monitoring conventions for each cluster live in their skills: `karolina-job`, `bsc-job`, `jeanzay-job`.

## Jean Zay (IDRIS) — H100 training + data backup

Third site. Login node `jean-zay3.idris.fr`; user `uyl37fq`. **Storage group `trg`** (where the checkout and data live) is separate from the **compute-hours project `lwy`** used for the SLURM account.

**You can `ssh jean-zay` from here** for read-only inspection (`squeue`, log tails, `ls`, `module avail`) when you need to check something. The route is a reverse SSH tunnel that cougar holds open through Karolina (ProxyJump karolina → `localhost:2222`); a `Connection refused` means the tunnel is down — see `docs/ssh_tunnel_jz.md`. On cougar the alias is `jean-zay` too.

- **Checkout:** `$TRG_WORK/code/OccAny` = `/lustre/fswork/projects/rech/trg/uyl37fq/code/OccAny` (`$TRG_WORK` = `/lustre/fswork/projects/rech/trg/uyl37fq`; HOME is `/linkhome/rech/genini01/uyl37fq`).
- **Env activation:** `source env_jz_h100.sh` — `module load arch/h100` + `pytorch-gpu/py3/2.5.0`, `PYTHONUSERBASE=$TRG_WORK/python_envs/occany`. No conda here. **In a non-interactive shell (slurm scripts, `srun bash script.sh`) `source ~/.bashrc` first** — `module` is only defined by the login profile, so `env_jz_h100.sh` fails (`module: command not found` → no torch) without it. The `slurm/jz_*.slurm` scripts already do this.
- **SLURM scripts:** `slurm/jz_*.slurm`. H100 partition (gpu_p6): `#SBATCH -C h100`, `#SBATCH --account=lwy@h100`, 4 GPUs/node (`--gres=gpu:4 --ntasks-per-node=4`, 24 cpus/task). Default QoS `qos_gpu_h100-t3` caps walltime at 20h; add `#SBATCH --qos=qos_gpu_h100-t4` for up to 100h (fewer GPUs).
- **Data roots:**
  - `/lustre/fsstor/projects/rech/trg/uyl37fq/datasets_preprocess_backup` — backup tier (fsstor), fed from Karolina.
  - `/lustre/fsn1/projects/rech/trg/uyl37fq/occany_data` — working tier (fsn1), fed from cougar. **fsn1 purges files untouched for 30 days.**
- **Sync scripts:**
  - `sh/sync_karolina_to_jeanzay.sh` — run **on Karolina**; rsyncs `$SCRATCH/data/*` to the fsstor backup through the tunnel (port 2222).
  - `sh/push_data_from_cougar_to_jeanzay.py` — run **on cougar**; parallel scene-level push of local data to fsn1 via the `jean-zay` alias (uses `--no-times --size-only` to dodge the 30-day purge).
  - `sh/pull_checkpoints_karolina_to_jeanzay.sh` — run **locally**; pulls a hardcoded checkpoint list from Karolina, then pushes to jean-zay `checkpoints/`. Karolina can also push straight through the tunnel: `rsync -e 'ssh -p 2222' <ckpt> uyl37fq@localhost:$TRG_WORK/code/OccAny/checkpoints/`.

## Critical: Never sync the cluster checkouts

`/home/it4i-anhquan/OccAny` (Karolina), `/home/vale/vale352205/code/OccAny` (BSC), and `/lustre/fswork/projects/rech/trg/uyl37fq/code/OccAny` (jean-zay) are the user's; assume they've synced what they want.

**Never on Karolina, BSC, or jean-zay:** `git clean -fd|-f|-x`, `git reset --hard`, `git checkout .`, `git restore .`, `git stash drop|clear`, `git pull|fetch+merge|rebase`, `rsync`/`scp` source local→cluster.

**Never locally without authorization:** `git push` (any form).

If a cluster op fails on sync state, **STOP and ask**. Read-only inspection is fine.

## Environment

```bash
conda activate occany    # Karolina — all you need
source env_bsc.sh        # BSC — module loads + activates ~/envs/maskgit venv
source env_jz_h100.sh    # Jean Zay — module load arch/h100 + pytorch-gpu (no conda)
```

On Karolina, `$PROJECT` (`/mnt/proj1/eu-25-92/`) and `$SCRATCH` (`/scratch/project/eu-25-92/`) are part of the contract — DeltaTok logs default to `$PROJECT/deltatok_log/`, data to `$SCRATCH/data/`.

## Key conventions

- **Training dataset strings** in configs and `sh/train_*.sh` are `eval()`'d by `occany.datasets.get_data_loader()` — must stay valid Python.
- **Vendored deps:** `third_party/{dust3r,croco/curope,Depth-Anything-3,sam3,GLD,deltatok,pyTorchChamferDistance}`. Path setup: `occany.utils.runtime_paths.prepend_vendored_import_paths()` (Python) or `occany_prepend_pythonpath` in `sh/train_common.sh`. One-time: `cd third_party/croco/models/curope && python setup.py install`.
- **Distributed launch:** all trainers support both torchrun and SLURM native (`SLURM_NTASKS` / `SLURM_PROCID`); `setup_cuda_distributed()` handles both.
- **Sharded extraction:** `--world` + `--pid` across `extract_occany_features.py`.
