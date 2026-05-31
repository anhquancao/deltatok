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

Covers `train_*.py`, `extract_*.py`, `inference.py`, `test_occ_rae.py`, `infer_*.py`, `sh/{train,eval,extract}_*.sh`. Enforced by `.claude/hooks/block-claude-md-rules.py`.

BSC is the secondary cluster, used only for jobs that explicitly target it — `slurm/bsc_*.slurm` and `sh/*_bsc.sh`. Reach it directly with `ssh bsc` (alias is configured locally; the two-hop `ssh karolina "ssh bsc ..."` also works).

## Cluster paths

|                       | Karolina (primary)                                                  | BSC (secondary)                                                |
| --------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| SSH (from local)      | `ssh karolina`                                                      | `ssh bsc`                                                      |
| Account / partition   | `eu-25-92` / `qgpu`                                                 | `ehpc793` / `acc` (QoS `acc_ehpc`, debug `acc_debug`)          |
| Repo checkout         | `/home/it4i-anhquan/OccAny`                                         | `/home/vale/vale352205/code/OccAny` (`$HOME/code/OccAny`)      |
| Data root             | `/scratch/project/eu-25-92/data` (= `$SCRATCH/data`)                | `/gpfs/scratch/ehpc793/occany_data`                            |
| Train logs / ckpts    | `$PROJECT/tb_log_occany/<EXP_NAME>` (= `/mnt/proj1/eu-25-92/...`)   | `/gpfs/scratch/ehpc793/tb_log_occany/<EXP_NAME>`               |
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

On Karolina, `$PROJECT` (`/mnt/proj1/eu-25-92/`) and `$SCRATCH` (`/scratch/project/eu-25-92/`) are part of the contract — eval defaults to `$PROJECT/data/...` and `$SCRATCH/...`.

## Evaluation

Always via shell presets (`EXP_LIST` + `EXP_ID`), never ad-hoc:

```bash
EXP_LIST=occany_plus EXP_ID=1 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany_plus EXP_ID=1 bash sh/compute_metric.sh
```

New presets → `sh/exp_lists/*.sh`, not hardcoded. Recon metrics: `extract_recon.py` → `compute_recon_metrics.py --exp_dir ...`.

Training wrappers: `sh/train_*.sh`; SLURM jobs: `slurm/{karolina,bsc,jz}_*.slurm` (per target cluster).

## Architecture

Script-driven. Root entrypoints (`inference.py`, `extract_output_occany.py`, `extract_recon.py`, `infer_trajectory.py`, `test_occ_rae.py`) orchestrate; reusable logic in `occany/`.

**Model paths:**
- **OccAny+ (DA3 + SAM3, primary):** `occany/model/model_da3.py`, `da3_inference.py`, `training_da3.py`, `semantic_inference.py`.
- **OccAny (Must3R + SAM2):** `occany/model/model_must3r.py`, `must3r_inference.py`.
- **DINOv3 ViT-H+/16 (recon-only, experimental):** `occany/model/{model_dinov3.py,dinov3_backbone.py}`, `training_dinov3.py`; wrappers `sh/train_occany_plus_recon_dinov3_vith16plus.sh` + matching `.slurm`. No semantic head.

**Training launchers:** run `launch_da3.py` / `launch_dinov3.py` — never `occany/training_*.py` directly. They call `prepend_vendored_import_paths()` (required for `dust3r`/`croco`/DA3 imports).

**Eval flow:** `extract_output_occany.py` (via `sh/eval_occany.sh`) saves voxels; `compute_metrics_from_saved_voxels.py` computes IoU/SSC. Loaders built by `prepare_eval_setting()` / `prepare_metric_eval_setting()` in `occany/datasets/eval_helper.py`. Voxel grids: KITTI 256×256×32 @ 0.2m, nuScenes 200×200×16 @ 0.4m.

**Inference output contracts:** `pts3d_*.npy` (point cloud → `vis_viser.py`); `voxel_predictions.pkl` (voxels → `vis_voxel.py`, `compute_metrics_from_saved_voxels.py`).

**OccRAE / DeltaTok:** caches DA3 layer-18 tokens for downstream training — see `docs/occrae.md`, `docs/deltatok.md`. Hydra configs in `configs/`; drivers `train_occrae.py`, `train_occrae_img_decoder.py`, `train_deltatok.py`. DeltaTok configs come in per-cluster variants (`configs/train_deltatok{,_geom}_{karolina,jeanzay}.yaml`) that override only dataset roots. `train_deltatok.py` resolves `model.occany_recon_ckpt` and `model.img_decoder.ckpt_path` **relative to the repo root**, so both must live under `checkpoints/` on each cluster — copy the img_decoder's `occrae_output/.../current.pt` → `checkpoints/occrae_img_decoder.pt`. Geom supervision (`*_geom*`) needs `training.use_decode_grad_checkpoint: true` on A100-40GB (Karolina) but disables it on H100-80GB (jean-zay).

**Vendored deps:** `third_party/{dust3r,croco/curope,Depth-Anything-3,sam3,GLD,InfiniDepth,deltatok}`. Path setup: `occany.utils.runtime_paths.prepend_vendored_import_paths()` (Python) or `occany_prepend_pythonpath` in `sh/train_common.sh`. One-time: `cd third_party/croco/models/curope && python setup.py install`.

## Key conventions

- **Training dataset strings** in `sh/train_occany_plus_*.sh` are `eval()`'d by `occany.datasets.get_data_loader()` — must stay valid Python.
- **Semantic mode strings:** `<source>@<model>` (e.g. `distill@SAM3`), parsed by `occany.utils.inference_helper.parse_semantic_mode()`.
- **Output resolution** (DA3 path) is dataset-specific. Use `occany.utils.resolution.get_output_resolution()` or eval presets (nuScenes `518×294`, KITTI `518×168`).
- **Sharded processing:** `--world` + `--pid` across extraction scripts.
