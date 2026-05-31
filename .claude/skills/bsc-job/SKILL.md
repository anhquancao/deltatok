---
name: bsc-job
description: Submit, monitor, and tail SLURM jobs on the BSC (MareNostrum) cluster for OccAny. Use whenever running training, evaluation, or feature extraction on BSC — the secondary GPU site, used only for jobs that explicitly target it.
---

# Running work on BSC (MareNostrum)

BSC is the secondary GPU site for OccAny. Primary GPU work goes to Karolina (`karolina-job` skill); BSC is used only for jobs that explicitly target it (`slurm/bsc_*.slurm`, `sh/*_bsc.sh`, `env_bsc.sh`). This skill captures the conventions.

## Cluster context

| Setting | Value |
|---|---|
| SSH alias (from local) | `bsc` (two-hop `ssh karolina "ssh bsc ..."` also works) |
| Account | `ehpc793` |
| Partition | `acc` |
| QoS | `acc_ehpc` (production), `acc_debug` (short/debug) |
| Repo path on cluster | `$HOME/code/OccAny` = `/home/vale/vale352205/code/OccAny` |
| Env activation | `source env_bsc.sh` (module loads + activates venv `~/envs/maskgit`) |
| Data root | `/gpfs/scratch/ehpc793/occany_data` |
| Train logs / ckpts | `/gpfs/scratch/ehpc793/tb_log_occany/<EXP_NAME>` |

> Note: many checked-in `slurm/bsc_*.slurm` scripts still carry `--account=ehpc551` (stale). The current allocation is **`ehpc793`** — prefer that.

## One-shot remote command

No conda on BSC — source `env_bsc.sh`, which `module purge`s, loads CUDA/python/etc., and activates the `~/envs/maskgit` venv.

```bash
ssh bsc "cd ~/code/OccAny && source env_bsc.sh && <command>"
```

## SLURM job submission

SLURM scripts live under `slurm/bsc_*.slurm`. They `cd "$HOME/code/OccAny"` and `source env_bsc.sh` internally.

```bash
ssh bsc "cd ~/code/OccAny && sbatch slurm/bsc_<name>.slurm"
```

Current scripts:
- `bsc_train_occany_plus_recon_1B.slurm` — main recon training (8 nodes × 4 GPU)
- `bsc_train_occany_plus_recon_dinov3_vith16plus.slurm` — DINOv3 ViT-H+/16 recon variant
- `bsc_train_occrae.slurm` / `bsc_train_occrae_overfit.slurm` — OccRAE flow-matching trainer
- `bsc_extract_occany_features.slurm` — feature extraction

### `acc` partition conventions

- `#SBATCH --account=ehpc793`, `#SBATCH --partition=acc`
- 4 GPUs/node: `--gres=gpu:4 --ntasks-per-node=4`, `--cpus-per-task=20`, often `--exclusive`
- Production: `--qos=acc_ehpc` (e.g. `--time=48:00:00`). Short/debug: `--qos=acc_debug` (`--time=2:00:00`, small node count).
- The BSC train wrappers (`sh/*_bsc.sh`) honor `NUM_NODE`, `NUM_GPU_PER_NODE`, `N_WORKERS`, `BATCH_SIZE` env overrides.

## Monitoring

```bash
# Queued/running jobs for the user
ssh bsc "squeue -u \$USER"

# Recent history with exit codes
ssh bsc "sacct -u \$USER --format=JobID,JobName,State,Elapsed,ExitCode,Start,End -X | tail -30"

# Tail a running job's log (path is the slurm script's --output= line)
ssh bsc "tail -f ~/code/OccAny/slurm/output/<name>_<jobid>.out"
```

## Data sync (Karolina → BSC)

Run **on Karolina** (it has a `bsc` ssh alias). Pushes to `/gpfs/scratch/ehpc793/occany_data`:

```bash
# verify-and-push
ssh karolina "cd ~/OccAny && bash sh/verify_karolina_to_bsc.sh"
# or sh/sync_karolina_to_bsc.sh / sh/push_data_to_bsc.sh
```

## Hard rules (from CLAUDE.md)

The cluster checkout is the user's — assume it's synced as they want. **Never on BSC:**
- `git clean -fd|-f|-x`, `git reset --hard`, `git checkout .`, `git restore .`
- `git stash drop|clear`
- `git pull`, `git fetch && git merge`, `git rebase`
- `rsync` / `scp` of source files local → cluster

If a BSC operation fails because of sync state, **stop and ask the user** — don't try to "fix" it. Read-only inspection (`git log`, `status`, `diff`, `show`, `cat`, `ls`, `squeue`) is fine.
