---
name: bsc-job
description: Submit, monitor, and tail SLURM jobs on the BSC (MareNostrum) cluster for DeltaTok/OccRAE. Use whenever running training, evaluation, or feature extraction on BSC — one of the two active GPU sites (alongside Jean Zay).
---

# Running work on BSC (MareNostrum)

BSC is one of the two active GPU sites for DeltaTok/OccRAE (the other is Jean Zay — `jeanzay-job` skill). 4× H100-64GB per node. This skill captures BSC-specific conventions; see CLAUDE.md for shared rules.

## Cluster context

| Setting | Value |
|---|---|
| SSH alias | `bsc` |
| Accounts | `ehpc1001` and `ehpc880` (both active; check the script's `--account=` line) |
| Partition | `acc` |
| QoS | `acc_ehpc` (prod, 72 h max) / `acc_debug` (2 h) |
| Node shape | 4 H100, `--cpus-per-task=20` |
| Repo path | `/gpfs/projects/ehpc1001/code/deltatok` |
| Env activation | `source env_bsc.sh` (module loads + venv; no conda) |
| Train logs / ckpts | `$SCRATCH/deltatok_log` |

Stale accounts in older scripts: `ehpc551`, `ehpc793`. The current allocations are `ehpc1001` and `ehpc880`.

## One-shot remote command

No conda on BSC — `env_bsc.sh` runs `module purge`, loads CUDA/python, and activates a venv.

```bash
ssh bsc "cd /gpfs/projects/ehpc1001/code/deltatok && source env_bsc.sh && <command>"
```

## SLURM job submission

Scripts are `slurm/deltatok/*_bsc.slurm` (tokenizer) and `slurm/deltatok_flow/*_bsc.slurm` (flow). Each has an `archive/` for retired arms.

**Submit from a login shell** — `module` is login-only:

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/<name>_bsc.slurm'"
```

Without `bash -lc`, `module` is undefined and the job dies in ~1 s with `module: command not found`.

### `acc` partition conventions

- `#SBATCH --partition=acc`, `#SBATCH --account=ehpc1001` (or `ehpc880`)
- 4 GPUs/node: `--gres=gpu:4 --ntasks-per-node=4`, `--cpus-per-task=20`, often `--exclusive`
- Production: `--qos=acc_ehpc` (72 h max). Debug: `--qos=acc_debug` (2 h).

### Walltime and chaining

The scheduler is `sched/backfill`. Walltime does **not** enter the priority formula, so shorter requests fit more backfill gaps. Measured: p90 wait 4 min at 12 h vs 8.6 h at 48 h. Chain at **40 h** rather than the 72 h cap. Use the `chain-slurm-jobs` skill for chained resume jobs. `training.exit_before_time_limit=true` keeps each split checkpoint-safe.

## Monitoring

```bash
# Queued/running jobs
ssh bsc "squeue -u \$USER"

# Recent history with exit codes
ssh bsc "sacct -u \$USER --format=JobID,JobName,State,Elapsed,ExitCode,Start,End -X | tail -30"

# Tail a running job's log
ssh bsc "tail -f /gpfs/projects/ehpc1001/code/deltatok/slurm/output/<name>_bsc_<jobid>.out"
```

## Hard rules (from CLAUDE.md)

The cluster checkout is the user's — assume it's synced as they want. **Never on BSC:**
- `git clean -fd|-f|-x`, `git reset --hard`, `git checkout .`, `git restore .`
- `git stash drop|clear`
- `git pull`, `git fetch && git merge`, `git rebase`
- `rsync` / `scp` of source files local → cluster

If a BSC operation fails because of sync state, **stop and ask the user** — don't try to "fix" it. Read-only inspection (`git log`, `status`, `diff`, `show`, `cat`, `ls`, `squeue`) is fine.
