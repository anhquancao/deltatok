---
name: karolina-job
description: Run, submit, monitor, and tail SLURM jobs on the Karolina cluster for OccAny. Use whenever executing training, evaluation, inference, or feature extraction that needs GPUs — all real compute happens on Karolina, never locally.
---

# Running work on Karolina

Karolina is the **primary** GPU site for OccAny — most Python and GPU work runs here, not locally. This skill captures the conventions. The secondary/tertiary sites have their own skills: `bsc-job` (BSC, secondary) and `jeanzay-job` (Jean Zay H100, tertiary).

## Cluster context

| Setting | Value |
|---|---|
| SSH alias | `karolina` |
| Account | `eu-25-92` |
| Partition | `qgpu` |
| Repo path on cluster | `/home/it4i-anhquan/OccAny` |
| Conda env | `occany` |

For jobs that explicitly target BSC (`env_bsc.sh`, `slurm/bsc_*.slurm`, `sh/*_bsc.sh`) or Jean Zay (`env_jz_*.sh`, `slurm/jz_*.slurm`), use the `bsc-job` / `jeanzay-job` skill instead.

## One-shot remote command

Always activate the conda env. `PROJECT` and `SCRATCH` env vars are part of the repo contract (datasets default to `$PROJECT/data/...`, processed artifacts to `$SCRATCH/...`).

```bash
ssh karolina "cd ~/OccAny && conda activate occany && <command>"
```

Examples:

```bash
# Reconstruction smoke test
ssh karolina "cd ~/OccAny && conda activate occany && python test_occ_rae.py --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth --input_dir ./demo_data/input --output_dir ./demo_data/output_occ_rae"

# Evaluation
ssh karolina "cd ~/OccAny && conda activate occany && EXP_LIST=occany_plus EXP_ID=1 bash sh/eval_occany.sh"
```

## Quick GPU one-shot (`srun` on `qgpu_exp`)

For a fast experiment that needs a real GPU but isn't worth an `sbatch` script (a smoke test, a one-off feature dump, debugging a script), run it through `srun` on the express partition. This is the non-interactive equivalent of the user's `qgpu_exp_alloc` alias:

```bash
# alias on Karolina:
# alias qgpu_exp_alloc='salloc -A $MY_PROJ -p qgpu_exp --gpus=1 -t 1:00:00 -c 16'

ssh karolina "cd ~/OccAny && srun -A eu-25-92 -p qgpu_exp --gpus=1 -t 1:00:00 -c 16 \
  bash -lc 'conda activate occany && python <script>.py <args>'"
```

- `qgpu_exp` caps walltime at **1 hour** — use the regular `qgpu` partition (via `sbatch`) for anything longer.
- `--gpus=1 -c 16` mirrors the alias; bump if the job needs more.
- `bash -lc` is needed so `.bashrc` is sourced inside the srun shell and `conda activate` works. Without `-lc`, `conda` is not on PATH.
- `srun` blocks until the job finishes and streams stdout/stderr back over ssh, so you see output live — handy for debugging. Use `sbatch` instead when you want to detach.

## SLURM job submission

SLURM scripts live under `slurm/karolina_*.slurm` (NOT `slurm/bsc_*.slurm`).

```bash
ssh karolina "cd ~/OccAny && sbatch slurm/karolina_<name>.slurm"
```

Common ones:
- `karolina_train_occany_plus_recon_1B.slurm` — main recon training
- `karolina_train_occany_plus_recon_1B_infinite_depth.slurm` — infinite-depth variant
- `karolina_train_occrae.slurm` — OccRAE flow-matching trainer
- `karolina_train_occrae_img_decoder.slurm` — MAE-style image decoder
- `karolina_train_deltatok.slurm` — DeltaTok
- `karolina_extract_*.slurm` / `karolina_eval_*.slurm` — extraction + eval

## Monitoring

```bash
# Currently queued/running jobs for the user
ssh karolina "squeue -u \$USER"

# Recent history with exit codes
ssh karolina "sacct -u \$USER --format=JobID,JobName,State,Elapsed,ExitCode,Start,End -X | tail -30"

# Tail a running job's log (path is set by the slurm script's --output= line)
ssh karolina "tail -f ~/OccAny/slurm/output/slurm-<jobid>.out"
```

When a job behaves oddly, the `slurm-log-summarizer` subagent (`.claude/agents/slurm-log-summarizer.md`) can do the round-trip in one call and return a tight summary.

## Hard rules (from CLAUDE.md)

These are enforced by the `.claude/hooks/block-claude-md-rules.py` hook, but worth knowing:

**Never run on Karolina via ssh:**
- `git clean -fd|-f|-x`, `git reset --hard`, `git checkout .`, `git restore .`
- `git stash drop|clear`
- `git pull`, `git fetch && git merge`, `git rebase`

**Never run locally without explicit user authorization:**
- `python ...` (training/eval/inference must go through Karolina)
- `rsync` / `scp` of source files local → karolina (the user syncs manually)
- `git push` to any remote

If a Karolina operation fails because of sync state, **stop and ask the user** — don't try to "fix" it. Read-only inspection (`git log`, `status`, `diff`, `show`, `cat`, `ls`) is fine.
