---
name: jeanzay-job
description: Submit, monitor, and tail SLURM jobs on the Jean Zay (IDRIS) H100 cluster for DeltaTok/OccRAE. Use whenever running training, evaluation, or feature extraction on Jean Zay — one of the two active GPU sites (alongside BSC).
---

# Running work on Jean Zay (IDRIS)

Jean Zay is one of the two active GPU sites for DeltaTok/OccRAE (the other is BSC — `bsc-job` skill). 4× H100-80GB per node. Also hosts a data-backup tier. This skill captures JZ-specific conventions; see CLAUDE.md for shared rules.

## Cluster context

| Setting | Value |
|---|---|
| SSH alias (from local & cougar) | `jean-zay` |
| Login node | `jean-zay3.idris.fr`, user `uyl37fq` |
| Compute account | `trg@h100` (H100), partition `gpu_p6` (`-C h100`) |
| Storage group | `trg` (checkout + data live here — **separate** from the `lwy` compute allocation) |
| Repo path on cluster | `$TRG_WORK/code/deltatok` = `/lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok` |
| Env activation | `source env_jz_h100.sh` (no conda — `module load arch/h100` + `pytorch-gpu/py3/2.5.0`) |

`$TRG_WORK` = `/lustre/fswork/projects/rech/trg/uyl37fq`; HOME is `/linkhome/rech/genini01/uyl37fq`.

## The SSH route is a reverse tunnel — it may be down

`ssh jean-zay` from here works through a reverse SSH tunnel that **cougar** holds open through Karolina (ProxyJump karolina → `localhost:2222`). It is reliable for **read-only inspection** (`squeue`, log tails, `ls`, `module avail`) and for `sbatch` submission. If you get `Connection refused`, the tunnel is down — see `docs/ssh_tunnel_jz.md` (cougar must re-open it with `autossh`). On cougar the alias is `jean-zay` too.

## One-shot remote command

No conda — source the H100 env. Storage group `trg` vs compute account `lwy` is intentional; don't "fix" it.

```bash
ssh jean-zay "cd \$TRG_WORK/code/deltatok && source env_jz_h100.sh && <command>"
```

There are also `env_jz_a100.sh` and `env_jz_v100.sh` for the other partitions; default to H100 (`env_jz_h100.sh`) unless the job targets A100/V100.

## SLURM job submission

SLURM scripts live under `slurm/*_jz.slurm`; DeltaTok training is under `slurm/deltatok/` and `slurm/deltatok_flow/` (each with an `archive/`). They `source env_jz_h100.sh` and `cd` to the fswork checkout internally.

**Always submit from a login shell — wrap the remote command in `bash -lc`:**

```bash
ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && sbatch slurm/<name>_jz.slurm'"
```

A bare `ssh jean-zay "... sbatch ..."` runs **non-login**, so the `module` shell function (defined only by the login profile) is never initialized in the submission shell and does **not** propagate into the job via `--export=ALL`. The batch script's `source ~/.bashrc` does **not** rescue this — `.bashrc` early-returns for non-interactive shells, so `module` stays undefined. Symptom: the job starts then dies in ~1s with `module: command not found` (in `*.err`) followed by `ModuleNotFoundError: No module named 'torch'` — `env_jz_h100.sh`'s trailing `echo`s still print to `*.out`, so the OUT log looks like the env was set even though no module loaded. Fix: resubmit with `bash -lc`.

Current scripts:
- `slurm/deltatok/train_deltatok_multitoken_sigreg_nozn_jz.slurm` — DeltaTok tokenizer (multitoken, SIGReg, no z-norm)
- `slurm/deltatok/train_deltatok_stage2_bottleneck_jz.slurm` — DeltaTok stage-2 bottleneck
- `slurm/deltatok_flow/train_deltatok_flow_mix_xxl_sigreg_jz.slurm` — flow matching (mix dataset, SIGReg)
- `slurm/deltatok_flow/train_deltatok_flow_once_xxl_sigreg_jz.slurm` — flow matching (once dataset)
- `slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_sigreg_jz.slurm` — flow matching (Waymo)

### H100 partition (gpu_p6) conventions

- `#SBATCH -C h100`, `#SBATCH --account=trg@h100`
- 4 GPUs/node: `--gres=gpu:4 --ntasks-per-node=4`, `--cpus-per-task=24` (96 cores/node), `--hint=nomultithread`
- Default QoS `qos_gpu_h100-t3` caps walltime at **20h**. For up to **100h** (fewer GPUs) add `#SBATCH --qos=qos_gpu_h100-t4`.
- Logs/ckpts go to the persistent **fswork** tier (e.g. `$TRG_WORK/deltatok_log`), NOT the working **fsn1** tier — `fsn1` purges files untouched for 30 days.

## Monitoring

```bash
# Queued/running jobs for the user
ssh jean-zay "squeue -u \$USER"

# Recent history with exit codes
ssh jean-zay "sacct -u \$USER --format=JobID,JobName,State,Elapsed,ExitCode,Start,End -X | tail -30"

# Tail a running job's log (path is the slurm script's --output= line, under the fswork checkout)
ssh jean-zay "tail -f \$TRG_WORK/code/deltatok/slurm/output/<name>_<jobid>.out"
```

## Data tiers and sync

- `/lustre/fsstor/projects/rech/trg/uyl37fq/datasets_preprocess_backup` — **backup** tier (fsstor), fed from Karolina via `sh/sync_karolina_to_jeanzay.sh` (run **on Karolina**; rsyncs through the tunnel on port 2222).
- `/lustre/fsn1/projects/rech/trg/uyl37fq/occany_dataset` — **working** tier (fsn1), fed from cougar via `sh/push_data_from_cougar_to_jeanzay.py` (run **on cougar**; uses `--no-times --size-only` to dodge the 30-day purge). **fsn1 purges files untouched for 30 days.**
- Checkpoints: `sh/pull_checkpoints_karolina_to_jeanzay.sh` (run **locally**), or Karolina pushes straight through the tunnel:
  `rsync -e 'ssh -p 2222' <ckpt> uyl37fq@localhost:$TRG_WORK/code/deltatok/checkpoints/`

## Hard rules (from CLAUDE.md)

The cluster checkout is the user's — assume it's synced as they want. **Never on Jean Zay:**
- `git clean -fd|-f|-x`, `git reset --hard`, `git checkout .`, `git restore .`
- `git stash drop|clear`
- `git pull`, `git fetch && git merge`, `git rebase`
- `rsync` / `scp` of source files local → cluster

If a Jean Zay operation fails because of sync state, **stop and ask the user** — don't try to "fix" it. Read-only inspection (`git log`, `status`, `diff`, `show`, `cat`, `ls`, `squeue`) is fine.
