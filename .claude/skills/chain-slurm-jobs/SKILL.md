---
name: chain-slurm-jobs
description: Submit N chained SLURM training jobs that each resume from the previous job's checkpoint, on Jean Zay or BSC. Use when the user asks to "run N more chained training jobs", "chain N jobs", "queue follow-on resume jobs", or otherwise wants a dependency chain of resume jobs for a long-running training run that exceeds a single walltime.
---

# Chain SLURM resume jobs

Long training runs outlive a single SLURM walltime (the Jean Zay `qos_gpu_h100-dev`
QoS caps at 2h). The pattern is to submit a chain of jobs where each one waits for
the previous to finish, then resumes from the checkpoint it left behind. The result
is uninterrupted training across many short allocations.

The mechanism is one flag on each `sbatch`:

- `--dependency=afterany:<prev_jobid>` — start only after `<prev_jobid>` ends, for
  **any** reason (success, failure, or walltime timeout). `afterany` (not `afterok`)
  is what you want: a job hitting its walltime is the normal case, and the chain must
  continue regardless.

Resuming needs no flag: training always loads
`<RESULTS_DIR>/<RUN_NAME>/ckpts/current.pth` when it exists.

## Cluster access

Chaining works on both active clusters. `module` is login-only on both, so always submit through a login shell.

| | Jean Zay | BSC |
|---|---|---|
| SSH alias | `jean-zay` | `bsc` |
| Submit | `ssh jean-zay "bash -lc '...'"` | `ssh bsc "bash -lc '...'"` |
| Working dir | `$TRG_WORK/code/deltatok` | `/gpfs/projects/ehpc1001/code/deltatok` |
| Walltime for chaining | Use QoS cap (20 h for `-t3`) | **Chain at 40 h** (not the 72 h cap — backfill rewards shorter requests) |
| Example script | `slurm/deltatok_flow/train_deltatok_flow_mix_xxl_sigreg_jz.slurm` | `slurm/deltatok/train_deltatok_compose_sigreg_bsc.slurm` |

BSC's scheduler is `sched/backfill` and walltime is absent from the priority formula. Measured: p90 queue wait 4 min at 12 h vs 8.6 h at 48 h. `training.exit_before_time_limit=true` keeps each split checkpoint-safe.

## Pre-flight checks (do these every time)

1. **Verify the cluster script matches local.** The user syncs files manually; never
   sync yourself. Compare checksums and stop if they differ — tell the user to sync first.
   ```bash
   md5sum slurm/<path>/<script>.slurm
   ssh <host> "bash -lc 'cd <working_dir> && md5sum slurm/<path>/<script>.slurm'"
   ```
2. **Check the queue for an existing tail job.** "N *more*" chained jobs should chain
   after whatever is already queued, not start a parallel chain. If a job is queued/running,
   set `prev` to its job id before the loop; if the queue is empty, the first job starts
   immediately.
   ```bash
   ssh <host> "bash -lc 'squeue -u \$USER -o \"%.10i %.22j %.9T %.11M %.18E %R\"'"
   ```
3. **Confirm a checkpoint exists** so the chain has something to resume (absent
   current.pth silently starts from scratch):
   ```bash
   ssh <host> "bash -lc 'ls -la <RESULTS_DIR>/<RUN_NAME>/ckpts/'"
   ```

## Submit the chain

Submit each job depending on the previous, capturing each job id (last field of
`Submitted batch job <id>`). Set `prev` to an existing tail job id to chain after it,
or leave it empty so the first job starts as soon as resources are free.

### Jean Zay

```bash
ssh jean-zay "bash -lc '
cd /lustre/fswork/projects/rech/trg/uyl37fq/code/deltatok || exit 1
prev=\"\"                      # set to an existing tail job id to chain after it
for i in 1 2 3 4; do          # N = number of chained jobs
  if [ -z \"\$prev\" ]; then
    out=\$(sbatch slurm/deltatok_flow/<SCRIPT>_jz.slurm)
  else
    out=\$(sbatch --dependency=afterany:\$prev slurm/deltatok_flow/<SCRIPT>_jz.slurm)
  fi
  echo \"job \$i: \$out (dep=\${prev:-none})\"
  prev=\$(echo \"\$out\" | awk \"{print \\\$NF}\")
done
'"
```

### BSC

Same pattern, different host and working dir. Chain at **40 h** walltime (see cluster access table).

```bash
ssh bsc "bash -lc '
cd /gpfs/projects/ehpc1001/code/deltatok || exit 1
prev=\"\"
for i in 1 2 3 4; do
  if [ -z \"\$prev\" ]; then
    out=\$(sbatch slurm/deltatok/<SCRIPT>_bsc.slurm)
  else
    out=\$(sbatch --dependency=afterany:\$prev slurm/deltatok/<SCRIPT>_bsc.slurm)
  fi
  echo \"job \$i: \$out (dep=\${prev:-none})\"
  prev=\$(echo \"\$out\" | awk \"{print \\\$NF}\")
done
'"
```

## Verify

Confirm each job past the first shows `afterany:<prev>(unfulfilled)` in the DEPENDENCY
column:

```bash
ssh <host> "bash -lc 'squeue -u \$USER -o \"%.10i %.22j %.9T %.11M %.18E %R\"'"
```

Report the job ids and their dependency edges back to the user as a table.

## Notes

- Every job in the chain resumes, including the first — it continues the existing run.
  To start over instead, change `RUN_NAME` or delete the run's `ckpts/`.
- To chain after a job submitted in a previous session, pass its id as the initial
  `prev`.
- `afterany` keeps the chain alive through timeouts/crashes. If you ever want the chain
  to stop on failure instead, switch to `afterok` — but that is not the default for
  walltime-bounded training.
