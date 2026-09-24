---
name: debug-queue-drip
description: Keep a batch of SLURM arms queued in the main QoS and move them one by one into the debug QoS as its slot frees, on BSC or Jean Zay. Use when the user asks to "switch jobs one by one to acc_debug", "drip the arms into the debug queue", "use the debug queue for the pending evals", or wants short jobs to jump the main queue without cancelling them.
---

# Drip arms into the debug queue

The debug QoS jumps the main queue but holds very few jobs per user. So submit every arm to the main QoS, then move one pending arm into the debug QoS each time a slot frees. Arms that start in the main queue first are left alone.

**Never `scancel` and resubmit to change QoS.** `scontrol update JobId=<id> QOS=<debug>` moves a pending job in place. It keeps the job ID, the log paths and its place in the main queue if the move is refused.

## Limits (measured 2026-09-24)

| | BSC | Jean Zay |
|---|---|---|
| Main QoS | `acc_ehpc`, priority 100, 72 h | `qos_gpu_h100-t3`, priority 50, 20 h |
| Debug QoS | `acc_debug`, priority 10000 | `qos_gpu_h100-dev`, priority 80 |
| Debug wall | 2 h | 2 h |
| Debug slots per user | **1 job** (pending or running) | 10 jobs, 32 GPUs |
| `scontrol update QOS=` by a user | works (`46485102`, `46485103`) | unverified |

When the slot is taken, BSC answers `Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)` and leaves the job untouched. A moved job's TimeLimit becomes 2:00:00 (was 1:30:00). The debug queue is not instant: `46485102` waited 12 min in `acc_debug`.

## Pre-flight

1. **Account.** The job's account must hold the debug QoS. On BSC `ehpc1001` does. `ehpc880` has only `acc_interactive` since 2026-09-24. Check with `sacctmgr -n -P show assoc user=$USER format=Account,QOS`.
2. **Walltime ≤ 2 h.** The move is refused above the cap. Submit short arms (evals, smokes) at ≤ 2 h. `LOWER_TIME=1` cuts longer limits to 2 h first. Only use it when the user asks, because it truncates a training run.
3. **Arms are PENDING in the main QoS.** List them in the order to promote.

## Run

Run from the repo root with `run_in_background: true`. The loop exits once no eligible arm is left waiting.

```bash
bash .claude/skills/debug-queue-drip/drip.sh bsc 46485102 46485103
bash .claude/skills/debug-queue-drip/drip.sh jean-zay <id> <id> ...
```

Each tick prints one line per arm (`<id> <state> <qos> <timelimit>`), then the actions, then `WAITING <n>`:

- **`MOVED <id>`**: now in the debug QoS.
- **`REFUSED <id> <msg>`**: the slot was taken or a limit was hit. It retries on the next tick.
- **`SKIP <id>`**: walltime over 2 h without `LOWER_TIME=1`. It stays in the main queue for good.

`tick.sh` is one pass on the cluster, and `drip.sh` loops it every 60 s over ssh. For a single manual move, run:

```bash
ssh bsc "bash -lc 'scontrol update JobId=<id> QOS=acc_debug'"
```

## After it exits

The loop only places jobs. It does not check that they run. Watch each moved arm until its first loss or `[Eval/` line, as CLAUDE.md requires. Report the arms that stayed in the main queue (`SKIP`, or started there first) separately.
