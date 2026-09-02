---
name: research-cycle
description: Manage the DeltaTok research lifecycle — the global roadmap (docs/ROADMAP.md, the CVPR paper skeleton: contributions, method, main results, ablations, TODO queue) and the per-hypothesis cycle in docs/research (task → analysis → impl → report → findings → next task), keeping thread ledgers and docs/README.md current. Use when the user says "add a TODO", "update the roadmap", "what's next", "new hypothesis", "new task", "start a cycle", "record results", "write the findings", "close the cycle", "what is open", or asks where a doc, deck, baseline or finding should go.
---

# Research cycle

Research in this repo is a loop. The roadmap feeds it and receives what it produces:

```
docs/ROADMAP.md TODO ──► task ──► analysis ──► impl ──► report ──► findings ──► next task
        ▲                                                                │
        └──────────── done / new TODO ◄──────────────────────────────────┘
```

Three files to read before touching anything:

| File | Role |
|---|---|
| `docs/ROADMAP.md` | The paper skeleton: contributions, method, main results, ablations, TODO queue with status |
| `docs/README.md` | Index of the docs: layout, threads, open hypothesis per thread |
| `docs/research/<thread>/README.md` | Thread ledger: one row per doc, stage, question, verdict |

## Where things live

| What | Path |
|---|---|
| Thread | `docs/research/<thread>/` — `tc_width`, `sigreg`, `compose`, `pair_sampling`, `flow` |
| Any doc | `<date>_<stage>_<slug>.<ext>`, stage ∈ `task` `analysis` `impl` `report` `findings`. Flat folder, no stage subfolders. |
| Cycle file | the `task` file, from `docs/research/TEMPLATE.md`; it gains §4 and §5 as the cycle advances |
| Deck | `<date>_report_<slug>_slides.html`, beside its task file, built per the CLAUDE.md slide rules |
| Cross-thread doc | `docs/research/` root, linked from every ledger it touches |
| Not research | `docs/infra/` (runbooks, incidents), `docs/occrae/`, `docs/proposals/` |
| Job records, cached logs | `../monitor_jobs/data/{monitor_jobs,archived_jobs}.json`, `../monitor_jobs/data/logs/<Cluster>/` |

A new thread needs a `README.md` with the same table columns as the others, and a row in `docs/README.md`.

## 0. Roadmap — `docs/ROADMAP.md`

The roadmap is the CVPR paper skeleton. Edit it in place; it is the one doc that is rewritten rather than superseded. Every section states what is measured, with the doc that holds the number, and what is missing.

- **Contributions.** The claims of the paper, one bullet each: thread, what it is, the deciding number with its epoch. A contribution is added or changed when a finding supports it, never ahead of one.
- **Method.** One table row per component: code, design doc, state today.
- **Main results.** The target setting (input, training sets, eval protocol, in-domain vs OOD), where the code stands against it, the baselines table (in checkout? run?), and the main table with unmeasured cells named as such.
- **Ablations.** One row per knob: knob, measured on, result, doc. A row is filled only from a `report` or `findings` doc. The gaps paragraph below the table is part of the section.
- **TODO.** The queue. Columns `# | Item | Thread | Paper | Status`; `Paper` names the section or table the item fills. Keep the user's wording for the item; add context links, not interpretation.

Status vocabulary, the only values allowed:

| Status | Meaning |
|---|---|
| `not started` | idea only; optional context link |
| `started: <task file>` | a task file exists |
| `running: <jobs>, read at ep N: <task file>` | arms submitted |
| `done: <findings or task file>` | §5 written; one line with the deciding number |
| `dropped: <reason>` | never started, with why |

- **Add a TODO.** Next `#`, the thread it belongs to (or `cross` for a new thread), status `not started`. Do not open a task file yet.
- **Start a TODO.** Run operation 1 below, then set the row to `started:`.
- **Finish a TODO.** After operation 4, set the row to `done:`, fill or update the ablation row or main-table cell it was for, and if §5 leaves a new question, add it as a new TODO row rather than editing the old one.
- **Never delete a row.** `done` and `dropped` rows stay; they are the history of the queue.

## The five cycle operations

### 1. Open — new hypothesis

1. Pick the thread. If the question spans two, put the file at `docs/research/` root and link it from both ledgers.
2. `cp docs/research/TEMPLATE.md docs/research/<thread>/<today>_task_<slug>.md`.
3. Fill the header line and §1: one falsifiable sentence, the falsifiers, the "not doing" list.
4. Fill §2 from the prior cycle and the ledger. Every number carries epoch, eval set, and a source file or job id.
5. Add a ledger row with verdict `open`, put the question in the open-hypothesis column of `docs/README.md`, and set the roadmap row to `started:`.

One hypothesis per task file. A doc that grows a second question gets that question split into its own file, as
`tc_width/2026-09-01_task_sigreg_weight_tc512.md` was split from the Q1–Q5 tracker.

### 2. Implement — §3

Commit hash, script, config keys, env knobs, `RUN_NAME`s, the pre-flight grep on the cluster copy, the exact
`sbatch` line, and the budget in min/ep × epochs. Launching follows `bsc-job` / `jeanzay-job` / `chain-slurm-jobs`.
Never sync the cluster copy; ask the user. Record job ids in the §4 tracking table as soon as they exist, and set
the roadmap row to `running:`.

A design that is a document of its own is a separate `<date>_impl_<slug>.md`, linked from §3.

### 3. Report — §4

Read at the epoch the task file fixed, never earlier. Numbers come from the cached stdout first:

```
../monitor_jobs/data/logs/<Cluster>/<name>_<jobid>.out
epoch line:  Epoch N, Iter I, Train: … (Recon: …, SIGReg: …, Compose: …), Eval: …
eval lines:  [Eval/<n> @ <Set>] LossRecon=…, …   and   ZPartRank=… ZTotalVar=…
```

TensorBoard mirrors at `/mnt/d/tb_logs/<root>/<run>/tb_logs/` are for images or a tag missing from stdout, read only
with the seek-skip walker from CLAUDE.md.

One table per eval set at the matched epoch, Δ% against the control named in §2. Tracking table:
job | arm | state | epoch | note. Deck beside the file, verified by rendering, never by reading.

### 4. Conclude — §5 and the ledger

Write the findings in the order of the falsifiers. Then:

- A result that overturns an earlier one gets `supersedes: <file>#<section>` in §5. The earlier file and its deck are not edited.
- Set the ledger verdict to one line carrying the number that decides it.
- Update the open-hypothesis column in `docs/README.md` and set the roadmap row to `done:`.
- A finding that is measured and not derivable from code or git also gets a memory note pointing at this file.

### 5. Close — → Next hypothesis

Link the next task file, or write `open` plus the one-line question left. A left-over question also becomes a new
roadmap TODO. Open the next file with operation 1 and name this one in its `prior cycle:` header.

## Status — "what is open", "what's next"

Answer in this order, from the files, not from memory:

1. **Queue.** Roadmap rows that are `not started` or `started`.
2. **In flight.** Roadmap rows that are `running`, cross-checked against `state_base` in `../monitor_jobs/data/monitor_jobs.json`, not against the doc.
3. **Unclosed cycles.** Task files whose `## → Next hypothesis` says `open`, and tracking rows still `PENDING` / `RUNNING`:

```bash
grep -nE "^\| [0-9]+ \|" docs/ROADMAP.md
grep -l "^open" docs/research/*/*_task_*.md
grep -hE "PENDING|RUNNING" docs/research/*/*_task_*.md
```

A roadmap row and its task file disagreeing on status is itself a finding to report.

## Reading rules that have each cost a wrong conclusion

- **Matched epochs only.** A curve truncated early reads as a floor (the compose-floor artifact, 2026-08-27).
- **Eval is not comparable across `1ecb17c`.** See `pair_sampling/2026-08-02_analysis_timestep_sampling_1ecb17c.html`.
- **`PredVsGT` is teacher-ceiling-bound.** Quote `PredVsOrig` for arm-vs-arm differences.
- **A SIGReg warmup spike is not a turnover.** Check the raw `SIGReg:` term around iter 2000 before reading a divergent arm; see `sigreg/2026-07-31_findings_pool_not_weight_neutral.html`.
- **Never quote whitened participation rank.** Raw `ZPartRank` only.
- **One seed.** Say so in every §5.

## Hard rules

- Nothing runs on this machine. Numbers come from cached logs or from a cluster job.
- Never `rsync`/`scp` to a cluster. Grep the remote copy and ask the user to sync.
- A job id is not a launch. Watch until the first loss line, per CLAUDE.md.
- Old docs are filed, not rewritten. A wrong old finding is superseded, not edited. The roadmap is the one exception.
