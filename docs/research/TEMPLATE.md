# <thread> — <one-line hypothesis>

File: `docs/research/<thread>/<YYYY-MM-DD>_task_<slug>.md`. Other stages are separate files: `_analysis_`, `_impl_`, `_report_` (decks end `_slides.html`), `_findings_`.

Created YYYY-MM-DD · thread `<thread>` · prior cycle: `<file>` · arms: `<run_name>` · jobs: `<Cluster>:<id>` · deck: `<file>_slides.html`

## 1 Hypothesis

One falsifiable sentence: what is predicted, on which metric, which eval set, at which epoch.

**Falsifiers.** One bullet per outcome and what each would mean.

**Not doing.** One bullet per deliberately excluded arm, with the reason.

## 2 Analysis

Why this is worth GPU hours before spending them: prior numbers with their source file, the mechanism, the
controls that already exist. Every number carries an epoch and an eval set.

## 3 Solution

Design and implementation: commit hash, script, config keys, env knobs, `RUN_NAME`s. The pre-flight grep on the
cluster copy and the exact `sbatch` line. Budget: min/ep × epochs → walltime.

## 4 Results

Matched-epoch table per eval set, Δ% against the control. Tracking table: job | arm | state | epoch | note.
Deck beside this file. Log and TB-mirror paths.

## 5 Findings

What the numbers mean, in the order of the falsifiers. What this overturns: `supersedes: <file>#<section>`.
Caveats: seeds, epochs, teacher ceilings.

## → Next hypothesis

Link to the next cycle file, or `open` plus the one-line question this leaves.
