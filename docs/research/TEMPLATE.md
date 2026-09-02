# <thread> — <one-line hypothesis>

File: `docs/research/plan/<YYYY-MM-DD>_<thread>_<slug>.md`. This is the `plan`. Its numbers go in a
`results/<date>_<thread>_<slug>.<ext>`; its verdict goes in a `analysis/<date>_<thread>_<slug>.md`. No numbers here except prior
ones that motivate the run.

Created YYYY-MM-DD · thread `<thread>` · prior cycle: `<file>` · arms: `<run_name>` · control: `<run_name>`
· jobs: `_pending_` · deck: `_pending_`

## 1 Hypothesis

One falsifiable sentence: what is predicted, on which metric, which eval set, at which epoch.

**Falsifiers.** One bullet per outcome and what each would mean.

**Not doing.** One bullet per deliberately excluded arm, with the reason.

## 2 Why it is worth the GPU hours

Prior numbers with their source file, the mechanism, and the controls that already exist. Every number carries an
epoch and an eval set.

## 3 How it is run

The patch, the arm and control `RUN_NAME`s, the script overrides, the pre-flight grep, the `sbatch` line, the
budget, and the job ids once submitted.

## 4 Outcome

`_pending_`, then links to the `results` and `analysis` files with the one-line verdict that decides the cycle.
