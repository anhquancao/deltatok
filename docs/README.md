# DeltaTok docs

Research is filed by **stage**, in three folders under [`research/`](research/README.md). A doc's **thread** — the
question it belongs to — is the tag in its filename.

| Stage | File | Holds |
|---|---|---|
| `plan` | `research/plan/<date>_<thread>_<slug>.<ext>` | The hypothesis, why it is worth GPU hours, and how it is run: the patch, the arm scripts, the `sbatch` line, the job ids |
| `results` | `research/results/<date>_<thread>_<slug>.<ext>` | Matched-epoch numbers and plots against the control. Decks end `_slides.html` |
| `analysis` | `research/analysis/<date>_<thread>_<slug>.<ext>` | What it means: a mechanism, a diagnosis, or the verdict on a plan |

**Start at [`ROADMAP.md`](ROADMAP.md)**: the CVPR paper skeleton with contributions, method, main results and
ablations. The work queue is [`TODO.md`](TODO.md); closed items move to [`DONE.md`](DONE.md).

## Layout

| Folder | Holds |
|---|---|
| `research/README.md` | The open questions, and one ledger section per thread: a row per doc with its stage, question and verdict. |
| `research/{plan,results,analysis}/` | Every research doc, one folder per stage. The date sorts each folder into a timeline. |
| `research/TEMPLATE.md` | The plan template. Copy it; never start from a blank file. |
| `infra/` | Runbooks and incidents that are not research: SSH tunnel, cpu-mem overcount, fsn1 purge, multicam eval bug, the Jean Zay scaling ladder. |
| `occrae/` | OccRAE architecture note and its five 2026-05-31 plans. Only the image decoder landed. |
| `proposals/` | EuroHPC access proposals (docx/pdf). |
| `ROADMAP.md` | The paper skeleton: contributions, method, main results, ablations. Each section says what is measured and what is missing. |
| `TODO.md` | The work queue: one row per item with its thread, the paper section it fills, and its status. |
| `DONE.md` | Items closed out of `TODO.md`, newest first. A row keeps its original `#`, so old references still resolve. |
| `deltatok.md` | Eval-only launch note. |

## Threads

Six, all ledgered in [`research/README.md`](research/README.md): `tc_width` (channel and token budget),
`sigreg` (making the code spread), `compose` (additive composition), `pair_sampling` (which frame pairs the
tokenizer trains on), `flow` (flow matching over frozen delta tokens), `pointdit` (which of pointdit's recipe
choices transfer). Cross-thread docs are tagged `cross`.

## Conventions

- **One hypothesis per plan.** A doc that answers several questions (Q1…Qn) is several plans. Say so at the top and split the live one out, as `research/plan/2026-09-01_tc_width_sigreg_weight_tc512.md` was.
- **Every number** carries an epoch, an eval set and a source: job id, log path or TB run. Compare at matched epochs only.
- **Never edit a landed analysis.** Write the new one and give it `supersedes: <file>#<section>`. Old decks stay as they were.
- **Ledgers are the index.** A new doc is not done until its thread section in `research/README.md` has a row and the open-questions table there is current.

Job records and cached stdout logs are in `../monitor_jobs/data/`.
