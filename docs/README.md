# DeltaTok docs

Research runs as a loop. One turn of the loop is one **cycle file**; cycles chain inside a **thread**.

**Start at [`ROADMAP.md`](ROADMAP.md)**: the CVPR paper skeleton with contributions, method, main results, ablations, and the TODO queue.

```
hypothesis → analysis → solution & implementation → results → findings → next hypothesis
```

## Layout

| Folder | Holds |
|---|---|
| `research/<thread>/` | Every doc of the thread, named `<date>_<stage>_<slug>.<ext>`. The stage says what the file is; the date sorts the folder into a timeline. |
| `research/<thread>/README.md` | The thread ledger: one row per doc with its stage, question and verdict. |
| `research/TEMPLATE.md` | The cycle template. Copy it; never start from a blank file. |
| `research/*.html` | Cross-thread docs: the 2026-08-18 task backlog and the 2026-09-02 four-week review. |
| `infra/` | Runbooks and incidents that are not research: SSH tunnel, cpu-mem overcount, fsn1 purge, multicam eval bug, the Jean Zay scaling ladder. |
| `occrae/` | OccRAE architecture note and its five 2026-05-31 plans. Only the image decoder landed. |
| `proposals/` | EuroHPC access proposals (docx/pdf). |
| `ROADMAP.md` | The paper skeleton: contributions, method, main results, ablations, TODO queue. Each section says what is measured and what is missing. |
| `deltatok.md` | Eval-only launch note. |

## Threads

| Thread | Question the thread is about | Open hypothesis (2026-09-02) |
|---|---|---|
| [`tc_width`](research/tc_width/README.md) | How many channels and tokens the delta code needs, and what sets its usable rank | Where `sigreg_weight` turns over at tc512 and whether the optimum scales with Cz. Arms BSC:45296347–49 pending, read at ep 67. |
| [`sigreg`](research/sigreg/README.md) | Making the code spread: estimator, pool, weight, geometry | Shares the `weight ∝ Cz` question with tc_width. |
| [`compose`](research/compose/README.md) | Additive composition of delta tokens | Composed ≠ autoregressive. Whether a short-schedule plain arm buys composability for free. |
| [`pair_sampling`](research/pair_sampling/README.md) | Which frame pairs and gaps the tokenizer trains on | None open. |
| [`flow`](research/flow/README.md) | Flow matching over frozen delta tokens | t≈0 starvation under `loss_mode=v`, and the compose-arm diagnostic in the 2026-09-01 numsteps analysis. |

## Conventions

- **One hypothesis per cycle file.** A doc that answers several questions (Q1…Qn) is several cycles. Say so at the top and split the live one out, as `tc_width/2026-09-01_task_sigreg_weight_tc512.md` was.
- **Names.** `<YYYY-MM-DD>_<stage>_<slug>.<ext>`, stage one of: `task` (a hypothesis and the arms that test it; the cycle file, which gains results and findings as it advances), `analysis` (mechanism or diagnosis), `impl` (design and implementation), `report` (numbers and plots; decks end in `_slides.html`), `findings` (what a result means). The date is when the work started.
- **Every number** carries an epoch, an eval set and a source: job id, log path or TB run. Compare at matched epochs only.
- **Never edit an old finding.** Write the new one and give it `supersedes: <file>#<section>`. Old decks stay as they were.
- **Cross-thread work** goes at `research/` root, and every thread it touches links it from its ledger.
- **Ledgers are the index.** A new doc is not done until its thread `README.md` has a row and this file's open-hypothesis column is current.

Manage the loop with the `research-cycle` skill. Job records and cached stdout logs are in `../monitor_jobs/data/`.
