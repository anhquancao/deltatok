# DeltaTok STATUS — the jobs board

What ran, what it reached, and what still has to be checked. One row per job that is queued, running, or finished
but not yet read into a `results/` doc. [`02-09-2026.md`](02-09-2026.md) keeps the *items*; this file keeps the
*jobs*. A row leaves once its read has landed and the TODO row is closed into [`../DONE.md`](../DONE.md).
Job states come from `../../monitor_jobs/data/monitor_jobs.json` (read the file, never the server); epochs come from
the cached `logs/BSC/*.out`. Every row says what to grep and where the number goes.

**As of 2026-09-06 23:32.**

## Queued and running

| Job | Arm | TODO | State | Check when it runs |
|---|---|---|---|---|
| BSC:45498520 | tc512 · sigreg 0.02 · compose 1.0 · `cov_weight=1e-4` | 6 | PENDING (Priority) since 23:28, 44 h, `ehpc1001` | startup print `cov_weight=0.0001` — a silent `0.0` is a stale trainer — and `Cov:` on the epoch line. ep-12 tripwire: train `ZPartRank` ≥ 65 and recon within 10% of BSC:45416718, else kill |
| BSC:45498521 | same, `cov_weight=3e-4` | 6 | PENDING (Priority) since 23:28, 44 h, `ehpc1001` | same, print `cov_weight=0.0003`. The hot rung: ~115% of the total loss at ep 4, so this is the one likely to trip the ep-12 wire |

Both are the `cov_weight` dose-response above BSC:45416718, which saturated `ZPartRank` at 121/512 with
`ZTotalVar` **5.6% past the 512 target and still rising** (540.7 at ep 72, up monotonically from 457.6 at ep 0,
crossing 512 at ep 17) — so any further `L_cov` drop is rank, not the trace correction that took 68% of
the first arm's. Added shape push over SIGReg 0.02 alone is ~17× at 1e-4 and ~52× at 3e-4, against ~5× at 3e-5
(κ ≈ 0.074 from `../research/plan/2026-09-02_sigreg_cov_penalty.md` §2, both terms sharing the trainer's
`scale` = 17.0). Scripts `slurm/deltatok/train_deltatok_compose_sigreg_covpen{1e-4,3e-4}_nozn_tc512_bsc.slurm`.

## Finished, read landed, TODO row still open

| Job | Arm | TODO | State | Read | Left to do |
|---|---|---|---|---|---|
| BSC:45296347 | tc512 · sigreg 0.02 · compose 1.0 | 2 | COMPLETED at ep 81/100, stopped by the 48 h wall, resumable | ep 67 and ep 81 in `../research/results/2026-09-04_tc_width_tc512_sigreg_weight_axis_slides.html` | Fill §4 of `../research/plan/2026-09-01_tc_width_sigreg_weight_tc512.md`; close TODO 2. It is the twin for TODO 6 and 8, so keep its `current.pth` |
| BSC:45297731 | tc512 · sigreg 0.04 | 2 | COMPLETED at ep 81/100 on the wall. Resubmit of BSC:45296348, which died at startup (`CUDA error: out of memory` in `set_device` on `as06r3b02`) | same deck | none |
| BSC:45296349 | tc512 · sigreg 0.08 | 2 | CANCELLED 2026-09-02 at ep 32, resumable | same deck: over-regularised, rank lower not higher | none; do not resume |
| BSC:45345063 | tc512 · sigreg 0.02 · `sigreg_compose_z` (sum) | 8 | CANCELLED 2026-09-04 15:46 mid-ep 67; last eval ep 66 | ep 65 in the same deck: the sum's 0.005 win reverses to +8.0% eval `LossRecon` at 0.02 | Fill §4 of `../research/plan/2026-09-02_sigreg_sum_at_weight_0.02.md`; close TODO 8 |
| BSC:45344713 | flow · pointditT (`logitnormal(-0.8,0.8)` + 10% `t=0`) from scratch | 7 | CANCELLED 2026-09-04 at ep 58 | ep 50 via evals BSC:45414635 / 45414644: `../research/results/2026-09-04_pointdit_lowt_numsteps_ep50_slides.html`, loses 19/20 cells | Remaining pointdit arms gated on TODO 11 |
| BSC:45417908 | flow · decoder noise probe (σ 0/0.32/0.55/0.82 on iter_100000, N=1) | 12 | COMPLETED 2026-09-04 16:20 | `../research/results/2026-09-04_flow_decoder_noise_probe.md` | none; it motivates TODO 12 |
| BSC:45416718 | tc512 · sigreg 0.02 · compose 1.0 · `cov_weight=3e-5` | 6 | COMPLETED at ep 72/100 on the 44 h wall, `current.pth` resumable | ep 67 in `../research/results/2026-09-06_sigreg_cov_penalty_tc512_slides.html`; plan §4–5 filled | Close TODO 6. Follow-on decided 2026-09-06: the cov dose-response BSC:45498520 / 45498521 above. **`sigreg 0.01 + cov 3e-5` is dropped, not deferred** — trace headroom is spent, so more `cov_weight` buys rank directly, and clearing the 0.0528 K / 0.0370 N axis best by a real margin settles the plateau confound without a second base. Keep `current.pth` for the optional ep-100 resume |
| BSC:45421190 + 45497311/19 + 45497830/31 | tc128 compose · decoder-only noise finetune `decode_noise_tau=0.8` | 12 | all COMPLETED; finetune reached ep 10/10, `epoch_5` and `epoch_10` both read | plan §4 filled and `../research/results/2026-09-06_flow_decoder_noise_finetune_slides.html` | Close TODO 12. Falsifier 1 fired at N=20 (best raymap −17.6%, matched −35.5%), falsifier 2 at N=1 (+17.7%, structured error). Adopt the finetuned `epoch_10` as the tokenizer for flow reads; re-read `2026-09-04_flow_bestofk_regressor_null.md` on it |

## Planned, not submitted

| TODO | What | Plan | Needs before `sbatch` | Check when it runs |
|---|---|---|---|---|
| 11 | best-of-K + K-spread evals on existing ckpts (minutes each) + one 72 h `train_fixed_t=0` regressor read at `iter_100000` | `../research/plan/2026-09-04_flow_bestofk_regressor_null.md` §3 | the eval flags and the regressor knob in §3; sync, then grep the remote file | falsifiers in §1 |

## Plans whose §4 is still `_Pending_` with data already on disk

- `../research/plan/2026-09-01_tc_width_sigreg_weight_tc512.md` — data in the 2026-09-04 weight-axis deck.
- `../research/plan/2026-09-02_sigreg_sum_at_weight_0.02.md` — data in the same deck, slide 6.
