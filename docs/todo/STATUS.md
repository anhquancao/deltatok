# DeltaTok STATUS — the jobs board

What ran, what it reached, and what still has to be checked. One row per job that is queued, running, or finished
but not yet read into a `results/` doc. [`02-09-2026.md`](02-09-2026.md) keeps the *items*; this file keeps the
*jobs*. A row leaves once its read has landed and the TODO row is closed into [`../DONE.md`](../DONE.md).
Job states come from `../../monitor_jobs/data/monitor_jobs.json` (read the file, never the server); epochs come from
the cached `logs/BSC/*.out`. Every row says what to grep and where the number goes.

**As of 2026-09-07 10:51.**

## Queued and running

| Job | Arm | TODO | State | Check when it runs |
|---|---|---|---|---|
| BSC:45498520 | tc512 · sigreg 0.02 · compose 1.0 · `cov_weight=1e-4` | 6 | RUNNING on `as02r3b26` since 2026-09-07 00:00, ep **17**/100 at 10:22, 44 h | ep-12 tripwire **passed on rank, failed on recon**: train `ZPartRank` 95.4 (bar 65) but `LossRecon` is +0.0% K / −0.9% N against the cov 0 control on an ep 12–17 mean, i.e. outside the "within 10% of BSC:45416718" clause the other way — it is 11% *worse* than 45416718. Not killed: it is the middle rung of the dose axis and the null is the finding. **Next read at the wall, ep ~72** |
| BSC:45498521 | same, `cov_weight=3e-4` | 6 | RUNNING on `as03r2b27` since 2026-09-07 00:00, ep **17**/100 at 10:22, 44 h | ep-12 tripwire passed: `ZPartRank` 129.1, `LossRecon` −8.9% K / −7.4% N vs control on the same mean. The hot rung did **not** trip the wire. **Next read at the wall, ep ~72** |

Both are the `cov_weight` dose-response above BSC:45416718. **First read landed 2026-09-07, ep 17, in slide 11 of
[`../research/results/2026-09-06_sigreg_cov_penalty_tc512_slides.html`](../research/results/2026-09-06_sigreg_cov_penalty_tc512_slides.html).**
**Read: more `L_cov` buys more rank, and the rank does not turn into performance.** Train `ZPartRank` is monotone in dose with no
ceiling — 62.8 (cov 0) → 93.1 (3e-5) → 104.2 (1e-4) → **141.0** (3e-4) — which kills the "~121/512 ceiling" in finding 1 of that deck.
None of it transfers. What the term moves is convergence *speed*, not the value converged to: the 3e-5 lead over cov 0 peaks at
−11.5% (ep 25) and decays to −7.7% (ep 70), still shrinking when that arm stopped. Above 3e-5 there is no return — 1e-4 lands on the
cov 0 control (+0.0% K / −0.9% N token, +5 to +11% worse under AR rollout) and 3e-4 fails to clear 3e-5 with 1.5× the rank.
One seed. Resolved configs differ only in `training.cov_weight`; no NaN and no grad-skip in either arm.

**Both jobs end on the 44 h wall at ~2026-09-08 19:50 CEST** (11h01m elapsed / 32h59m left at 10:51). At the observed
0.61 h/epoch they reach **ep ~71–72**, not ep 100, and `exit_before_time_limit=true` stops them cleanly. That is a matched
read: BSC:45416718 (`cov 3e-5`) also stopped at ep 72 on its own 44 h wall, and the cov 0 control BSC:45296347 has data
through ep 81. **The ep-17 conclusion above is interim — re-check all three claims at ep 72 before it is quoted anywhere else:**

| # | Claim to re-test at ep 72 | Holds if | Dies if |
|---|---|---|---|
| 1 | `L_cov` moves convergence *speed*, not the endpoint | the 3e-5 lead over cov 0 keeps decaying past −7.7% (it went −11.5% at ep 25 → −7.7% at ep 70) | the gap stops shrinking and settles at a stable non-zero margin |
| 2 | Nothing above 3e-5 pays | 1e-4 is still level with the cov 0 control on `LossRecon` and still worse under AR rollout | 1e-4 crosses below the control and closes on 3e-5 |
| 3 | Rank does not buy recon | 3e-4 still fails to clear 3e-5 despite 1.5× the `ZPartRank` | 3e-4 crosses below 3e-5 late, which revives the rank story |

Reads go in slide 11 of the deck (currently written as an ep-17 interim, and it says both arms are still running).
Grep the four `slurm/output/*.out` logs, not TB — every scalar needed is on the `[KEpoch …]`, `[Train]` and `[Eval/…]` lines.
Slides 1–10 of that deck still carry the original "cov 3e-5 wins token recon" verdict and now disagree with slide 11;
fold them in once the ep-72 numbers settle which reading is right. Scripts `slurm/deltatok/train_deltatok_compose_sigreg_covpen{1e-4,3e-4}_nozn_tc512_bsc.slurm`.

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
