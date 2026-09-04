# DeltaTok STATUS — the jobs board

What ran, what it reached, and what still has to be checked. One row per job that is queued, running, or finished
but not yet read into a `results/` doc. [`02-09-2026.md`](02-09-2026.md) keeps the *items*; this file keeps the
*jobs*. A row leaves once its read has landed and the TODO row is closed into [`../DONE.md`](../DONE.md).
Job states come from `../../monitor_jobs/data/monitor_jobs.json` (read the file, never the server); epochs come from
the cached `logs/BSC/*.out`. Every row says what to grep and where the number goes.

**As of 2026-09-04 17:04 (monitor refresh).**

## Queued and running

| Job | Arm | TODO | State | Read at | Check | Result goes to |
|---|---|---|---|---|---|---|
| BSC:45421190 | tc128 compose · decoder-only noise finetune, `decode_noise_tau=0.8`, encoder frozen | 12 | PENDING on `ehpc1001`, 8 h, 10 ep from `epoch_100` | ep 10 (~5.5 h after start), ep 5 as the backstop | First 60 s: `.out` must print `decode_noise_tau=0.8`, `encoder_blocks ... trainable=   0.000M` and `Load ckpt from: .../compose1.0/ckpts/epoch_100.pth`; `LossRecon` at iter 0 near the source arm's terminal value, not a fresh-init one. Then two `slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm` reads with `DELTATOK_CKPT` pointed at the finetuned ckpt — flow N=1/20 (`MSEToken` must stay 0.6677) and the `--noise_sigmas 0.0,0.32,0.55,0.82` ladder | `../research/plan/2026-09-04_flow_decoder_noise_finetune.md` §4, then a `results/` deck |
| BSC:45416718 | tc512 · sigreg 0.02 · compose 1.0 · `cov_weight=3e-5` | 6 | PENDING on `ehpc880`, 44 h, est. start 2026-09-04 18:10 | ep 67 (~45 h after start) | First 60 s: `.out` must print `cov_weight=3e-05`; a silent `0.0` is a stale trainer. Then eval `LossRecon_Comp` per eval set vs twin BSC:45296347 at ep 67, and vs the 0.01 arm BSC:45106935 (a win under ~6% is inside the weight plateau) | `../research/plan/2026-09-02_sigreg_cov_penalty.md` §4, then a `results/` deck |

## Finished, read landed, TODO row still open

| Job | Arm | TODO | State | Read | Left to do |
|---|---|---|---|---|---|
| BSC:45296347 | tc512 · sigreg 0.02 · compose 1.0 | 2 | COMPLETED at ep 81/100, stopped by the 48 h wall, resumable | ep 67 and ep 81 in `../research/results/2026-09-04_tc_width_tc512_sigreg_weight_axis_slides.html` | Fill §4 of `../research/plan/2026-09-01_tc_width_sigreg_weight_tc512.md`; close TODO 2. It is the twin for TODO 6 and 8, so keep its `current.pth` |
| BSC:45297731 | tc512 · sigreg 0.04 | 2 | COMPLETED at ep 81/100 on the wall. Resubmit of BSC:45296348, which died at startup (`CUDA error: out of memory` in `set_device` on `as06r3b02`) | same deck | none |
| BSC:45296349 | tc512 · sigreg 0.08 | 2 | CANCELLED 2026-09-02 at ep 32, resumable | same deck: over-regularised, rank lower not higher | none; do not resume |
| BSC:45345063 | tc512 · sigreg 0.02 · `sigreg_compose_z` (sum) | 8 | CANCELLED 2026-09-04 15:46 mid-ep 67; last eval ep 66 | ep 65 in the same deck: the sum's 0.005 win reverses to +8.0% eval `LossRecon` at 0.02 | Fill §4 of `../research/plan/2026-09-02_sigreg_sum_at_weight_0.02.md`; close TODO 8 |
| BSC:45344713 | flow · pointditT (`logitnormal(-0.8,0.8)` + 10% `t=0`) from scratch | 7 | CANCELLED 2026-09-04 at ep 58 | ep 50 via evals BSC:45414635 / 45414644: `../research/results/2026-09-04_pointdit_lowt_numsteps_ep50_slides.html`, loses 19/20 cells | Remaining pointdit arms gated on TODO 11 |
| BSC:45417908 | flow · decoder noise probe (σ 0/0.32/0.55/0.82 on iter_100000, N=1) | 12 | COMPLETED 2026-09-04 16:20 | `../research/results/2026-09-04_flow_decoder_noise_probe.md` | none; it motivates TODO 12 |

## Planned, not submitted

| TODO | What | Plan | Needs before `sbatch` | Check when it runs |
|---|---|---|---|---|
| 11 | best-of-K + K-spread evals on existing ckpts (minutes each) + one 72 h `train_fixed_t=0` regressor read at `iter_100000` | `../research/plan/2026-09-04_flow_bestofk_regressor_null.md` §3 | the eval flags and the regressor knob in §3; sync, then grep the remote file | falsifiers in §1 |

## Plans whose §4 is still `_Pending_` with data already on disk

- `../research/plan/2026-09-01_tc_width_sigreg_weight_tc512.md` — data in the 2026-09-04 weight-axis deck.
- `../research/plan/2026-09-02_sigreg_sum_at_weight_0.02.md` — data in the same deck, slide 6.
