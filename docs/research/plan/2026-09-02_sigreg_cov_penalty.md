# sigreg — attack the rank ceiling with a direct covariance penalty, not more SIGReg weight

Created 2026-09-02 · thread `sigreg` · prior cycle: `2026-09-02_sigreg_sum_at_weight_0.02.md`
· arm: `..._sigreg0.02_ns1024_pool8192_compose1.0_cov3e-5` · control: BSC:45296347 (same recipe, `cov_weight=0`)
· jobs: _pending_ · deck: _pending_ · TODO 6

## 1 Hypothesis

**Adding `L_cov = ‖E[z zᵀ] − I‖²_F / Cz` on the same pooled rows SIGReg already sees raises eval
`ZPartRank` past the ~90/512 ceiling the whole weight axis is stuck at, and eval `LossRecon_Comp`
follows it down.**

The threshold is quantitative, not directional. At the twin's ep-33 state (`ZTotalVar` 595.9,
`ZPartRank` 87.6) the identity in §2 makes halving `L_cov` exactly equivalent to **`ZPartRank` ≥ 150**
at fixed trace. So: `ZPartRank` ≥ 150 by ep 33 and eval `LossRecon_Comp` below BSC:45296347 at
matched ep 67, both eval sets.

**Falsifiers.**

- **Rank rises past 150 and recon improves** → the ceiling was the estimator, not an equilibrium with
  recon. The covariance term becomes the default third loss and the SIGReg weight axis is re-read on
  top of it.
- **Rank rises and recon is flat or worse** → rank is not causal for recon, and the r = −0.999
  correlation across the 5 A/B arms (`../../todo/02-09-2026.md` → `2026-08-26_tc_width_compose_convergence_sigreg_tc.md`)
  was a time confound. This is the most informative outcome and it kills the thread's central claim.
- **Rank flat** → the ceiling is not statistical leverage. It is either an equilibrium with recon or
  the linear bottleneck's own limit, and the next lever is `bottleneck_mlp`, not a regulariser.
- **`ZTotalVar` falls to ~512 with `ZPartRank` flat** → the penalty took the cheap scale win and
  nothing else. 26.5% of the available `L_cov` drop at ep 33 is pure trace (see §2), so this outcome
  must be excluded explicitly, not read off `L_cov` alone.
- **Training destabilises** → the weight is hot. Read `Train/ZTotalVar` and the grad-skip warnings;
  the fallback rung is 1e-5.

**Not doing.**

- **Whitening before the isotropy term.** Degenerate for this question. With a detached whitener
  `W = Σ̂^{-1/2}` the whitened rows have identity sample covariance *by construction*, so the CF test
  can no longer see anisotropy and **every** Σ is a fixed point. It would convert SIGReg into a pure
  marginal-Gaussianity test — the opposite of attacking rank. Only pool staleness leaves any residual
  signal, and that measures drift, not spectrum shape.
- **Per-channel rescale before the isotropy term.** A diagonal gauge change. Its fixed point equalises
  `diag(Σ)` only, while the collapse lives in the off-diagonal / rotated basis: trace/Cz is already
  1.16 here and per-channel std was measured to be the wrong instrument
  (`../analysis/2026-07-28_sigreg_z_spread.html`). Worse, per-channel rescale nearly *doubles* the
  measured participation ratio with the code unchanged, so it moves the readout and not the state.
  Plain VICReg's variance term fails here for the same reason; the Frobenius form below is VICReg's
  variance **and** covariance terms in one expression.
- **Swapping SIGReg out.** Forfeits comparability to every arm in the ledger, including the
  no-SIGReg twins. This arm is SIGReg 0.02 **plus** the penalty; the only difference from
  BSC:45296347 is one added loss term.
- **A weight bracket up front.** One arm at the shape-pressure-matched weight, with an ep-12 tripwire
  (§4) that reroutes to 1e-4 or 1e-5 without waiting for ep 67.
- **A warm-started fine-tune off the twin's `current.pth`.** `INIT_CKPT` is weights-only
  (`deltatok_trainer.py:844`), so `iter`/`global_epoch` reset and the cosine LR restarts at peak.
  That needs its own matched-compute control — two jobs to answer a weaker question than one
  from-scratch arm answers directly.

## 2 Analysis

### The rank ceiling is real, and both knobs that reach it are exhausted

`Train/ZPartRank` on the twin BSC:45296347, Cz=512, from the cached stdout:

| ep | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 33 |
|---|---|---|---|---|---|---|---|---|---|
| `ZPartRank` | 19.5 | 35.2 | 48.4 | 60.6 | 71.3 | 78.0 | 83.9 | 86.8 | 87.6 |
| Δ per 4 ep | | +15.7 | +13.2 | +12.2 | +10.7 | +6.7 | +5.9 | +2.9 | |

Decelerating to a stop at ~18% of the 512-dim budget, with `ZTotalVar` pinned at 594–596 the whole
way. The other axes agree: rank stays in a 37–97 band from Cz=64 to Cz=1024 (`Cz` vs rank r = −0.10,
`2026-08-27_tc_width_tc_sigreg_ab_slides.html`), and on the weight axis 0.005 → 0.01 → 0.02 gives
36.7 → 97.2 → 87.6 while 0.08 *lowers* it to 38.7 · 54.3 at twice the loss (`../../todo/02-09-2026.md`, TODO 2
note). Width does nothing and the weight has turned over. Nothing left to turn.

### The identity that makes `L_cov` a rank objective

With `T = ZTotalVar = tr Σ`, `P = ZPartRank = T²/Σλ²` and `C = Cz`:

```
‖Σ − I‖²_F / C  =  T² / (C·P)  −  2T/C  +  1
```

At fixed trace it is strictly decreasing in `P`. The penalty *is* the rank objective, where the CF
statistic reaches rank only through a second-order effect on 1D marginals. Evaluated on the twin:

| ep | 4 | 12 | 20 | 24 | 28 | 33 |
|---|---|---|---|---|---|---|
| `L_cov` (train rows) | 40.6 | 12.9 | 8.26 | 7.45 | 6.90 | **6.59** |

and the inverse map at the ep-33 trace: `L_cov` 6.59 → 3.30 **is** `ZPartRank` 87.6 → 150; `L_cov` →
1.0 would be `ZPartRank` ≈ 298. Uncentered vs centered is inert here — the off-center energy is
`ZRowMeanSquare·Cz − ZTotalVar` = 0.05 of a 595.9 trace (0.009%).

**The scale caveat.** Holding `P` at 87.6 and pulling `T` from 595.9 to 512 alone takes `L_cov` 6.59 →
4.845. So 26.5% of the headroom is a trace correction the penalty can bank without touching rank.
Decompose every read through the identity; never quote `ΔL_cov` as evidence of rank.

### SIGReg and `L_cov` as functions of the same spectrum

Same population minimizer (`z ~ N(0, I)` zeroes both); the difference is how they weight the two things
a spectrum can be wrong about. With `m = T/C` the mean eigenvalue and `Var(λ) = (1/C)·Σ(λᵢ − m)²`:

```
L_cov  =        (m − 1)²  +      1      · Var(λ)                       (exact)
SIGReg ≈  κ · [ (m − 1)²  +  2/(C+2)   · Var(λ) ]                     (leading order, Gaussian slices)
```

The second line: a slice `⟨z, a⟩` has variance `σₐ² = aᵀΣa`, its Epps–Pulley error is `κ·(σₐ² − 1)²` to
second order, and for `a` uniform on the sphere `E[aᵀΣa] = m`, `Var[aᵀΣa] = 2·Var(λ)/(C+2)`. **At C = 512
the shape term carries 1/257 of the weight of the scale term**, and no SIGReg parameter can rebalance
them — both come out of the same per-slice error, so `sigreg_weight` multiplies both (plus the floor
bias and the non-Gaussianity term). That is the 0.08 arm.

On the twin at ep 33: `m` = 1.164, `(m−1)²` = 0.027, `Var(λ)` = 6.56. `L_cov` = 6.59, 99.6% of it the
rank collapse. SIGReg (÷κ) = 0.027 + 0.026 — a 0.16 scale offset and 424 dead dimensions weigh the same.
Two prior facts fall out: the measured 1.9% marginal shift at PR 748/1536 is `√(2/(C+2))·√Var(λ)`
(concentration of measure), and shape pressure `∝ 1/C` at fixed weight predicts a usable rank roughly
constant in *absolute* dims across Cz — the observed 37–97 band from Cz=64 to Cz=1024.

Dropped by the approximation: SIGReg's higher-moment (tail) term, which `L_cov` does not have at all.
Hence on top of SIGReg, not instead of it.

### Why the statistic is the suspect

At the tc1536 arm where SIGReg's leverage was measured, a PR = 748/1536 code shifts each 1D marginal's
std by 1.9%, and the observed statistic decomposes as ~26% finite-sample floor, ~35% anisotropy, ~40%
residual non-Gaussianity (`../analysis/2026-07-28_sigreg_z_spread.html` + the 2026-07-30 review). Only
about a third of what `sigreg_weight` buys is spent on spectrum shape — which is why raising it to 0.08
made everything worse *including* rank.

Honest correction to that note's framing at this operating point: leverage scales as `√(2/P)`, so at
P ≈ 88 the marginal shift is ~7–8%, not 1.9% — SIGReg is a **less** weak lever at tc512 than at tc1536.
The conditioning gap survives anyway. For `L_cov` the finite-sample floor on true N(0,I) is
`(C+1)/S` = 513/8704 ≈ **0.059** against a signal of 6.59, i.e. **~110:1**, and the whole signal is
spectrum. That is the argument: not that SIGReg is blind, but that it spends two thirds of its
pressure elsewhere while this term spends all of it on the thing being measured.

### Cost

Two `(Cz, Cz)` Grams per micro-batch: 128 live rows and 2048 pooled rows per rank at Cz=512 →
~1.1 GFLOP/step, plus one 1 MB all-reduce. Against a 4-GPU H100 step this should be under 1%;
**verify it** against the twin's measured **35.8 min/epoch** (BSC:45296347, ep 49 at 23:21:38) rather
than assuming it.

### Why tc512

The twin exists, reads at ep 67, and tc512 has the most unused budget (88/512 = 17%) — at tc128 rank
is already 90/128. The risk is that tc512 is a *stuck* arm: worst rank of all six widths from ~ep4 on,
cause unknown (`../../todo/02-09-2026.md` history, `2026-08-27_tc_width_tc_sigreg_ab_slides.html`). If this arm
lifts rank but not recon, re-run at tc256 (rank 96.5, healthy) before concluding anything about
mechanism.

## 3 Solution

Four edits, all additive and default-off. The single-term path stays bit-identical at `cov_weight=0`.
Reviewed 2026-09-02 against the trainer; the earlier five-edit version is cut as follows.

**No new diagnostic.** `L_cov` is an exact function of two scalars every arm already logs:
`T²/(C·P) − 2T/C + 1` from `ZTotalVar` and `ZPartRank` (§2; the off-center energy is 0.009% of the
trace). The twin's `L_cov` row in §2 was read exactly that way. So no `ZCovErr` in `z_spread.py` —
compute it offline from the logs, on any arm, old or new.

**Not needed, and why.**

- **A `_need_z` gate refactor** for a future cov-only arm. §1 rules that arm out ("never a swap"), so
  the existing `self.sigreg is not None` gates already produce `z_bneck`. Assert instead.
- **A separate `cov_warmup`.** Two additive terms with their own linear ramps do not compound; a
  second ramp is a knob nobody will sweep. Reuse `sigreg_warmup` (2000 here). `grad_clip=0.1` bounds
  the early step anyway.
- **A `CovPenalty(nn.Module)`.** No parameters, no buffers, no state. A function.
- **The `no_sync()` caveat.** The trainer never calls `no_sync()`; every micro-batch backward runs
  through DDP. Same situation as SIGReg, whose comment already carries it.

**1 — new `occrae/cov_penalty.py`, one function.** Same pooling and collective contract as
`SIGReg.forward`; `sigreg.py` is not touched.

```python
def cov_penalty(live, pool):
    """||E[z z^T] - I||_F^2 / Cz over live + pooled rows. Third term beside SIGReg, never a swap."""
    C = live.shape[-1]                                              # feature dim (Cz)
    s = live.reshape(-1, C).float()                                 # (L, C) the only rows with grad
    gram, n = s.T @ s, s.shape[0]                                   # (C, C) differentiable
    if pool is not None and pool.numel():
        p = pool.reshape(-1, C).float()                             # (P, C) detached FIFO rows
        gram = gram + p.T @ p                                       # (C, C) out-of-place: keeps the live graph
        n += p.shape[0]
    gram = gram / n                                                 # (C, C) second-moment estimate
    if dist.is_available() and dist.is_initialized():
        # Equal-weight rank average, differentiable; DDP's 1/W grad average keeps the scale (as SIGReg).
        gram = autograd_all_reduce(gram, op=dist.ReduceOp.SUM) / dist.get_world_size()   # (C, C)
    eye = torch.eye(C, device=gram.device, dtype=gram.dtype)        # (C, C)
    return (gram - eye).square().sum() / C                          # scalar, 0 iff E[z z^T] = I
```

**2 — `occrae/deltatok_trainer.py`.** Beside the SIGReg build (~line 617):

```python
# Direct second-moment penalty on the rows SIGReg pools: attacks the spectrum shape the sliced CF
# statistic reaches only second-hand. On top of sigreg_weight, never instead -- shares its pool,
# its warmup ramp and its `scale`.
self._cov_weight = float(self.cfg.training.get("cov_weight", 0.0))
assert self._cov_weight == 0 or self._sigreg_weight > 0, "cov_weight needs sigreg_weight > 0"
```

plus `cov_weight=<resolved>` in the master-rank startup print, or a stale cluster trainer reads
0.0 and the arm is a clean null for the wrong reason. In the loss block (~1115), inside the existing
`if self.sigreg is not None:` after the SIGReg add, reusing its `live, pool, scale, ramp`
(`_sigreg_pooled` **banks the pool as a side effect**; it is called once):

```python
if self._cov_weight > 0:
    with torch.autocast(device_type="cuda", enabled=False):
        loss_cov = cov_penalty(live, pool)
    loss_total = loss_total + (self._cov_weight * ramp * scale) * loss_cov
```

with `loss_cov = None` initialised beside `loss_sigreg`, and the bookkeeping every other term has:
`cum_cov`/`n_cov`, `stats["cov"]`, `Train/LossCov`, and `("cov", "Cov")` in the epoch-echo tuple
(~1680) so the epoch stdout line carries it.

**3 — `configs/deltatok/train_deltatok.yaml`**, after `sigreg_compose_z` (~line 128):

```yaml
  cov_weight: 0.0  # + ||E[zz^T]-I||_F^2/Cz on the SIGReg pool: spectrum shape directly. Needs sigreg_weight > 0
```

The pool is shared: `sigreg_pool_samples` governs both terms and stays at 8192 (not weight-neutral,
`../analysis/2026-07-31_sigreg_pool_not_weight_neutral.html`).

**4 — the arm.** Copy the twin's script, never rewrite:

```bash
cp slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm \
   slurm/deltatok/train_deltatok_compose_sigreg_covpen_nozn_tc512_bsc.slurm
```

Change `--job-name`, `--output`, `--error`, `RUN_NAME` together, and add exactly one override:

- `COV_WEIGHT=${COV_WEIGHT:-3e-5}` → `training.cov_weight=${COV_WEIGHT}`
- `RUN_NAME=deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_compose${COMPOSE_WEIGHT}_cov${COV_WEIGHT}`
- `SIGREG_WEIGHT` default 0.005 → **0.02**, the twin's value, so a bare `sbatch` cannot produce a
  0.005 arm by omission.
- `--time=40:00:00` → **`44:00:00`**. The twin runs 35.8 min/ep (ep 49 at 29 h 15 min training clock,
  2026-09-02 23:21), so ep 67 lands at 40.0 h — on the wall, and `exit_before_time_limit` would stop
  it at ep 66. Backfill at 44 h fits gaps almost as well as 40 h.
- everything else byte-identical: tc512, compose 1.0, ns1024, pool8192, warmup2000, max_gap 9,
  bsize 2, account `ehpc880` (the twin's; ehpc1001 already carries BSC:45344713 and BSC:45345063).

**Where 3e-5 comes from.** Two anchors, both from the twin at ep 33. *Loss-share parity* with SIGReg
(`0.02 × 17.0 × 0.0039` = 1.48% of a 0.0896 total) gives `w = 0.0148 × 0.0896 / (17.0 × 6.59)` = 1.2e-5 —
but that is too timid to separate the hypotheses. Per §2, SIGReg's per-axis gradient is
`∝ C(m−1) + 2(λᵢ−1)`, the covariance term's is `∝ 2(λᵢ−1)`, so at 1e-5 the added *redistribution* push is
only ~1.3–2× what SIGReg 0.02 already delivers (κ ≈ 0.07–0.12 from the CF quadrature; the observed
0.0039 sits below the leading-order 0.0063, hence the range) — roughly the 0.04–0.06 arm's shape push,
and a 2× step was already flat. **At 3e-5 the added push is ~4–6× SIGReg's own: at or above what the
0.08 arm delivered, with ~1–2% of its added scale stiffness and none of its floor or tail terms.** That
is the comparison the arm exists to make. Loss share: 3.8% at ep 33, 11.5% at ep 4 (`L_cov` 40.6). It is a
leading-order anchor, which is why §4 has an ep-12 tripwire in both directions.

**Pre-flight, then submit** (the user syncs manually — if a grep is empty the cluster copy is stale;
ask, do not rsync):

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  grep -n \"cov_weight\\|cov_penalty\" occrae/deltatok_trainer.py configs/deltatok/train_deltatok.yaml && \
  ls occrae/cov_penalty.py && \
  grep -E \"RUN_NAME|cov_weight|SIGREG_WEIGHT=|job-name|time=\" slurm/deltatok/train_deltatok_compose_sigreg_covpen_nozn_tc512_bsc.slurm'"

ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  sbatch slurm/deltatok/train_deltatok_compose_sigreg_covpen_nozn_tc512_bsc.slurm'"
```

**Budget.** 35.8 min/ep measured on the twin → ep 67 at ~40 h, ep 73 inside a 44 h job. No resume
needed for the read. No separate smoke job: the first epoch line lands ~36 min in and carries every
tripwire below.

## 4 Results

_Pending._

### How to read it

**Primary: eval `LossRecon_Comp` vs BSC:45296347 at matched ep 67, per eval set, not pooled.**
Secondary: eval `LossRecon`, the three `PredVsOrig` geometry losses, `ZPartRank`, `ZTotalVar`,
and `L_cov` from them via the §2 identity. Rank is the *mechanism* readout, not the verdict — whether rank pays downstream is a
separate open question (`../analysis/2026-07-28_sigreg_z_spread.html`, "How to apply").

| read | twin BSC:45296347 | this arm must show |
|---|---|---|
| ep 0, first 60 s | — | startup print `cov_weight=3e-05`; a silent 0.0 is a stale trainer |
| ep 0 | Train 0.2638, Eval 0.1225, 35.8 min/ep | `Cov:` in the epoch line, ~7–40; epoch time within 1% |
| ep 4 | Train 0.1795, Eval 0.0948, PR 19.5 | recon not more than ~10% behind the twin |
| **ep 12, tripwire** | PR 48.4, `L_cov` 12.9 | **PR ≥ 65.** Below → kill, resubmit at 1e-4. Recon >10% behind → 1e-5 |
| ep 33 | Train 0.0896, Eval 0.0507, PR 87.6, `L_cov` 6.59 | **PR ≥ 150** (= `L_cov` 3.30 at fixed trace) |
| ep 67 | _(twin read pending, TODO 2)_ | eval `LossRecon_Comp` below the twin, both sets |

Decompose every `L_cov` move with `T²/(C·P) − 2T/C + 1` before calling it rank: 26.5% of the ep-33
headroom is trace alone. `ZTotalVar` drifting to 512 with `ZPartRank` flat is the scale-only null.

The raw `SIGReg:` stdout scalar stays comparable across this flag (same statistic, same rows), unlike
the `sigregsum` arm — so it doubles as a check on whether the two terms agree about the code.

### Tracking

| Job | Arm | State | Notes |
|---|---|---|---|
| BSC:45296347 | tc512 plain sigreg 0.02 | RUNNING ep 33 | **the twin for this read** |
| _pending_ | tc512 sigreg 0.02 + `cov_weight=3e-5` | not submitted | this arm |

Logs: `slurm/output/train_deltatok_compose_sigreg_covpen_nozn_tc512_bsc_<jobid>.{out,err}`.
TB mirror: `/mnt/d/tb_logs/deltatok_log/<run>/tb_logs/`.

## 5 Findings

_Pending. Write in the order of the falsifiers._

## → Next hypothesis

`open`. Four branches, decided by §5:

- **Rank and recon both move** — re-read the `sigreg_weight` axis with the penalty on, and re-test at
  tc128 / tc1024 to see whether the 37–97 dim band was ever about width.
- **Rank moves, recon does not** — the r = −0.999 rank↔recon correlation was a time confound. Retire
  `ZPartRank` as a target and settle the downstream question directly with the flow eval on this
  checkpoint vs the twin.
- **Rank does not move** — the ceiling is not statistical. Next lever is the bottleneck itself
  (`bottleneck_mlp=true`), not another regulariser.
- **tc512 turns out to be the confound** — repeat at tc256, where rank is healthy at 96.5.
