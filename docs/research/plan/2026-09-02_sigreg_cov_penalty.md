# sigreg — attack the rank ceiling with a direct covariance penalty, not more SIGReg weight

Created 2026-09-02 · thread `sigreg` · prior cycle: `2026-09-02_sigreg_sum_at_weight_0.02.md`
· arm: `..._sigreg0.02_ns1024_pool8192_compose1.0_cov3e-5` · control: BSC:45296347 (same recipe, `cov_weight=0`)
· jobs: BSC:45416718 · deck: [`../results/2026-09-06_sigreg_cov_penalty_tc512_slides.html`](../results/2026-09-06_sigreg_cov_penalty_tc512_slides.html) · TODO 6
· twin read: `../results/2026-09-04_tc_width_tc512_sigreg_weight_axis_slides.html`

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
  correlation across the 5 A/B arms (`../viewer.html` · the queue → `2026-08-26_tc_width_compose_convergence_sigreg_tc.md`)
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

| ep | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 33 | 49 | 67 | 81 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ZPartRank` | 19.5 | 35.2 | 48.4 | 60.6 | 71.3 | 78.0 | 83.9 | 87.6 | 97.1 | 101.8 | 101.5 |

Gaining +15.7 per 4 ep at the start, +2.9 by ep 33, and **flat from ep 49 on** — saturated at ~20% of
the 512-dim budget, with `ZTotalVar` at 594–650 the whole way. The twin ran to ep 81 (48 h wall,
`current.pth` resumable), so this tail is measured, not extrapolated.

The other axes agree. Rank stays in a 37–97 band from Cz=64 to Cz=1024 (`Cz` vs rank r = −0.10,
`2026-08-27_tc_width_tc_sigreg_ab_slides.html`). On the weight axis the completed read
(`../results/2026-09-04_tc_width_tc512_sigreg_weight_axis_slides.html`) is a **plateau, not a
turnover**: eval `ZPartRank` at ep 67 goes 36.7 (0.005) → 97.2 (0.01) → 89.2 (0.02) → 97.6 (0.04),
i.e. saturated near 100/512 across a 4× weight range, then *falls* to 46.5 at 0.08 with 1.8× the loss.
Width does nothing and the weight is spent. Nothing left to turn.

### The identity that makes `L_cov` a rank objective

With `T = ZTotalVar = tr Σ`, `P = ZPartRank = T²/Σλ²` and `C = Cz`:

```
‖Σ − I‖²_F / C  =  T² / (C·P)  −  2T/C  +  1
```

At fixed trace it is strictly decreasing in `P`. The penalty *is* the rank objective, where the CF
statistic reaches rank only through a second-order effect on 1D marginals. Evaluated on the twin:

| ep | 4 | 12 | 20 | 28 | 33 | 49 | 67 | 81 |
|---|---|---|---|---|---|---|---|---|
| `L_cov` (train rows) | 40.6 | 12.9 | 8.26 | 6.90 | **6.59** | **5.99** | 6.26 | 6.59 |

**`L_cov` is not monotone on the twin.** It bottoms at ep 49 and climbs back to its ep-33 value by
ep 81 — rank is flat at ~102 while `ZTotalVar` drifts 604 → 650, so the late rise is pure trace. The
untouched objective therefore *loses* ground on the very quantity this arm adds. Compare at matched
epochs only.

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

**Fit κ on a high-anisotropy state, not a low one.** `SIGReg_measured = κ·(model term) + floor`, and the floor is
~26% of the statistic and does not shrink as the anisotropy does. Dividing the raw scalar by the model term
therefore over-estimates κ exactly where `Var(λ)` is small — i.e. on the cov arms. Measured: twin ep 33 → **0.074**,
twin ep 67 → 0.046, cov arm ep 67 → **0.172**. Use the twin's ep-33 fit; the cov arm's is inflated by the floor,
not by a real change in the constant.

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
**verify it** against the twin's measured **34.9 min/epoch** (BSC:45296347, ep 81 at 47:44:23) rather
than assuming it.

### Why tc512

The twin exists, reads at ep 67 and ep 81, and tc512 has the most unused budget (102/512 = 20%) —
at tc128 rank is already 90/128 (73%).

**The "stuck arm" caveat is resolved and no longer applies.** tc512's worst-of-six rank was the
inherited `sigreg_weight` 0.005, not the width: at 0.01–0.04 its eval recon at ep 67 is 0.0416–0.0441
pooled against 0.0414 for tc128 and 0.0416 for tc256, versus 0.0589 at 0.005. Width is a wash on token
recon at the right weight, and tc512 reaches it using a fifth of its channels. Decoded geometry still
prefers narrow on KITTI — tc128's `Pointmap_PredVsGT` sits 0.007 above the teacher floor against 0.196
for this twin — which is a `tc_width` question, not this arm's.

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
- `--time=40:00:00` → **`44:00:00`**. The twin runs 34.9 min/ep and hit ep 67 at 39 h 35 min of
  training clock, so a 40 h job would be stopped at ep 66 by `exit_before_time_limit`. 44 h reaches
  ~ep 75, and backfill at 44 h fits gaps almost as well as 40 h.
- everything else byte-identical: tc512, compose 1.0, ns1024, pool8192, warmup2000, max_gap 9,
  bsize 2, account `ehpc880` (the twin's; ehpc1001 already carries BSC:45344713 and BSC:45345063).

**The `17.0` factor applies to `L_cov` identically.** Both terms are added as `(weight * ramp * scale) * loss` with
the same `scale` out of one `_sigreg_pooled` call (`deltatok_trainer.py:1163` and `:1167`), and it is exactly
`(live 512 + pool 8192) / live 512` = 17.0. Anchored on the twin, the cov term's share of the training loss is
3.8% / 12.5% / 37.5% at ep 33 and 11.5% / 38.5% / 115% at ep 4 for `cov_weight` 3e-5 / 1e-4 / 3e-4. `1e-3` is
excluded on this alone: 385% of the loss at ep 4.

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

**Read 2026-09-06 from `logs/BSC/deltatok_covpen_bsc_45416718.out`, ep 67 matched. The arm stopped itself at
ep 72/100 on the 44 h wall (`exit_before_time_limit`), `current.pth` resumable. `cov_weight=3e-05` printed at
startup and `Cov:` carried the epoch line throughout, so the knob was live.**

**Verdict: the primary read passes and the secondary bar is cleared, but the rank threshold fails.**
Recon improved on both eval sets against both twins; `ZPartRank` rose +28% at ep 33 where the hypothesis
demanded +71%. That is falsifier 1's direction without its magnitude.

| ep 67, eval | twin 45296347 (cov 0) | axis-best 45106935 (0.01) | **this arm (cov 3e-5)** |
|---|---|---|---|
| `LossRecon_Comp` KITTI | 0.0561 | 0.0528 | **0.0511** (−8.9% / −3.2%) |
| `LossRecon_Comp` nuScenes | 0.0401 | 0.0370 | **0.0367** (−8.5% / −0.8%) |
| `LossRecon` KITTI / nuScenes | 0.0520 / 0.0363 | 0.0493 / 0.0340 | **0.0480 / 0.0336** |
| eval `ZPartRank` K / N | 89.2 / 98.0 | 97.2 / 113.0 | **104.1 / 114.6** |
| train `ZPartRank` / `ZTotalVar` | 101.8 / 635.0 | 114.8 / 621.7 | **120.2 / 536.7** |
| train `L_cov` (via §2 identity) | 6.26 | 5.15 | **3.58** |
| epoch-line Train / Eval | 0.0765 / 0.0442 | — | **0.0715 / 0.0408** |

Every tripwire in "How to read it" passed except the ep-33 rank bar: ep 4 recon 0.1171 K vs the twin's 0.1209
(ahead, not 10% behind); ep 12 train PR **80.2** against the ≥ 65 kill line; ep 33 train PR **111.9** against
the **≥ 150** bar, `L_cov` 3.69 against 3.30.

**The rank ceiling moved but did not lift.** Train `ZPartRank` by epoch, arm vs twin:

| ep | 0 | 4 | 8 | 12 | 20 | 28 | 33 | 49 | 67 | 72 |
|---|---|---|---|---|---|---|---|---|---|---|
| cov 3e-5 | 66.8 | 50.5 | 66.4 | 80.2 | 99.6 | 108.1 | 111.9 | 117.3 | 120.2 | 121.1 |
| twin | 55.4 | 19.5 | 35.2 | 48.4 | 71.3 | 83.9 | 87.6 | 97.1 | 101.8 | 102.2 |

The arm reaches the twin's *terminal* rank by ep 20 and is +18% at ep 67 — but it saturates on the same shape,
+3.8 over the last 23 epochs. A new plateau at ~121/512 (24%), not an escape.

**`L_cov` decomposition, the check §2 demands.** Using `T²/(C·P) − 2T/C + 1` at ep 67, twin (635.0, 101.8)
`L_cov` 6.26 → arm (536.7, 120.2) 3.58:

- holding rank at the twin's 101.8 and moving trace alone → 4.43, i.e. **68% of the drop is trace**;
- holding trace and moving rank alone → 5.24, **32% is rank**.

At ep 33 the split is 55% trace / 45% rank. So the penalty banked more of the cheap scale win than §2's 26.5%
estimate — `ZTotalVar` runs 505–540 against the twin's 590–640, near the 512 target. **This is not the
scale-only null**: rank rose materially at every epoch. But `L_cov` alone overstates the rank effect by ~2×.

**Cost: +2.1%, above the 1% budget.** Ep 67 at 40:25:17 against the twin's 39:35:27 — 35.7 vs 34.9 min/epoch.
**Stability: clean.** No grad-skip or NaN warnings in 72 epochs; `SIGReg:` tracks the twin (0.0028 vs 0.0038 at
ep 67), so the two terms agree about the code.

### How to read it

**Primary: eval `LossRecon_Comp` vs BSC:45296347 at matched ep 67, per eval set, not pooled.**
Secondary: eval `LossRecon`, the three `PredVsOrig` geometry losses, `ZPartRank`, `ZTotalVar`,
and `L_cov` from them via the §2 identity. Rank is the *mechanism* readout, not the verdict — whether rank pays downstream is a
separate open question (`../analysis/2026-07-28_sigreg_z_spread.html`, "How to apply").

| read | twin BSC:45296347 | this arm must show |
|---|---|---|
| ep 0, first 60 s | — | startup print `cov_weight=3e-05`; a silent 0.0 is a stale trainer |
| ep 0 | Train 0.2638, Eval 0.1225, 34.9 min/ep | `Cov:` in the epoch line, ~7–40; epoch time within 1% |
| ep 4 | Train 0.1795, Eval 0.0948, PR 19.5 | recon not more than ~10% behind the twin |
| **ep 12, tripwire** | PR 48.4, `L_cov` 12.9 | **PR ≥ 65.** Below → kill, resubmit at 1e-4. Recon >10% behind → 1e-5 |
| ep 33 | Train 0.0896, Eval 0.0507, PR 87.6, `L_cov` 6.59 | **PR ≥ 150** (= `L_cov` 3.30 at fixed trace) |
| **ep 67, the read** | Train 0.0765, Eval 0.0442, train PR 101.8, `L_cov` 6.26 · eval `LossRecon_Comp` **0.0561** K / **0.0401** N, `LossRecon` 0.0520 / 0.0363, eval PR 89.2 / 98.0 | eval `LossRecon_Comp` below the twin, both sets |
| ep 75 | _(past the twin, inside the 44 h wall)_ | free tail; the twin's own ep 81 is 0.0563 K / 0.0392 N |

Decompose every `L_cov` move with `T²/(C·P) − 2T/C + 1` before calling it rank: 26.5% of the ep-33
headroom is trace alone. `ZTotalVar` drifting to 512 with `ZPartRank` flat is the scale-only null.

The raw `SIGReg:` stdout scalar stays comparable across this flag (same statistic, same rows), unlike
the `sigregsum` arm — so it doubles as a check on whether the two terms agree about the code.

**A second bar, because 0.02 is not the best point on the weight axis.** The plateau 0.01–0.04 spans
6% of eval recon, and 0.02 is its worst arm: at ep 67 the 0.01 twin BSC:45106935 reads
`LossRecon_Comp` 0.0528 K / 0.0370 N against 0.0561 / 0.0401 here. So beating BSC:45296347 by less
than that is **inside the weight plateau** and does not separate "the penalty works" from "you moved
along the weight axis". Report against both twins; only clearing 0.0528 / 0.0370 is a new best.
0.02 is kept as the base anyway so the arm stays a one-term diff against a matched control.

### Tracking

| Job | Arm | State | Notes |
|---|---|---|---|
| BSC:45296347 | tc512 plain sigreg 0.02 | COMPLETED ep 81 | **the twin for this read**; 48 h wall, `current.pth` resumable |
| BSC:45106935 | tc512 plain sigreg 0.01 | COMPLETED ep 67 | secondary bar — the axis best, see "A second bar" above |
| BSC:45416718 | tc512 sigreg 0.02 + `cov_weight=3e-5` | COMPLETED ep 72, wall | **this arm**; read at ep 67 above, `current.pth` resumable to ep 100 |
| BSC:45498520 | tc512 sigreg 0.02 + `cov_weight=1e-4` | PENDING, 44 h, `ehpc1001` | dose-response, from scratch. Cov loss share 12.5% at ep 33 |
| BSC:45498521 | tc512 sigreg 0.02 + `cov_weight=3e-4` | PENDING, 44 h, `ehpc1001` | dose-response, from scratch. 37.5% at ep 33; the hot rung |

Logs: `slurm/output/train_deltatok_compose_sigreg_covpen_nozn_tc512_bsc_<jobid>.{out,err}`.
TB mirror: `/mnt/d/tb_logs/deltatok_log/<run>/tb_logs/`.

## 5 Findings

> **Reversed 2026-09-07 by the dose-response, at matched ep 29.** Everything below was written from the single `3e-5` arm at
> ep 67 and reads the +18% rank / −8.9% recon pair as causal. It is not. Train `ZPartRank` is monotone in dose with no ceiling —
> 84.8 (cov 0) → 109.1 (3e-5) → 115.8 (1e-4) → **163.3** (3e-4), a 1.9× spread — while eval `LossRecon` moves a few percent and
> out of dose order: only 3e-5 beats the control, 1e-4 is +8.0% K / +5.3% N *behind* it, 3e-4 clears the control but not 3e-5.
> **`cov_weight` is not adopted, and `ZPartRank` is a diagnostic, not a target.** The r = −0.999 correlation below compares arms
> that differ in convergence; it breaks as soon as rank is pushed directly at a fixed recipe. Slide 11 of
> `../results/2026-09-06_sigreg_cov_penalty_tc512_slides.html` carries the axis; slides 1–10 still show the ep-67 reading.


**Falsifier 1, partially.** Rank rose and recon followed, but not past 150. `ZPartRank` 101.8 → 120.2 at ep 67
(+18%) with eval `LossRecon_Comp` −8.9% KITTI / −8.5% nuScenes against the matched control. The direction the
thread predicted is real and the term is cheap, so **`cov_weight` becomes the default third loss** — but the
ceiling it buys is ~121/512, not the ~300 an `L_cov` → 1.0 would imply. The estimator was *part* of the ceiling,
not all of it.

**It is a new best on the weight axis, narrowly.** The plan's second bar was 0.0528 K / 0.0370 N (the 0.01 arm).
The arm reads 0.0511 / 0.0367, so it clears both — but the nuScenes margin is **0.8%**, well inside the 6% weight
plateau. The KITTI margin (3.2%) is the one carrying the claim. The follow-on is a **dose-response at fixed `sigreg 0.02`**
(`BSC:45498520` at 1e-4, `BSC:45498521` at 3e-4), not a second base. A `sigreg 0.01 + cov` arm was dropped: the
weight axis is fully mapped at ep 67 (0.01 → 0.0528 K / 0.0370 N, 0.04 → 0.0544 / 0.0392, 0.02 → 0.0561 / 0.0401),
so a higher dose landing materially below **0.0528 / 0.0370** kills "cov only moves you along the weight axis"
without spending an arm on a second base — no `sigreg_weight` ever reached there.

**Falsifier 4, the scale-only null, is excluded but it is closer than §2 predicted.** 68% of the ep-67 `L_cov`
drop is trace, not 26.5%. Rank still moved at every epoch, so the null does not hold — but any future arm must
quote the decomposition, never `ΔL_cov`.

**`ZPartRank` survives as a target, provisionally.** The r = −0.999 correlation was not a pure time confound:
here rank and recon moved together across two arms at *matched* epochs and matched compute. That is one A/B, not
a causal proof, and the ~121 plateau says rank is not the only thing recon wants.

**The arm is at its equilibrium, not out of wall time.** At ep 72 `ZTotalVar` is 540.7 against the 512 target with
`ZPartRank` flat at 121.1 (120.2 at ep 67 — +0.9 in five epochs). Put `T = 512` into the §2 identity and the floor is
`512/P − 1` = **3.26** against the arm's actual 3.58, so only **9%** of the remaining `L_cov` is trace and any further
drop would have to be rank almost by construction. **The penalty never pulled trace down at all:** `ZTotalVar` rises
essentially monotonically from its global minimum of 457.6 at ep 0, crosses 512 upward at ep 17 (513.5), never returns
below it, and ends 5.6% *above* the target still climbing. The largest single-epoch decrease anywhere after ep 3 is
−0.85. So the arm started under-scaled and grew through the target; the scale term is now being slowly lost, and
further `L_cov` reduction cannot come from trace in either direction. Resuming to ep 100 would buy recon, not rank.
More weight moves this; more epochs will not. That is the argument for the dose-response.

**Cost is 2.1%, not the <1% claimed.** Two `(Cz, Cz)` Grams plus a 1 MB all-reduce cost 0.8 min/epoch at tc512.
Acceptable, but it scales as `Cz²` — re-measure before turning this on at tc1024.

## → Next hypothesis

**Running: the `cov_weight` dose-response at fixed `sigreg 0.02`** — `BSC:45498520` (1e-4) and `BSC:45498521` (3e-4),
44 h each on `ehpc1001`, both from scratch. The added shape push over `sigreg 0.02` alone is ~5× at 3e-5, ~17× at
1e-4 and ~52× at 3e-4, while scale stiffness rises only +2.0% / +6.8% / +20% — a ratio no `sigreg_weight` reaches
(matching 52× would need weight ~1.0, and 0.08 already broke). Read at matched ep 67 against **0.0528 K / 0.0370 N**,
the axis best: clearing it kills the plateau confound outright. Watch the ep-12 tripwire (`ZPartRank` ≥ 65) on
3e-4 especially — its loss share is 115% at ep 4.

The four branches below still frame the read:

- **Rank and recon both move** — re-read the `sigreg_weight` axis with the penalty on, and re-test at
  tc128 / tc1024 to see whether the 37–97 dim band was ever about width.
- **Rank moves, recon does not** — the r = −0.999 rank↔recon correlation was a time confound. Retire
  `ZPartRank` as a target and settle the downstream question directly with the flow eval on this
  checkpoint vs the twin.
- **Rank does not move** — the ceiling is not statistical. Next lever is the bottleneck itself
  (`bottleneck_mlp=true`), not another regulariser.
- **tc512 turns out to be the confound** — repeat at tc256, where rank is healthy at 96.5. Less
  likely now that the stuck-arm caveat is resolved (§2), but tc128 still wins KITTI decoded geometry.
