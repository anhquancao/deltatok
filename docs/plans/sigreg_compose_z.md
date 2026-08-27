# SIGReg on the composed sum

## Context

In the compose arm, SIGReg currently sees **one randomly chosen hop** per micro-batch
(`occrae/deltatok_trainer.py:1055-1057`):

```python
hop_idx = torch.randint(0, 2, ()).item()   # 0 or 1: t0->t1 or t1->t2
z_bneck = z_compose[:, hop_idx]            # (B, N, K, Cz)
```

The composed sum `z_comp = z_a + z_b` is built at `:948`, decoded for `loss_compose`, and then
discarded — it never reaches the regulariser. The random pick exists only to hold the live row
count equal to the single-pair arms, so `sigreg_weight` transfers between compose and plain arms.

Goal: regularise `z_comp` too, so the composed code is held to the same distribution as the hops.

## Design

**Concatenate all three streams into the one existing SIGReg call.** No rescaling of `z_comp`, no
second pool, no second all-reduce.

```python
z_bneck = torch.cat([z_compose[:, 0], z_compose[:, 1], z_comp], dim=0)  # (3B, N, K, Cz)
```

`_sigreg_pooled` reshapes to `(-1, Cz)` at `:693`, so the cat axis is cosmetic. Live rows go
`L -> 3L`.

### Why one pooled call, not three separate statistics

`occrae/sigreg.py:61-90` builds **one** empirical CF as a plain mean over all rows (live + pool),
then a single `|CF_emp - phi|^2`. So a cat does not pin three distributions — it pins the 2:1
*mixture* of hops and sums.

That is the same structure the working arm already uses. `max_gap=9` means `gap ~ U[1,9]`, so
gap-1 and gap-9 codes — which have genuinely different natural scales — already share one pool and
one CF test, and that arm trains well. Pooling heterogeneous streams is the established, measured
behaviour here, not a new risk. Three separate statistics would be a departure from the precedent,
not a return to it.

### The one thing that is genuinely different

Across gaps the encoder has a free per-gap gain: it can normalise each gap's delta to unit scale
independently, and nothing forbids that. `sigreg_gap_sigma` tried to *undo* that normalisation and
measured 6.5-13.9% worse on every geometry loss, which is evidence the free gain is what the
encoder wants.

`z_comp = z_a + z_b` is an algebraic identity, so it has no free gain. With both hop marginals
pinned to `N(0,I)`:

```
Var(z_a + z_b) = 2I + 2*sym(Cov(z_a, z_b))
```

Pinning the sum to `N(0,I)` as well has its joint optimum at `sym(Cov) = -0.5*I` — hops that are
on average anti-aligned (mean cosine ~ -0.5). Consecutive ego-motion is positively correlated, so
that optimum is not reachable without cost.

Two reasons to run it anyway:

- SIGReg is a soft penalty at `sigreg_weight=0.005`, against recon + compose. It applies mild
  pressure, it does not impose a constraint.
- The mixture test is weaker still: the model can partly satisfy it by scale-splitting the streams
  (hops slightly below unit, sum slightly above) rather than by decorrelating anything.

The realistic outcomes are a small effect, some scale-splitting, or a slow hop-scale shrink. Which
one happens is the experiment. Under `z_norm=false` the decoder sees raw z, so a shrink migrates
into decoder weights — the same dynamic as the stage-2 bottleneck scale collapse. The diagnostics
below are what tell these apart.

## Code changes

1. **`_compose_forward` (`occrae/deltatok_trainer.py:961`)** — return `z_comp` alongside `z`. It is
   already computed at `:948`; only the return tuple changes.

2. **Compose branch (`:1055-1057`)** — replace the `hop_idx` pick with the 3-way cat, gated on the
   new flag. Off = today's random hop, bit-identical.

3. **`_sigreg_pooled` (`:671-705`)** — unchanged. One stream, one FIFO buffer, existing `scale`.

4. **`sigreg_gap_sigma` branch (`:1058-1060`)** — currently divides the chosen hop by
   `sqrt(per-sample gap)`. With three streams it would need three divisors (`g1`, `g2`, `g1+g2`)
   and would otherwise mis-scale silently. Assert the two flags are mutually exclusive; gapsig is
   measured-negative, so there is no arm that needs both.

5. **Config (`configs/deltatok/train_deltatok.yaml`, near `:127`)** — add `sigreg_compose_z: false`.
   It must live in the **base** yaml: OmegaConf struct mode rejects
   `training.sigreg_compose_z=true` for an absent key even though the trainer reads it via `.get()`.
   The alternative is a `+` prefix in the slurm `EXTRA_CFG`.

6. **Init assert (`:633-639`)** — extend the existing `compose_weight > 0` block: `sigreg_compose_z`
   requires `compose_weight > 0` and `sigreg_weight > 0`.

## Pool sizing

Triple `sigreg_pool_samples` to **24576** in the arm, so the estimator matches today's exactly.

`scale = (C/W + L)/L` where `C = sigreg_pool_samples`, `W` = world size, `L` = live rows
(`:697-704`). Tripling both `C` and `L` leaves it invariant. This is exact, not approximate, at the
live configuration: `cap_local = ceil(8192/4) = 2048` triples to exactly `6144`, every FIFO entry's
row count triples uniformly, so eviction pops the same entries and `scale` is step-for-step
identical. The `ceil` and the `while len(buf) > 1` overshoot guard never bite here — a single
micro-batch is ~768 rows against a 6144 cap.

The pool's history span in micro-batches is also preserved, so this does **not** re-trigger the
`pool=32768` failure mode (`scale` 11.7 -> 43.7, 21 optim steps of frozen history, Eval stuck at
0.12). See `deltatok-sigreg-pool-size-not-weight-neutral`.

Leaving the pool at 8192 would instead drop `scale` by ~3x and shorten the history span by 3x —
a different regime, confounding the arm.

## Diagnostics (required to make the arm falsifiable)

Nothing today measures any of the quantities this change acts on. Eval builds `z_bneck` from a
single gap-1 pair and calls `z_spread.update(z_bneck)` (`:1289`); `z_comp` never appears in the
eval path at all.

Add a **train-time** readout in the compose branch — `z_comp` is already in hand, so this costs no
extra encode or forward:

- `ZPartRank` per stream (hop A, hop B, comp) via three `ZSpreadStats` accumulators.
- `ZTotalVar` per stream — this is what separates scale-splitting from a genuine shrink.
- Mean hop-hop cosine `<z_a, z_b> / (|z_a||z_b|)` — the direct readout of the anti-alignment
  pressure, and the only way to see whether SIGReg is moving `Cov` at all.

Without these, a null result is uninterpretable: a flat `Eval/LossRecon_Comp` could mean the term
did nothing, or that it was bought off by scale-splitting.

Unverified: whether `ZSpreadStats.update` (`occrae/z_spread.py:39`) tolerates three accumulators
per step without the distributed `summary()` collectives (`:52`) getting out of order. Check before
implementing.

## Arm setup

New arm script derived from `slurm/deltatok/train_deltatok_compose_sigreg_bsc.slurm`
(`sigreg_num_slices=3072`, `sigreg_warmup=2000`, `max_gap=9`, `z_norm=false`).

Changes: `training.sigreg_compose_z=true`, `training.sigreg_pool_samples=24576`.

**Force a new `RUN_NAME`.** Training always resumes from `<RESULTS_DIR>/<RUN_NAME>/ckpts/current.pth`
when it exists, so relaunching the existing arm with the flag flipped would silently change the
objective mid-curve.

**Memory, verify before launch.** `x_t` in `occrae/sigreg.py:66` is `(L_live, 3072, 17)` fp32:
~53 MB -> ~160 MB at 3x rows, plus cos/sin transients of the same size. Order +0.5 GB peak on an arm
that already halved `bsize`. This is arithmetic, not measured.

## How to judge it

Primary: `Eval/LossRecon_Comp` against the matched compose+sigreg twin at the same epoch.

The stdout `SIGReg:` scalar is **not comparable** across this change — the row population changed,
and one third of the rows are now exact linear combinations of the other two thirds, so the
effective sample count behind the CF is below `n_pooled` and the finite-sample floor sits higher.
Do not read the absolute value as progress or regression.

Watch for the shrink signature: hop `ZTotalVar` drifting below 1 while comp `ZTotalVar` rises
toward it, with recon flat. That is scale-splitting, and it means the term was bought off.

## Not doing

- **Three separate SIGReg statistics.** Would give a real per-stream constraint, but departs from
  the pooled-mixture structure the multi-gap arm already validates, and costs 3 all-reduces plus
  3 pools.
- **Rescaling `z_comp`** (`/sqrt(2)`, gap-aware `/sqrt(t2-t0)`, or scale-free `/rms.detach()`).
  Deliberately excluded — the point of this arm is to test the plain additive application first.
- **A one-sided rank floor** (hinge on participation ratio) instead of a distribution match. The
  fallback if this arm shows the shrink signature.
- **Touching the eval path.** Train-time diagnostics are enough to read this arm.
