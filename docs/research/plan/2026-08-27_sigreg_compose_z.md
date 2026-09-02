# SIGReg on the composed sum

## Context

In the compose arm, SIGReg sees **one randomly chosen hop** per micro-batch
(`occrae/deltatok_trainer.py:1071-1073`):

```python
hop_idx = torch.randint(0, 2, ()).item()              # 0 or 1: pick hop t0→t1 or t1→t2
z_bneck = z_compose[:, hop_idx]                              # (B, N, K, Cz)
```

The composed sum `z_comp = z_a + z_b` is built at `:966`, decoded for `loss_compose`, then
discarded — it never reaches the regulariser. The random pick exists only to hold the live row
count equal to the single-pair arms, so `sigreg_weight` transfers between compose and plain arms.

## Goal

`z_a`, `z_b` and `z_comp` all belong to **one distribution**. They are the same kind of object: a
delta over a gap. `max_gap=9` already draws hop gaps from `U[1,9]` and pools them through one CF
test, so `z_comp` — a delta over gap `g1+g2` — is another draw from that same population, not a
foreign stream. Put all three in the one existing SIGReg call.

## Code changes

1. ✅ **`_compose_forward` (`:944-978`)** — `z_comp` stacked as `z[:, 2]` (`:978`), return stays
   4 values `(loss_recon, loss_compose, z, step_t)` with `z` shape `(B, 3, N, K, Cz)`.

2. ✅ **Read the flag in `__init__`** (`:644`):
   `self._sigreg_compose_z = bool(self.cfg.training.get("sigreg_compose_z", False))`.

3. ✅ **Compose branch (`:1072-1084`)** — gated on `_sigreg_compose_z`. Flag on: 3-way cat of
   `z_compose[:, 0..2]` → `(3B, N, K, Cz)`. Off: random hop, bit-identical to before.

   `_sigreg_pooled` reshapes to `(-1, Cz)` at `:710`, so the cat axis is cosmetic. Live rows
   `L -> 3L`.

4. ✅ **`_sigreg_pooled` (`:688-719`)** — unchanged. One stream, one FIFO buffer, existing `scale`.

5. ✅ **Mutual-exclusion assert (`:648-650`)** — `sigreg_compose_z` and `sigreg_gap_sigma` cannot
   both be true. Three streams would need three divisors (`g1`, `g2`, `g1+g2`); gapsig is
   measured-negative, so no arm needs both.

6. ✅ **Config (`configs/deltatok/train_deltatok.yaml:128`)** — `sigreg_compose_z: false`.

7. ✅ **Init assert (`:645-647`)** — `sigreg_compose_z` requires `compose_weight > 0` and
   `sigreg_weight > 0`.

8. ✅ **Diagnostics** — `ZSpreadStats` accumulator constructed in `__init__` (`:655-656`),
   `_log_z_spread` helper (`:658-686`), `.update(z_bneck)` in the compose SIGReg block (`:1084`),
   epoch flush before `return stats` (`:1216`). Runs with or without `sigreg_compose_z`.

   `ZSpreadStats.update` (`occrae/z_spread.py:38`) does no collectives, only local float64 moment
   accumulation. `summary()` (`:51`) holds the all-reduces, so one train accumulator alongside the
   eval ones is safe as long as every rank flushes in the same order.

## Pool sizing

**Keep `sigreg_pool_samples` at 8192.** The arm takes the `scale` drop rather than compensating for
it, matching the standing `# NOT weight-neutral; keep 8192 (2026-07-31)` note in the arm script.

`scale = (P + L)/L` where `P` = pooled rows, `L` = live rows (`:704`). `cap = ceil(8192/4) = 2048`
rows per rank. Tripling `L` at a fixed pool drops `scale` ~3x:

| arm | `L` | pool | `scale` |
|---|---|---|---|
| plain, N=1 | 128 | 2048 | 17.0 |
| plain, N=2 | 256 | 2048 | 9.0 |
| compose_z, N=1 | 384 | 2048 | 6.33 |
| compose_z, N=2 | 768 | 2048 | 3.67 |

The pool sits at exactly `cap` in every arm because eviction is row-granular (`:703`). Row count
holds; what shortens ~3x is the pool's **history span in micro-batches** (16 -> 5.3 at N=1).

This is an accepted confound, not a neutral change. `sigreg_weight` is not comparable between the
compose_z arm and its twin at equal nominal weight — the effective pressure is `weight * scale`, so
the arm runs at ~1/3 the twin's. Read the arm against the twin on recon, not on `SIGReg:`.

Tripling the pool to 24576 would restore `scale` to 17.0 and preserve the 16-micro-batch span. It
was rejected to keep the pool knob fixed across the whole sweep. The failure mode that motivates
caution here is `pool=32768` (`scale` 11.7 -> 43.7, 21 optim steps of frozen history, Eval stuck at
~0.10-0.12): see `../analysis/2026-07-31_sigreg_pool_not_weight_neutral.html`.

## Arm setup

New arm script derived from `slurm/deltatok/train_deltatok_compose_sigreg_bsc.slurm`
(`sigreg_num_slices=3072`, `sigreg_warmup=2000`, `max_gap=9`, `z_norm=false`).

Change: `training.sigreg_compose_z=true`. Pool stays at the script's `sigreg_pool_samples=8192`.

**Force a new `RUN_NAME`.** Training always resumes from `<RESULTS_DIR>/<RUN_NAME>/ckpts/current.pth`
when it exists, so relaunching the existing arm with the flag flipped would silently change the
objective mid-curve.

**Memory, verify before launch.** `x_t` in `occrae/sigreg.py:66` is `(L_live, 3072, 17)` fp32:
~53 MB -> ~160 MB at 3x rows, plus cos/sin transients of the same size. Order +0.5 GB peak on an
arm that already halved `bsize`. Arithmetic, not measured.

## How to judge it

Primary: `Eval/LossRecon_Comp` against the matched compose+sigreg twin at the same epoch.

The stdout `SIGReg:` scalar is **not comparable** across this change — the row population changed,
and one third of the rows are exact linear combinations of the other two thirds, so the effective
sample count behind the CF is below `n_pooled` and the finite-sample floor sits higher. Do not read
the absolute value as progress or regression.

Watch `Train/ZTotalVar` drifting below 1 with recon flat: the code shrinking rather than the
streams merging. Under `z_norm=false` the decoder sees raw z, so a shrink migrates into decoder
weights — the same dynamic as the stage-2 bottleneck scale collapse.

## Not doing

- **Separate SIGReg statistics per stream.** The goal is one distribution, so one pooled test is
  the matching objective. Would also cost extra all-reduces and pools.
- **Rescaling `z_comp`** (`/sqrt(2)`, gap-aware `/sqrt(t2-t0)`, or scale-free `/rms.detach()`).
  The point of this arm is the plain additive application.
- **Per-stream diagnostic accumulators** (hop A / hop B / comp separately). One pool, one readout.
- **A hop-hop cosine readout.** It measured anti-alignment pressure, which only arises if hops are
  individually pinned to unit variance. They are not — the pool mixes gaps.
- **Touching the eval path.** Train-time diagnostics are enough to read this arm.
