# DeltaTok: additive delta-token composition

## Context

DeltaTok today learns one thing: `decode(encode(f_i, f_j), f_i) ≈ f_j` for a single sampled pair per
sequence (`occrae/deltatok_trainer.py:990-1012`). Long-range transitions are reached only by chaining
the *decoder* — `_rollout_from_z` (`occrae/deltatok_shared.py:343`) decodes `d12` onto `f1`, feeds the
prediction back, decodes `d23`. Error compounds, and there is no algebra on the latent itself.

The goal is to make the delta token additive, so a transition can be built in latent space:

```
f1,f2 -> d12    f2,f3 -> d23    f3,f4 -> d34     (encode adjacent hops)
d13 = d12 + d23         d14 = d12 + d23 + d34    (compose: plain addition)
f1 + d13 -> f3          f2 + d24 -> f4           (apply: one decode, no chaining)
```

Half of this already exists. `f_i + d_ij -> f_j` is literally `net.decode(z, x_prev, ...)`, and
`max_gap=9` already trains `encode(f1, f3)` in-distribution, so a long-range delta is not a new object.
The new capability is producing `d13` from `(d12, d23)` **without touching frames**, and supervising it.

Nothing in `docs/`, `docs/plans/`, or the code mentions composition, additivity, or latent algebra —
grep for `compos*`/`additiv*`/`abelian`/`d12` returns zero real hits. This is new ground, and there is
no reason for `d12 + d23 ≈ d13` to hold at initialisation. It has to be trained in explicitly.

Two facts from the existing research make it feasible:
- **`z_norm=false` on every production arm.** The z LayerNorm pins each token to `|z|=sqrt(Cz)` on a
  zero-mean hyperplane (`occrae/deltatok_trainer.py:174-181`); a sum of two such tokens is off that
  sphere. Additivity is only representable with the norm off, which the arms already do.
- **SIGReg supplies the scale anchor.** At `sigreg_weight>0` the code is zero-mean with
  `TotalVar/Cz ≈ 1.006`. The nosigreg mg9 arm has per-element RMS ~14 and worst-channel mean offset
  408 — a raw sum doubles that offset. Build on a SIGReg arm.

## Design

**Composition is plain addition.** No new parameters, no compose head, no cumsum. A composed delta is
the sum of the adjacent deltas it spans, selected by a mask:

```python
# z_adj: (B, L, N, K, Cz) — L adjacent hops per sequence
hop  = torch.arange(L, device=z_adj.device)                            # (L,)
mask = ((hop >= i[..., None]) & (hop < j[..., None])).to(z_adj.dtype)  # (B, n_comp, L)
z_comp = torch.einsum('bcl,blnkd->bcnkd', mask, z_adj)                 # (B, n_comp, N, K, Cz)
```

Because there are no new params, **existing checkpoints load unchanged** and a compose arm can be
warm-started from the mg9/SIGReg checkpoint via `INIT_CKPT` (`sh/train_deltatok.sh:35-37`).

**Per micro-batch:**

1. Sample a window of `L+1` timesteps per sequence. Draw a total span `s ~ U[L, max_gap]` once per
   micro-batch (mirroring how `gap` is drawn today), then per-sequence `t0 ~ U[0, T-s)` and `L-1`
   interior cut points, giving sorted `t0 = τ0 < τ1 < … < τL = t0+s`. Per-hop gaps vary; the composed
   span still reaches `max_gap`, so composed deltas stay in the range `encode()` was trained on.
2. Run the frozen backbone on those `L+1` timesteps only (new `_extract_window_feats`) →
   `feats (B, L+1, N, P, C)`.
3. Build `rope_local` / `rope_global` **once** and reuse for every encode and decode in the step.
   `_compute_global_rope` randomly permutes camera↔position while `self.training`
   (`occrae/deltatok_trainer.py:273-274`); rebuilding it per call would make `d12` and `d23`
   incomparable within the same step.
4. `z_adj = net.encode(feats[:, :-1], feats[:, 1:], ...)` → `(B*L, N, K, Cz)`, reshape to `(B, L, …)`.
5. **Base pair recon** (existing loss, now averaged over all `L` hops instead of one pair):
   `log_cosh(net.decode(z_adj, feats[:, :-1]), feats[:, 1:].detach())`.
6. **Composed recon**: sample `n_comp` spans `(i, j)` with `j-i >= 2` per sequence — draw
   `span ~ U[2, L]` then `i ~ U[0, L-span]`, which weights 2-hop and L-hop equally. Mask-sum to
   `z_comp`, gather anchor `feats[b, i]` and target `feats[b, j]`, and
   `log_cosh(net.decode(z_comp, anchor), target.detach())`.
7. `loss_total = loss_recon + compose_weight * loss_compose` (+ existing SIGReg term).

**No latent-consistency term.** `d12+d23` is never matched against `encode(f1,f3)`. Additivity is
enforced functionally through the decoder only.

**SIGReg row count.** `z_adj` has `L×` more rows than today's single pair. `sigreg_pool_samples` is
documented as *not* weight-neutral (`docs/journal/deltatok_sigreg_pool_not_weight_neutral_2026-07-31.html`)
— raising 8192→32768 at fixed weight stopped learning entirely, via `scale = pool/live` in
`_sigreg_pooled` (`occrae/deltatok_trainer.py:685`). So by default feed SIGReg **one randomly chosen
hop per sequence**, keeping the live row count identical to a non-compose arm and the sweep results
transferable. Composed deltas are never fed to SIGReg.

## Files to change

### 1. `occrae/deltatok_shared.py` — new `_extract_window_feats`

Add next to `_extract_pair_feats` (line 156). Same gather, generalised from 2 timesteps to `S`, and
**no folding into the batch axis**:

```python
def _extract_window_feats(self, imgs, num_cameras, step_t):
    """Encode only the timesteps in step_t (B, S) -> feats (B, S, N, P, C), H, W."""
```

Leave `_extract_pair_feats` untouched — every existing caller (eval, flow trainer, AR rollout) stays
bit-identical. The ~8 duplicated gather lines are deliberate: refactoring the shared helper would put
the three running BSC arms at risk.

### 2. `occrae/deltatok_trainer.py` — the training path

- `__init__` (~line 604, beside the SIGReg block): read `compose_weight`, `compose_chain_len`,
  `compose_num_pairs`, `compose_sigreg_one_hop`. When `compose_weight > 0`, assert
  `model.deltatok.z_norm == false` (additivity is unrepresentable on the LN sphere),
  `compose_chain_len >= 2`, `compose_chain_len <= max_gap < T`, and
  `bottleneck_recon_weight == 0` (the compose path calls `encode`/`decode` directly, so it does not
  produce `z_pre`/`z_rec`; stage-2 bottleneck is a separate arm).
- New `_compose_forward(imgs, num_cameras)` implementing steps 1-6 above and returning
  `(loss_recon, loss_compose, z_for_sigreg)`.
- `train_one_epoch` (line 990): branch on `compose_weight > 0`. The `else` branch is the current code,
  unchanged. Add `loss_compose` to `loss_total` next to the `loss_bneck` term (line 1024), and log
  `Train/LossCompose` beside the existing scalars (line 1090).

### 3. Eval diagnostics — `occrae/deltatok_trainer.py` + `occrae/metric.py`

Two new always-on keys on the T=5 eval windows, reusing the existing `_encode_pair_deltas` helper.
They make composition measurable on *any* arm, including the checkpoints from the in-flight SIGReg
sweep, so the baseline is quantified before a compose arm is launched.

- `LossRecon_Comp` — sum hops `0..t` (a `tril` mask-matmul over `z (B, T-1, N, K, Cz)`), decode once
  from frame 0. Directly comparable to the existing `LossRecon_AR` on the same frames: latent
  composition vs decoder chaining.
- `LossRecon_Direct` — `decode(encode(f0, f_{t+1}), f0)`. The ceiling for `LossRecon_Comp`: the gap
  between them is exactly what additivity costs.

Add both to `_EVAL_LOSS_KEYS` (`occrae/metric.py:13-25`) **first** — `DeltaTokEvalMetric.update`
raises `AttributeError` on any unregistered key. Per-loader logging at `deltatok_trainer.py:1413`
iterates `results.items()`, so it picks them up automatically.

### 4. `configs/deltatok/train_deltatok.yaml` — new keys under `training:`

```yaml
  compose_weight: 0.0          # 0 = off; existing single-pair sampling path is bit-identical
  compose_chain_len: 4         # L adjacent hops -> L+1 frames per sequence
  compose_num_pairs: 2         # composed spans (i,j), j-i>=2, supervised per sequence
  compose_sigreg_one_hop: true # feed SIGReg one hop/sequence so pressure matches non-compose arms
```

All reads go through `cfg.training.get(...)`, so overlays and every existing SLURM arm keep working.

### 5. `slurm/deltatok/train_deltatok_compose_bsc.slurm` — new arm

`cp slurm/deltatok/train_deltatok_maxgap_sigreg_nozn_bsc.slurm` then edit surgically. Differences:

- `--account=ehpc1001` (the source script carries `ehpc880`; per CLAUDE.md new files use `ehpc1001`,
  which the newest tc768 arm BSC:44478797-99 already uses).
- `CONFIG_NAME=train_deltatok_nt10_bsc` — `num_timesteps=10`, needed for `L=4` at `max_gap=9`.
- Add `training.compose_weight`, `compose_chain_len=4`, `compose_num_pairs=2`.
- `training.bsize=2 training.effective_bsize=16` (grad_cum 2). The step does ~4× the tokenizer work of
  a baseline arm — 4 encodes + 6 decodes vs 1 + 1, over a 2.5× larger frozen-backbone forward. Halving
  bsize keeps the memory profile close to the arms that are known to fit.
- `INIT_CKPT` documented in the header comment for warm-starting from the mg9/SIGReg checkpoint.

## Verification

1. **Config no-op.** Confirm a baseline arm is unaffected: run 20 iters of
   `train_deltatok_maxgap_sigreg_nozn_bsc.slurm` on `acc_debug` and check `Train/LossRecon` matches the
   pre-change values for the same seed.
2. **Smoke test the compose arm** on `qos=acc_debug` (2 h) before any production submit:
   ```bash
   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_bsc.slurm'"
   ```
   Watch in the background until `RUNNING` *and* a first loss line lands. Grep the `.out` for
   `LossCompose` — it must be finite and above `LossRecon` at iter 0 (an untrained sum should decode
   badly; if they are equal at iter 0, the mask-sum is collapsing to a single hop).
3. **Baseline additivity measurement.** Run `EVAL_ONLY=1` against an existing mg9/SIGReg checkpoint and
   read `Eval/<test>/LossRecon_Comp` vs `LossRecon_Direct` vs `LossRecon_AR`. This quantifies how far
   from additive the current tokenizer is, with no training — the number that says whether the compose
   loss is doing anything.
4. **Success criterion after training.** `LossRecon_Comp` falls toward `LossRecon_Direct`, and
   `LossRecon_Comp < LossRecon_AR` (one-shot composition beats decoder chaining). Watch `ZPartRank`
   and `ZTotalVar` — the max_gap sweep saw hard collapse on some arms (`ZPartRank` → ~1.7), so a
   collapsing compose arm must be killed, not waited out.
5. **Sync check before every submit.** The user syncs manually; grep the remote copy of the new slurm
   script and `occrae/deltatok_trainer.py` before `sbatch`, and ask for a sync if stale.

## Not doing

- **No latent-consistency loss.** `d12+d23` is never matched to `encode(f1,f3)`.
- **No compose head or gated sum.** Plain addition, zero new parameters.
- **No flow-trainer changes.** `occrae/deltatok_flow_trainer.py` and its configs are untouched; the
  additive latent is a property a later flow arm can exploit, not part of this change.
- **K-slot alignment is not addressed.** Nothing binds slot `k` of `d12` to slot `k` of `d23` beyond
  the shared `z_embed` init — the same weak-binding class as the camera-swap bug. Adding a loss is the
  cheap first test; if it plateaus, slot binding is the next suspect.
- **No local run.** Everything executes on BSC.
