# DeltaTok: additive delta-token composition

## Context

DeltaTok learns `decode(encode(f_i, f_j), f_i) ≈ f_j` for one pair per sequence
(`occrae/deltatok_trainer.py:990-1012`). Longer transitions exist only by chaining the decoder
(`_rollout_from_z`, `occrae/deltatok_shared.py:343`), and error compounds. Goal: an additive delta
token, so a span is composed in latent space. From a sampled triplet (1, 3, 5):

```
f1,f3 -> d13    f3,f5 -> d35    (encode the two hops)
d15 = d13 + d35                 (compose: plain addition)
f1 + d15 -> f5                  (apply: one decode, no chaining)
```

Apply is already `net.decode(z, x_prev, ...)`, and `max_gap=9` trains long-gap `encode()`
in-distribution. The new part is producing `d15` without frames. That does not hold at init; the
compose loss trains it in. Two prerequisites:

- **`z_norm=false`.** The z LayerNorm pins each token to `|z|=sqrt(Cz)` on a zero-mean hyperplane
  (`occrae/deltatok_trainer.py:174-181`); a sum of two such tokens leaves that sphere. Production
  arms already run the norm off.
- **A SIGReg base.** SIGReg keeps the code zero-mean with `TotalVar/Cz ≈ 1.006`, so sums stay
  in-range. Build on a SIGReg arm, not nosigreg.

## Design

**Composition is plain addition** — no new parameters, no compose head:

```python
z = z.reshape(B, 2, N, K, Cz)  # (B, 2, N, K, Cz) the 2 hops of triplet (τ0, τ1, τ2)
z_a = z[:, 0]                  # (B, N, K, Cz) d(τ0 -> τ1), e.g. d13
z_b = z[:, 1]                  # (B, N, K, Cz) d(τ1 -> τ2), e.g. d35
z_comp = z_a + z_b             # (B, N, K, Cz) d(τ0 -> τ2), e.g. d15
```

One triplet, two hops, one `+`. No mask, no sub-span selection, no chain-length config. All the
variety comes from step 1 resampling `(τ0, τ1, τ2)` every micro-batch, so a fixed pair of hops
still covers every gap combination. Longer chains are a separate arm (see *Not doing*).

Zero new params: existing checkpoints load unchanged, so the arm warm-starts from the mg9/SIGReg
checkpoint via `INIT_CKPT` (`sh/train_deltatok.sh:35-37`).

**Per micro-batch:**

1. **Sample timesteps.** 3 distinct per sequence, uniform (`rand(B, T).argsort()[:, :3]`, sorted)
   → `τ0 < τ1 < τ2`. Per-sequence, never shared across the batch. `T=10` caps the composed span at
   9 = `max_gap`, so composed deltas stay in the trained gap range; per-hop gaps land in [1, 8].
2. **Backbone.** Frozen backbone on those 3 timesteps only (new `_extract_window_feats`) →
   `feats (B, 3, N, P, C)`.
3. **One DDP forward.** Everything trainable runs in a single `self.tokenizer(...)` call, via the
   new `compose` flag. Correctness, not style: `net.encode`/`net.decode` on the unwrapped module
   skip DDP's reducer hooks, gradients never all-reduce, and ranks silently diverge. The single
   forward also builds `rope_local`/`rope_global` once; `_compute_global_rope` permutes
   camera↔position while `self.training` (`occrae/deltatok_trainer.py:273-274`), and only a shared
   build keeps the two hops comparable.
4. **Encode, compose, decode.** `z = encode(feats[:, :-1], feats[:, 1:], ...)` → `(B*2, N, K, Cz)`;
   `z_comp = z_a + z_b` as above; decode the 2 hops and the composed delta in one batched `decode`
   (rows are independent, bit-identical to separate calls).
5. **Base pair recon.** Existing loss, over both hops:
   `log_cosh(x_hat_hops, feats[:, 1:].detach())`.
6. **Composed recon.** Anchor `feats[:, 0]`, target `feats[:, 2]`, loss
   `log_cosh(x_hat_comp, target.detach())`.
7. **Total.** `loss_total = loss_recon + compose_weight * loss_compose` (+ existing SIGReg term).

**No latent-consistency term.** `d12+d23` is never matched against `encode(f1,f3)`. Additivity is
enforced through the decoder only.

**SIGReg row count.** A triplet gives 2× the rows of one pair, and SIGReg weight is not
row-count-neutral (`scale = pool/live`, `occrae/deltatok_trainer.py:685`;
`../analysis/2026-07-31_sigreg_pool_not_weight_neutral.html`). `compose_sigreg_one_hop`
feeds one random hop per sequence (true) or both (false). This arm runs **false**: 2 hops ×
bsize 2 = the bsize-4 arms' live row count, so sweep results transfer. Composed deltas never reach
SIGReg.

## Files to change

### 1. `occrae/deltatok_shared.py` — new `_extract_window_feats`

Next to `_extract_pair_feats` (line 156): the same gather, generalised from 2 timesteps to `S`, no
folding into the batch axis. Leave `_extract_pair_feats` untouched; every existing caller stays
bit-identical.

```python
def _extract_window_feats(self, imgs, num_cameras, step_t):
    """Encode only the timesteps in step_t (B, S) -> feats (B, S, N, P, C), H, W."""
```

### 2. `occrae/deltatok_trainer.py` — training path

- **`DeltaTokModule.forward` (line 430):** one new kwarg `compose=False` (bool). Rows arrive
  hop-major, 2 per sequence, so everything is derived: `z.reshape(-1, 2, N, K, Cz)` then
  `z_a + z_b`, and the anchor is `x_prev.reshape(-1, 2, N, P, C)[:, 0]` (row `2b` of `x_prev` is
  already `feats[b, 0]`). Returns `(x_hat_hops, x_hat_comp, z_adj)`.
- **`__init__` (~line 604, beside the SIGReg block):** read the two `compose_*` keys. When
  `compose_weight > 0`, assert `model.deltatok.z_norm == false`, `max_gap >= 2`, and
  `bottleneck_recon_weight == 0` (the compose branch produces no `z_pre`/`z_rec`).
- **New `_compose_forward(imgs, num_cameras)`:** design steps 1-6, returns
  `(loss_recon, loss_compose, z_adj)`.
- **`train_one_epoch` (line 990):** branch on `compose_weight > 0`; the `else` branch is the
  current code, unchanged. Add `loss_compose` to `loss_total` next to `loss_bneck` (line 1024);
  log `Train/LossCompose` (line 1090).

### 3. Eval diagnostics — `occrae/deltatok_trainer.py` + `occrae/metric.py`

Two always-on keys on the T=5 eval windows, reusing `_encode_pair_deltas`. They measure composition
on any checkpoint, so the baseline exists before a compose arm launches.

- **`LossRecon_Comp`** — `z.cumsum(dim=1)` over `z (B, T-1, N, K, Cz)` gives the sum of hops `0..t`
  at every `t`; decode all of them from frame 0. Latent composition vs decoder chaining, on the
  same frames as `LossRecon_AR`.
- **`LossRecon_Direct`** — `decode(encode(f0, f_{t+1}), f0)`. The ceiling for `LossRecon_Comp`;
  the gap is what additivity costs.

Register both in `_EVAL_LOSS_KEYS` (`occrae/metric.py:13-25`) **first** —
`DeltaTokEvalMetric.update` raises `AttributeError` on unregistered keys. Per-loader logging
(`deltatok_trainer.py:1413`) picks them up automatically.

### 4. `configs/deltatok/train_deltatok.yaml` — new keys under `training:`

```yaml
  compose_weight: 0.0          # 0 = off; existing single-pair sampling path is bit-identical
  compose_sigreg_one_hop: true # feed SIGReg one hop/sequence (false = both)
```

All reads go through `cfg.training.get(...)`, so overlays and existing SLURM arms keep working.

### 5. `slurm/deltatok/train_deltatok_compose_bsc.slurm` — new arm

`cp slurm/deltatok/train_deltatok_maxgap_sigreg_nozn_bsc.slurm`, then edit:

- **`--account=ehpc1001`** — the source carries stale `ehpc880`.
- **`CONFIG_NAME=train_deltatok_nt10_bsc`** — `num_timesteps=10` caps the composed span at
  `max_gap=9`.
- **Compose keys:** `training.compose_weight` (`COMPOSE_WEIGHT` env, default 1.0 — both terms are
  same-scale log-cosh), `compose_sigreg_one_hop=false`.
- **`training.bsize=2 training.effective_bsize=16`** (grad_cum 2). A triplet step is 2 encodes +
  3 decodes vs 1 + 1, so at half bsize it holds ~1.2× the activations of the bsize-4 arms known to
  fit. OOM fallback: `bsize=1`, grad_cum 4.
- **`INIT_CKPT`** in the header comment, pointing at the mg9/SIGReg checkpoint.

## Verification

1. **Config no-op.** 20 iters of `train_deltatok_maxgap_sigreg_nozn_bsc.slurm` on `acc_debug`;
   `Train/LossRecon` must match pre-change values at the same seed.
2. **Smoke test** on `qos=acc_debug` (2 h) before any production submit:
   ```bash
   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_bsc.slurm'"
   ```
   Watch in the background until `RUNNING` and a first loss line. `LossCompose` must be finite and
   above `LossRecon` at iter 0; equal values mean `z_comp` collapsed to a single hop.
3. **Baseline additivity.** `EVAL_ONLY=1` on an existing mg9/SIGReg checkpoint; read
   `Eval/<test>/LossRecon_Comp` vs `LossRecon_Direct` vs `LossRecon_AR`. Quantifies non-additivity
   before any training.
4. **Success criterion.** `LossRecon_Comp` falls toward `LossRecon_Direct` and beats
   `LossRecon_AR`. Watch `ZPartRank` / `ZTotalVar`; the max_gap sweep saw hard collapse
   (`ZPartRank` → ~1.7). Kill a collapsing arm, do not wait it out.
5. **Sync check.** The user syncs manually. Grep the remote slurm script and
   `occrae/deltatok_trainer.py` before every `sbatch`; ask for a sync if stale.

## Not doing

- **No latent-consistency loss.** `d12+d23` is never matched to `encode(f1,f3)`.
- **No compose head or gated sum.** Plain addition, zero new parameters.
- **No flow-trainer changes.** `occrae/deltatok_flow_trainer.py` and its configs are untouched.
- **Longer chains are extrapolation.** Only 2-hop sums are supervised, and 2-term additivity does
  not guarantee 3- or 4-term. `LossRecon_Comp` on the T=5 eval windows (up to 4 hops) measures
  exactly that. If it fails, a longer-chain arm generalises the same code: sample `L+1` timesteps
  and sum `L` hops.
- **K-slot alignment is not addressed.** Nothing binds slot `k` of `d12` to slot `k` of `d23`
  beyond the shared `z_embed` init — the same weak-binding class as the camera-swap bug. If the
  compose loss plateaus, slot binding is the next suspect.
- **No local run.** Everything executes on BSC.
