# Plan: Replace flow-model cross-attention conditioning with a pooled frame-0 context token

## Context

The DeltaTok flow transformer (`occrae/network/efficient_transformer.py`, class `Transformer`)
currently conditions on the first frame via **per-slot cross-attention**: a
`cross_cond` tensor of shape `(B, N, Hp, Wp, C)` (camera `i`'s frame-0 patch grid is
seen only by delta slot `i`). This per-slot cross-attention is the only thing that
binds each generated delta to its camera, and that binding is weak (gated `alpha_cross`
starts at 0; eroded by symmetric cross-camera spatial attention), which motivated the
recent `camera_embed` swap fix and is contributing to the overfitting the user is seeing.

We want to **remove cross-attention from the flow path** and instead **concat frame-0
with the delta tokens** so the existing self-attention handles conditioning — making the
per-camera binding structural. Decisions confirmed with the user:
- **Pooled frame-0 as a clean leading temporal slot** (mean-pool each camera's `P`
  frame-0 patches to one token, prepend as transition slot 0, kept un-noised and out of
  the loss). Reuses the existing `context` machinery in `flow_noising`/`flow_loss`/
  `flow_euler_sample`. Smallest diff; frame-0 is compressed `P→1`.
- **Config flag `model.cond_mode: cross | concat`**, default `cross` (keeps current
  behavior, legacy CLIP path, and old checkpoints working). New runs set `concat`.

Key enabling facts found during exploration:
- `flow_euler_sample` already keeps the first `context` temporal slots fixed
  (`generation_helper.py:68-70, 89-90`) — **no sampler change needed**.
- `cross_dim=None` already builds the Transformer with cross-attention disabled
  (`efficient_transformer.py:61-65, 211-230`) — **no transformer change needed**.
- Temporal self-attention folds cameras into the batch (`b (t s) d -> (b s) t d`,
  `efficient_transformer.py:110`), so a frame-0 token at spatial slot `n` only mixes with
  camera `n`'s deltas → structural per-camera binding for free.
- `DeltaTokModule.decode` (`deltatok_trainer.py:356`) is the existing concat reference.

## Approach

Add a `cond_mode` switch in the flow trainer. In `concat` mode: build the model with
`cross_dim=None`, prepend a pooled+normalized frame-0 token as temporal slot 0, set
`context=1` everywhere (noising, loss, sampling), and drop slot 0 from the sampled output.
The frame-0 patch grid is still used full-resolution by the tokenizer decode rollout
(`_rollout_from_z(self.deltatok, feats[:, 0], ...)`) — only the *flow conditioning* changes.

## Files to modify

### 1. `configs/train_deltatok_flow.yaml` (+ overlays inherit)
Add to the `model:` block (around lines 52-86):
```yaml
  cond_mode: cross   # cross | concat  (concat = pooled frame-0 context slot, no cross-attn)
```
No change needed in the `_karolina`/`_jeanzay`/`_overfit` overlays — they inherit; the
concat experiment is selected per-run via `--cfg model.cond_mode=concat`.

### 2. `occrae/deltatok_flow_trainer.py` (the bulk of the change)

**a. Read the flag in `__init__`** (near `self.num_views`, ~line 80):
```python
self.cond_mode = str(self.cfg.model.get("cond_mode", "cross"))
assert self.cond_mode in ("cross", "concat")
self.n_ctx = 1 if self.cond_mode == "concat" else 0   # clean leading temporal slots
```

**b. `get_network("vit")`** (~lines 196-215): make `cross_dim` depend on the mode so the
concat model never builds the cross-attention params:
```python
cross_dim = 1536 if self.cond_mode == "cross" else None
model = Transformer(..., cross_dim=cross_dim, ...)
```

**c. New helper `_build_concat_context`** (next to `_build_cross_cond`, ~line 311),
reusing `ctx_norm` for scale-matching with the unit-scale delta tokens:
```python
def _build_concat_context(self, x0, H, W):
    """Pooled frame-0 token per camera, in flow latent layout (B, C, 1, N, 1)."""
    B, N, P, C = x0.shape                                   # frame-0 patches
    ctx = x0.mean(dim=2)                                    # (B, N, C) mean-pool patches -> 1 tok/cam
    ctx_norm = str(self.cfg.model.get("ctx_norm", "layernorm"))
    if ctx_norm == "layernorm":
        ctx = F.layer_norm(ctx.float(), (C,)).to(x0.dtype)  # (B, N, C) unit scale (match delta tokens)
    return ctx.permute(0, 2, 1)[:, :, None, :, None].contiguous()  # (B, C, 1, N, 1) prepend slot
```

**d. A small shared input builder** so train/eval/visualize don't drift. Returns the flow
target (with the context slot prepended in concat mode) and the cross_cond:
```python
def _flow_inputs(self, feats, H, W):
    z = self._encode_deltas(feats, H, W)                          # (B, T-1, N, C)
    x_spatial = z.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()  # (B, C, T-1, N, 1)
    if self.cond_mode == "concat":
        f0 = self._build_concat_context(feats[:, 0], H, W)        # (B, C, 1, N, 1)
        T = x_spatial.shape[2] + 1
        assert T <= self.num_views, f"T={T} exceeds num_views={self.num_views}"
        x_target = torch.cat([f0, x_spatial], dim=2)             # (B, C, T, N, 1) slot0 clean
        return z, x_target, None                                  # no cross_cond
    cross_cond = self._build_cross_cond(feats[:, 0], H, W)        # (B, N, Hp, Wp, C)
    return z, x_spatial, cross_cond
```

**e. `train_one_epoch`** (~lines 445-459): replace the x_spatial/cross_cond build + the
`context=0` calls with `self._flow_inputs(...)` and `context=self.n_ctx` in both
`flow_noising` and `flow_loss`. The model forward already takes `cross_cond` (None in
concat mode).

**f. `eval_one_epoch`** — two spots:
- Flow-loss spot (~lines 618-620, 679-689): use `_flow_inputs` + `context=self.n_ctx`.
- Sampling spot (~lines 624-634): build init noise with the clean slot prepended in
  concat mode and drop it after, e.g.:
  ```python
  if self.cond_mode == "concat":
      f0 = self._build_concat_context(feats[:, 0], H, W)            # (B, C, 1, N, 1)
      zr = torch.cat([f0, torch.randn(... T-1 ...)], dim=2)         # (B, C, T, N, 1)
  else:
      zr = torch.randn_like(x_spatial)
  gen = flow_euler_sample(..., context=self.n_ctx, cross_cond=cross_cond, ...)
  gen = gen[:, :, self.n_ctx:]                                       # drop the context slot
  z_hat = gen.squeeze(-1).permute(0, 2, 3, 1).contiguous()          # (B, T-1, N, C)
  ```
  The AR rollout (`_rollout_from_z(self.deltatok, feats[:, 0], z_hat, ...)`) is unchanged.

### 3. `visualize_deltatok_flow.py` (~lines 258-272)
Mirror the eval sampling branch (reuse `trainer._build_concat_context` /
`trainer.cond_mode` / `trainer.n_ctx`) so the viz script samples correctly in concat mode.

### No changes needed
- `occrae/network/efficient_transformer.py` — `cross_dim=None` disables cross-attn; the
  `if cross_cond is not None` guard (`:416`) skips the cross block when None.
- `occrae/generation_helper.py` — `flow_euler_sample` `context` handling already correct.
- `eval_occ_rae.py` — already calls `flow_euler_sample` without `cross_cond`.
- Legacy CLIP path in `generation_helper.generate_samples` — untouched (`cond_mode=cross`).

## Verification

All runs on Karolina (`ssh karolina`, `conda activate occany`); never run locally. Verify
the cluster copy matches local before any SLURM submit, and ask the user to sync.

1. **Smoke / overfit, concat mode** (fast sanity that shapes + context wiring are correct):
   ```
   bash sh/train_deltatok_flow.sh   # with CONFIG_NAME=...overfit, EXTRA_CFG="model.cond_mode=concat"
   ```
   Expect: no shape errors; sanity-check eval prints `LossFlow` and the `LossPointmap`/
   `LossDepth` rollout metrics; train loss decreases.
2. **Confirm cross mode still works** (regression): run the same overfit with
   `model.cond_mode=cross` and confirm parity with current behavior, and that an existing
   cross checkpoint still loads (`--ckpt ...`, strict=False).
3. **Param-count check**: in concat mode the model log should show fewer params (no
   `cross_cond_emb` / `cross_spatial_pos`), confirming `cross_dim=None` took effect.
4. **Visualization**: run `visualize_deltatok_flow.py` in concat mode on a few scenes;
   confirm forecast depth/RGB panels render and the generated views stay matched to their
   cameras (the swap behavior this change targets).
5. **Overfitting comparison**: the user's actual goal — compare train vs eval `LossFlow`
   curves between `cross` and `concat` to see whether the structural binding reduces the
   gap.
