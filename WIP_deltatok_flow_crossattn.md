# WIP: Delta-token flow matching (cross-attention conditioning)

Status snapshot so work can resume later. Branch: `revert`.

## Goal

Adapt `occrae/deltatok_flow_trainer.py` so the flow model no longer diffuses raw OccRAE
patch tokens. Instead it diffuses the **DeltaTok delta tokens** (the tokens
`test_deltatok_ar.py` works with): one 1536-d token per camera per frame transition,
produced by the frozen `DeltaTokModule.encode` (`occrae/deltatok_trainer.py:179`).
Context = the first frame; at eval the sampled deltas are decoded autoregressively with
the frozen DeltaTok decoder from the GT first frame, stitched into full tokens, and
decoded with OccRAE for pointmap/depth/ray metrics + viz.

## Decisions made (with user)

1. **Conditioning = cross-attention** on ALL frame-0 patch tokens (user first said
   "concat all tokens", then approved switching to cross-attention after trade-off
   discussion). Rationale: the diffused stream is tiny ((T−1)×N tokens) vs ~777+ context
   tokens — in-sequence context would burn ~99% of compute on discarded context updates,
   force `num_views≈800`, and pin training to a single resolution. Cross-attention
   removes the resolution constraint (multi-res dataset strings OK again) and makes
   sampling cheap. `Block.forward` already unpacks `gamma_cross, alpha_cross` AdaLN
   chunks; `CrossAttention` already exists in `occrae/network/transformer_block.py:116`;
   `flow_euler_sample` already forwards `cross_cond` (`occrae/generation_helper.py:73`).
2. **Frozen DeltaTok**: layer-12 surround ckpt
   `/mnt/proj1/eu-25-92/deltatok_log/deltatok_surround_constGlobalRope_layer12/ckpts/current.pth`,
   `model.encode_layer: 12`, arch block copied from `configs/train_deltatok.yaml`
   (`num_hidden_layers: 12`, swiglu/qk_norm/gated_attn true, mlp_ratio 4). Load with
   `strict=True`, strip `module.`/`_orig_mod.` prefixes, `.eval().requires_grad_(False)`,
   cast bf16, never DDP-wrap.
3. **Data pipeline**: switch flow trainer from `PreprocessedSequenceDataset` to the
   `get_data_loader()` pipeline of `DeltaTokTrainer` (reuse `_normalize_batch`,
   `num_cameras` surround handling, per-test-loader eval).

Original (superseded) plan file: `/home/acao/.claude/plans/adapt-the-occrae-deltatok-flow-trainer-p-clever-biscuit.md`
— still valid except the "in-sequence context slots" layout, replaced by cross-attention.

## Design (current)

- Flow latent `x = (B, C, T-1, N_cam, 1)` — delta tokens only. Spatial attention mixes
  cameras within a step; temporal attention mixes the T−1 delta steps.
- `context=0` everywhere (`flow_noising`, `flow_loss`, `flow_euler_sample`) — conditioning
  is purely via `cross_cond`.
- `cross_cond` = frame-0 spatial feats `feats[:, 0]` `(B, N, P, C)`, layer-normed
  (parameter-free `F.layer_norm(ctx.float(), (C,))`, config `model.ctx_norm:
  layernorm|none`; delta tokens exit a LayerNorm so they're ~unit-scale, raw DA3 feats
  are not), reshaped to `(B, N, Hp, Wp, C)` and passed to `Transformer.forward`.
  Keep the **un-normalized** `feats[:, 0]` for the DeltaTok decode rollout at eval.
- Transformer cfg: `num_views` = max T−1 (small, e.g. 8); `ref_spatial_size =
  (model.max_num_cameras=6, 1)`; `out_dim=1536`; `cross_dim=1536` enables the cross path.
- AdaLN-zero: `time_embed` last layer is zero-init → `alpha_cross` starts at 0 → cross
  branch gated off at init; no extra init needed.

### Tensor flow (train step)
```
batch = self._normalize_batch(self._next_train_batch())
imgs (B, V, 3, H, W), num_cameras = batch.get("num_cameras", 1), V = T*N
tokens, feats, _, _, H, W = self._extract_pair_feats(imgs, num_cameras)   # feats (B,T,N,P,C)
z = self._encode_deltas(feats, H, W)                                      # (B, T-1, N, C) frozen
x_spatial = z.permute(0,3,1,2).unsqueeze(-1)                               # (B, C, T-1, N, 1)
cross_cond = self._build_cross_cond(feats, H, W)                           # (B, N, Hp, Wp, C)
z_t, e, t = self.flow_noising(x_spatial, context=0, mu, sigma)             # t: (B, T-1)
pred = self.vit(x=z_t, ada_cond=t, cross_cond=cross_cond)
loss = self.flow_loss(pred, x_spatial, z_t, e, t, context=0)
```

### Eval step (per test loader, mirrors `DeltaTokTrainer.eval_one_epoch` :895)
1. Teacher-forced flow loss (same as train).
2. Sample: `zr = randn_like(x_spatial)`; `gen = flow_euler_sample(self._ema_model(), zr,
   pred_mode=cfg, context=0, num_steps=eval_num_steps, cross_cond=cross_cond,
   autocast_ctx=self.autocast)` inside `ema_scope()`.
3. `z_hat = gen.squeeze(-1).permute(0, 2, 3, 1).to(feats.dtype)`  # (B, T-1, N, C)
4. New `_rollout_from_sampled_z(feats, z_hat, H, W, num_cameras)`: copy of
   `_autoregressive_rollout` (`deltatok_trainer.py:700-740`) minus `net.encode` —
   `x_hat_prev = feats[:,0]`; loop `x_hat_t = self.deltatok.decode(z_hat[:, t-1],
   x_hat_prev, rope_local, rope_global)`; feed back; returns `(B*(T-1), N, P, C)`.
5. `full = self._reconstruct_full_tokens(tokens, x_hat, B, V, num_cameras)` (alias);
   `decoded = self._decode_tokens(full, height, width, num_cameras)` (alias);
   `height, width = batch["output_resolution_hw"]`.
6. Losses via existing `_compute_frame_losses`: forecast slice `slice(num_cameras, V)`,
   context slice `slice(0, num_cameras)`. Also a GT-delta rollout
   (`_rollout_from_sampled_z(feats, z, ...)`) as tokenizer upper bound (`*_tok` keys).
   Keys: `LossFlow, LossPointmap/Depth/Raymap, *_tok, *_ctx`. Don't use
   `DeltaTokEvalMetric` (its `_EVAL_LOSS_KEYS` are hardcoded for the tokenizer trainer,
   `occrae/metric.py:13`) — manual per-loader sums + one stacked `dist.all_reduce`,
   logged under `Eval/<test_name>/<key>`.
7. Viz: keep `_log_viz_sample` block; `context_set = set(range(num_cameras))`;
   `batch["timesteps"]` is a `(B,V)` tensor from `_normalize_batch` — indexing works.

## Work completed so far

### `occrae/network/efficient_transformer.py` (partially done)
- DONE: import `CrossAttention` from `occrae.network.transformer_block` (line 9).
- DONE: `Block.__init__` takes `use_cross_attn=False`; builds `self.cross_attn =
  CrossAttention(dim, heads, dropout)` + `self.ln_cross = RMSNorm(...)` when True, else
  `self.cross_attn = None` (replaced the commented-out lines).

Nothing else has been edited yet.

## Remaining work (in order)

### 1. Finish `efficient_transformer.py`
- `Block.forward`: replace the commented cross-attention line (~:104) with:
  ```python
  if self.cross_attn is not None and cross_cond is not None:
      x = x + alpha_cross * self.cross_attn(modulate(self.ln_cross(x), gamma_cross), cross_cond, cross_mask)
  ```
  (applies to full x incl. register token; `gamma_cross/alpha_cross` rows for registers
  are zero — fine).
- `TransformerEncoder.__init__(..., use_cross_attn=False)` → pass through to `Block`.
- `Transformer.__init__`: when `cross_dim is not None`:
  - keep `self.cross_cond_emb` (the existing Linear/SiLU/Linear works on the last dim of
    `(B, N, Hp, Wp, cross_dim)` tensors);
  - add `self.cross_ref_spatial_size = (37, 37)` (or param), `self.cross_spatial_pos =
    nn.Parameter(zeros(1, 37*37, hidden))`, `self.cross_camera_pos = nn.Embedding(16, hidden)`;
  - pass `use_cross_attn=cross_dim is not None` into `TransformerEncoder`;
  - init: `trunc_normal_(cross_spatial_pos, std=0.02)`, `normal_(cross_camera_pos.weight, std=0.02)`.
- Add `interpolate_cross_pos_encoding(self, h, w)`: like `interpolate_pos_encoding`
  (:217) but no CLS slot, over `cross_spatial_pos` with ref `cross_ref_spatial_size`.
- `Transformer.forward` cross handling (replace the `.unsqueeze(1)` block at :309-310):
  ```python
  if cross_cond is not None:
      if cross_cond.dim() == 5:   # (B, N_cam, Hp, Wp, cross_dim) token-sequence conditioning
          bc, n_cam, ch, cw, _ = cross_cond.shape
          cc = self.cross_cond_emb(cross_cond)                                   # (B,N,Hp,Wp,D)
          cc = cc + self.interpolate_cross_pos_encoding(ch, cw).view(1, 1, ch, cw, -1)
          cam = self.cross_camera_pos(torch.arange(n_cam, device=cc.device)).view(1, n_cam, 1, 1, -1)
          cross_cond = (cc + cam).reshape(bc, n_cam * ch * cw, -1)
      else:                        # legacy (B, cross_dim) single-vector conditioning
          cross_cond = self.cross_cond_emb(cross_cond).unsqueeze(1)
  ```

### 2. Rewrite `occrae/deltatok_flow_trainer.py`
Per the design above. Key points:
- Remove `collate_preprocessed` (lines 38-135) + its imports (`PreprocessedSequenceDataset`,
  `build_sequence_records`, `_normalize_resolutions`, `ProcessedRootBatchSampler`,
  `InputProcessor`, `crop_resize_if_necessary`, `to_tensor`, `PIL.Image`, `random`,
  `DataLoader`, `intrinsics_c2w_to_raymap`).
- Add imports: `torch.nn.functional as F`, `from occany.datasets import get_data_loader`,
  `from occrae.deltatok_trainer import DeltaTokModule, DeltaTokTrainer`.
- Class-level bound-method aliases (they only touch `self.cfg/occ_rae/_num_prefix_tokens/autocast`):
  `_normalize_batch` (:494), `_extract_pair_feats` (:664), `_reconstruct_full_tokens`
  (:869), `_decode_tokens` (:848), `_set_train_loader_epoch` (:486), `_build_occ_rae`
  (:1215 — the version with `requires_grad_(False)` + bf16 + optional img_decoder).
- `__init__` order: `super().__init__(args)` → set `self.autocast` → `self._build_occ_rae()`
  → `backbone = self.occ_rae.model._get_pretrained_backbone()`;
  `self._num_prefix_tokens = 1 + int(backbone.num_register_tokens)`;
  `self._patch_size = int(backbone.patch_size)` → `self.deltatok = self._build_deltatok(backbone)`
  → `self.vit = self.get_network("vit")` → optim/EMA/criteria as before.
- `get_network("vit")`: `Transformer(out_dim=1536, num_views=cfg.model.num_views,
  cross_dim=1536, hidden_dim, proj=1, depth, heads, mlp_dim=hidden*4, dropout,
  is_causal=cfg, use_trajectory_cond=False, trajectory_length=0,
  ref_spatial_size=(int(cfg.model.get("max_num_cameras", 6)), 1))`. Keep existing
  ckpt-resume logic. Old flow ckpts won't load (shape mismatch) — fine, new runs only.
- New `_build_deltatok(backbone)` (decision #2 above).
- New `_encode_deltas(feats, H, W)`:
  ```python
  B, T, N, P, C = feats.shape
  x_prev = feats[:, :-1].reshape(B*(T-1), N, P, C); x = feats[:, 1:].reshape(...)
  rope_local = self.deltatok._compute_rope(H, W, feats.device, feats.dtype)
  rope_global = self.deltatok._compute_global_rope(H, W, N, feats.device, feats.dtype)
  with torch.no_grad(), self.autocast:
      z = self.deltatok.encode(x_prev, x, rope_local, rope_global)   # (B*(T-1), N, 1, C)
  return z.reshape(B, T-1, N, C)
  ```
  Assert `T-1 <= self.num_views` (temporal_pos capacity).
- New `_build_cross_cond(feats, H, W)`: layernorm (cfg `ctx_norm`) + reshape to
  `(B, N, H//patch, W//patch, C)`.
- `train_one_epoch` / `eval_one_epoch` / `fit`: per the design + eval sections above.
  `fit` dataset block = copy of `DeltaTokTrainer.fit` (:1246-1296: `get_data_loader` for
  `cfg.dataset.train_dataset` with `per_dataset_sampling`, `self.test_loaders` dict from
  `cfg.dataset.test_dataset` split on `+`). Keep flow trainer's virtual-epoch loop,
  `_next_train_batch`/`_reset_train_iterator` (switch to `_set_train_loader_epoch`),
  EMA, cosine LR, checkpoint cadence.
- Drop `_tokens_to_spatial`/`_spatial_to_tokens` and `cfg.dataset.cond_num` usage.

### 3. Configs
- `configs/train_deltatok_flow.yaml`:
  - dataset: delete `preprocess_data_root, subsampling_rate, max_stride, frame_stride,
    cond_num, processed_roots, max_*_samples`; add `per_dataset_sampling: true` +
    `train_dataset`/`test_dataset` strings copied from `configs/train_deltatok_karolina.yaml:6-49`
    (multi-resolution is fine now with cross-attention).
  - model: `encode_layer: 12`; `num_views: 8` (max T−1 delta slots); add
    `max_num_cameras: 6`, `ctx_norm: "layernorm"`, `deltatok:` block (copy from
    `configs/train_deltatok.yaml:73-82`), `deltatok_ckpt: /mnt/proj1/eu-25-92/deltatok_log/deltatok_surround_constGlobalRope_layer12/ckpts/current.pth`.
  - training: add `val_num_workers: 2`, `eval_num_visualizations: 8`,
    `sanity_check_num_items: 2`; bump `num_workers` to 4.
- `configs/train_deltatok_flow_karolina.yaml`: drop `preprocess_data_root` override;
  thin overlay (paths in the copied strings are already Karolina-style `/scratch/project/eu-25-92/...`).
- `configs/train_deltatok_flow_overfit.yaml`: tiny counts instead of `max_train_samples`,
  e.g. `train_dataset: "2 @ Occ3dNuscenesSeqMultiView(... split='val' ...)"`, test = same;
  `warm_up: 0`, small `virtual_epoch`.

### 4. `train_deltatok_flow.py` entrypoint
- Replace hand-rolled `VENDORED_IMPORT_PATHS` (lines 19-33) with
  `prepend_vendored_import_paths(..., extra=["third_party/pyTorchChamferDistance",
  "third_party/GLD/src", "third_party/deltatok"])` as in `train_deltatok.py:18-25`
  (**required**: `occrae.deltatok_trainer` imports `models.gated_attn`/`models.qk_norm`
  from `third_party/deltatok`).
- Remove `--preprocess_data_root` arg/override (lines 77-82, 159-160) and the `--overfit`
  `max_train_samples` override (lines 95, 165-167).
- Add `--deltatok_ckpt` → `cfg.model.deltatok_ckpt`; define `--resume` (trainer reads it).
- Update docstring.

### 5. Verification (Karolina only — never run locally; **user must sync first**)
1. `ssh karolina "cd ~/deltatok && conda activate occany && source env_bsc.sh && python -c 'import train_deltatok_flow'"`
2. Overfit smoke: `python train_deltatok_flow.py --config-name train_deltatok_flow_overfit
   --run-name flow_delta_smoke --cfg model.vit_size=tiny training.max_iter=20
   training.virtual_epoch=10 training.eval_num_steps=5` — sanity eval must run end-to-end
   (sampling → DeltaTok rollout → OccRAE decode → viz PNGs under `.../ckpts/eval_viz/`).
3. Check the `deltatok_ckpt` load log line; `*_tok` (GT-delta rollout) metrics should
   roughly match `test_deltatok_ar.py` "ar" numbers on the same test set.
4. LossFlow decreases on the overfit config; short 2-GPU run for DDP sanity (frozen
   `deltatok`/`occ_rae` are non-DDP by design).

## Pitfalls already identified
- Frozen modules stay out of DDP and the optimizer.
- Cast frozen DeltaTok to bf16 once (rope cache is dtype-keyed, `deltatok_trainer.py:130`).
- `flow_euler_sample` `alpha` offsets now spread over only T−1 slots — default 0.5 is fine.
- Single-camera test sets: DeltaTok global layers degenerate to local-only (N>1 checks) — expected.
- `DeltaTokEvalMetric` keys are hardcoded — don't reuse; manual all_reduce instead.
