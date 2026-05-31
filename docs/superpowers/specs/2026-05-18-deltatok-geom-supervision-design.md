# DeltaTok train-time geometric supervision — design

**Date:** 2026-05-18
**Status:** Spec — pending plan
**Entrypoint touched:** `sh/train_deltatok.sh` → `train_deltatok.py` → `occrae/deltatok_trainer.py`

## Motivation

`DeltaTokTrainer.train_one_epoch` currently optimises only the token-residual
loss `_log_cosh(x_hat, x)` between predicted and ground-truth OccRAE layer-18
spatial features. Geometric supervision (depth, pointmap, raymap) is computed
only in `eval_one_epoch`, after `OccRAE.decode()` is called on stitched full
tokens.

This proposal promotes the eval-time geometric losses into the training step:
during training, the predicted delta-token `x_hat` is stitched back into full
tokens, decoded through the **frozen** OccRAE decoder, and supervised against
ground-truth depth/pointmap/raymap labels. The token loss is retained as a
dense per-patch regulariser.

The change is local to the DeltaTok pipeline. No other entrypoints are
affected.

## Decisions (set during brainstorming)

1. **Supervision target** — ground-truth labels (`PredVsGT`). Decoded
   `depth`/`pointmap`/`ray` are compared against
   `batch["gt_{depth,pointmap,raymap,mask}"]` etc.; OccRAE's decode of GT
   tokens is **not** used as a teacher.
2. **Loss mix** — keep the existing token loss; **add** pointmap/depth/raymap
   geometric losses. Total = weighted sum of all four.
3. **Rollout mode** — teacher-forced only (`x_hat = tokenizer(x_prev_gt,
   x_gt, ...)`). No autoregressive rollout in training. AR continues to be
   measured in eval.
4. **Decode context** — decode **only the predicted-frame views** (no anchor
   frames included as cross-view context). Simpler, single decode pass per
   step. Anchor experiments can come later.
5. **Eval criteria stay as-is** (`PointmapLoss(lambda_c=0.0, loss_type="L2")`
   etc.) so eval numbers remain comparable to past TB runs. New **train
   criteria** mirror `sh/train_occany_plus_recon_1B.sh`
   (`PointmapLoss(lambda_c=1.0, loss_type="L1")`,
   `RaymapLoss(lambda_c=1.0, loss_type="L1")`, identical `DepthLosses`).
6. **Scale-invariant depth loss** — out of scope for this iteration. Recon
   training used it, but DeltaTok train will start without it.

## Background facts verified during brainstorming

- DA3-giant in this repo instantiates `vit_giant2` with the default
  `num_register_tokens=0` (`third_party/Depth-Anything-3/.../vision_transformer.py:639`,
  `dinov2.py:49-57`). **There are no register tokens.**
- The CLS token *is* present and always prepended
  (`vision_transformer.py:160, 267`); the layer-18 export keeps it intact in
  `raw_state` (`vision_transformer.py:380-382`).
- `DeltaTokTrainer._num_prefix_tokens = 1 + backbone.num_register_tokens`
  resolves to **1** (CLS only). The existing `_reconstruct_full_tokens`
  helper already splices this one CLS token back from GT — the same
  mechanism is used in the new train path.
- `OccRAE.decode()` (`occany/model/occ_rae.py:127`) is decorated
  `@torch.no_grad()`, blocking gradient flow back to its input tokens. A
  separate gradient-bearing path is required for training (`decode_grad`
  below).


## Architecture   

### A. New method on `OccRAE`: `decode_grad`

Add a sibling to `OccRAE.decode()` in `occany/model/occ_rae.py`. Identical
body, **no** `@torch.no_grad()` decorator. Used only by DeltaTok training; all
existing callers (`inference.py`, `extract_*`, eval loops) keep using
`decode()`.

```python
def decode_grad(
    self,
    latents: Dict[str, object],
    pose_from_depth_ray: bool = False,
    pose_from_cam_dec: bool = False,
    point_from_depth_and_pose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Same as :meth:`decode` but does NOT wrap in ``torch.no_grad`` so
    gradients can flow back to ``latents['tokens']``. Model weights remain
    frozen via ``requires_grad_(False)``; this method only allows the
    autograd graph to extend into the inputs.
    """
    x = latents["tokens"]
    h, w = latents["H"], latents["W"]
    start_layer = self.encode_layer + 1
    return self.model.inference_batch_from_layer(
        x=x, start_layer=start_layer, h=h, w=w, local_x=None,
        pose_from_depth_ray=pose_from_depth_ray,
        pose_from_cam_dec=pose_from_cam_dec,
        point_from_depth_and_pose=point_from_depth_and_pose,
    )
```

OccRAE weights are already `requires_grad_(False)` in
`DeltaTokTrainer._build_occ_rae`, so this only opens the autograd graph;
parameters never update.

### B. Train criteria

Add three new attributes to `DeltaTokTrainer.__init__` alongside the existing
eval criteria. The existing criteria (`pointmap_criterion`,
`depth_criterion`, `raymap_criterion`) keep their current settings and
remain used by `_compute_frame_losses` / `_pred_vs_orig` in
`eval_one_epoch`.

```python
# Eval criteria (unchanged) — keep eval metrics comparable to past TB runs:
self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
self.depth_criterion    = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
self.raymap_criterion   = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")

# Train criteria (new) — mirror sh/train_occany_plus_recon_1B.sh:
self.train_pointmap_criterion = PointmapLoss(lambda_c=1.0, gt_scale=True, loss_type="L1")
self.train_depth_criterion    = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
self.train_raymap_criterion   = RaymapLoss(lambda_c=1.0, gt_scale=True, loss_type="L1")
```

Note: `DepthLosses` is L1 internally regardless of any switch
(`losses_da3.py:415`). The train/eval `depth_criterion`s end up functionally
identical; the duplicate exists for symmetry with pm/raymap and to keep the
train-loss path from reaching across to "eval criteria".

### C. Train-step pipeline

Edit `train_one_epoch` (`deltatok_trainer.py:744-819`). Replace the
single-line `_log_cosh` loss assembly with:

#### 1. Pair-index tracking (during/after subsampling)

```python
T = batch["imgs"].shape[1] // num_cameras
num_pairs      = x_prev.shape[0]                            # = B * (T-1)
pair_batch_idx = torch.arange(num_pairs, device=imgs.device) // (T - 1)
pair_t         = torch.arange(num_pairs, device=imgs.device) %  (T - 1) + 1
# pair_t[i] = timestep of x (the prediction target) for pair i, in [1, T).
# pair_t[i] - 1 corresponds to x_prev.

pairs_per_batch = int(self.cfg.training.get("pairs_per_batch", 0))
if pairs_per_batch > 0 and num_pairs > pairs_per_batch:
    keep = torch.randperm(num_pairs, device=imgs.device)[:pairs_per_batch]
    x_prev, x = x_prev[keep], x[keep]
    pair_batch_idx = pair_batch_idx[keep]
    pair_t         = pair_t[keep]
```

#### 2. Forward through DeltaTok and compute token loss

Unchanged:

```python
with self.autocast:
    x_hat = self.tokenizer(x_prev, x, H, W, num_cameras=num_cameras)
with torch.autocast(device_type="cuda", enabled=False):
    L_token = _log_cosh(x_hat.float(), x.detach().float()).mean()
```

#### 3. Gather GT for x

```python
num_selected      = x_prev.shape[0]
cam_offsets       = torch.arange(num_cameras, device=pair_t.device)
pair_view_start_x = pair_t * num_cameras                                    # (num_selected,)
pair_views_x      = pair_view_start_x[:, None] + cam_offsets[None, :]       # (num_selected, num_cameras)

def gather_x(tensor_BV):
    """(B, V, ...) → (num_selected, num_cameras, ...) — pull views for x."""
    return tensor_BV[pair_batch_idx[:, None], pair_views_x]

gt_d   = gather_x(batch["gt_depth"]).to(self.device)
gt_pm  = gather_x(batch["gt_pointmap"]).to(self.device)
gt_ray = gather_x(batch["gt_raymap"]).to(self.device)
gt_m   = gather_x(batch["gt_mask"]).to(self.device)
c2w    = gather_x(batch["gt_c2w"]).to(self.device)
intr   = gather_x(batch["gt_intrinsics"]).to(self.device)
```

#### 4. Stitch GT CLS into predicted-spatial tokens, then decode with grad

```python
prefix = self._num_prefix_tokens  # = 1 (CLS only, no register tokens)
B, V, N_tok, C = tokens.shape
tokens_btnc = tokens.view(B, T, num_cameras, N_tok, C)

# GT CLS for the views corresponding to x.
gt_cls_x = tokens_btnc[
    pair_batch_idx[:, None],
    pair_t[:, None],
    cam_offsets[None, :],
    :prefix,
]                                                                      # (num_selected, num_cameras, prefix, C)

full_tokens_x = torch.cat([gt_cls_x, x_hat.to(tokens.dtype)], dim=2)   # (num_selected, num_cameras, N_tok, C)

H_img, W_img = batch["output_resolution_hw"]
with self.autocast:
    decoded = self.occ_rae.decode_grad(
        {"tokens": full_tokens_x, "H": H_img, "W": W_img}
    )
# decoded["depth"]:    (num_selected, num_cameras, H_img, W_img)
# decoded["pointmap"]: (num_selected, num_cameras, H_img, W_img, 3)
# decoded["ray"]:      (num_selected, num_cameras, H_img, W_img, 6)
```

#### 5. Apply train criteria + weighted sum

```python
w = self.cfg.training.loss_weights

with torch.autocast(device_type="cuda", enabled=False):
    L_pm,  _ = self.train_pointmap_criterion(
                  decoded["pointmap"].float(), gt_pm.float(),
                  mask=gt_m, confidence=decoded.get("pointmap_conf"))
    L_d,   _ = self.train_depth_criterion(
                  decoded["depth"].float().reshape(-1, 1, H_img, W_img),
                  gt_d.float().reshape(-1, 1, H_img, W_img),
                  confidence=None,
                  mask=gt_m.float().reshape(-1, 1, H_img, W_img))
    L_ray, _ = self.train_raymap_criterion(
                  decoded["ray"].float(), decoded.get("ray_conf"),
                  c2w.float(), intr.float(), gt_ray.float())

    L_total = (w.token    * L_token
             + w.pointmap * L_pm
             + w.depth    * L_d
             + w.raymap   * L_ray)

(L_total / self.cfg.training.grad_cum).backward()
```

If `decoded` does not expose `pointmap_conf` (i.e. the OccAny+ recon head does
not emit per-pointmap confidence), `PointmapLoss` with `lambda_c=1.0` and
`confidence=None` falls back to plain masked L1
(`losses_da3.py:308`). Functionally fine; the `lambda_c` setting still aligns
with the recon training regime for callers that *do* pass confidence.

#### 6. Logging

Replace the single `Train/LossRecon` scalar at `deltatok_trainer.py:811` with
per-component + total:

```
Train/LossToken      Train/LossPointmap     Train/LossDepth
Train/LossRaymap     Train/LossTot
```

(`Train/LossRecon` can optionally be retained as an alias of
`Train/LossToken` for backward TB compat — decided at implementation time.)

### D. Config knobs

Add to `configs/train_deltatok.yaml`:

```yaml
training:
  loss_weights:
    token:    1.0
    pointmap: 1.0
    depth:    1.0
    raymap:   1.0
```

Defaults of `1.0` everywhere mirror `sh/train_occany_plus_recon_1B.sh`
(`--lambda_depth 1.0 --lambda_pointmap 1.0`). The token weight is set to
`1.0` as a placeholder — `_log_cosh` on layer-18 tokens is typically O(0.01-0.1),
so geometric terms naturally dominate by ~10-100x, which is intended (geometry
is the new primary signal; the token term is a stabilising regulariser). Bump
`w_token` upward only if the token loss visibly drops to noise without
converging.

`configs/train_deltatok_karolina.yaml` inherits unchanged.

## Files touched

- `occany/model/occ_rae.py` — add `decode_grad` method.
- `occrae/deltatok_trainer.py`
  - `__init__`: add the three `train_*_criterion` attributes.
  - `train_one_epoch`: replace the single token-loss assembly with the
    pair-index gather, OccRAE decode, criteria application, weighted sum,
    and per-component logging described above.
- `configs/train_deltatok.yaml` — add `training.loss_weights` block.
- `configs/train_deltatok_karolina.yaml` — no change required (inherits).

No changes to `sh/train_deltatok.sh`, `train_deltatok.py`, or eval paths.

## Non-goals (explicit out-of-scope)

- AR rollout in training (TF only).
- Anchor / context-frame stitching in the train-time decode (decode only x).
- Scale-invariant depth loss term.
- Changing eval criteria (they stay at L2 / `lambda_c=0.0`).
- Tuning OccRAE decoder weights (frozen throughout).

## Memory expectations

With the default Karolina setup (`bsize=1`, `pairs_per_batch=1`, `T≥2`,
`num_cameras` up to 6), each training step adds **one OccRAE
`decode_grad` forward over `num_cameras` views** plus its backward. Autograd
activations from the DA3 decode trunk are the dominant new cost. If memory
becomes a blocker on the giant backbone, the OccRAE decode trunk can be
wrapped in `torch.utils.checkpoint` — defer until measured.

## Open implementation questions (for the plan stage)

1. Does `OccRAE.decode_grad`'s output dict expose `pointmap_conf`, or only
   `ray_conf` / `depth_conf`? Confirm at plan time; affects whether the
   `confidence=` arg to `train_pointmap_criterion` is `None` or
   `decoded["pointmap_conf"]`.
2. Verify by quick smoke test that gradients actually flow from `L_pm`
   through `decode_grad` into the tokenizer params (no stray detach inside
   `inference_batch_from_layer`).
3. Keep `Train/LossRecon` as an alias of `Train/LossToken` (for TB-run
   continuity), or rename outright. Cosmetic.

## Success criteria

- `train_one_epoch` runs at parity wall-clock with current loop modulo one
  extra OccRAE decode forward+backward over `num_cameras` views per step.
- Eval scalars under `Eval/<test>/LossPointmap`, `…/LossDepth`,
  `…/LossRaymap` (the *PredVsGT* family) **decrease** vs. the
  geometric-loss-disabled baseline.
- `Eval/<test>/LossRecon` (token-recon eval scalar) does **not** materially
  regress.
- `Train/LossToken`, `Train/LossPointmap`, `Train/LossDepth`,
  `Train/LossRaymap` all show non-degenerate trajectories (decreasing or
  plateauing — not exploding or stuck at init).
