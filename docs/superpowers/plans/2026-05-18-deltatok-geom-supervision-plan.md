# DeltaTok Train-Time Geometric Supervision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote DeltaTok's eval-time geometric losses (pointmap / depth / raymap) into the training step, so the tokenizer is supervised against GT geometry through a gradient-bearing OccRAE decoder rather than only against per-patch token residuals.

**Architecture:** Add a no-grad-free sibling `OccRAE.decode_grad` so autograd can flow back through the frozen DA3 decoder into the tokenizer. In `DeltaTokTrainer.train_one_epoch`, after computing `x_hat`, stitch GT CLS into the predicted spatial tokens (per pair), decode with `decode_grad`, gather matching GT views, apply pointmap/depth/raymap criteria, and backprop a weighted sum that also retains the existing `_log_cosh` token loss as a regulariser.

**Tech Stack:** PyTorch, DA3-Giant (frozen via OccRAE), Hydra configs, SLURM via `karolina-job` skill.

**Spec:** `docs/superpowers/specs/2026-05-18-deltatok-geom-supervision-design.md`

---

## Files touched

- **Modify** `occany/model/occ_rae.py` — add `decode_grad` method after `decode` (around line 163).
- **Modify** `occrae/deltatok_trainer.py`:
  - `DeltaTokTrainer.__init__` (lines 444-446) — add three `train_*_criterion` attributes.
  - `DeltaTokTrainer.train_one_epoch` (lines 744-819) — replace single-line `_log_cosh` loss assembly with pair-index gather, `decode_grad`, weighted multi-term loss, and per-component logging.
- **Modify** `configs/train_deltatok.yaml` — add `training.loss_weights` block.
- **No change** to `sh/train_deltatok.sh`, `train_deltatok.py`, `configs/train_deltatok_karolina.yaml` (inherits), or any eval path.

## GPU caveat

Per `CLAUDE.md`: this repo has no local GPU. Every smoke step in this plan that needs CUDA / OccRAE / dataset loading must run on Karolina via `ssh karolina "conda activate occany && …"` or via the `karolina-job` skill for SLURM submissions. The Karolina checkout at `/home/it4i-anhquan/OccAny` is the user's — do **not** rsync/push/pull from this session.

---

### Task 1: Add `OccRAE.decode_grad`

**Files:**
- Modify: `occany/model/occ_rae.py` (insert after `decode` at line 163)

- [ ] **Step 1: Add `decode_grad` immediately after `decode`**

Insert the following method right after the closing of `decode` (currently ends at line 163, just before the `# Decode to multi-level features` divider at line 165):

```python
    def decode_grad(
        self,
        latents: Dict[str, object],
        pose_from_depth_ray: bool = False,
        pose_from_cam_dec: bool = False,
        point_from_depth_and_pose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Same as :meth:`decode` but without ``@torch.no_grad`` so gradients
        can flow back to ``latents['tokens']``. OccRAE weights stay frozen via
        ``requires_grad_(False)``; this method only opens the autograd graph
        into the inputs. Used by the DeltaTok train-time geometric loss; all
        existing callers (inference, extraction, eval) keep using
        :meth:`decode`.
        """
        x = latents["tokens"]
        h = latents["H"]
        w = latents["W"]
        start_layer = self.encode_layer + 1

        return self.model.inference_batch_from_layer(
            x=x,
            start_layer=start_layer,
            h=h,
            w=w,
            local_x=None,  # layer 18 is local → x == local_x
            pose_from_depth_ray=pose_from_depth_ray,
            pose_from_cam_dec=pose_from_cam_dec,
            point_from_depth_and_pose=point_from_depth_and_pose,
        )
```

- [ ] **Step 2: Karolina smoke test — confirm gradient flow**

Addresses spec open question 2 (no stray `detach` / `no_grad` inside `inference_batch_from_layer`'s call chain). Run:

```bash
ssh karolina "conda activate occany && cd /home/it4i-anhquan/OccAny && python -c \"
import torch
from occany.model.occ_rae import OccRAE
occ = OccRAE(weights_path='checkpoints/occany_plus_recon_1B.pth', device='cuda', encode_layer=18)
occ.eval().requires_grad_(False)
imgs = torch.randn(1, 2, 3, 168, 518, device='cuda')
with torch.no_grad():
    lat = occ.encode(imgs)
tokens = lat['tokens'].detach().clone().requires_grad_(True)
out = occ.decode_grad({'tokens': tokens, 'H': 168, 'W': 518})
loss = out['depth'].mean() + out['pointmap'].mean() + out['ray'].mean()
loss.backward()
assert tokens.grad is not None and tokens.grad.abs().sum() > 0, 'no gradient reached tokens'
print('OK: |grad|=', tokens.grad.abs().mean().item())
\""
```

Expected stdout: `OK: |grad|= <some positive float>`.
Failure modes:
- `RuntimeError: element 0 of tensors does not require grad` → there is still a `@torch.no_grad()` or `.detach()` inside `forward_from_layer` / `_process_depth_output`; bisect to find it.
- Loss tensor is fine but `tokens.grad is None` → the autograd graph was severed by an in-place op; same bisect.

- [ ] **Step 3: Commit**

```bash
git add occany/model/occ_rae.py
git commit -m "feat(occrae): add decode_grad sibling for training-time geometric loss"
```

---

### Task 2: Add `training.loss_weights` block to `configs/train_deltatok.yaml`

**Files:**
- Modify: `configs/train_deltatok.yaml` (append under `training:` block)

- [ ] **Step 1: Append `loss_weights` at the end of the `training:` section**

Currently the file's `training:` block ends at `sanity_check_num_items: 2` (line 105). Append:

```yaml
  # Weights for the multi-term training loss. `token` is the existing
  # _log_cosh token-residual regulariser; pointmap/depth/raymap mirror
  # sh/train_occany_plus_recon_1B.sh (--lambda_depth 1.0 --lambda_pointmap 1.0).
  # Geometric terms naturally dominate by ~10-100x because _log_cosh on
  # layer-18 tokens is O(0.01-0.1) — intended (geometry is the new primary
  # signal; token is the regulariser). Bump `token` only if its trajectory
  # degenerates to noise.
  loss_weights:
    token:    1.0
    pointmap: 1.0
    depth:    1.0
    raymap:   1.0
```

- [ ] **Step 2: Confirm `configs/train_deltatok_karolina.yaml` inherits cleanly**

```bash
grep -n "loss_weights" configs/train_deltatok_karolina.yaml || echo "OK: no override (inherits)"
```

Expected: `OK: no override (inherits)`.

- [ ] **Step 3: Do not commit yet — bundle with Tasks 3 and 4**

Skip; the criteria added in Task 3 and the consumer in Task 4 are useless without each other.

---

### Task 3: Add train criteria to `DeltaTokTrainer.__init__`

**Files:**
- Modify: `occrae/deltatok_trainer.py:444-446`

- [ ] **Step 1: Add the three `train_*_criterion` attributes**

Locate the existing eval-criteria block at lines 444-446:

```python
        self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
        self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.raymap_criterion = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
```

Replace with:

```python
        # Eval criteria — keep at L2 / lambda_c=0 so eval scalars stay comparable
        # to past TB runs. Do NOT touch these without coordinating an eval rerun.
        self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
        self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.raymap_criterion = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")

        # Train criteria — mirror sh/train_occany_plus_recon_1B.sh
        # (--lambda_pointmap 1.0 --lambda_depth 1.0). DepthLosses is L1 internally
        # regardless of any switch (losses_da3.py:415), so the train depth
        # criterion is functionally identical to the eval one; the duplicate
        # exists for symmetry and so the train path doesn't reach across to
        # "eval criteria".
        self.train_pointmap_criterion = PointmapLoss(lambda_c=1.0, gt_scale=True, loss_type="L1")
        self.train_depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.train_raymap_criterion = RaymapLoss(lambda_c=1.0, gt_scale=True, loss_type="L1")
```

- [ ] **Step 2: Do not commit yet — bundle with Task 4**

---

### Task 4: Rewrite `train_one_epoch` to apply geometric losses through `decode_grad`

**Files:**
- Modify: `occrae/deltatok_trainer.py:744-819`

This is the central change. Replace the entire `train_one_epoch` method body. The outer loop / grad-accumulation / logging-cadence scaffolding stays; only the per-step loss assembly changes.

- [ ] **Step 1: Replace the full `train_one_epoch` method**

Replace lines 744-819 with:

```python
    def train_one_epoch(self):
        self.tokenizer.train()
        cum_loss = 0.
        num_batches = 0
        last_update_time = time.time()
        window_loss = deque(maxlen=self.cfg.training.grad_cum)
        self.optim.zero_grad(set_to_none=True)

        epoch = int(self.cfg.training.global_epoch)
        self._set_train_loader_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Training (Epoch {epoch})",
            disable=not self.is_master,
        )

        w = self.cfg.training.loss_weights

        for batch in pbar:
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                break

            batch = self._normalize_batch(batch)
            num_batches += 1
            update_grad = (num_batches % self.cfg.training.grad_cum) == 0

            self.adapt_learning_rate()

            imgs = batch["imgs"].to(self.device, non_blocking=True)
            num_cameras = batch.get("num_cameras", 1)
            tokens, _feats, x_prev, x, H, W = self._extract_pair_feats(
                imgs, num_cameras=num_cameras
            )

            # ------------------------------------------------------------
            # Pair-index tracking. After pairs_per_batch subsampling, each
            # row i of (x_prev, x) corresponds to batch item pair_batch_idx[i]
            # and timestep transition pair_t[i]-1 -> pair_t[i]. We need this
            # to gather the correct GT views for x.
            # ------------------------------------------------------------
            T = imgs.shape[1] // num_cameras
            num_pairs = x_prev.shape[0]                                  # = B * (T-1)
            pair_batch_idx = torch.arange(num_pairs, device=imgs.device) // (T - 1)
            pair_t = torch.arange(num_pairs, device=imgs.device) % (T - 1) + 1

            pairs_per_batch = int(self.cfg.training.get("pairs_per_batch", 0))
            if pairs_per_batch > 0 and num_pairs > pairs_per_batch:
                keep = torch.randperm(num_pairs, device=imgs.device)[:pairs_per_batch]
                x_prev = x_prev[keep]
                x = x[keep]
                pair_batch_idx = pair_batch_idx[keep]
                pair_t = pair_t[keep]

            # ------------------------------------------------------------
            # 1. DeltaTok forward + token-residual regulariser.
            # ------------------------------------------------------------
            with self.autocast:
                x_hat = self.tokenizer(x_prev, x, H, W, num_cameras=num_cameras)

            with torch.autocast(device_type="cuda", enabled=False):
                L_token = _log_cosh(x_hat.float(), x.detach().float()).mean()

            # ------------------------------------------------------------
            # 2. Gather GT for the predicted views. Views are camera-major
            #    within each timestep (see _normalize_batch lines 504-512):
            #    view index = timestep * num_cameras + camera.
            # ------------------------------------------------------------
            cam_offsets = torch.arange(num_cameras, device=pair_t.device)
            pair_view_start_x = pair_t * num_cameras                            # (num_selected,)
            pair_views_x = pair_view_start_x[:, None] + cam_offsets[None, :]    # (num_selected, num_cameras)

            def gather_x(tensor_BV):
                """(B, V, ...) -> (num_selected, num_cameras, ...) for the views of x."""
                return tensor_BV[pair_batch_idx[:, None], pair_views_x]

            gt_d = gather_x(batch["gt_depth"]).to(self.device)
            gt_pm = gather_x(batch["gt_pointmap"]).to(self.device)
            gt_ray = gather_x(batch["gt_raymap"]).to(self.device)
            gt_m = gather_x(batch["gt_mask"]).to(self.device)
            c2w = gather_x(batch["gt_c2w"]).to(self.device)
            intr = gather_x(batch["gt_intrinsics"]).to(self.device)

            # ------------------------------------------------------------
            # 3. Stitch GT CLS onto x_hat, then decode through the frozen
            #    DA3 decoder *with gradient*. Anchor frames are excluded —
            #    single decode pass over the predicted views only (spec D4).
            #    _num_prefix_tokens == 1 for DA3-Giant (CLS only, no register
            #    tokens).
            # ------------------------------------------------------------
            prefix = self._num_prefix_tokens
            B_full, V_full, N_tok, C_tok = tokens.shape
            tokens_btnc = tokens.view(B_full, T, num_cameras, N_tok, C_tok)
            gt_cls_x = tokens_btnc[
                pair_batch_idx[:, None],
                pair_t[:, None],
                cam_offsets[None, :],
                :prefix,
            ]   # (num_selected, num_cameras, prefix, C_tok)

            full_tokens_x = torch.cat([gt_cls_x, x_hat.to(tokens.dtype)], dim=2)
            # full_tokens_x: (num_selected, num_cameras, N_tok, C_tok)

            with self.autocast:
                decoded = self.occ_rae.decode_grad(
                    {"tokens": full_tokens_x, "H": H, "W": W}
                )
            # decoded["depth"]:    (num_selected, num_cameras, H, W)
            # decoded["pointmap"]: (num_selected, num_cameras, H, W, 3)
            # decoded["ray"]:      (num_selected, num_cameras, H, W, 6)
            # decoded["ray_conf"]: (num_selected, num_cameras, H, W)
            # NOTE: `pointmap_conf` is NOT exposed by _process_depth_output
            # (model_da3.py:761-769), so we pass confidence=None below.

            # ------------------------------------------------------------
            # 4. Apply train criteria + weighted sum. With confidence=None
            #    and lambda_c=1.0, PointmapLoss falls back to plain masked L1
            #    (losses_da3.py:308-339).
            # ------------------------------------------------------------
            with torch.autocast(device_type="cuda", enabled=False):
                L_pm, _ = self.train_pointmap_criterion(
                    decoded["pointmap"].float(),
                    gt_pm.float(),
                    mask=gt_m,
                    confidence=None,
                )
                L_d, _ = self.train_depth_criterion(
                    decoded["depth"].float().reshape(-1, 1, H, W),
                    gt_d.float().reshape(-1, 1, H, W),
                    confidence=None,
                    mask=gt_m.float().reshape(-1, 1, H, W),
                )
                L_ray, _ = self.train_raymap_criterion(
                    decoded["ray"].float(),
                    decoded.get("ray_conf"),
                    c2w.float(),
                    intr.float(),
                    gt_ray.float(),
                )

                L_total = (
                    w.token    * L_token
                    + w.pointmap * L_pm
                    + w.depth    * L_d
                    + w.raymap   * L_ray
                )

            (L_total / self.cfg.training.grad_cum).backward()

            if update_grad:
                nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.cfg.training.grad_clip)
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

            loss_val = L_total.detach().item()
            cum_loss += loss_val
            window_loss.append(loss_val)

            if update_grad:
                if self.distributed:
                    mini_batch_loss = self.all_gather(torch.tensor(window_loss).mean())
                else:
                    mini_batch_loss = torch.tensor(window_loss).mean()

                if self.is_master:
                    now = time.time()
                    elapsed = max(now - last_update_time, 1e-6)
                    speed_samples_per_sec = self.cfg.training.bsize / elapsed
                    last_update_time = now

                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.cfg.training.iter)
                    # Per-component scalars. Train/LossRecon is kept as an alias
                    # of Train/LossToken so existing TB runs that plot LossRecon
                    # stay continuous (spec open question 3 — resolved: alias).
                    self.log_add_scalar('Train/LossToken',    L_token.detach(),  self.cfg.training.iter)
                    self.log_add_scalar('Train/LossRecon',    L_token.detach(),  self.cfg.training.iter)
                    self.log_add_scalar('Train/LossPointmap', L_pm.detach(),     self.cfg.training.iter)
                    self.log_add_scalar('Train/LossDepth',    L_d.detach(),      self.cfg.training.iter)
                    self.log_add_scalar('Train/LossRaymap',   L_ray.detach(),    self.cfg.training.iter)
                    self.log_add_scalar('Train/LossTot',      mini_batch_loss,   self.cfg.training.iter)
                    self.log_add_scalar('Train/SpeedSamplesPerSec', speed_samples_per_sec, self.cfg.training.iter)

                    pbar.set_postfix(loss=mini_batch_loss.item())

                self.cfg.training.iter += 1

        return cum_loss / max(1, num_batches)
```

- [ ] **Step 2: Commit Tasks 2-4 together**

```bash
git add configs/train_deltatok.yaml occrae/deltatok_trainer.py
git commit -m "feat(deltatok): train-time geometric supervision via decode_grad

Add pointmap/depth/raymap losses to the DeltaTok train step. Each pair's
predicted x_hat is stitched with GT CLS and decoded through the frozen
DA3 decoder via OccRAE.decode_grad, then supervised against batch GT
(PredVsGT). The existing _log_cosh token residual is retained as a
regulariser. Loss weights are configurable under training.loss_weights
(defaults all 1.0; geometric terms naturally dominate). Eval criteria
are untouched; train criteria mirror sh/train_occany_plus_recon_1B.sh
(L1 / lambda_c=1 for pointmap+raymap). Train/LossRecon kept as alias
of Train/LossToken for TB continuity."
```

---

### Task 5: End-to-end smoke run on Karolina

Verifies that a few train iterations of the new loss path run without shape errors and produce non-degenerate per-component scalars.

**Files:** read-only (`sh/train_deltatok.sh`, `configs/train_deltatok_karolina.yaml`).

- [ ] **Step 1: Confirm the Karolina checkout matches local HEAD**

Per CLAUDE.md, the Karolina checkout is the user's — do not push/pull/sync from this session. Ask the user to confirm by running locally `git rev-parse HEAD` and on Karolina `git rev-parse HEAD`, then comparing. If they differ, stop and ask the user how they want to sync.

- [ ] **Step 2: Submit a tiny-iter smoke run via the `karolina-job` skill**

Use the `karolina-job` skill (per CLAUDE.md, the canonical wrapper for SLURM submissions). Submit a job that:
- runs `sh/train_deltatok.sh` (single GPU, no DDP needed for smoke)
- overrides Hydra: `training.max_iter=30 training.warm_up=5 training.eval_num_items=2 training.sanity_check_num_items=2`
- writes logs/ckpts to a throwaway `exp_name` like `deltatok_geom_smoke`

Let the job complete (a few minutes on `qgpu`). Do not hand-roll `sbatch` — let the skill handle submission and tailing.

- [ ] **Step 3: Inspect first ~10 logged steps**

Pull the smoke run's stdout (or TB scalar dump) and verify:

| Scalar | Expected ballpark at step 1 | Expected trajectory over 30 iters |
|---|---|---|
| `Train/LossToken` | O(0.01-0.1) | flat or slowly decreasing |
| `Train/LossPointmap` | finite, O(0.1-10) | not NaN, not exploding |
| `Train/LossDepth` | finite, O(0.1-10) | not NaN |
| `Train/LossRaymap` | finite, O(0.1-10) | not NaN |
| `Train/LossTot` | weighted sum of above | monotone-ish |
| `Eval/<test>/LossRecon` (sanity) | within ~2x of pre-change baseline | — |

Specific failure modes to watch for:
- `RuntimeError: element 0 of tensors does not require grad` → Task 1 Step 2's check missed something; bisect inside `inference_batch_from_layer` / `_process_depth_output`.
- Any loss `NaN` at step 0 → re-check `gt_mask` dtype handling in `DepthLosses` (it must be float-castable; the reshape `gt_m.float().reshape(-1, 1, H, W)` should handle it).
- `Train/LossToken` exploding while geometric terms drop → confirm `L_token` is actually included in `L_total` (the spec calls it out as a stabiliser; dropping it can cause encoder collapse).
- `IndexError` in `gather_x` → re-verify `num_cameras = batch.get("num_cameras", 1)` matches what `_normalize_batch` produced; views must be camera-major within each timestep (lines 504-512).

- [ ] **Step 4: Report and stop**

Summarise to the user in 3-5 bullets: did the run complete? per-component first/last-step scalar values? sanity-check eval losses? anomalies?

Do **not** push, merge, or kick off any longer training run without explicit user direction. The longer-horizon training run that validates spec success criteria (PredVsGT eval losses decreasing vs. the geom-disabled baseline) is a separate experiment the user will trigger.

---

## Non-goals — explicit out of scope

From the spec § Non-goals — must not creep into any task:
- AR rollout in training (TF only).
- Anchor / context-frame stitching in the train-time decode.
- Scale-invariant depth loss term.
- Changing eval criteria (they stay at L2 / `lambda_c=0`).
- Tuning OccRAE decoder weights (already frozen via `requires_grad_(False)` in `_build_occ_rae`; `decode_grad` only opens the autograd graph into inputs).

## Self-review notes

- **Spec coverage.** § A (decode_grad) → Task 1. § B (train criteria) → Task 3. § C (steps 1-6: pair tracking, forward, GT gather, stitch+decode, criteria+sum, logging) → Task 4. § D (config) → Task 2. Smoke run (Task 5) addresses spec open questions 1 and 2; open question 3 (LossRecon alias) is resolved inline in Task 4 (keep as alias).
- **Pointmap confidence.** Open question 1 resolved during plan write: `_process_depth_output` (model_da3.py:761-769) returns only `depth_conf` and `ray_conf` — no `pointmap_conf`. So Task 4 passes `confidence=None` to `train_pointmap_criterion`; `PointmapLoss(lambda_c=1.0, confidence=None)` falls back to plain masked L1 (losses_da3.py:308), which is fine.
- **Shape consistency.** `gather_x` returns `(num_selected, num_cameras, ...)`; this matches the criteria's accepted shapes (PointmapLoss `(..., 3)`, DepthLosses `(B, 1, H, W)` after reshape, RaymapLoss `(B, T, H, W, 6)`).
- **No placeholders.** Every step has an exact path, code block, or shell command. No "TODO", no "implement later", no "similar to Task N".
