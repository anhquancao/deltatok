# Multi-camera pointmap eval metric is broken (per-timestep decode) — 2026-08-04

**One line:** `_decode_tokens` decodes multi-camera items **one timestep at a time**, so each
timestep's pointmap comes out in its own local frame while the GT pointmap is in the global
timestep-0 frame. The result is a constant pointmap error floor for any 2-camera eval
(nuScenes `fixed_cams=[0,1]`). **Eval-only bug — does NOT affect training.** Present
byte-for-byte in both the old (4d5d690) and current code.

## RESOLVED — the consec-vs-random training regression: varied timesteps is the fix

The question that started this investigation — why the current-code `..._consec5_...` run
underperforms the old-code `..._pairslice` run — is answered, and it is **separate from the
eval bug below**. **Training DeltaTok on varied-interval timesteps recovers old-code
performance.** `deltatok_l12_dtok64_tc1536_nozn_randint5_vpt1to2_nosigreg` (BSC, random-interval
sampling, no SIGReg) trains well and does not show the consec5 plateau (~0.093 Eval).

- **Root cause:** consecutive 5-frame windows give adjacent-frame deltas that are too small —
  frame_{t+1} ≈ frame_t, so the tokenizer learns near-identity and the K=64 delta code is
  starved of signal. Random-interval sampling spreads timesteps across the record (irregular,
  larger gaps → larger, varied deltas), which is what the old code did (`min_memory_num_views`
  + spread timesteps). What the tokenizer must encode — the delta — is the lever; SIGReg,
  num_timesteps, reverse_seq, and eval set were all ruled out earlier.
- **Not** the multi-camera pointmap eval bug below: that is eval-only, orthogonal, and shows up
  in `LossPointmap_*`. This regression is a real training-signal effect, visible in the correct
  metrics (`LossRecon`, depth, `PredVsOrig`).
- **Actionable:** default DeltaTok training to `timestep_sampling='random'`.
  `configs/deltatok/train_deltatok_randint_bsc.yaml` +
  `slurm/deltatok/train_deltatok_multitoken_nosigreg_nozn_randint_bsc.slurm` are the reference.
- **Open:** SIGReg on top of the recovered sampling —
  `..._sigreg_nozn_randint_bsc.slurm` (BSC 44167693, RUN_NAME `...randint5_..._sigreg0.05_...`)
  submitted to confirm SIGReg still shapes z without giving back the sampling win.

## Symptom

`Eval/206 @ Occ3dNuscenesSeqMultiView/LossPointmap_PredVsGT(_AR)` looks **stuck** while depth
and recon descend normally. KITTI (1 camera) pointmap descends fine.

## Evidence (TensorBoard scalars, run `deltatok_surround_layer12_bsize2_dtok64_tc768_..._nozn`, JZ)

| metric | step 1125 | step 3375 | step 50625 | verdict |
|---|---|---|---|---|
| nuScenes `LossPointmap_OrigVsGT` (decoder + **perfect GT tokens**) | 15.587 | 15.587 | 15.587 | **constant floor** |
| nuScenes `LossPointmap_PredVsGT` | 17.05 | 16.09 | 15.54 | pinned to floor |
| nuScenes `LossPointmap_PredVsGT_AR` | 21.25 | 16.53 | 15.51 | pinned to floor |
| nuScenes `LossPointmap_PredVsOrig` (**tokenizer only**) | 4.58 | 3.18 | (descends) | **healthy** |
| nuScenes `LossDepth_PredVsGT_AR` | 7.83 | 4.46 | 3.07 | healthy |
| nuScenes `LossRecon` | 0.106 | 0.087 | 0.0286 | healthy |
| KITTI `LossPointmap_OrigVsGT` | 2.455 | 2.455 | 2.455 | low floor |
| KITTI `LossPointmap_PredVsGT` | 15.9 | 11.1 | 2.59 | descends → floor |

The smoking gun: `OrigVsGT` feeds the decoder **perfect GT tokens** and still gets **15.587
frozen** on nuScenes vs **2.455** on KITTI. `PredVsGT` is pinned to that 15.587 floor — the
tokenizer cannot beat it no matter how well it trains. Meanwhile the tokenizer-only metric
`PredVsOrig` descends, so the tokenizer is healthy; the metric is broken.

## Root cause

`occrae/deltatok_shared.py::_decode_tokens` (~line 410):

```python
if num_cameras <= 1:
    return self.occ_rae.decode({"tokens": tokens, ...})       # KITTI: all T frames, ONE call, shared frame
# num_cameras > 1:
for t in range(T):
    decoded_parts.append(self.occ_rae.decode({"tokens": tokens_by_t[:, t], ...}))  # nuScenes: per-timestep
```

The multi-camera branch decodes each timestep separately, so each timestep's pointmap is
produced in *that timestep's* reference frame (its camera-0), while the GT pointmap
(`_get_views`: `in_camera0 = affine_inverse(views[0]['camera_pose'])`) is in the **global
timestep-0** frame. For timesteps 1..T-1 the mismatch is the ego-motion offset between t0 and
t — a constant no training can reduce. Depth is per-view / frame-independent, so it's immune
(descends). KITTI takes the single-call branch (all frames share one frame) → fine.

## Why the floor magnitude varies (and why consecutive-eval runs still "go down")

The floor ≈ ego-motion displacement between t0 and t, so it scales with **how far apart the
eval timesteps are**:
- Old spread eval (`min_memory_num_views` + random-spread timesteps) → frames far apart →
  large floor (~15.5), hit within ~4 epochs.
- New consecutive eval (`num_timesteps=5`, adjacent frames) → small ego-motion → **much lower
  floor** → `PredVsGT_AR` descends a long way before flattening. So on the randint/consec5 BSC
  runs the nuScenes pointmap **is still descending** — consistent with the bug, just a lower floor.

**BSC confirmation (PENDING):** `scratchpad/tb_bsc.py` (nuScenes pointmap OrigVsGT / PredVsGT /
PredVsGT_AR for `..._randint5_..._nosigreg` and `..._consec5_..._nosigreg`) was **Killed on the
BSC login node** (cgroup limit; event files carry image summaries). Rerun on a compute node
(`srun ... python`) or with a memory-capped reader (`EventAccumulator` +
`size_guidance={IMAGES:0,...}`). Expect consec5's `OrigVsGT` floor << 15.5, with `PredVsGT_AR`
descending toward it — which is why the live randint run's nuScenes `PredVsGT_AR` is still going down.

## Scope — does NOT affect training

- `_decode_tokens` is called **only** in `eval_one_epoch` (`deltatok_trainer.py` 1262/1263/1289),
  which is `@torch.no_grad()`. Never in `train_one_epoch`.
- Training loss is pure feature-space (`_log_cosh(x_hat, x)` + optional SIGReg / feature-loss /
  bottleneck). No pointmaps, no `_decode_tokens`, no camera poses. No gradient path.
- Identical in old and new code → orthogonal to the consec-vs-random training regression.

## Fix — APPLIED 2026-08-04

`_decode_tokens` collapsed to a **single `occ_rae.decode` over all V views** (dropped the
per-timestep loop), so every view shares view-0's (t0, cam0) frame. Verified safe:
`inference_batch_from_layer` uses `ref_view_strategy="first"` and is view-count-agnostic;
`decode_to_image` already decodes all views in one call; the per-timestep split dated to the
initial commit (no OOM rationale). `num_cameras` arg kept for caller parity but unused.

**VALIDATION (in progress):** BSC `44160801` (randint nosigreg) resubmitted with the fix,
resuming from ~epoch 5. Two separate checks:

- **OOM safety — CONFIRMED.** `44160801` passed the startup sanity-check eval
  (`eval_one_epoch(sanity_check=True)`, `deltatok_trainer.py:1558`), which runs the exact V=10
  nuScenes single-call decode (5 ts x 2 cam through the giant's blocks 13-40) before the epoch
  loop. No OOM at 4 H100 / eval bsize — so no view chunking needed.
- **Metric value** is NOT verified by the sanity pass — scalar logging is gated on
  `not sanity_check` (`:1471`). Confirm at the **first real epoch-end eval** that nuScenes
  `LossPointmap_OrigVsGT` drops from ~15.5 to a KITTI-like value, with `PredVsGT` tracking it.

## Meanwhile (A/B guidance)

For the consec-vs-random and any multi-camera comparison, **ignore the nuScenes
`LossPointmap_*` scalars** — use `LossRecon` (the per-epoch "Eval" summary), depth, and
`PredVsOrig`, which are all correct.

## Pointers

- Bug: `occrae/deltatok_shared.py::_decode_tokens` (~410–430)
- Callers (eval-only): `occrae/deltatok_trainer.py` 1262, 1263, 1289
- GT frame: `occany/datasets/base_seq_dataset.py::_get_views` (`in_camera0`)
- Extraction scripts: `scratchpad/tb_extract.py` (JZ), `scratchpad/tb_bsc.py` (BSC)
