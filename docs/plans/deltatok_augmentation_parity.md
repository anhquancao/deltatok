# Plan: DeltaTok-Parity Augmentations (Random Per-Pair Stride + Horizontal Flip)

## Objective
Bring our DeltaTok training pipeline (`occrae/deltatok_trainer.py` +
`occrae/dataset/preprocessed_sequence.py`) closer to the upstream
`third_party/deltatok` Kinetics training recipe by reproducing two augmentations
that we currently do not implement:

1. **`time_stride_range: (1/25, 1/3)`** — a *randomized per-pair temporal gap*.
   In upstream `datasets/base.sample_frame_indices`, each of the V−1 deltas
   between consecutive sampled frames is drawn independently from
   `U(min_dt, max_dt)` *in seconds*. Pairs in the same clip therefore typically
   have **different** strides.
2. **`horizontal_flip: True`** — a 50% NumPy mirror applied to all V frames in a
   sequence (upstream `datasets/base._augment`, base.py:140).

Our `PreprocessedSequenceDataset` is the opposite on both axes:
- `generate_temporal_sequences` (preprocessed_sequence.py:155) bakes a single
  *equispaced* stride list `[k · subsampling_rate for k in 0..max_stride]` into
  the cached records — every pair in a sequence has the same gap, fixed at
  cache time.
- `collate_preprocessed` (deltatok_trainer.py:205) does no horizontal flipping
  of images, intrinsics, or `cam2worlds`.

The goal of this plan is to add both augmentations *without* breaking the
existing record-cache contract, the multi-resolution collate, or the GT
raymap/pointmap consistency required by the eval path.

## Scope & Impact

* **Dataset loading.** `generate_temporal_sequences` gains the ability to
  emit windows whose consecutive gaps are each drawn from a discrete
  `[min_stride, max_stride]` range (per-pair stride) instead of a single fixed
  list. The on-disk record schema and the cache key gain new fields.
* **Trainer config.** `configs/train_deltatok.yaml` gets a new
  `dataset.stride_range` block (and an optional `horizontal_flip` flag under
  `dataset` / `augmentation`). Existing `subsampling_rate` / `max_stride` keys
  remain valid — they are auto-mapped to a degenerate single-stride range so
  old configs still work.
* **Collate.** `collate_preprocessed` (deltatok_trainer.py:205) gains an
  optional horizontal-flip step that flips `imgs`, mirrors `intrinsics.cx`, and
  inverts the camera x-axis in `cam2worlds`. `gt_raymap` and `gt_pointmap` are
  recomputed from the flipped intrinsics + c2w *inside the same collate call*,
  so downstream losses see a self-consistent GT.
* **Cache invalidation.** Because the stride enumeration changes, the
  sequence-record cache version (`_SEQ_CACHE_VERSION`) bumps. Existing
  `.pkl` caches under each processed root are ignored on first run and
  rewritten.
* **Out of scope.** No changes to OccRAE, DA3, the DeltaTok module, the
  optimizer, the LR schedule, or the eval visualization helpers. No change to
  inference (`inference.py`, `extract_recon.py`) — augmentation is training-only.

## Key Files
1. `occrae/dataset/preprocessed_sequence.py` — sequence enumeration, cache
   key, `SequenceRecord` schema.
2. `occrae/deltatok_trainer.py` — `collate_preprocessed`, optional flip path,
   GT recomputation.
3. `configs/train_deltatok.yaml` — new augmentation knobs.
4. `docs/occrae.md` (touch only if the augmentation knobs are user-facing).

## Background: mapping seconds → frames

Deltatok trains on Kinetics videos at ~25–30 fps; `(1/25, 1/3) s` is roughly
`(1, 8)` raw frames. Our preprocessed sequences are stored in *raw frame index*
units, so the mapping needs the dataset native fps:

| Dataset | fps  | `1/25 s` | `1/3 s` |
| ------- | ---- | -------- | ------- |
| KITTI   | 10   | 0.4 fr   | 3.3 fr  |
| Waymo   | 10   | 0.4 fr   | 3.3 fr  |
| nuScenes (RGB) | 12 | 0.5 fr | 4.0 fr |

So a deltatok-equivalent range in our units is roughly `stride ∈ {1, 2, 3}`
raw frames at 10 fps, or `{1, 2, 3, 4}` at 12 fps. Our current default
`subsampling_rate=5, max_stride=3` already produces 0.5 s pair gaps at 10 fps —
*longer* than deltatok's max. Without the range change we cannot reach the
short end of deltatok's distribution.

## Proposed Solution (Implementation Steps)

### Step 1: Per-pair stride enumeration in `generate_temporal_sequences`

* Add new params to `generate_temporal_sequences` (preprocessed_sequence.py:155):
  - `min_stride: int` (raw frames between consecutive sampled frames, inclusive)
  - `max_stride: int` (inclusive)
  - `num_steps: int` (number of *pairs*; equals V−1 for our DeltaTok use case)
  - keep `subsampling_rate` for backward compat — it scales both bounds.
* Inside the per-camera loop, replace the fixed `strides` list with an
  enumeration of all length-`num_steps` tuples
  `(d_1, …, d_{num_steps})` where each `d_i ∈ [min_stride, max_stride]`.
  For each starting `frame_id`:
  1. Compute the cumulative offsets `0, d_1, d_1+d_2, …`.
  2. Check every offset frame stem exists.
  3. If yes, emit `SequenceRecord` with `strides=tuple(cum_offsets)`.
* Cap explosion: when `(max_stride − min_stride + 1) ** num_steps` exceeds a
  configurable budget (e.g. 64), sample a deterministic random subset per
  starting frame using a seed derived from `(scene_name, frame_id)` so caches
  are reproducible.
* Update `frame_stride` jump logic to step past the *minimum* possible window
  end (`frame_id + min_stride * num_steps`) when `frame_stride is None`, so
  windows can still overlap as needed.

### Step 2: Cache schema + version bump

* Add `min_stride`, `max_stride`, `num_steps` to `expected_params` in
  `_load_cached_root_records` and `_save_cached_root_records`.
* Bump `_SEQ_CACHE_VERSION` (preprocessed_sequence.py top) so existing pickles
  are invalidated. New cache file name pattern:
  `v{V}_sub{sub}_stride{min}-{max}_steps{N}_fs{fs}.pkl`.
* `SequenceRecord` already carries `strides` as a tuple; no schema break, just
  new contents (variable per-pair gaps instead of `[0, k, 2k, 3k]`).

### Step 3: Config plumbing

In `configs/train_deltatok.yaml`:

```yaml
dataset:
  preprocess_data_root: /gpfs/scratch/ehpc558/quan/occrae_data
  subsampling_rate: 1            # 1 = raw frame units
  stride_range: [1, 3]           # per-pair gap, inclusive (replaces max_stride)
  num_pairs: 3                   # = num_views - 1; equals 3 for V=4
  enumeration_budget: 64
  horizontal_flip: true
  processed_roots: []
  max_train_samples: -1
  max_val_samples: -1
```

Backward-compat shim in `fit()` (deltatok_trainer.py:828): if
`stride_range` is missing but `max_stride` is set, derive
`stride_range = [subsampling_rate, max_stride * subsampling_rate]` and
`num_pairs = max_stride`.

### Step 4: Horizontal flip in `collate_preprocessed`

In `collate_preprocessed` (deltatok_trainer.py:205), after the per-batch
crop/resize loop and before `torch.stack`:

```python
flip = horizontal_flip_enabled and random.random() < 0.5  # ONE coin per item
if flip:
    item_imgs = [img.flip(-1) for img in item_imgs]
    if has_gt:
        # cx -> W - cx
        for K in item_intrinsics: K[0, 2] = width - 1 - K[0, 2]
        # invert camera local x-axis
        for c2w in item_c2w: c2w[:3, 0] *= -1
```

Important constraints:

1. **One coin per record, not per frame** — all V frames must share the flip,
   otherwise temporal correspondence between `frame_t` and `frame_{t+1}` is
   destroyed and DeltaTok can't learn delta tokens.
2. **GT recomputation must happen *after* flipping**, since the existing
   raymap/pointmap recomputation block (deltatok_trainer.py:286) reads
   intrinsics/c2w. The current code already recomputes raymap from
   intrinsics + c2w in the collate, so as long as we flip *before* that
   block, no extra change is needed.
3. **Depth is invariant under horizontal flip** (it's a per-pixel scalar), so
   `gt_depth` only needs the spatial flip applied alongside `imgs`.

### Step 5: Sanity-check overfit run

Run an overfit on a single sequence (`max_train_samples: 1`) with both new
augmentations on:

```bash
ssh karolina "conda activate occany && source env_bsc.sh && \
  python train_deltatok.py --config-name train_deltatok \
    dataset.max_train_samples=1 dataset.horizontal_flip=true \
    dataset.stride_range='[1,3]' dataset.num_pairs=3 \
    training.max_iter=2000 training.virtual_epoch=200"
```

Expected behavior:

* Train loss still drives to near-zero on the single sample (augmentation does
  not prevent overfit — it just changes the input distribution).
* Eval visualizations show flipped frames roughly half the time, with
  consistent cameras (predicted depth/pointmap should still register against
  flipped GT).
* TensorBoard `Train/SpeedSamplesPerSec` is within ~10% of the pre-change
  baseline (the only added work is a per-batch image flip and ~6 scalar
  multiplications on intrinsics/c2w).

### Step 6: Full training comparison (optional, after sanity check)

Launch one run with augmentations off and one with augmentations on, same
seed, same `max_iter`. Compare:

* `Eval/LossRecon` at matched iterations
* `Eval/LossDepth` and `Eval/LossPointmap` on the held-out val split

Augmentation should *not* materially improve eval loss in the short term — the
val set is small — but the model should generalize better at long horizons.

## Risks & Open Questions

* **Sequence enumeration blow-up.** With `stride_range=[1,3]` and
  `num_pairs=3` the budget per starting frame is 27 — manageable. With wider
  ranges, the deterministic-subset sampler in Step 1 must kick in or the cache
  pickle will balloon. We should log the per-root sequence count in
  `train_stats` and warn if it exceeds e.g. 5× the pre-change baseline.
* **Deltatok semantics vs ours.** Upstream samples a *continuous* time delta
  and snaps to the nearest decoded frame. We can only sample at the discrete
  raw-frame grid. This loses jitter at sub-frame resolution but is the closest
  faithful analog given that our preprocessing already discretizes time.
* **fps assumption.** The `stride_range` defaults assume 10 fps; the user
  must override per-dataset if mixing nuScenes (12 Hz) with KITTI/Waymo
  (10 Hz). Consider documenting per-dataset defaults in
  `docs/occrae.md` once the knob lands.
* **Camera flip and SE(3) consistency.** Flipping the camera local x-axis
  changes the chirality of the camera frame. Our raymap is recomputed from
  intrinsics+c2w in the same collate call so it stays consistent, but any
  downstream code that reads `gt_c2w` *and* assumes a right-handed camera
  basis would break. Audit `occany.utils.helpers.intrinsics_c2w_to_raymap`
  and the pointmap loss before merging.

## Rollout

1. Land Step 1–3 behind a config gate (`stride_range` default falls back to
   the old behavior). Verify cache rebuild on the dev split.
2. Land Step 4 behind `dataset.horizontal_flip` (default `false`).
3. Run Step 5 sanity check.
4. Flip both defaults to `true` only after Step 6 shows no eval regression.
