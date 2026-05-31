# Unified View Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `BaseSeqDatasetMultiView._get_views` with a single contiguous-window algorithm that returns exactly `mem_views` views per item by decomposing the per-batch budget into `actual_vpt` random cameras × matching number of timesteps. Then run the unified training via the updated `sh/train_occany_plus_recon_1B_infinite_depth.sh`.

**Architecture:**
- One unified `_get_views` for all datasets (temporal-only or surround). `actual_vpt` is drawn uniformly in `[1, num_views_per_timestep]` per item, then the budget `mem_views` is split into `num_full_T = mem_views // actual_vpt` full timesteps plus an optional partial last timestep of `mem_views % actual_vpt` cameras. A contiguous window of `timesteps_needed = num_full_T + (1 if partial else 0)` timesteps is chosen, the same cameras (anchor at index 0) are used across every timestep, and the partial timestep takes the first `partial` entries of that camera order. Padding by random repetition handles sequences shorter than the window.
- `_select_cameras` always picks a random anchor per item (no fixed-anchor mode).
- Legacy constructor params (`min_views_per_timestep`, `min_num_timesteps`, `no_partial_views`, `anchor_cam`) are kept on `BaseSeqDatasetMultiView.__init__` as ignored args so the deprecated `seq_surround` script keeps parsing. The new algorithm reads none of them.
- The target training script (`train_occany_plus_recon_1B_infinite_depth.sh`) is updated to `min_memory_num_views=2` per dataset block, with the obsolete knobs stripped.

**Tech Stack:** Python (numpy), bash entrypoints.

---

## File Structure

**Modified:**
- `occany/datasets/base_seq_dataset.py` — rewrite `_get_views`, simplify `_select_cameras`, drop `_build_view_indices`, mark legacy ctor params as ignored, drop stale `__getitem__` comment about bump-up.
- `sh/train_occany_plus_recon_1B_infinite_depth.sh` — strip obsolete knobs, set `min_memory_num_views=2` across train blocks, clean up `no_partial_views=True` from eval blocks.

**Untouched (intentional):**
- `sh/train_occany_plus_recon_1B_infinite_depth_seq_surround.sh` — keeps its values; runs under the new algorithm but the legacy knobs become no-ops.
- `sh/train_occany_plus_recon_1B_no_sam_distill.sh` — its `min_views_per_timestep=1` becomes a no-op (equivalent to new default).
- `test_view_sampling.py` — old assertions no longer hold; left as-is since the user can rewrite or delete later.

---

## Tasks

### Task 1: Simplify `_select_cameras` to always-random anchor

**Files:**
- Modify: `occany/datasets/base_seq_dataset.py:263-280`

- [ ] **Step 1: Replace `_select_cameras` body**

Open `occany/datasets/base_seq_dataset.py`, find the existing `_select_cameras` method (lines 263-280) and replace it with:

```python
    def _select_cameras(self, max_vpt, actual_vpt, rng):
        # Always pick a random anchor per call; anchor occupies index 0.
        anchor = int(rng.integers(0, max_vpt))
        if actual_vpt >= max_vpt:
            others = np.arange(max_vpt)
            others = others[others != anchor]
            rng.shuffle(others)
            return np.concatenate([[anchor], others])
        if actual_vpt == 1:
            return np.array([anchor])
        others = np.arange(max_vpt)
        others = others[others != anchor]
        rest = rng.choice(others, size=actual_vpt - 1, replace=False)
        return np.concatenate([[anchor], rest])
```

The only change vs. the current version is removing the `if self.anchor_cam is None: ... else: anchor = self.anchor_cam` branch.

- [ ] **Step 2: Commit**

```bash
git add occany/datasets/base_seq_dataset.py
git commit -m "refactor: always randomize anchor camera in _select_cameras"
```

---

### Task 2: Rewrite `_get_views` with the unified algorithm

**Files:**
- Modify: `occany/datasets/base_seq_dataset.py:301-380` (the `_get_views` body up through the end of the view-building loop's index construction — stop before frame loading at the `frames = [seq[i] for i in memory_view_indices]` line, which is preserved unchanged)

- [ ] **Step 1: Replace `_get_views` head through view-index construction**

The current `_get_views` runs from roughly line 301 through line 376 (everything up to and including the `if len(memory_view_indices) < memory_num_views:` padding block). Replace that entire span with the following — keep the code that follows (`frames = [seq[i] for i in memory_view_indices]` onward) verbatim.

```python
    def _get_views(self, seq_idx, resolution, memory_num_views, rng, is_eval=False):
        scene_idx, seq, ts = self.seqs[seq_idx]
        scene_name = self.scenes[scene_idx]
        preprocessed_scene_dir = osp.join(self.ROOT, scene_name)

        seq_len = len(seq)
        max_vpt = self.num_views_per_timestep
        num_avail_T = seq_len // max_vpt
        assert num_avail_T > 0, (
            f"seq_len ({seq_len}) must be >= num_views_per_timestep ({max_vpt})"
        )

        # 1. Choose cameras for this item.
        if self.fixed_cams is not None:
            selected_cams = np.array(self.fixed_cams, dtype=int)
            actual_vpt = len(selected_cams)
        else:
            actual_vpt = int(rng.integers(1, max_vpt + 1))
            selected_cams = self._select_cameras(max_vpt, actual_vpt, rng)

        # 2. Decompose mem_views into full timesteps + optional partial last.
        num_full_T = memory_num_views // actual_vpt
        partial = memory_num_views % actual_vpt
        timesteps_needed = num_full_T + (1 if partial else 0)

        # 3. Pick a contiguous window of timesteps.
        if num_avail_T >= timesteps_needed:
            start_T = int(rng.integers(0, num_avail_T - timesteps_needed + 1))
            chosen_T = list(range(start_T, start_T + timesteps_needed))
        else:
            chosen_T = list(range(num_avail_T))

        # 4. Optional reversal (same coin per item).
        if self.reverse_seq and rng.random() < 0.5:
            chosen_T = list(reversed(chosen_T))

        # 5. Build view indices. Same selected_cams (anchor first) across every
        # full timestep; partial last timestep takes selected_cams[:partial],
        # which always includes the anchor at index 0.
        memory_view_indices = []
        for i, t in enumerate(chosen_T):
            base = t * max_vpt
            cams = selected_cams if i < num_full_T else selected_cams[:partial]
            memory_view_indices.extend(int(base + c) for c in cams)

        # 6. Pad with random repetition if the sequence couldn't supply
        # `timesteps_needed` timesteps (preserves uniform batch tensor shape).
        if len(memory_view_indices) < memory_num_views:
            num_repeats = memory_num_views - len(memory_view_indices)
            repeated = rng.choice(memory_view_indices, size=num_repeats, replace=True)
            memory_view_indices = list(memory_view_indices) + [int(i) for i in repeated]

        assert len(memory_view_indices) == memory_num_views, (
            f"expected exactly {memory_num_views} indices, got {len(memory_view_indices)}"
        )

```

(The original code continued here with `frames = [seq[i] for i in memory_view_indices]`. Leave that and everything after it untouched.)

- [ ] **Step 2: Delete the now-unused `_build_view_indices` helper**

Find the `_build_view_indices` method (currently at `occany/datasets/base_seq_dataset.py:282-299`) and delete it entirely — the new `_get_views` inlines its logic.

- [ ] **Step 3: Clean up the stale comment in `__getitem__`**

In `__getitem__` at `occany/datasets/base_seq_dataset.py:469-473`, the comment block reads:

```python
        # _get_views may bump memory_num_views up (min_num_timesteps) or return
        # fewer views when the source sequence is shorter than the requested
        # budget. Sync to the actual returned count so view['idx'] and
        # view['memory_num_views'] reflect reality, and drop ray_map indices
        # that fall past it so gen_view_idx lookups stay in range.
        memory_num_views = len(views)
```

Replace it with the narrower (still true) version:

```python
        # Sync to the actual returned count so view['idx'] and
        # view['memory_num_views'] reflect reality, and drop ray_map indices
        # that fall past it so gen_view_idx lookups stay in range.
        memory_num_views = len(views)
```

Under the new algorithm, `_get_views` returns **exactly** `memory_num_views` views, so this `len(views)` assignment is now a no-op — but keeping it costs nothing and keeps `view['idx']` symmetric.

- [ ] **Step 4: Mark legacy constructor params as ignored**

At `occany/datasets/base_seq_dataset.py:220-245` (the `BaseSeqDatasetMultiView.__init__` head), keep the existing signature so legacy scripts don't crash, but add a one-line comment above the four legacy assignments to mark them deprecated. Locate the block (after the `super().__init__` call style), and just above `self.min_views_per_timestep = min_views_per_timestep`, insert this single comment line:

```python
        # DEPRECATED: legacy knobs kept for backward compat with seq_surround
        # script; ignored by the unified _get_views.
```

Do not remove the four assignments — leave them so `self.min_views_per_timestep`, `self.min_num_timesteps`, `self.no_partial_views`, and `self.anchor_cam` remain readable from `self`.

- [ ] **Step 5: Commit**

```bash
git add occany/datasets/base_seq_dataset.py
git commit -m "refactor: unified _get_views with contiguous-window budget split"
```

---

### Task 3: Update `train_occany_plus_recon_1B_infinite_depth.sh` to the new config

**Files:**
- Modify: `sh/train_occany_plus_recon_1B_infinite_depth.sh:34-77`

The current script (already has `BATCH_SIZE=2`, `aug_crop=128`, `transform=SeqColorJitter`, `aug_focal=0.9` from earlier edits) still has the obsolete knobs and uses larger per-dataset `min_memory_num_views`. We strip the obsolete knobs and set `min_memory_num_views=2` across all train blocks.

- [ ] **Step 1: Rewrite the train_dataset and test_dataset blocks**

Open the script and replace the `$CMD \ --train_dataset="..." --test_dataset="..." \` span (lines 33-77 of the current file) with:

```bash
$CMD \
    --train_dataset="5000 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        2000 @ VKittiSeqMultiView(VKITTI_PROCESSED_ROOT='$SCRATCH/data/vkitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=10, ray_map_prob=-1, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$SCRATCH/data/ddad_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$SCRATCH/data/pandaset_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True) + \
        5000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$SCRATCH/data/once_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3', load_infinidepth_pseudo=True)"  \
    --test_dataset="200 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$SCRATCH/data/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=5, max_memory_num_views=5, reverse_seq=False, \
        z_far=50, split='val', seed=42, \
        resolution=[(518, 168)], distill_model_name='SAM3', base_model='da3') + \
        200 @ Occ3dNuscenesSeqMultiView(NUSCENES_PREPROCESSED_ROOT='$SCRATCH/data/occ3d_nuscenes_processed', \
        seq_pkl_name='seq_surround_temporal_sub1_stride9_all.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, num_views_per_timestep=6, \
        z_far=50, split='val', seed=42, fixed_cams=[0,1], \
        resolution=[(518, 266)], distill_model_name='SAM3', base_model='da3')" \
```

Changes from the current file:
- All five train blocks: `min_memory_num_views` set to `2`; `min_views_per_timestep=*`, `min_num_timesteps=2`, `no_partial_views=True` removed.
- Eval blocks: `no_partial_views=True` removed (along with the trailing comma that preceded it in the same line).
- Sample weights, `seq_pkl_name`, `num_views_per_timestep`, augmentation args, resolutions, `frame_interval`, `z_far`, `reverse_seq`, distillation/model knobs, and `load_infinidepth_pseudo=True` are all preserved.
- `ray_map_prob=-1` stays (per earlier decision: no raymap).

The rest of the file (lines 1-32 and 78 onward, including all `--lr`, `--lambda_*`, `--infinidepth_pseudo_supervision`, the trailing comment block) is unchanged.

- [ ] **Step 2: Verify shell syntax**

Run: `bash -n /home/acao/code/OccAny/sh/train_occany_plus_recon_1B_infinite_depth.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add sh/train_occany_plus_recon_1B_infinite_depth.sh
git commit -m "train: unified recon_1B_infinite_depth config with budget-based view sampling"
```

---

### Task 4: Smoke-test on Karolina

CLAUDE.md forbids running code locally — all Python execution must go via `ssh karolina`. This task validates the Python module imports, the dataset constructs, and `_get_views` returns the expected length under representative configs.

**Files:**
- Run: read-only checks via `ssh karolina`

- [ ] **Step 1: Import check**

Run:
```bash
ssh karolina "cd $(pwd) && conda activate occany && python -c 'from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView; print(\"import ok\")'"
```
Expected: `import ok`

If this fails with `AttributeError` or `ImportError`, re-read Task 2 — most likely a missing import or a typo in the new block.

- [ ] **Step 2: Algorithm length invariant (no real data needed)**

Save the following file as `scripts/check_unified_sampling.py` (a local-only sanity script — delete after running):

```python
"""One-shot sanity check for the unified _get_views algorithm.

Builds a mock subclass that bypasses real npz loading, then drives
_get_views directly with a range of (mem_views, N) combinations and
asserts that every call returns exactly mem_views indices.
"""
import os
import sys
import numpy as np

from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()

from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


def make_instance(max_vpt, num_timesteps):
    obj = BaseSeqDatasetMultiView.__new__(BaseSeqDatasetMultiView)
    obj.num_views_per_timestep = max_vpt
    obj.fixed_cams = None
    obj.reverse_seq = True
    obj.ROOT = "/tmp/does_not_matter"

    # Fake one scene with num_timesteps * max_vpt frames.
    obj.scenes = ["scene0"]
    seq = list(range(num_timesteps * max_vpt))
    ts = list(range(num_timesteps * max_vpt))
    obj.seqs = [(0, seq, ts)]
    obj.frames = list(range(num_timesteps * max_vpt))
    return obj


def main():
    rng = np.random.default_rng(seed=0)
    # Replace _get_views' downstream npz loading by monkeypatching:
    # we only exercise the index-selection portion. We'll call the
    # method but stop before frame loading by capturing memory_view_indices
    # via a thin wrapper.
    from occany.datasets import base_seq_dataset as bsd

    original = bsd.BaseSeqDatasetMultiView._get_views

    captured = {}

    def patched(self, seq_idx, resolution, memory_num_views, rng, is_eval=False):
        # Inline copy of the index-selection portion of _get_views.
        scene_idx, seq, ts = self.seqs[seq_idx]
        seq_len = len(seq)
        max_vpt = self.num_views_per_timestep
        num_avail_T = seq_len // max_vpt
        assert num_avail_T > 0
        actual_vpt = int(rng.integers(1, max_vpt + 1))
        selected_cams = self._select_cameras(max_vpt, actual_vpt, rng)
        num_full_T = memory_num_views // actual_vpt
        partial = memory_num_views % actual_vpt
        timesteps_needed = num_full_T + (1 if partial else 0)
        if num_avail_T >= timesteps_needed:
            start_T = int(rng.integers(0, num_avail_T - timesteps_needed + 1))
            chosen_T = list(range(start_T, start_T + timesteps_needed))
        else:
            chosen_T = list(range(num_avail_T))
        if self.reverse_seq and rng.random() < 0.5:
            chosen_T = list(reversed(chosen_T))
        memory_view_indices = []
        for i, t in enumerate(chosen_T):
            base = t * max_vpt
            cams = selected_cams if i < num_full_T else selected_cams[:partial]
            memory_view_indices.extend(int(base + c) for c in cams)
        if len(memory_view_indices) < memory_num_views:
            num_repeats = memory_num_views - len(memory_view_indices)
            repeated = rng.choice(memory_view_indices, size=num_repeats, replace=True)
            memory_view_indices = list(memory_view_indices) + [int(i) for i in repeated]
        captured["indices"] = memory_view_indices
        captured["actual_vpt"] = actual_vpt
        captured["selected_cams"] = selected_cams.tolist()
        return memory_view_indices  # not a real views list, but enough for length check

    bsd.BaseSeqDatasetMultiView._get_views = patched
    try:
        for N in [1, 5, 6]:
            obj = make_instance(max_vpt=N, num_timesteps=20)
            for mem_views in [2, 5, 10, 12, 30]:
                if N * 20 < mem_views:
                    # seq too short — padding will kick in; still expect exact length
                    pass
                for trial in range(50):
                    result = patched(obj, seq_idx=0, resolution=None,
                                     memory_num_views=mem_views, rng=rng)
                    assert len(result) == mem_views, (
                        f"N={N}, mem_views={mem_views}, trial={trial}: "
                        f"got len(views)={len(result)}"
                    )
        print("all length invariants hold")
    finally:
        bsd.BaseSeqDatasetMultiView._get_views = original


if __name__ == "__main__":
    main()
```

Run:
```bash
ssh karolina "cd $(pwd) && conda activate occany && python scripts/check_unified_sampling.py"
```
Expected: `all length invariants hold`

If the assertion fires, read the printed `N`, `mem_views`, `trial` — those are the inputs that broke the invariant, then revisit Task 2 step 1.

- [ ] **Step 2.5: Delete the sanity script**

```bash
rm scripts/check_unified_sampling.py
```

(No commit — temporary script.)

- [ ] **Step 3: Launch the training run**

CLAUDE.md and project conventions say training is launched via SLURM or directly. The exact submission is the user's call — once the smoke check passes, the unified config is ready to launch via either:

```bash
ssh karolina "cd $(pwd) && bash sh/train_occany_plus_recon_1B_infinite_depth.sh"
```
or via SLURM (`sbatch slurm/train_occany_plus.slurm` if applicable). Hand off to user.

---

## Self-Review

- **Spec coverage:** The four design pieces from the brainstorm are each covered — `_get_views` rewrite (Task 2), `_select_cameras` random anchor (Task 1), legacy params kept-but-ignored (Task 2 step 4), target script updated (Task 3). Smoke verification covers length invariants (Task 4).
- **Placeholder scan:** No TBDs, no "implement later", no "add error handling". Each code step shows the actual code to write.
- **Type/name consistency:** The methods referenced match between tasks (`_select_cameras`, `_get_views`, `_build_view_indices` (to delete), `__init__` legacy params). The shell script identifiers (`min_memory_num_views`, `num_views_per_timestep`, `fixed_cams`, etc.) match what `BaseSeqDatasetMultiView.__init__` accepts.
- **Scope check:** Two files modified, focused on one behavior change. Single-plan-sized.

---
