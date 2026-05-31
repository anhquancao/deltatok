# Trajectory eval: single-camera temporal sampling

Why the CVPR recon model (`occany_cvpr26/sh/train_occany_plus_recon_1B.sh`) beats
the current distill checkpoints on the nuScenes trajectory eval
(`slurm/eval_trajectory.slurm`), and how to close the gap by changing only the
train view sampling.

## What the trajectory eval actually feeds

- Recon-only pose recovery: `infer_trajectory.py` builds **25 consecutive
  single-camera frames** (`_build_occany_recon_views`), monotonic
  `timestep=0..24`, view 0 = frame 0, then calls
  `inference_occany_da3(..., pose_from_depth_ray=True)`.
- `NuscenesVistaDataset` takes frames 0..24 consecutively; the GT trajectory is
  defined at 2 Hz (`src_indices = arange(len)*5.0`) and interpolated to 25 frames.
- So the eval distribution is: **one camera, followed over time**. 25 frames
  @10 Hz ≈ 5 keyframes @2 Hz — a ~2.0 s span, in-distribution for a 2 Hz trainer.

## Diagnosis

The differentiator is the **fraction of single-camera temporal training
samples**, not the loss recipe or the trajectory horizon.

- CVPR mix: `seq_exact_len_*` entries (single-camera by construction) +
  `seq_surround_all` → ~64% single-camera temporal.
- Current distill mix
  (`sh/train_occany_plus_recon_1B_infinite_depth_sam3_distill.sh`): all surround
  entries use `seq_surround_temporal_*` with `num_views_per_timestep=5/6`. A clean
  single-camera chain only appears when the sampler happens to draw `actual_vpt=1`
  (~17-20% of samples). The single-camera temporal signal is diluted.

Ruled out (each checked, none is the cause):

- **Ray maps / forecasting** — model is reconstruction-only; `ray_map_prob=-1`
  everywhere relevant.
- **Cadence mismatch** — 25@10 Hz = 5@2 Hz; the 2.0 s extent is in-distribution
  for CVPR's 5-10 frame @2 Hz training.
- **Anchor / ordering** — index 0 is the temporal window start in BOTH train and
  eval (modulo the shared 50% `reverse_seq`); only the anchor *camera* is random
  in multi-cam, and the eval has a single camera anyway.
- **Sequence length** — current's budget (max 28) already covers the eval's 25.

## The fix: add single-camera temporal entries (keep the recipe)

`sh/train_occany_plus_recon_1B_infinite_depth_sam3_distill_cvpr_sampling.sh`
(wired into `slurm/jz_train_occany_plus_recon_1B_infinite_depth_sam3_distill_cvpr_sampling.slurm`):
copy the distill script verbatim — all 5 original entries, all loss / distill /
InfiniDepth flags, same test set — and **ADD** 4 single-camera temporal entries
(Waymo/DDAD/Pandaset/Once) that reuse the SAME `seq_surround_temporal_sub5_stride9_all.pkl`.

### Why not `num_views_per_timestep=1`

`generate_surround_temporal_sequences` (`occrae/util/seq_helper.py:66`) lays the
pkl out **timestep-major**:

```
[cam0_t0, cam1_t0, ..., camN_t0,  cam0_t1, cam1_t1, ..., camN_t1,  ...]
```

The unified sampler (`occany/datasets/base_seq_dataset.py:_get_views`) with
`num_views_per_timestep=1` treats *every flat entry as its own timestep*
(`num_avail_T = seq_len // 1`) and slides a contiguous flat window. On a
timestep-major layout that walks **cameras within a timestep**, and the anchor is
always camera 0 (`rng.integers(0, max_vpt=1) == 0`). That is neither
single-camera nor random — it does NOT mimic the eval. (`anchor_cam` is a stored
field that the sampler ignores: line 274 always re-draws the anchor.)

### Correct mechanism: keep `num_views_per_timestep`, force `actual_vpt=1`

Keep `num_views_per_timestep` at the real camera count (Waymo/Once 5,
DDAD/Pandaset 6) so the sampler sees true timesteps (`base = t * num_cameras`,
`num_avail_T = seq_len // num_cameras = 10`). Then force one camera per timestep
via the budget:

```
min_memory_num_views == max_memory_num_views == min_num_timesteps == 10
```

Walking through `_get_views`:

- `memory_num_views` is fixed at 10.
- `vpt_cap = memory_num_views // min_num_timesteps = 10 // 10 = 1`
- `upper = min(camera_pool, memory_num_views, vpt_cap) = 1` → `actual_vpt = 1`
- `_select_cameras(camera_pool, 1, rng)` draws ONE random anchor camera in
  `[0, n_cams)`.
- `T = memory_num_views // actual_vpt = 10` → indices `[t*n_cams + c for t in 0..9]`
  = one random camera followed across all 10 timesteps. `reverse_seq=True` still
  flips direction 50% of the time.

### `min_num_timesteps` / no `max_num_timesteps`

`min_num_timesteps` guarantees each sample spans ≥ that many timesteps by capping
cameras-per-timestep: `vpt_cap = memory_num_views // min_num_timesteps`. Default
`1` lets surround samples take many cameras; setting it equal to the (fixed)
budget pins `actual_vpt=1`. There is **no `max_num_timesteps`** parameter and none
is needed: with `min_memory == max_memory == min_num_timesteps`, the timestep
count is exactly fixed (`T=10`). Strictly single-camera AND variable-length would
require separate entries per length (or a new sampler knob), because a single
`min_num_timesteps` against a `memory_num_views` *range* lets `actual_vpt` exceed 1
(e.g. `10 // 5 = 2`).

## Files

- `sh/train_occany_plus_recon_1B_infinite_depth_sam3_distill_cvpr_sampling.sh`
- `slurm/jz_train_occany_plus_recon_1B_infinite_depth_sam3_distill_cvpr_sampling.slurm`
  (Jean Zay H100, 4 nodes × 4 GPUs, `DATA_ROOT` on fsn1, logs/ckpts on `$TRG_WORK`)
- EXP_NAME: `occany_plus_recon_1B_infinite_depth_sam3_distill_cvpr_sampling`
