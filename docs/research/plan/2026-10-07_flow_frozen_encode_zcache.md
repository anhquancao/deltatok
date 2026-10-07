# Plan: flow trains from cached GT delta tokens; DA3 runs on frame 0 only

**Date:** 2026-10-07 · **Thread:** flow · **Cluster:** BSC
· **Reference:** control `BSC:46503892` (3.24 s/iter, 4 GPU, bsize 16), posecond `BSC:47029284`
· jobs: _pending_ · prior cycle: —

## Overview

- **Offline, once:** `extract_deltatok_flow_zcache.py` writes the GT delta tokens z of every consecutive frame pair,
  one copy per height. Only the frozen DA3 + DeltaTok encode is cached.
- **Loader:** unchanged, still opens all 10 frames (images, `gt_c2w`, ...). With `zcache_root` set it also attaches
  the window's 9 z rows from the cache.
- **Trainer:** DA3 runs on frame 0 only, for the cross-attn condition. The DeltaTok encode is skipped.

| File | Change |
|---|---|
| `occany/datasets/base_seq_dataset.py` | `_load_raw_frame` helper (done); `zcache_root` kwarg; attach `zc_z` |
| `occrae/deltatok_shared.py` | `_normalize_batch` passes `zc_z`; `T >= 2` assert moved in `_extract_pair_feats` (done) |
| `occrae/deltatok_flow_trainer.py` | cache branch at the top of `_encode_inputs` |
| `extract_deltatok_flow_zcache.py` | new (done) |
| `configs/deltatok_flow/` | 2 new `*_zcache_bsc.yaml` |
| `slurm/` | extraction array (done) + the control's zcache twin |

## 0. Cache layout

- **Root:** `/gpfs/scratch/ehpc1001/quan/deltatok_flow_zcache/l12_dtok64_tc1536_sigreg0.02_ep70`.
  Tokenizer: `epoch_70` of `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.02_ns3072_pool24576_compose1.0_decnoise0_sw0_seed1_rclip0`.
- **One dir per height and scene:** `<root>/<W>x<H>/<DatasetClass>/<scene>/`
  - `z.npy`: `(n_pairs, 64, 1536)` float32, exactly the encoder's output (smoke: rel L2 0 vs the online encode).
  - `pairs.json`: `[[frame_a, frame_b], ...]`, one entry per row.
- **Heights:** 280, 266, 210, 168, all 518 wide. 294 is dropped.
- **Size:** 393 KB per pair. ~1.18M pairs over the 5 train sources (ONCE 974k) → ~460 GB per height, ~1.85 TB total.

## 1. `occany/datasets/base_seq_dataset.py`

- **`_load_raw_frame`:** done. The npz load + skew fix moved out of `_get_views`, outputs unchanged.
- **`__init__`:** kwarg `zcache_root=None` → `self.zcache_root`; `self._zc = {}` (per-worker open-scene cache).
- **`__getitem__`, after the `is_good_type` loop, before the deepcopy (L320):**
  ```python
  if self.zcache_root:                                              # train-only: z from the cache
      ids = [v['frame_id'] for v in views]                          # T frame ids of the window
      views[0]['zc_z'] = self._read_zcache(views[0]['scene_name'], ids, resolution)  # (T-1, K, C) float32
  ```
- **New `_read_zcache(scene, frame_ids, resolution)`:** mmap `<W>x<H>/.../z.npy`, pick the T-1 consecutive pairs
  via `pairs.json`, keep the last 32 scenes open in `self._zc`.

## 2. `occrae/deltatok_shared.py:_normalize_batch`

Before `return out` (L154):
```python
if "zc_z" in batch[0]:                                    # z-cache loader
    out["zc_z"] = batch[0]["zc_z"]                        # (B, T-1, K, C) float32
```

## 3. `occrae/deltatok_flow_trainer.py:_encode_inputs` (L421)

Insert at the top:
```python
if "zc_z" in batch:                                       # z-cache: DA3 on frame 0 only, z read from disk
    assert not want_tokens, "z-cache has no full tokens; feat loss off"
    _, feats, _, _, H, W = self._extract_pair_feats(imgs[:, :num_cameras], num_cameras=num_cameras, return_pairs=False)  # (B, 1, N, P, C)
    z = batch["zc_z"].to(self.device, non_blocking=True).unsqueeze(2)  # (B, T-1, 1, K, C)
    return None, feats[:, 0].contiguous(), z, H, W
```
`_build_pose_cond` is unchanged: `gt_c2w` comes from the loader as today.

## 4. `extract_deltatok_flow_zcache.py` (done)

- Builds the frozen encoder from the flow config, lists every scene's unique consecutive pairs over all train
  windows, shards scenes `[pid::world]`.
- Units of 512 pairs; one DataLoader over all units' frames prefetches while the GPU runs DA3 per frame and
  DeltaTok per pair, at each height. Frames are resized with the loader's own function.
- Writes `pairs.json` + `z.npy` (tmp + rename) per height; skips scenes whose 4 `z.npy` exist.
- `--verify N`: N random train windows through the online 10-frame encode vs the cache. Pass: relative L2 ≤ 1e-2.

## 5. Configs (`configs/deltatok_flow/`)

- **`train_deltatok_flow_alldata_ctx2fwd8_zcache_bsc.yaml`:** `cp train_deltatok_flow_alldata_ctx2fwd8_bsc.yaml`.
  Add `zcache_root='<root>'` to each of the 5 train strings. Test strings unchanged.
- **`train_deltatok_flow_alldata_ctx2fwd8_multires_zcache_bsc.yaml`:** `cp` of the multires config. Same
  `zcache_root`. `resolution=[(518, 280), (518, 266), (518, 210), (518, 168)]`.

## 6. Slurm

- **`slurm/extract_deltatok_flow_zcache_bsc.slurm`:** done. `--array=0-19`, 1 GPU, `--time=04:00:00`,
  `--world 20 --pid $SLURM_ARRAY_TASK_ID`.
- **`slurm/deltatok_flow/train_deltatok_flow_alldata_xxl_tc1536mg9_sigreg002seed1_ctx2fwd8_zcache_bsc.slurm`:**
  `cp` of the control slurm.
  - `CONFIG_NAME=train_deltatok_flow_alldata_ctx2fwd8_zcache_bsc`, `RUN_NAME=..._xxl_dit_zcache`, 12 h.
  - Job name, output and error changed together.

## 7. Pre-flight

1. **Sync:** monitor-sync, then md5 every touched file on BSC.
2. **Small extraction + verify:** one acc_debug task with `--world 200 --pid 0 --verify 32`. Pass: relative L2 ≤ 1e-2.
3. **Full extraction:** submit the 20-task array.
4. **Zcache twin of the control:** launch, watch to the first loss line.
   - s/iter vs the control's 3.24.
   - Loss at iters 500 / 1000 vs the control's 2.374 / 0.891 (same sampler draws).
