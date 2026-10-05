# VGGT-World technical details — checkpoints, stages, rollout stride, eval protocol

Created 2026-10-03 · thread `flow` · plan: `plan/2026-10-01_flow_vggt_world_baseline_eval.md` · deck: `results/2026-10-04_flow_vggt_world_baseline_slides.html`
· paper: arXiv 2603.12655 · code: `third_party/VGGT-World` · our runner: `eval_vggt_world.py`

**Use stride 1.** It is the authors' default rollout. Paper appendix B.3: "we maintain a consistent
chunk-size-4 configuration and autoregressively generate future frames using a sliding window with a stride of 1".
Their mid-term eval code does the same. Stride 2 was our own invention; keep it only as a comparison.

Paper quotes come from the arXiv HTML page, not the PDF. Wording may differ slightly.

## Model

VGGT with a flow-matching (FM) model between the two halves of its aggregator.

- **`aggregator.part1(frames)`:** image → tokens per frame. Input in [0, 1]; it normalises internally (`aggregator.py:325`).
- **`fm.sample_euler(cond, shape_like, steps, patch_hw)`:** 2 frames of tokens in → the next 2 frames out.
  Their evals use 50 Euler steps.
- **`aggregator.part2(tokens)` + `depth_head` + `camera_head`:** tokens of all frames → depth, confidence and pose,
  in frame-0 coordinates. Depth has no metric scale.

## Checkpoints

Both from the README's OneDrive `ckpt/` folder, both 7,623,097,862 B. On BSC: `/gpfs/scratch/ehpc1001/quan/vggt_world/`.

| File | md5 | Stage | `prev_epoch` | Train steps | Train time | EMA |
|---|---|---|---|---|---|---|
| `kitti.pt` | `fa5b697452bd866ae85280728bf8d6f5` | 1 | 15 | 25,632 | 13.4 h | none (`ema_models` is `None`) |
| `cityscapes.pt` | `1b6428e61d3140ccac8562fa8b0e728b` | probably 2 | 6 | 14,230 | 13.8 h | none |

- **The checkpoints store no config.** The stage is inferred, not read.
- **`kitti.pt` is stage 1.** The repo has only a stage-1 config for KITTI (`training_fm/config/default_kitti.yaml`).
- **`cityscapes.pt` is probably stage 2.** It is epoch 6, and `eval/cityscapes_val_mid.py` defaults to
  `exp000_cityscapes_finetune_1/ckpts/checkpoint_6.pt`. Only Cityscapes has a stage-2 config (`default_cityscapes_stage2.yaml`).
- **Same VGGT, unrelated flow models.** 1,797 / 1,797 non-FM tensors are byte-identical (VGGT frozen). The FM
  (432.6M params) has global cosine 0.017 between the two files, relative diff 1.40 (√2 ≈ 1.41 for unrelated vectors).
  So `cityscapes.pt` is **not** fine-tuned from `kitti.pt`, nor the reverse. README: stage 2 resumes from a Cityscapes `stage1.pt`.
  The `_tok` rows are therefore identical across checkpoints.
- **We load `ckpt["model"]`** (online weights), as their eval scripts do. There are no EMA weights to compare.
- **Load check:** `missing=0`, 456 unexpected keys, all `point_head` / `track_head` (disabled in our build).

## Stage 1 vs stage 2

Both stages train the same FM model. They differ in what it is conditioned on (paper §3.3).

- **Stage 1, "teacher forcing":** the 2 condition frames are always real. Predict the next 2. Training chunks of 4 frames.
- **Stage 2, "trajectory-consistent flow forcing":** a fine-tune that shows the model its own errors. Chunks of 5 frames.
  - Sample frames 3–4 from real frames 1–2, without gradient (`_forward_stage_2`, `vggt.py:62`; 25 Euler steps).
  - Condition = real frame 2 plus a blend of real and predicted frame 3. Paper eq. 11: `c_mix = (1−λ)·Z + λ·Ẑ`.
    λ goes from 0 (all real) to 1 (all predicted) over training.
  - Train to predict frames 4–5 from that condition.
- **Why it matters:** stage 2 trains exactly the (real, predicted) input that stride 1's second call uses.
  A stage-1 model has never seen a predicted frame as input.

## Rollout stride

Both modes call the same FM model (2 in → 2 out). They differ in how far the input window moves per call.
Our windows give frames 0–1 and forecast frames 2–9 (`_forecast_tokens` in `eval_vggt_world.py`).

```
stride 1 (default)                 stride 2 (ours, comparison only)
call 1: [0 1] → 2 3                call 1: [0 1] → 2 3
call 2:   [1 2] → (3) 4            call 2:       [2 3] → 4 5
call 3:     [2 3] → (4) 5          call 3:             [4 5] → 6 7
  ...                              call 4:                   [6 7] → 8 9
call 7:           [6 7] → (8) 9
(bracketed = predicted again, discarded)
7 calls                            4 calls
```

- **Stride 1, call 2 input:** real frame 1 plus predicted frame 2, the stage-2 training input.
- **Stride 2, call 2 input:** two predicted frames, never seen in training.
- **From call 3 on,** both modes condition on predictions only. Neither matches training past call 2.
- **Cost:** stride 1 runs at 1.2–1.3x the wall time of stride 2 (KITTI 10.7 vs 8.8 s/window, 1 H100).

## Their eval protocol

From `eval/*_val_{short,mid}.py` and paper appendix B.2. Images are centre-cropped to 224×448.

| | KITTI | Cityscapes |
|---|---|---|
| Frame spacing | every 2nd frame at 10 FPS = 200 ms (`step = 2`) | every 3rd frame at 16 FPS = 187.5 ms (`step = 3`) |
| Short-term | input [9, 11] → scored at 13 (1 step, 200 ms) | input [13, 16] → scored at 19 (1 step, 187.5 ms) |
| Mid-term | input [5, 7] → scored at 13 (3 steps, 600 ms) | input [7, 10] → scored at 19 (3 steps, 562.5 ms) |
| Mid-term calls | (5, 7) → 9, 11; then (real 7, pred 9) → 11, **13** | (7, 10) → 13, 16; then (real 10, pred 13) → 16, **19** |

The mid-term eval is stride 1 cut short after 2 calls.

**They score one frame per window, depth only** (`kitti_val_mid.py:219-290`, `cityscapes_val_mid.py:256-285`).

- **Frame:** only the last one (KITTI 13, Cityscapes 19). The frames in between are generated, not scored.
- **Metrics:** depth AbsRel and δ1 (< 1.25). No pose, pointmap, raymap or Chamfer.
- **Scale:** per-frame median(GT) / median(pred) before scoring. Metric scale is never tested.
- **KITTI GT:** projected LiDAR (`val_depth/.../proj_depth/groundtruth`).
- **Cityscapes GT:** VGGT's own depth on the real frame. It measures agreement with VGGT, not accuracy.
- **Decode window:** part2 sees only the last 4–5 frames (KITTI mid: real 7, predicted 9, 11, 13).

## How our benchmark differs

- **Frame spacing:** 0.5 s on all three sets, 2.5x theirs. KITTI and Waymo are subsampled every 5th frame. nuScenes keyframes are 2 Hz.
- **Horizon:** 8 forecast steps (4 s). They never score past 3 steps (0.6 s).
- **Scored frames and metrics:** we average all 8 forecast frames, and add pointmap, raymap and Chamfer, which test pose.
  Our `_oracle` row is one median scale per window, not per frame. Paper numbers are not comparable with ours.
- **Resolution:** `--resolution native` feeds VGGT the loader size (518×168 KITTI, 518×266 nuScenes / Waymo).
  `448` resizes to their 224×448.
- **Domain:** `kitti.pt` is in-domain only on KITTI. `cityscapes.pt` is out of domain on all three sets.
- **Metric scale:** VGGT depth is up to scale. We fit one scale per window to DA3METRIC-LARGE on the context frames
  (`_da3_scale`, recipe of `extract_recon.py:404-434`). The `_oracle` row uses median(GT) / median(pred) on the forecast frames instead.
- **Input normalisation:** our loader gives DA3/ImageNet-normalised images (`base_seq_dataset.py:285`).
  VGGT gets `imgs * std + mean`; DA3 gets `imgs` as is.

## Runs

All 1 GPU, native resolution, 50 FM steps, full sets (KITTI 2,013, nuScenes 2,382, Waymo test 2,018 windows).
Output: `results/vggt_world_alldata_ctx2fwd8/<ckpt>_<rollout>_native/` on BSC; `<ckpt>` is the checkpoint, not the dataset.

| Checkpoint | Rollout | Jobs (KITTI / nuScenes / Waymo) |
|---|---|---|
| `kitti.pt` | stride 2 | `BSC:46921883` / `46921884` / `46921885` |
| `kitti.pt` | stride 1 | `BSC:46936417` / `46936418` / `46936419` |
| `cityscapes.pt` | stride 1 | `BSC:46962868` / `46962869` / `46962870` |
