# Plan: pretrained VGGT-World on the all-data ctx2fwd8 flow benchmark

**Date:** 2026-10-01 · **Thread:** flow · **Cluster:** BSC
· **Reference:** `BSC:46861552` / `53` / `55` / `31` (all-data ctx2fwd8 flow, 1/5/10/20 steps, Chamfer), `results/deltatok_flow_chamfer_alldata_ctx2fwd8`
· **Model:** VGGT-World (arXiv 2603.12655), vendored at `third_party/VGGT-World`, published checkpoints only (no retrain)
· jobs: _pending_ · deck: _pending_ · prior cycle: `results/2026-09-30_flow_eval_numsteps_alldata_ctx2fwd8_slides.html`

Zero-shot first. A retrain (FM only, VGGT frozen, our 2 Hz all-data mix) is decided after this read: if the `_tok`
row is already bad, the frozen VGGT is the limit and a retrain cannot help.

Nothing under `third_party/` is edited.

## 0. What is scored

- **Windows:** `dataset.test_dataset` of `configs/deltatok_flow/eval_deltatok_flow_alldata_ctx2fwd8_bsc.yaml`, read
  through the unchanged `_build_test_loaders`.

  | Set | W × H | Patch grid (14 px) | Stride |
  |---|---|---|---|
  | KITTI val | 518 × 168 | 37 × 12 | 2 |
  | nuScenes val (`occ3d_nuscenes_val_preprocessed`, `_fs1_` pkl) | 518 × 266 | 37 × 19 | 2 |
  | Waymo test | 518 × 266 | 37 × 19 | 16 |

- **Frames:** 10 consecutive frames of cam 0. Frames 0–1 are context, 2–9 are forecast (`slice(2, 10)`).
- **Metrics:** `_compute_frame_losses` (`occrae/deltatok_shared.py:415`, criteria as `occrae/deltatok_flow_trainer.py:181-183`,
  `gt_scale=True`) and `compute_chamfer_metrics` (`occrae/chamfer_metrics.py`, own Umeyama Sim(3)).
- **FD-OccAny:** N/A. VGGT-World produces neither RGB nor OccAny tokens.
- **1 GPU**, the same as the reference evals.

## 1. Checkpoint

- Local: `ssh -N -R 15432 bsc` (background).
- BSC login node, `http(s)_proxy=socks5h://localhost:15432`: resolve the README's OneDrive share via
  `api.onedrive.com/v1.0/shares/u!<b64(url)>/root/children`, `curl` `kitti_checkpoint.pt` and the Cityscapes
  stage-2 `.pt` into `/gpfs/scratch/ehpc1001/quan/vggt_world/`, record md5 + size.
- Fallback: the user downloads in a browser; the `.pt` is copied to the same folder.
- `DA3METRIC-LARGE` is already in `/gpfs/scratch/ehpc1001/hf_cache/hub/`.

## 2. `eval_vggt_world.py` (new)

`cp eval_deltatok_flow_sampler.py eval_vggt_world.py`, then:

- **Keep:** the arg-parser skeleton, the Hydra `compose` (used only for `dataset.test_dataset` and
  `training.val_bsize`), `_build_test_loaders`, `_sanitize`, the per-set JSON dump.
- **Drop:** the DeltaTok trainer build, the step / mode / noise sweeps, `_bank_z_basis`, `--fvd` and `_frechet_distance`.
- **Args:** `--ckpt`, `--fm_steps 50`, `--rollout {stride2,stride1}` (default `stride2`),
  `--resolution {native,448}` (default `native`), `--da3_metric_model depth-anything/DA3METRIC-LARGE`, `--seed`,
  `--output_dir results/vggt_world_alldata_ctx2fwd8`.
- **Model:**
  - Prepend `third_party/VGGT-World` to `sys.path`.
  - `VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)`.
  - `load_state_dict(ckpt["model"], strict=False)`; print the missing / unexpected counts.
  - `.eval()`, CUDA, bf16 autocast. Same for `DepthAnything3.from_pretrained(--da3_metric_model)`, `requires_grad_(False)`.
- **Criteria shim:** an object holding `device` and the three criteria, so `_compute_frame_losses` is called unchanged.

**Per batch:**

1. List → dict as `_normalize_batch` (`occrae/deltatok_shared.py:105`). `imgs` (B, 10, 3, H, W) is dust3r `ImgNorm`,
   in [-1, 1]; `x01 = (imgs + 1) / 2` for VGGT.
2. `--resolution 448`: resize `x01` to 448 × 224 for VGGT only; upsample depth back to (H, W) bilinearly and rescale K.
3. **Context:** `cond = model.aggregator.part1(x01[:, 0:2])`.
4. **Rollout:**
   - `stride2`: `model.fm.sample_euler(cond, shape_like=(B, 2, N, C), steps=fm_steps, patch_hw=(h // 14, w // 14))`
     → frames 2–3; the condition becomes the 2 new frames → 4–5, 6–7, 8–9. 4 FM calls.
   - `stride1`: as `third_party/VGGT-World/eval/kitti_val_mid.py`. The condition is (newest frame, latest
     prediction); each call keeps 1 new frame. 7 FM calls.
5. **Decode once, all 10 frames in frame-0 coordinates:** `agg = part2([cat(GT0, GT1, pred 2..9)])`.
   - `depth, conf = model.depth_head(agg, images=x01, patch_start_idx=...)`.
   - `pose_enc = model.camera_head(agg)[-1]` → `pose_encoding_to_extri_intri(pose_enc, (h, w))` → `w2c`, `K`;
     `c2w = closed_form_inverse_se3(w2c)`.
6. **Metric scale from DA3METRIC-LARGE**, context frames 0–1, predicted K (mirrors `extract_recon.py:404-434`):
   - `m = da3_metric(imgs_da3[:, 0:2])`; `m.depth = apply_metric_scaling(m.depth, K[:, 0:2])`.
   - `non_sky = compute_sky_mask(m.sky, 0.3)`;
     `align = compute_alignment_mask(conf[:, :2], non_sky, depth[:, :2], m.depth, median_conf)`.
   - `s = least_squares_scale_scalar(m.depth[align], depth[:, :2][align])`, one scalar per window.
   - Empty mask or non-finite `s` → `s = 1`, counted.
7. **Scale and build:** `depth *= s`; `c2w[..., :3, 3] *= s`;
   pointmap = unproject(depth, K, c2w) (B, 10, H, W, 3);
   `ray = intrinsics_c2w_to_raymap(K, c2w, H, W)` (`occany/utils/helpers.py:67`) (B, 10, H, W, 6).
8. **Score** `{depth, pointmap, ray}` with `_compute_frame_losses(..., slice(2, 10), ray_conf=None)` and Chamfer.

**Rows per set:**

- main: DA3-metric scale;
- `_oracle`: per-window median scale vs GT depth on the forecast frames;
- `_tok`: `part2` on GT `part1` tokens of all 10 frames, DA3 scale;
- diagnostics: `s_da3 / s_oracle` median, p10, p90; the `s = 1` fallback count.

**Output:** `<output_dir>/<ckpt_stem>_<rollout>_<resolution>/metrics.json` and `[Eval/<set>/...]` lines on stdout.

## 3. `slurm/eval_vggt_world_alldata_ctx2fwd8_bsc.slurm` (new)

`cp slurm/eval_deltatok_flow_numsteps_alldata_ctx2fwd8_bsc.slurm`, then:

- `#SBATCH`: `--job-name=eval_vggt_world_alldata_ctx2fwd8`, matching `--output` / `--error`, `--account=ehpc1001`,
  `--qos=acc_ehpc`, 1 GPU, `--time=06:00:00`.
- Vars: `CKPT=/gpfs/scratch/ehpc1001/quan/vggt_world/kitti_checkpoint.pt`, `ROLLOUT=stride2`, `RESOLUTION=native`,
  `NUM_ITEMS=1000000`, `OUTPUT_DIR=results/vggt_world_alldata_ctx2fwd8`, `HF_HUB_OFFLINE=1`,
  `HF_HOME=/gpfs/scratch/ehpc1001/hf_cache`.
- Body: one `python eval_vggt_world.py ...`; drop `CFG_ARGS`.
- If `results/` is not already covered: add the output dir to `.gitignore` and to the `Deltatok` `excludes` in
  `../monitor_jobs/data/projects.json`.

## 4. Pre-flight

1. BSC login node, CPU, `source env_bsc.sh`: import `vggt.models.vggt`, `vggt.models.flux_modules`,
   `vggt.utils.pose_enc`, `depth_anything_3.api`. Missing packages are reported, not installed.
2. Grep the BSC copies of both new files; the user syncs.
3. Smoke on `acc_debug`, `--num_items 8`, all 3 sets, `native` and `448`. Print:
   - missing / unexpected keys (only track-head keys expected missing);
   - `imgs` range before and after the [0, 1] map;
   - `s_da3 / s_oracle` (≈ 1 on KITTI; if not, retry DA3 on ImageNet-normalised input);
   - per-frame depth loss, `_tok` < forecast;
   - seconds per window.
4. Prod: `ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/eval_vggt_world_alldata_ctx2fwd8_bsc.slurm'"`.
   Watch with a background `until` loop to the first `[Eval/` line. Then the Cityscapes stage-2 checkpoint with
   `ROLLOUT=stride1`.
