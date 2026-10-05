# Plan: baselines dump depth + c2w; one script scores and draws them

**Date:** 2026-10-05 · **Thread:** flow · **Cluster:** BSC
· **Reference:** VGGT-World `cityscapes.pt` stride1 `BSC:46962868`–`70` (old inline scoring)
· jobs: _pending_ · prior cycle: `plan/2026-10-04_flow_gen3r_baseline_eval.md`

## 0. Layout

- **Generators** (`eval_vggt_world.py`, `eval_gen3r.py`) only forecast. They write `<output_dir>/<run>/<set>/<n:05d>.npz`, where `n` is the window index in loader order.
  - Keys: superseded by `plan/2026-10-05_flow_deltatok_dump_for_scorer.md` §0. They are `depth` (N,) and `point` (N, 3) on GT-mask pixels, `c2w` (T, 4, 4), `scene_name`, `frame_stems`, and `depth_dense` / `point_dense` for the first 16 windows. Format is masked fp32 + zlib.
  - All T frames are saved with the DA3 scale applied.
  - About 31 GB per model for the three sets.
- **Scorer** `eval_forecast_metrics.py --pred_dir <run>` rebuilds the same loader and loads window `n` for loader item `n`.
  - It asserts that the names match.
  - It runs the current scoring code unchanged: the main row and `_oracle`.
  - `--viz N` draws the trainer panel for the first N windows.
  - It writes `<pred_dir>/<set>.json` and `<pred_dir>/eval_viz/`.

## 1. `eval_vggt_world.py` (about −110 / +12)

- **L2–13 docstring:** "Dump pretrained VGGT-World forecasts (depth, K, c2w at DA3 scale) on the DeltaTok flow-eval windows; score with `eval_forecast_metrics.py`." Keep the usage block.
- **Delete:**
  - L41, the chamfer path.
  - L69–71 and L73, the loss / helpers / chamfer / viz imports.
  - L93 `--no_chamfer` and L96 `--viz`.
  - L145–152 `_Scorer`.
  - L251–284 `_oracle_scale`, `_scaled_outputs`, `_chamfer_rows`.
  - L318–319 `sums` / `ratios`.
  - L338–341 `tok_gt` and `dec_gt`: the true-frame decode.
  - L358–369, the viz block.
  - L370–389, the loss loops.
  - L406–419, the results JSON.
- **L304:** `batcher = DeltaTokSharedMixin()`. Only `_normalize_batch` is used, and the mixin has no `__init__`.
- **L347–357** become:
  ```python
  s_fc, fb_fc = _da3_scale(m_depth, m_sky, dec_fc[0], dec_fc[1], dec_fc[2])
  depth = dec_fc[0] * s_fc[:, None, None, None]                                      # (B, T, H, W) metres
  c2w = dec_fc[3].clone()
  c2w[..., :3, 3] *= s_fc[:, None, None]                                             # (B, T, 4, 4)
  for b in range(B):
      torch.save({"depth": depth[b].cpu(), "K": dec_fc[2][b].cpu(), "c2w": c2w[b].cpu(),
                  "scene_name": batch["scene_name"][b], "frame_stems": list(batch["frame_stems"][b])},
                 os.path.join(set_dir, f"{n_items + b:05d}.pt"))
  n_items += B
  n_fallback += int(fb_fc.sum())
  ```
  At the start of each set: `set_dir = os.path.join(output_dir, _sanitize(test_name))` and `os.makedirs`.
- **DBG:** keep the x01 line and print `s_da3 / fallback`. At the end of each set, print `wrote {n_items} windows, {n_fallback} DA3 fallbacks`.

## 2. `eval_forecast_metrics.py` (new, ~220 lines)

`cp eval_vggt_world.py eval_forecast_metrics.py` on the current file, then strip it.

- **Drop:** VGGT and DA3 imports; the tensorboardX stub; `_load_vggt_world`, `_forecast_tokens`, `_decode`, `_da3_scale`.
- **Keep verbatim:** `_sanitize`, `_build_test_loaders`, `_Scorer`, `_oracle_scale`, `_scaled_outputs`, `_chamfer_rows`.
- **Args:**
  - `--config-dir`, `--config-name`, `--num_items`, `--test_filter`, `--no_chamfer`, `--verbose_batches`, `--viz`: as now.
  - `--bsize 4`.
  - `--pred_dir` (required).
- **Per batch:**
  ```python
  preds = [torch.load(os.path.join(set_dir, f"{n_items + b:05d}.pt")) for b in range(B)]
  for b, p in enumerate(preds):
      assert (p["scene_name"], tuple(p["frame_stems"])) == (batch["scene_name"][b], tuple(batch["frame_stems"][b])), (n_items + b, p["scene_name"])
  depth, K, c2w = (torch.stack([p[k] for p in preds]).to(device) for k in ("depth", "K", "c2w"))  # (B, T, H, W), (B, T, 3, 3), (B, T, 4, 4)
  s_or = _oracle_scale(depth, gt_depth, gt_mask, fslice)                                     # (B,) on top of the saved scale
  rows = {"": _scaled_outputs(depth, K, c2w, torch.ones_like(s_or)), "_oracle": _scaled_outputs(depth, K, c2w, s_or)}
  ```
  - Then the current loss loop, with Chamfer on `""` only. JSON keys are unchanged.
  - `ScaleRatio_p*` is computed from `1 / s_or` over all windows.
- **Viz:** the current block (L358–369), with `extra_panels` set to two BEVs (`c2w[b]`, `batch["gt_c2w"][b]`) and `col_titles=["RGB", "Pred Depth", "BEV (pred)", "BEV (GT)"]`.
- **DBG, first `--verbose_batches` batches:** per frame, the depth L1 and the step length vs GT for window 0. This is moved from `eval_gen3r.py:364-373`.

## 3. `eval_gen3r.py`: the same cut

- Apply the same deletions and save block as §1, with the viz block, `wan_vae` RGB decode and step-length DBG removed. That is about −100 lines.
- Output: `<output_dir>/<ctx_mode>_step<slot_step>_native/<set>/`.

## 4. Slurm

- **`slurm/eval_vggt_world_alldata_ctx2fwd8_bsc.slurm`:**
  - Header: "Dump VGGT-World forecasts for one ctx2fwd8 eval set, 1 GPU."
  - `OUTPUT_DIR` default `/gpfs/scratch/ehpc1001/quan/forecast_preds/vggt_world_alldata_ctx2fwd8`; time `16:00:00`.
- **`slurm/eval_gen3r_alldata_ctx2fwd8_bsc.slurm`:** `OUTPUT_DIR` default `/gpfs/scratch/ehpc1001/quan/forecast_preds/gen3r_alldata_ctx2fwd8`.
- **`slurm/eval_forecast_metrics_alldata_ctx2fwd8_bsc.slurm`** (new, a copy of the VGGT-World script):
  - Rename the job name, output and error.
  - Require `PRED_DIR`; drop `CKPT` / `ROLLOUT` / `RESOLUTION`; time `03:00:00`.
  - `srun python eval_forecast_metrics.py --config-name ... --num_items "$NUM_ITEMS" --pred_dir "$PRED_DIR" ${EXTRA_ARGS:-}`.

## 5. Pre-flight

1. BSC reachable, then sync with `monitor-sync`. Md5 the three `.py` files and three slurm scripts.
2. The reference exists: `results/vggt_world_alldata_ctx2fwd8/cityscapes_stride1_native/*.json` and the DBG lines in `slurm/output/eval_vggt_world_alldata_ctx2fwd8_bsc_46962868.out`.
3. At least 70 GB free on `/gpfs/scratch/ehpc1001`.

## 6. Runs (VGGT-World `cityscapes.pt`)

1. **Smoke:**
   - KITTI generator: `NUM_ITEMS=4`, `acc_debug`, 30 min.
   - One scorer job with `--dependency=afterok`.
   - Pass: the name assert holds, and `s_da3` plus the frame 2–9 depth L1 match `46962868`'s first-batch DBG to 3 decimals.
2. **Full:**
   - 3 generators, 16 h, `acc_ehpc`.
   - 3 scorers, 3 h, `afterok`.
   - Pass: every non-`_tok` key within 2% of the old JSON. The match is statistical, not exact.
     The old second `_da3_scale` call (on the `tok` decode) drew a CUDA `randperm`, so the flow noise differs from batch 2 onward.
3. **Panels:** a scorer job per set with `NUM_ITEMS=2 EXTRA_ARGS="--viz 2"` on the full dump. This replaces the `eval_vggt_world --viz` jobs in `plan/2026-10-04_flow_baseline_forecast_viz.md`.
4. **Gen3R:** follows `plan/2026-10-04_flow_gen3r_baseline_eval.md` §5–6, with one scorer job per generator.
