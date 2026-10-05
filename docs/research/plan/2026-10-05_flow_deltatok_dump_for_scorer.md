# Plan: DeltaTok-flow dumps depth + point + c2w; scorer drops LossRaymap and adds pose errors

**Date:** 2026-10-05 · **Thread:** flow · **Cluster:** BSC
· **Reference:** DeltaTok-flow evals `BSC:46861907`–`18` (flow `iter_050000`, tokenizer `epoch_70`, 1 GPU)
· jobs: _pending_ · prior cycle: `plan/2026-10-05_flow_forecast_dump_then_score.md`

## 0. Layout

- **Every file is masked fp32 + zlib** (`np.savez_compressed`): `depth` and `point` only on the window's GT-mask (LiDAR) pixels.
  Every metric reads only those pixels, so the metrics stay exact. Each file keeps its mask as bits (`np.packbits`), so `read_window` decodes it alone; the scorer checks it against the loader's mask.
- **DeltaTok file:** `<dump_dir>/<mode>_steps<n>_sigma<s>/<set>/<n:05d>.npz`, holding:
  - `depth` (N,) float32: the decoded DA3 depth on the N GT pixels, metric with no fit.
  - `point` (N, 3) float32: `decoded["pointmap"]` = depth · ray dir + ray origin (`model_da3.py:472`).
  - `c2w` (T, 4, 4), padded from (T, 3, 4): a pinhole fit to the decoded rays, from `DA3Wrapper._process_ray_pose_estimation` (`model_da3.py:378`). This is the fit `pose_from_depth_ray` and the BEV panels use. No second decode.
  - `scene_name`, `frame_stems`.
- **Baseline files** (VGGT-World, Gen3R): the same keys. The generator unprojects `point` with its own K and c2w.
- **Dense for panels:** in every model's dump, the first 16 windows per set also keep `depth_dense` (T, H, W) and `point_dense` (T, H, W, 3).
  - Set by `N_DENSE` in `occrae/forecast_dump.py`, which writes and reads every dump.
  - Cost is about 1 GB per model run before zlib.
  - The scorer's `--viz N` draws them (N ≤ 16).
- **Window index:** `n = items_seen + b` in the trainer's eval loop, at 1 GPU (`eval_deltatok_flow_sampler.py:267`).
  The per-set configs that the scorer reads hold the same `test_dataset` strings as `eval_deltatok_flow_alldata_ctx2fwd8_bsc`.
- **Storage:** 16 B per GT pixel. LiDAR density is not measured yet; at 10% that is about 12.5 GB per model run, before zlib.

## 1. `occrae/deltatok_flow_trainer.py` (+11)

- **L88**, after `_per_window`:
  `self._dump_dirs = None  # {test_name: dir} for eval_forecast_metrics.py; set by the sampler's --dump_dir`
- **After L970** (`decoded_tok`), inside `if "gt_mask" in batch:`:
  ```python
  if self._dump_dirs is not None:  # eval_forecast_metrics.py input; c2w is a pinhole fit to the decoded rays
      with torch.autocast("cuda", enabled=False):
          c2w, _ = self.occ_rae.model._process_ray_pose_estimation(
              decoded["ray"].float(), decoded["ray_conf"].float(), height, width)          # (B, V, 3, 4) frame-0 coords
      for b in range(B):
          torch.save({"depth": decoded["depth"][b].float().cpu(), "point": decoded["pointmap"][b].float().cpu(),  # (V, H, W), (V, H, W, 3)
                      "c2w": c2w[b].float().cpu(), "scene_name": batch["scene_name"][b],
                      "frame_stems": list(batch["frame_stems"][b])},
                     os.path.join(self._dump_dirs[test_name], f"{items_seen + b:05d}.pt"))
  ```

## 2. `eval_deltatok_flow_sampler.py` (+9)

- **Arg:** `--dump_dir` (default `None`), with help "Write eval_forecast_metrics.py inputs: <dump_dir>/<mode>_steps<n>_sigma<s>/<set>/<n>.npz."
- **Pass loop:** next to `eval_viz_dir` (L308):
  ```python
  if args.dump_dir:  # one dir per pass, as eval_viz_dir
      trainer._dump_dirs = {name: os.path.join(args.dump_dir, f"{mode}_steps{n_steps}_sigma{sigma}", _sanitize(name))
                            for name in trainer.test_loaders}
      for d in trainer._dump_dirs.values():
          os.makedirs(d, exist_ok=True)
  ```

## 3. `eval_forecast_metrics.py`: point branch, no raymap, pose errors

- **`_Scorer`:** drop `raymap_criterion`.
- **`_scaled_outputs(depth, point, c2w, scale)`:** scales depth, points and the c2w translation about the frame-0 origin.
  Scaling commutes with unprojection, so the baseline numbers are unchanged.
- **`_frame_losses(scorer, dec, batch, fslice)`** replaces the mixin's `_compute_frame_losses`. It holds that method's pointmap and depth calls verbatim (`deltatok_shared.py:416-428`) and no raymap.
- **New `_pose_errors(c2w, gt_c2w, fslice)`:** ATE / RTE / RRE with no fit or alignment, in frame-0 coordinates at the saved scale.
  The steps are the 8 moves into each forecast frame, `(f-1 → f)` for f = 2…9, so the step from the last context frame into frame 2 counts. Each metric is an RMSE per window.
  ```python
  P, P_gt = c2w.double(), gt_c2w.double()                                             # (B, T, 3|4, 4), (B, T, 4, 4)
  if P.shape[-2] == 3:  # DeltaTok saves (3, 4)
      P = torch.cat([P, P_gt[..., 3:, :]], dim=-2)                                    # (B, T, 4, 4) last row [0, 0, 0, 1]
  ate = (P[:, fslice, :3, 3] - P_gt[:, fslice, :3, 3]).norm(dim=-1).pow(2).mean(1).sqrt()  # (B,) camera positions c2w[:3, 3], m
  cur = torch.arange(fslice.start, P.shape[1], device=P.device)                      # (F,) forecast frames
  step = torch.linalg.inv(P[:, cur - 1]) @ P[:, cur]                                 # (B, F, 4, 4) predicted motion f-1 -> f
  step_gt = torch.linalg.inv(P_gt[:, cur - 1]) @ P_gt[:, cur]                        # (B, F, 4, 4)
  E = torch.linalg.inv(step_gt) @ step                                               # (B, F, 4, 4) per-step error
  rte = E[..., :3, 3].norm(dim=-1).pow(2).mean(1).sqrt()                             # (B,) m
  cos = ((E[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)   # (B, F)
  rre = torch.rad2deg(torch.arccos(cos)).pow(2).mean(1).sqrt()                       # (B,) deg
  ```
- **New `_depth_metrics(depth, gt_depth, gt_mask, fslice)`:** per window, over the valid GT pixels of the forecast frames.
  ```python
  d, g = depth[:, fslice], gt_depth[:, fslice]                                        # (B, F, H, W)
  m = gt_mask[:, fslice].float()                                                      # (B, F, H, W)
  n = m.sum((1, 2, 3)).clamp_min(1)                                                   # (B,)
  abs_rel = ((d - g).abs() / g.clamp_min(1e-6) * m).sum((1, 2, 3)) / n               # (B,)
  ratio = torch.maximum(d / g.clamp_min(1e-6), g / d.clamp_min(1e-6))                 # (B, F, H, W)
  delta1 = ((ratio < 1.25).float() * m).sum((1, 2, 3)) / n                            # (B,)
  ```
  The keys are `AbsRel` and `Delta1`, with `_oracle` twins.
- **No alignment anywhere:** predictions are scored at the metric scale they were saved with.
  - **`occrae/chamfer_metrics.py`:** `compute_chamfer_metrics(..., align: bool = True)`. L160 becomes `if align:` `P = umeyama_alignment(P, G, mask)[-1]`.
    The default keeps the trainer's `eval_chamfer` unchanged.
  - **Scorer `_chamfer_rows`:** passes `align=False`.
  - **Chamfer for `_oracle`:** now differs from the main row, so it is computed there too.
  - **Unchanged:** AbsRel, δ1 and the pose errors are raw, with no median or Umeyama fit.
  - **`_oracle` (GT-median scale)** stays as the one labelled diagnostic row.
- **Loading:** zero-filled dense `depth` / `point`, scattered back on the batch's GT mask. An assert checks the pixel count.
  Panels use `depth_dense` and skip windows without it (any beyond the first 16).
- **JSON keys:**
  - Drop `LossRaymap*`.
  - Add `ATE`, `RTE` and `RRE` for the main row. Add `ATE_oracle` and `RTE_oracle` for the oracle row; RRE does not depend on scale.
  - These are not comparable with VGGT-World's Table 5, which uses Umeyama alignment.

## 4. Unaligned Chamfer everywhere

- **`occrae/chamfer_metrics.py`:** the `align` default flips to `False`, so the trainer's `training.eval_chamfer` is unaligned too.
- **`test_chamfer_parity.py:173`:** passes `align=True`, because it checks against Gen3R's aligned reference.

## 5. Rerun matrix: every Chamfer number in the deck, through dump + score

All runs use 1 GPU, `NUM_ITEMS=1000000` (the full stride sets), and `afterok` scorers with `slurm/eval_forecast_metrics_alldata_ctx2fwd8_bsc.slurm`.

| Model | Dump jobs | Scorers | Storage |
|---|---|---|---|
| DeltaTok-flow `iter_050000`, steps 1 / 5 / 10 / 20 | 6: set × {`NUM_STEPS=1,5`, `10,20`}, 12 h | 12 | 4 runs |
| VGGT-World `cityscapes.pt` stride1 | 3, 16 h | 3 | 1 run |
| VGGT-World `kitti.pt` stride1 (appendix) | 3, 16 h | 3 | 1 run |
| Gen3R (`plan/2026-10-04_flow_gen3r_baseline_eval.md`) | 3 after its smoke | 3 | 1 run |

- **DeltaTok command:** `slurm/eval_deltatok_flow_numsteps_alldata_ctx2fwd8_bsc.slurm` with
  `OUTPUT_DIR=results/deltatok_flow_dump_alldata_ctx2fwd8 EXTRA_ARGS="--test_filter Kitti --dump_dir /gpfs/scratch/ehpc1001/quan/forecast_preds/deltatok_flow_alldata_ctx2fwd8"`.
  Filters are `Kitti` / `Nuscenes` / `Waymo`. Steps are split 2 per job: the old evals took 3.6–4.0 h per step count with Chamfer.
- **VGGT-World:** stride1 only; the stride2 runs are not redone. Dump root: `/gpfs/scratch/ehpc1001/quan/forecast_preds/vggt_world_alldata_ctx2fwd8`.
- **`training.eval_chamfer`** stays `false` in the dump jobs; the scorer runs Chamfer.
- **Storage:** 7 model runs × ~12.5 GB at 10% LiDAR density, about 90 GB. Measure the density and `du -sh` on the smoke dump.
- **Deck:** rebuild the Chamfer slides of `results/2026-10-04_flow_vggt_world_baseline_slides.html` from the new JSONs. Add depth AbsRel / δ1 and pose ATE / RTE / RRE, and drop Raymap.

## 6. Checks

1. **Pre-flight:** check free space on `/gpfs/scratch/ehpc1001` against the §5 storage rule.
2. **Smoke:**
   - KITTI dump with `NUM_ITEMS=4 NUM_STEPS=5`, `acc_debug`, 30 min, then the scorer.
   - Pass: the scorer's `LossDepth` and `LossPointmap` equal the dump job's own `[Eval/...]` line to 4 decimals, and the name assert holds.
3. **Full:**
   - The dump job's trainer `LossDepth` and `LossPointmap` equal `46861907`–`18` at every step count.
   - Chamfer has no parity target: those evals aligned with Sim(3) and the scorer does not.
