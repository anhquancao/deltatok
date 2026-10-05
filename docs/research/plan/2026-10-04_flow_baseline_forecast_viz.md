# Plan: forecast-frame and BEV panels for the baseline decks

**Date:** 2026-10-04 · **Thread:** flow · **Cluster:** BSC
· **Deck:** `results/2026-10-04_flow_vggt_world_baseline_slides.html` · prior cycle: `plan/2026-10-01_flow_vggt_world_baseline_eval.md`

Reuse the flow trainer's eval panel, `_log_viz_sample` + `_build_bev_panel` in `occrae/visualization_helper.py`, for every model.

- **Panel:** one row per frame; context rows get a cyan border.
- **Columns:** depth at 0–50 m with `depth2rgb`, RGB where the model has it, and BEV camera paths.
- **Windows:** `--num_items 2` picks the first 2 windows of each set's eval order.
  `ResizedDataset_MUSt3R.set_epoch` permutes with seed 777, so every script gets the same 2 windows, the ones the scored runs started with.
- **No shared axes:** each BEV panel autoscales, and each model shows its own scored depth (DA3 scale for the baselines).

## 1. DeltaTok-flow: no code change

`slurm/viz_forecast_deltatok_flow_alldata_ctx2fwd8_bsc.slurm` (new):
- It is a copy of `slurm/eval_deltatok_flow_numsteps_alldata_ctx2fwd8_bsc.slurm` with `NUM_STEPS=5`, `NUM_ITEMS=2`,
  `training.eval_num_visualizations=2`, `--viz_rgb` instead of `--fvd`, and 30 min.
- Panels: `<OUTPUT_DIR>/<run>/eval_viz/ode_steps5_sigmaNone/eval_depth/<set>/*.jpg`.
- Columns: `Pred Depth (Flow) | Pred Depth (GT-z) | Pred RGB (Flow) | Pred RGB (GT-z) | GT RGB | BEV (Flow) | BEV (GT-z) | BEV (GT)`.

## 2. VGGT-World and Gen3R: `eval_forecast_metrics.py --viz N`

- The scorer draws the panel for the first N windows from the dumped depth and c2w, into `<pred_dir>/eval_viz/`.
- Columns: `RGB | Pred Depth | BEV (pred) | BEV (GT)`.
- Run: the scorer slurm on a full dump with `NUM_ITEMS=2 EXTRA_ARGS="--viz 2"`, `--time=00:30:00`. One job per `SET`.
  Dumps: `plan/2026-10-05_flow_forecast_dump_then_score.md`.

## 3. Deck

- Insert slides after slide 6 (before Caveats). Per set, show the DeltaTok and VGGT-World panels of the same window side by side.
- Crop to the forecast rows for frames 3, 6 and 9.
- Check that the scene names in the panel filenames match across models.
- Verify with headless Chrome: `scrollHeight − clientHeight == 0` on every slide, and screenshot each one.

## 4. Jobs

1. **BSC reachable**, which it was not on 2026-10-04. Then sync with `monitor-sync` and md5 the edited files.
2. **Panel jobs:** 1 DeltaTok, plus 3 scorer jobs per baseline once its dump exists, 30 min each.
