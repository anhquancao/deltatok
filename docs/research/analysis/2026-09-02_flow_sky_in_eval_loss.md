# Sky in the flow eval loss

Created 2026-09-02 · thread `flow` · stage `analysis` · answers roadmap TODO 9 · code reading only, nothing measured

## Answer

**Sky is dropped from the geometry losses and kept at full weight in the flow loss.** It is never detected as sky.
The geometry losses are masked by "the LiDAR returned a point here", and the sky has no return.

## The mask

`_compute_frame_losses` (`occrae/deltatok_shared.py:415-437`) is the one place the eval geometry losses are formed,
and both trainers call it — `deltatok_flow_trainer.py:827-831` and `deltatok_trainer.py:1421`. It passes
`batch["gt_mask"]` to the pointmap and depth criteria, and **nothing to the raymap criterion**, which is depth-free.

`gt_mask` is the dataset's `valid_mask`, or `gt_depth > 0` if the loader does not ship one
(`occrae/deltatok_shared.py:143-145`). Every eval loader ships one. It is built at
`occany/datasets/base_seq_dataset.py:281-284`:

```
valid_mask = (depthmap > 0) & (depthmap < z_far) & isfinite(pts3d)
```

from `dust3r.utils.geometry.depthmap_to_absolute_camera_coordinates` (`third_party/dust3r/.../geometry.py:196-212`),
with `z_far` set per view at `:279`.

GT depth is a **sparse LiDAR reprojection**, not a dense map: `_resize_image_and_sparse_depthmap`
(`occany/datasets/base_seq_dataset.py:96-117`) reprojects the point cloud into the resized camera and leaves every
un-hit pixel at 0. Sky pixels get no return, so `depthmap = 0`, so `valid_mask = 0`.

## What that means

- **Sky is excluded implicitly, as a side effect.** There is no sky head, no segmentation, no
  `compute_sky_mask`. The repo *has* one — `occany/da3_inference.py:388`, `extract_recon.py:406` — but it is used
  for DA3 pseudo-supervision and recon extraction, never on this path.
- **Everything else with no return is dropped identically.** Above the LiDAR's vertical FOV, glass and specular
  surfaces, dynamic-object dropouts, and every pixel beyond `z_far`. Sky is not separable from these in the number.
- **`z_far=50`, not 80, on every eval set.** All 12 `configs/deltatok_flow/*.yaml` and 7 `configs/deltatok/*.yaml`
  test-dataset strings pass `z_far=50` explicitly, so far-field geometry is not evaluated. The 80 m is a red
  herring from two places, neither on this path: a dead comment in `third_party/dust3r/.../geometry.py:196-198`
  (`# Invalid any depth > 80m` followed by the no-op `valid_mask = valid_mask`), and `z_far=80` on dust3r's own
  spring / pointodyssey / dynamic_replica loaders. The class default is `z_far=0`, which disables the clip
  entirely (`base_stereo_view_dataset.py:34`), so the 50 comes only from the config.
- **The reduction is a batch-level masked mean**: `sum(mask * err) / sum(mask)`, over the whole `(B, views, H, W)`
  tensor at once (`occany/loss/losses_da3.py:325-338` pointmap, `:407-441` depth). Views with more returns dominate
  the reported scalar. `gt_scale=True` and `alpha=0` (`deltatok_flow_trainer.py:147-149`), so depth is a plain
  masked L1 at metric scale and the gradient term is off.
- **A model pays nothing for what it puts in the sky** on `LossDepth` / `LossPointmap` and their `_tok` twins.

## The flow objective does not exclude it

`flow_loss` (`occrae/deltatok_flow_trainer.py:507-543`) builds `mask = torch.ones_like(x)` and zeroes only the
**context frames**. It operates in delta-token latent space `(B, C, T, N, K)`, which has no pixel notion at all, so
`Eval/Loss` and `MSEToken` include the sky patch tokens at full weight. The model is trained to predict the sky's
delta tokens and scored on doing so; it is only the decoded-geometry readout that ignores the result.

## Open

- **The valid fraction is not measured**, per eval set. KITTI is 64-beam over a 518×168 crop, nuScenes 32-beam over
  518×266, so `sum(mask)` almost certainly differs several-fold between the two eval sets. Cross-dataset loss
  comparisons are averages over very different pixel counts. Worth logging `mask.mean()` alongside the losses.
- **Whether that is the eval protocol the paper wants.** Depth forecast scored only on LiDAR hits under 50 m is a
  defensible protocol, but it must be stated, and it interacts with the FVD plan (TODO 10), which would score the
  full frame including sky.
