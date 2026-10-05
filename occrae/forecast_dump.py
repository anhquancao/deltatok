"""Forecast dumps: written by eval_vggt_world.py, eval_gen3r.py and the flow trainer
(eval_deltatok_flow_sampler.py --dump_dir), read by eval_forecast_metrics.py.

One <set_dir>/<n:05d>.npz per loader window n, fp32 + zlib, frame-0 coords, metres:
  depth (N,), point (N, 3)                          on the window's N GT-mask (LiDAR) pixels
  mask (packed bits), mask_shape (T, H, W)          where those pixels sit; read_window() unpacks
  c2w (T, 4, 4), scene_name, frame_stems
  depth_dense (T, H, W), point_dense (T, H, W, 3)   first N_DENSE windows only, for panels
"""

import os

import numpy as np
import torch

N_DENSE = 16  # windows per set that keep dense depth and points


def save_windows(set_dir, n0, depth, point, c2w, batch):
    """Write windows n0..n0+B-1: depth (B, T, H, W), point (B, T, H, W, 3), c2w (B, T, 3|4, 4)."""
    mask = batch["gt_mask"].to(depth.device).bool().reshape(depth.shape)          # (B, T, H, W) LiDAR pixels
    c2w44 = torch.eye(4, device=c2w.device).repeat(*c2w.shape[:2], 1, 1)          # (B, T, 4, 4)
    c2w44[..., :3, :] = c2w[..., :3, :].float()
    for b in range(depth.shape[0]):
        dense = {}
        if n0 + b < N_DENSE:
            dense = {"depth_dense": depth[b].float().cpu().numpy(),               # (T, H, W)
                     "point_dense": point[b].float().cpu().numpy()}               # (T, H, W, 3)
        np.savez_compressed(os.path.join(set_dir, f"{n0 + b:05d}.npz"),
                            depth=depth[b][mask[b]].float().cpu().numpy(),         # (N,)
                            point=point[b][mask[b]].float().cpu().numpy(),         # (N, 3)
                            mask=np.packbits(mask[b].cpu().numpy()),               # (ceil(T*H*W / 8),) uint8
                            mask_shape=np.array(mask[b].shape),                    # (3,) = T, H, W
                            c2w=c2w44[b].cpu().numpy(), scene_name=batch["scene_name"][b],
                            frame_stems=list(batch["frame_stems"][b]), **dense)


def read_window(path):
    """One dump, no loader needed: depth (T, H, W) and point (T, H, W, 3), 0 off the mask;
    mask (T, H, W); c2w (T, 4, 4); scene_name, frame_stems; depth_dense / point_dense if kept."""
    with np.load(path) as f:
        w = {k: f[k] for k in f.files}
    shape = tuple(w.pop("mask_shape"))                                             # (T, H, W)
    w["mask"] = np.unpackbits(w["mask"], count=int(np.prod(shape))).reshape(shape).astype(bool)  # (T, H, W)
    depth = np.zeros(shape, np.float32)                                            # (T, H, W)
    depth[w["mask"]] = w["depth"]                                                  # same row-major order as the save
    point = np.zeros(shape + (3,), np.float32)                                     # (T, H, W, 3)
    point[w["mask"]] = w["point"]
    w["depth"], w["point"] = depth, point
    return w


def load_windows(set_dir, n0, batch, device):
    """Read windows n0..n0+B-1 and check them against the batch. Returns depth (B, T, H, W),
    point (B, T, H, W, 3), c2w (B, T, 4, 4), and per window its depth_dense (T, H, W) or None."""
    mask = batch["gt_mask"].cpu().bool().reshape(batch["gt_depth"].shape).numpy()  # (B, T, H, W) LiDAR pixels
    wins = [read_window(os.path.join(set_dir, f"{n0 + b:05d}.npz")) for b in range(mask.shape[0])]
    dense = []
    for b, w in enumerate(wins):
        assert (str(w["scene_name"]), tuple(w["frame_stems"].tolist())) == \
            (batch["scene_name"][b], tuple(batch["frame_stems"][b])), (n0 + b, str(w["scene_name"]))
        assert np.array_equal(w["mask"], mask[b]), (n0 + b, "GT mask differs from the dump")
        d = None
        if "depth_dense" in w:
            d = torch.from_numpy(w["depth_dense"]).to(device)
        dense.append(d)
    depth, point, c2w = (torch.from_numpy(np.stack([w[k] for w in wins])).to(device)
                         for k in ("depth", "point", "c2w"))                       # (B, T, H, W), (B, T, H, W, 3), (B, T, 4, 4)
    return depth, point, c2w, dense
