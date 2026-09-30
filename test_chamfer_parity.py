#!/usr/bin/env python3
"""Check occrae/chamfer_metrics_torch.py against Gen3R's original (open3d + pytorch3d).

Inputs are real GT pointmaps (--dataset, default nuScenes val). The "prediction" is
the GT under a known similarity transform plus depth-proportional noise and a
far-field bias, so alignment and both NN directions do real work.

Stages, each on identical inputs:
  1 umeyama   ours (float64) vs Gen3R (float32)
  2 voxel     ours vs open3d voxel_down_sample, compared as sorted sets
  3 fps       ours vs pytorch3d sample_farthest_points on the same ordered cloud
  4 knn       ours vs pytorch3d knn_points on the same sampled clouds
  5 e2e       full metric; open3d's voxel order is not reproducible, so the gap
              is judged against ours rerun over --num_orders random voxel orders
  6 sparse    a window cut below num_points: pytorch3d zero-pads, ours caps
  7 speed     s/window for Gen3R vs ours vs hybrid, and ours on a trainer-shaped batch
  8 hybrid    occrae/chamfer_metrics.py (the eval's) vs Gen3R end to end
  9 pure      the hybrid with pytorch3d FPS swapped for torch FPS (zero-padded like pytorch3d)

Needs open3d + pytorch3d (Jean Zay env). Usage
-----
python test_chamfer_parity.py
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

from occany.utils.runtime_paths import prepend_vendored_import_paths

REPO_ROOT = prepend_vendored_import_paths(Path(__file__).resolve().parent)

from occany.datasets import get_data_loader  # noqa: E402
from occrae import chamfer_metrics_torch as ours  # noqa: E402
from occrae import chamfer_metrics as hybrid  # noqa: E402  torch voxel/NN + pytorch3d FPS (the eval's)

torch.backends.cuda.matmul.allow_tf32 = False  # Gen3R's float32 Umeyama must not run in TF32

# The eval set's nuScenes entry (clean val folder, cam 0, 10 frames); stride 10 spreads windows over scenes.
NUSCENES_JZ = ("Occ3dNuscenesSeqMultiView(NUSCENES_PREPROCESSED_ROOT="
               "'/lustre/fsn1/projects/rech/trg/uyl37fq/occany_dataset/occ3d_nuscenes_val_preprocessed', "
               "seq_pkl_name='seq_surround_temporal_sub1_stride9_fs1_occ3d_nuscenes_val_all.pkl', "
               "num_timesteps=10, num_views_per_timestep=6, fixed_cams=[0], "
               "z_far=50, split='val', seed=42, window_stride=10, resolution=[(518, 266)])")


def timed(fn):
    """(result, seconds) with the GPU drained on both sides."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def load_gen3r():
    """Gen3R's eval_utils.py by path, skipping the gen3r package imports."""
    path = Path(REPO_ROOT) / "third_party/Gen3R/gen3r/utils/eval_utils.py"
    spec = importlib.util.spec_from_file_location("gen3r_eval_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_windows(dataset, num_windows, device):
    """[(G (F, H, W, 3), mask (F, H, W))], all views of each window."""
    loader = get_data_loader(f"{num_windows} @ {dataset}", batch_size=1, num_workers=2, shuffle=False, drop_last=False)
    for obj in (getattr(loader, "sampler", None), getattr(loader, "dataset", None)):
        if hasattr(obj, "set_epoch"):
            obj.set_epoch(0)  # ResizedDataset needs it, as in the trainer
    out = []
    for batch in loader:
        G = torch.stack([v["pts3d"] for v in batch], dim=1)[0].float()  # (V, H, W, 3)
        if "valid_mask" in batch[0]:
            m = torch.stack([v["valid_mask"] for v in batch], dim=1)[0]  # (V, H, W)
        else:
            m = torch.stack([v["depthmap"] for v in batch], dim=1)[0] > 0  # (V, H, W)
        out.append((G.to(device), m.bool().reshape(G.shape[:-1]).to(device)))
        if len(out) == num_windows:
            break
    return out


def make_pred(G, gen):
    """GT under a known sim(3) plus noise that grows with depth and a far-field bias."""
    q = torch.randn(4, generator=gen, dtype=torch.float64)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    R = torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=torch.float64)  # (3, 3)
    s = 0.5 + torch.rand((), generator=gen, dtype=torch.float64)
    t = 5 * torch.randn(3, generator=gen, dtype=torch.float64)
    Gd = G.double().cpu()  # (F, H, W, 3)
    depth = Gd.norm(dim=-1, keepdim=True)  # (F, H, W, 1)
    noise = 0.02 * depth * torch.randn(Gd.shape, generator=gen, dtype=torch.float64)  # (F, H, W, 3)
    bias = (depth > depth.median()).double() * torch.tensor([0.5, 0.0, 0.3], dtype=torch.float64)  # (F, H, W, 3)
    P = s * ((Gd + noise + bias) @ R.T) + t  # (F, H, W, 3)
    return P.float().to(G.device)


def torch_fps_padded(points, K):
    """pytorch3d sample_farthest_points for one cloud (1, M, 3): K > M zero-pads, as pytorch3d does."""
    M = points.shape[1]
    k = min(K, M)
    idx = ours.sample_farthest_points(points, torch.tensor([M], device=points.device), k)  # (1, k)
    out = points.new_zeros(1, K, 3)  # (1, K, 3)
    out[:, :k] = points[:, idx[0]]
    return out, idx


def pure_chamfer(P, G, m):
    """hybrid.compute_chamfer_metrics with its pytorch3d FPS swapped for torch_fps_padded."""
    fps = hybrid.sample_farthest_points
    hybrid.sample_farthest_points = torch_fps_padded
    try:
        return hybrid.compute_chamfer_metrics(P, G, m)
    finally:
        hybrid.sample_farthest_points = fps


def sort_rows(x):
    a = x.detach().cpu().numpy()
    return a[np.lexsort(a.T[::-1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=NUSCENES_JZ)
    ap.add_argument("--num_windows", type=int, default=20)
    ap.add_argument("--num_orders", type=int, default=10)
    ap.add_argument("--num_points", type=int, default=20000)
    ap.add_argument("--sparse_points", type=int, default=5000)
    ap.add_argument("--batch_windows", type=int, default=8)  # trainer call = 2 x val_bsize windows
    args = ap.parse_args()

    import open3d as o3d
    from pytorch3d.ops import knn_points, sample_farthest_points
    g3 = load_gen3r()
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)

    windows = load_windows(args.dataset, args.num_windows, device)
    print(f"[data] {len(windows)} windows, frames x H x W = {tuple(windows[0][1].shape)}, "
          f"valid pts/window {min(int(m.sum()) for _, m in windows)}..{max(int(m.sum()) for _, m in windows)}", flush=True)

    worst = {k: 0.0 for k in ("umeyama", "voxel", "knn")}
    fps_same, fps_total, voxel_count_mismatch = 0, 0, 0
    e2e_rows = []
    times = []  # (gen3r s, ours s, hybrid s, pure s) per window
    hybrid_rows = []  # (gen3r, hybrid, pure) chamfer
    for w, (G, m) in enumerate(windows):
        P = make_pred(G, gen)  # (F, H, W, 3)
        diag = (G[m].max(0).values - G[m].min(0).values).norm().item()

        # 1 umeyama
        P_g3 = g3.umeyama_alignment(P, G, m)[-1]  # (F, H, W, 3) float32
        P_al = ours.umeyama_alignment(P, G, m)  # (F, H, W, 3) float64
        worst["umeyama"] = max(worst["umeyama"], (P_g3.double() - P_al)[m].abs().max().item() / diag)

        # 2 voxel: same float64 input to both
        pts = P_al[m]  # (N, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        v_o3d = np.asarray(pcd.voxel_down_sample(voxel_size=0.005).points)  # (M, 3)
        v_ours = ours.voxel_down_sample(pts, 0.005)  # (M, 3)
        if v_o3d.shape[0] != v_ours.shape[0]:
            voxel_count_mismatch += 1
        else:
            worst["voxel"] = max(worst["voxel"], np.abs(sort_rows(torch.from_numpy(v_o3d)) - sort_rows(v_ours)).max())

        # 3 fps: our voxel order into both
        K = min(args.num_points, v_ours.shape[0])
        _, idx_p3d = sample_farthest_points(v_ours[None], K=K)  # (1, K)
        idx_ours = ours.sample_farthest_points(v_ours[None], torch.tensor([v_ours.shape[0]], device=device), K)  # (1, K)
        fps_same += (idx_p3d[0] == idx_ours[0]).sum().item()
        fps_total += K

        # 4 knn on the same sampled clouds
        Gs = G.double()[m]  # (N, 3)
        Gs = Gs[ours.sample_farthest_points(Gs[None], torch.tensor([Gs.shape[0]], device=device),
                                            min(args.num_points, Gs.shape[0]))[0]]  # (K, 3)
        Ps = v_ours[idx_ours[0]]  # (K, 3)
        d_p3d = knn_points(Ps[None], Gs[None], K=1).dists[0, :, 0].sqrt()  # (K,)
        d_ours = ours.nn_dist(Ps, Gs)  # (K,)
        worst["knn"] = max(worst["knn"], ((d_p3d - d_ours).abs() / d_ours.clamp(min=1e-9)).max().item())

        # 5 end to end
        (acc, comp, cd, rel, _), t_g3 = timed(lambda: g3.compute_chamfer_metrics(P, G, m))
        mine, t_ours = timed(lambda: ours.compute_chamfer_metrics(P[None], G[None], m[None], num_points=args.num_points))
        (_, _, cd_h, _, _), t_h = timed(lambda: hybrid.compute_chamfer_metrics(P, G, m))
        (_, _, cd_p, _, _), t_p = timed(lambda: pure_chamfer(P, G, m))
        times.append((t_g3, t_ours, t_h, t_p))
        hybrid_rows.append((cd.item(), cd_h.item(), cd_p.item()))
        print(f"[e2e {w:2d}] chamfer gen3r {cd.item():.5f} hybrid {cd_h.item():.5f} pure {cd_p.item():.5f}", flush=True)
        orders = [ours.compute_chamfer_metrics(P[None], G[None], m[None], num_points=args.num_points,
                                               generator=torch.Generator().manual_seed(r))["chamfer"].item()
                  for r in range(args.num_orders)]
        e2e_rows.append((cd.item(), mine["chamfer"].item(), float(np.std(orders)),
                         acc.item(), mine["accuracy"].item(), comp.item(), mine["completeness"].item()))
        print(f"[e2e {w:2d}] chamfer gen3r {cd.item():.5f} ours {mine['chamfer'].item():.5f} "
              f"order-std {np.std(orders):.5f} | acc {acc.item():.5f}/{mine['accuracy'].item():.5f} "
              f"comp {comp.item():.5f}/{mine['completeness'].item():.5f} "
              f"rel% {rel.item():.4f}/{mine['relative_percent'].item():.4f}", flush=True)

    print(f"[1 umeyama] max |gen3r - ours| / scene diag = {worst['umeyama']:.2e}")
    print(f"[2 voxel]   count mismatches {voxel_count_mismatch}/{len(windows)}; max |diff| (sorted sets) = {worst['voxel']:.2e} m")
    print(f"[3 fps]     identical indices {fps_same}/{fps_total} ({100 * fps_same / fps_total:.3f}%)")
    print(f"[4 knn]     max relative distance diff = {worst['knn']:.2e}")
    e = np.array(e2e_rows)
    gap = np.abs(e[:, 0] - e[:, 1])
    print(f"[5 e2e]     |gen3r - ours| chamfer: mean {100 * (gap / e[:, 0]).mean():.3f}% max {100 * (gap / e[:, 0]).max():.3f}%; "
          f"gap / order-std: median {np.median(gap / np.maximum(e[:, 2], 1e-12)):.2f} max {(gap / np.maximum(e[:, 2], 1e-12)).max():.2f}")

    # 6 sparse: keep sparse_points valid pixels of the first window
    G, m = windows[0]
    keep = torch.nonzero(m.reshape(-1))[:, 0]
    keep = keep[torch.randperm(keep.numel(), generator=gen)[:args.sparse_points].to(keep.device)]
    ms = torch.zeros(m.numel(), dtype=torch.bool, device=m.device)
    ms[keep] = True
    ms = ms.view(m.shape)
    P = make_pred(G, gen)
    _, _, cd_g3, _, _ = g3.compute_chamfer_metrics(P, G, ms)
    cd_ours = ours.compute_chamfer_metrics(P[None], G[None], ms[None], num_points=args.num_points)["chamfer"].item()
    print(f"[6 sparse]  {args.sparse_points} pts < {args.num_points}: chamfer gen3r {cd_g3.item():.5f} ours {cd_ours:.5f}")

    # 7 speed: first window excluded (CUDA/open3d warm-up); batch = 2 x batch_windows, as the trainer calls it
    t = np.array(times[1:])
    print(f"[7 speed]   per window: gen3r {t[:, 0].mean():.3f} s, ours {t[:, 1].mean():.3f} s, "
          f"hybrid {t[:, 2].mean():.3f} s, pure {t[:, 3].mean():.3f} s (mean of {len(t)})")
    h = np.array(hybrid_rows)
    hg = np.abs(h[:, 0] - h[:, 1]) / h[:, 0]
    print(f"[8 hybrid]  |gen3r - hybrid| chamfer: mean {100 * hg.mean():.3f}% max {100 * hg.max():.3f}% "
          f"(voxel order only; sparse windows included)")
    for name, col in (("gen3r", 0), ("hybrid", 1)):
        pg = np.abs(h[:, col] - h[:, 2]) / h[:, col]
        print(f"[9 pure]    |{name} - pure| chamfer: mean {100 * pg.mean():.3f}% max {100 * pg.max():.3f}%")
    nb = min(args.batch_windows, len(windows))
    Gb = torch.stack([g for g, _ in windows[:nb]])  # (nb, F, H, W, 3)
    mb = torch.stack([mm for _, mm in windows[:nb]])  # (nb, F, H, W)
    Pb = torch.stack([make_pred(g, gen) for g, _ in windows[:nb]])  # (nb, F, H, W, 3)
    for rep in range(2):
        _, tb = timed(lambda: ours.compute_chamfer_metrics(Pb.repeat(2, 1, 1, 1, 1), Gb.repeat(2, 1, 1, 1, 1),
                                                           mb.repeat(2, 1, 1, 1), num_points=args.num_points))
        print(f"[7 speed]   ours, trainer batch of {2 * nb} windows (rep {rep}): {tb:.2f} s = {tb / (2 * nb):.3f} s/window")


if __name__ == "__main__":
    main()
