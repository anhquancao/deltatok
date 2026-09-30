"""Point-cloud Accuracy / Completeness / Chamfer, ported from Gen3R.

Source: third_party/Gen3R/gen3r/utils/eval_utils.py (JaceyHuang/Gen3R @ 1df25b6),
``umeyama_alignment`` + the masked branch of ``compute_chamfer_metrics``.
Pure torch, so it needs neither open3d nor pytorch3d and runs on CUDA or ROCm:
  open3d  voxel_down_sample      -> voxel_down_sample (same per-voxel mean)
  pytorch3d sample_farthest_points -> sample_farthest_points (start index 0)
  pytorch3d knn_points(K=1)       -> nn_dist (exact)
Everything after the alignment runs in float64, as Gen3R does (open3d returns
float64). Deferred: the eval uses Gen3R's copy (chamfer_metrics.py). Departures:
  - voxel output order: open3d's hash order vs sorted here, so FPS starts at a
    different point and picks a slightly different subset;
  - fewer than num_points points: pytorch3d zero-pads the sample, here it is capped.
"""
import torch


def umeyama_alignment(P, G, mask, with_scale=True):
    """Gen3R's similarity alignment of P onto G over masked points, in float64.

    P, G: (F, H, W, 3); mask: (F, H, W) bool. Returns P aligned, (F, H, W, 3) float64.
    Scale is Gen3R's sqrt(var G / var P), not Umeyama's trace form.
    """
    P, G = P.double(), G.double()
    P_valid = P[mask]  # (N, 3)
    G_valid = G[mask]  # (N, 3)
    if P_valid.shape[0] < 3:
        return P.clone()

    P_mean = P_valid.mean(dim=0, keepdim=True)  # (1, 3)
    G_mean = G_valid.mean(dim=0, keepdim=True)  # (1, 3)
    P_centered = P_valid - P_mean  # (N, 3)
    G_centered = G_valid - G_mean  # (N, 3)

    H = P_centered.T @ G_centered  # (3, 3)
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    if torch.linalg.det(U @ V.T) < 0:
        V[:, -1] *= -1
    R = V @ U.T  # (3, 3)

    if with_scale:
        P_var = (P_centered ** 2).sum()
        G_var = (G_centered ** 2).sum()
        s = torch.sqrt(G_var / P_var) if P_var > 0 else P.new_ones(())
    else:
        s = P.new_ones(())

    t = G_mean.squeeze() - s * (R @ P_mean.T).squeeze()  # (3,)
    P_aligned = s * (P.reshape(-1, 3) @ R.T) + t  # (F*H*W, 3)
    return P_aligned.view(P.shape)  # (F, H, W, 3)


def voxel_down_sample(points, voxel_size):
    """open3d PointCloud.voxel_down_sample: mean of the points in each voxel.

    points: (N, 3) float64. Returns (M, 3) float64, sorted by voxel key.
    """
    lo = points.min(dim=0).values - voxel_size * 0.5  # (3,) open3d's voxel_min_bound
    ijk = torch.floor((points - lo) / voxel_size).long()  # (N, 3) voxel index
    _, inv = torch.unique(ijk, dim=0, return_inverse=True)  # inv (N,) -> voxel id; no flat key, so no overflow on wild preds
    M = int(inv.max()) + 1
    sums = points.new_zeros(M, 3).index_add_(0, inv, points)  # (M, 3)
    counts = torch.bincount(inv, minlength=M).to(points.dtype)  # (M,)
    return sums / counts[:, None]  # (M, 3)


def sample_farthest_points(points, lengths, K):
    """pytorch3d sample_farthest_points, random_start_point=False, over padded clouds.

    points: (B, P, 3); lengths: (B,) valid points per cloud. Returns idx (B, K);
    entries past min(K, length) repeat already-picked points and must be dropped.
    """
    B, P, _ = points.shape
    valid = torch.arange(P, device=points.device)[None] < lengths[:, None]  # (B, P)
    min_d = torch.full((B, P), float("inf"), dtype=points.dtype, device=points.device)  # (B, P)
    min_d[~valid] = -1.0  # padding is never the farthest
    rows = torch.arange(B, device=points.device)  # (B,)
    sel = torch.zeros(B, dtype=torch.long, device=points.device)  # (B,) start at index 0
    idx = torch.empty(B, K, dtype=torch.long, device=points.device)  # (B, K)
    for k in range(K):
        idx[:, k] = sel
        last = points[rows, sel]  # (B, 3)
        d = ((points - last[:, None]) ** 2).sum(-1)  # (B, P) squared distance to the last pick
        min_d = torch.minimum(min_d, d)  # (B, P) distance to the picked set
        sel = min_d.argmax(dim=1)  # (B,) farthest remaining point
    return idx


def nn_dist(x, y, chunk=2048):
    """Euclidean distance from each point of x to its nearest point in y (pytorch3d
    knn_points K=1, then sqrt). x: (N, 3), y: (M, 3). Returns (N,)."""
    out = []
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i:i + chunk], y, compute_mode="donot_use_mm_for_euclid_dist")  # (n, M) exact, no TF32 matmul
        out.append(d.min(dim=1).values)  # (n,)
    return torch.cat(out)  # (N,)


def compute_chamfer_metrics(P, G, mask, voxel_size=0.005, num_points=20000, generator=None):
    """Gen3R's masked compute_chamfer_metrics, batched over windows.

    P, G: (B, F, H, W, 3) pred / GT pointmaps; mask: (B, F, H, W) valid GT pixels.
    Per window: align P to G (Umeyama + scale), voxel-downsample both masked clouds,
    FPS each to num_points, then nearest-neighbour distances both ways.
    generator: shuffles each cloud before FPS (parity test only; None = sorted order).
    Returns dict of (B,) float64: accuracy (P->G), completeness (G->P),
    chamfer (their mean), relative_percent (chamfer / GT bbox diagonal * 100).
    Windows with < 3 valid points are NaN.
    """
    B = P.shape[0]
    clouds = []  # 2B clouds: pred of each window, then GT of each window
    for b in range(B):
        m = mask[b].bool()  # (F, H, W)
        P_al = umeyama_alignment(P[b], G[b], m)  # (F, H, W, 3) float64
        clouds.append((P_al[m], G[b].double()[m]))  # (N, 3) each
    flat = [c[0] for c in clouds] + [c[1] for c in clouds]  # 2B x (N, 3)

    down = []
    for pts in flat:
        if pts.shape[0] < 3:
            down.append(pts)
            continue
        d = voxel_down_sample(pts, voxel_size)  # (M, 3)
        if generator is not None:
            d = d[torch.randperm(d.shape[0], generator=generator).to(d.device)]
        down.append(d)

    lengths = torch.tensor([d.shape[0] for d in down], device=P.device)  # (2B,)
    Pmax = max(int(lengths.max()), 1)
    padded = P.new_zeros(2 * B, Pmax, 3, dtype=torch.float64)  # (2B, Pmax, 3)
    for i, d in enumerate(down):
        padded[i, :d.shape[0]] = d
    K = min(num_points, Pmax)
    idx = sample_farthest_points(padded, lengths, K)  # (2B, K)
    sampled = [padded[i, idx[i, :min(K, int(lengths[i]))]] for i in range(2 * B)]  # 2B x (k, 3)

    out = {k: torch.full((B,), float("nan"), dtype=torch.float64) for k in
           ("accuracy", "completeness", "chamfer", "relative_percent")}
    for b in range(B):
        Ps, Gs = sampled[b], sampled[B + b]  # (k, 3) each
        if clouds[b][1].shape[0] < 3:
            continue
        acc = nn_dist(Ps, Gs).mean()  # P -> G
        comp = nn_dist(Gs, Ps).mean()  # G -> P
        cd = (acc + comp) / 2
        diag = (Gs.max(dim=0).values - Gs.min(dim=0).values).norm()  # GT bbox diagonal
        out["accuracy"][b] = acc.item()
        out["completeness"][b] = comp.item()
        out["chamfer"][b] = cd.item()
        out["relative_percent"][b] = (cd / diag * 100).item()
    return out
