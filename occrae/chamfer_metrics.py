"""Point-cloud Accuracy / Completeness / Chamfer, from Gen3R's gen3r/utils/eval_utils.py
(JaceyHuang/Gen3R @ 1df25b6). Pure torch: open3d voxel_down_sample, pytorch3d FPS and
knn_points are torch twins (test_chamfer_parity.py: chamfer within 0.15% of Gen3R).
Only the masked branch is ported. Batched port (deferred): occrae/chamfer_metrics_torch.py.
"""
import torch

from typing import Tuple
from occrae.chamfer_metrics_torch import sample_farthest_points as fps_indices


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


def sample_farthest_points(points, K):
    """pytorch3d sample_farthest_points for one cloud (1, M, 3): K > M zero-pads, as pytorch3d does."""
    M = points.shape[1]
    k = min(K, M)
    idx = fps_indices(points, torch.tensor([M], device=points.device), k)  # (1, k)
    out = points.new_zeros(1, K, 3)  # (1, K, 3)
    out[:, :k] = points[:, idx[0]]
    return out, idx


def nn_dist(x, y, chunk=2048):
    """Euclidean distance from each point of x to its nearest point in y (pytorch3d
    knn_points K=1, then sqrt). x: (N, 3), y: (M, 3). Returns (N,)."""
    out = []
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i:i + chunk], y, compute_mode="donot_use_mm_for_euclid_dist")  # (n, M) exact, no TF32 matmul
        out.append(d.min(dim=1).values)  # (n,)
    return torch.cat(out)  # (N,)


def umeyama_alignment(P: torch.Tensor, G: torch.Tensor, mask: torch.Tensor, with_scale: bool=True):
    """
    Align predicted point cloud P to ground truth point cloud G using Umeyama algorithm.
    
    Args:
        P: (F, H, W, 3) predicted point cloud
        G: (F, H, W, 3)/[N, 3] ground truth point cloud  
        mask: (F, H, W) boolean mask indicating valid points
        with_scale: whether to allow scaling transformation
        
    Returns:
        tuple: (R, t, s, P_aligned) where:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
            s: scalar scale factor
            P_aligned: (F, H, W, 3) aligned point cloud
    """
    # Extract valid points using mask
    if G.ndim == 4:
        P_valid = P[mask]  # (N, 3)
        G_valid = G[mask]  # (N, 3)
    else:
        P_valid = P[mask.bool()]  # (N, 3)
        G_valid = G[0]  # (N, 3)
    
    if P_valid.shape[0] < 3:
        # Not enough points for alignment, return identity transformation
        R = torch.eye(3, device=P.device, dtype=P.dtype)
        t = torch.zeros(3, device=P.device, dtype=P.dtype)
        s = torch.ones(1, device=P.device, dtype=P.dtype)
        P_aligned = P.clone()
        return R, t, s, P_aligned
    
    # Center the point clouds
    P_mean = P_valid.mean(dim=0, keepdim=True)  # (1, 3)
    G_mean = G_valid.mean(dim=0, keepdim=True)  # (1, 3)
    
    P_centered = P_valid - P_mean  # (N, 3)
    G_centered = G_valid - G_mean  # (N, 3)
    
    # Compute covariance matrix
    H = (P_centered.T @ G_centered).float()  # (3, 3)
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(H)
    
    # Ensure proper rotation matrix (handle reflection case)
    V = Vt.T
    if torch.linalg.det((U @ V.T).float()) < 0:
        V[:, -1] *= -1
    
    # Compute rotation matrix
    R = V @ U.T  # (3, 3)
    
    # Compute scale factor
    if with_scale:
        # Scale factor based on variance ratio
        P_var = (P_centered ** 2).sum()
        G_var = (G_centered ** 2).sum()
        s = torch.sqrt(G_var / P_var) if P_var > 0 else torch.ones(1, device=P.device, dtype=P.dtype)
    else:
        s = torch.ones(1, device=P.device, dtype=P.dtype)
    
    # Compute translation
    t = G_mean.squeeze() - s * (R @ P_mean.T).squeeze()  # (3,)
    
    # Apply transformation to all points
    P_aligned = s * (P.view(-1, 3) @ R.T) + t  # (F*H*W, 3)
    P_aligned = P_aligned.view(P.shape)  # (F, H, W, 3)
    
    return R, t, s, P_aligned


def compute_chamfer_metrics(
    P: torch.Tensor, 
    G: torch.Tensor, 
    mask: torch.Tensor, 
    squared: bool=False, 
    require_downsample: bool=True, 
    in_mm: bool=False, 
    voxel_size: float=0.005, 
    align: bool=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gen3R's compute_chamfer_metrics, masked branch: [align: Umeyama + scale fit],
    voxel-downsample both masked clouds, FPS each to 20k points, then
    nearest-neighbour distances both ways. Under 20k points FPS zero-pads the
    sample, as in Gen3R.

    Metrics:
        - accuracy: mean distance from predicted points to ground-truth points (P -> G)
        - completeness: mean distance from ground-truth points to predicted points (G -> P)
        - chamfer: average of accuracy and completeness
        - relative_percent: chamfer normalized by GT scene extent, in percent

    Args:
        P: Predicted points, shape `(F, H, W, 3)` or `(1, F, H, W, 3)`.
        G: Ground-truth points, shape `(F, H, W, 3)` or `(1, F, H, W, 3)`.
        mask: Boolean valid-point mask with shape `(F, H, W)`.
        squared: If `True`, keep squared KNN distances; otherwise use Euclidean distances.
        require_downsample: If `True`, run voxel downsampling before FPS.
        in_mm: If `True`, converts accuracy/completeness/chamfer from meters to millimeters.
        voxel_size: Voxel size of the downsampling.
        align: If `True`, Umeyama sim(3)-fit P to G first (Gen3R's protocol).

    Returns:
        accuracy, completeness, chamfer, relative_percent: Tensors of shape `(1,)`;
        P: Scored prediction (F, H, W, 3).
    """
    P, G = P.float(), G.float()
    if P.ndim == 5:
        P = P.squeeze(0)
    if G.ndim == 5:
        G = G.squeeze(0)

    # align P to G, extract valid points
    if align:  # default False: score metric predictions as they are
        P = umeyama_alignment(P, G, mask)[-1]
    P_flat = P[mask][None, ...]  # (1, N, 3)
    G_flat = G[mask][None, ...]  # (1, N, 3)

    if require_downsample:
        # float64, as open3d returns
        P_flat_downsampled = voxel_down_sample(P_flat[0].double(), voxel_size)[None]  # (1, M, 3)
        G_flat_downsampled = voxel_down_sample(G_flat[0].double(), voxel_size)[None]  # (1, M, 3)

        # select 20000 points as valid using Farthest Point Sampling
        P_flat, _ = sample_farthest_points(P_flat_downsampled, K=20000)  # (1, K, 3)
        G_flat, _ = sample_farthest_points(G_flat_downsampled, K=20000)  # (1, K, 3)

    # P -> G
    dist_pg = nn_dist(P_flat[0], G_flat[0])[None]  # (1, N)
    if squared:
        dist_pg = dist_pg ** 2
    accuracy = dist_pg.mean(dim=1)

    # G -> P
    dist_gp = nn_dist(G_flat[0], P_flat[0])[None]  # (1, N)
    if squared:
        dist_gp = dist_gp ** 2
    completeness = dist_gp.mean(dim=1)

    chamfer = (accuracy + completeness) / 2

    mins = G_flat.squeeze(0).min(dim=0)[0]  # (3,)
    maxs = G_flat.squeeze(0).max(dim=0)[0]  # (3,)
    dist = (maxs - mins).norm()
    relative_percent = (chamfer / dist) * 100  # %
    
    if in_mm:
        accuracy = accuracy * 1000
        completeness = completeness * 1000
        chamfer = chamfer * 1000
    
    return accuracy, completeness, chamfer, relative_percent, P
