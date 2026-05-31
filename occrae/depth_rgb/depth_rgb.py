"""Encode metric depth maps as RGB images and decode them back.

Implements the Vision Banana depth-RGB bijection from Section 3.2 of
"Image Generators are Generalist Vision Learners" (arXiv:2604.20329).
"""

import numpy as np

CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1],
], dtype=np.float64)

for _i in range(len(CORNERS) - 1):
    assert np.sum(np.abs(CORNERS[_i + 1] - CORNERS[_i])) == 1, \
        f"Corners {_i} and {_i+1} do not differ in exactly one coordinate"

_STARTS = CORNERS[:-1]   # (7, 3)
_ENDS = CORNERS[1:]      # (7, 3)
_DIRS = _ENDS - _STARTS  # (7, 3)
_DOT_DD = np.sum(_DIRS * _DIRS, axis=1)  # (7,), all 1.0 for unit edges


# c controls the depth range where most color resolution is spent;
# 95% of the color path covers 0 to ~10.4*c meters (≈52m at c=5).
def power_forward(d: np.ndarray, lam: float = -3.0, c: float = 5.0) -> np.ndarray:
    """Map depth d >= 0 to normalized distance f in [0, 1)."""
    d = np.asarray(d, dtype=np.float64)
    return 1.0 - (1.0 - d / (lam * c)) ** (lam + 1.0)


def power_inverse(f: np.ndarray, lam: float = -3.0, c: float = 5.0) -> np.ndarray:
    """Map normalized distance f in [0, 1) back to depth d >= 0."""
    f = np.asarray(f, dtype=np.float64)
    return lam * c * (1.0 - (1.0 - f) ** (1.0 / (lam + 1.0)))


def _f_to_rgb(f: np.ndarray) -> np.ndarray:
    """Map f in [0, 1) to RGB in [0, 1]^3 via the cube edge path."""
    s = f * 7.0
    i = np.clip(np.floor(s).astype(int), 0, 6)
    t = s - i
    t = t[..., np.newaxis]  # (..., 1)
    return (1.0 - t) * CORNERS[i] + t * CORNERS[i + 1]


def _rgb_to_f(rgb: np.ndarray) -> np.ndarray:
    """Map RGB in [0, 1]^3 to f in [0, 1) via nearest-segment projection."""
    shape = rgb.shape[:-1]
    p = rgb.reshape(-1, 3)  # (N, 3)

    # Project onto all 7 segments at once: (N, 7)
    # p_expanded: (N, 1, 3), _STARTS: (7, 3) -> diff: (N, 7, 3)
    diff = p[:, np.newaxis, :] - _STARTS[np.newaxis, :, :]  # (N, 7, 3)
    dot_pd = np.sum(diff * _DIRS[np.newaxis, :, :], axis=2)  # (N, 7)
    t = np.clip(dot_pd / _DOT_DD[np.newaxis, :], 0.0, 1.0)  # (N, 7)

    # Closest point on each segment: (N, 7, 3)
    q = _STARTS[np.newaxis, :, :] + t[:, :, np.newaxis] * _DIRS[np.newaxis, :, :]
    dist_sq = np.sum((p[:, np.newaxis, :] - q) ** 2, axis=2)  # (N, 7)

    best = np.argmin(dist_sq, axis=1)  # (N,)
    t_best = t[np.arange(len(t)), best]  # (N,)
    f = (best + t_best) / 7.0
    return f.reshape(shape)


def depth_to_rgb(depth: np.ndarray, lam: float = -3.0, c: float = 100.0) -> np.ndarray:
    """(H, W) float meters -> (H, W, 3) float in [0, 1]."""
    f = power_forward(depth, lam, c)
    return _f_to_rgb(f)


def rgb_to_depth(rgb: np.ndarray, lam: float = -3.0, c: float = 100.0) -> np.ndarray:
    """(H, W, 3) float in [0, 1] -> (H, W) float meters."""
    f = _rgb_to_f(rgb)
    return power_inverse(f, lam, c)