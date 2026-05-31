"""Tests for the Vision Banana depth-RGB bijection."""

import time
import numpy as np
import pytest
from occrae.depth_rgb import (
    CORNERS, depth_to_rgb, power_forward, power_inverse, rgb_to_depth,
)


def test_power_forward_zero():
    assert power_forward(0.0) == 0.0


def test_power_forward_monotonic():
    d = np.linspace(0, 100, 1000)
    f = power_forward(d)
    assert np.all(np.diff(f) > 0)


def test_power_forward_large_depth():
    f = power_forward(1e9)
    assert f < 1.0
    assert f > 0.999


def test_power_round_trip():
    rng = np.random.default_rng(42)
    d = rng.uniform(0, 100, 1000)
    np.testing.assert_allclose(power_inverse(power_forward(d)), d, atol=1e-9)


def test_corner_ordering():
    assert tuple(CORNERS[0]) == (0, 0, 0)
    assert tuple(CORNERS[-1]) == (1, 1, 1)
    for i in range(len(CORNERS) - 1):
        diff = np.abs(CORNERS[i + 1] - CORNERS[i])
        assert np.sum(diff) == 1, f"Pair {i}-{i+1} differs in != 1 coordinate"


def test_float_round_trip():
    rng = np.random.default_rng(7)
    d = rng.uniform(0, 50, 1000)
    recovered = rgb_to_depth(depth_to_rgb(d))
    assert np.max(np.abs(recovered - d)) < 1e-6


def test_decoder_noise_robustness():
    rng = np.random.default_rng(99)
    d = rng.uniform(0.5, 20, 1000)
    rgb = depth_to_rgb(d)
    rgb_noisy = rgb + rng.uniform(-0.02, 0.02, rgb.shape)
    rgb_noisy = np.clip(rgb_noisy, 0, 1)
    recovered = rgb_to_depth(rgb_noisy)
    rel_err = np.abs(recovered - d) / d
    assert np.median(rel_err) < 0.05


def test_endpoints():
    np.testing.assert_allclose(depth_to_rgb(np.array(0.0)), [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(depth_to_rgb(np.array(1e6)), [1, 1, 1], atol=1e-3)


def test_shape_and_vectorization():
    rng = np.random.default_rng(0)
    big = rng.uniform(0, 30, (128, 96))

    t0 = time.perf_counter()
    out = depth_to_rgb(big)
    t_big = time.perf_counter() - t0

    assert out.shape == (128, 96, 3)

    small = rng.uniform(0, 30, (1,))
    n = 128 * 96
    t0 = time.perf_counter()
    for _ in range(n):
        depth_to_rgb(small)
    t_small_total = time.perf_counter() - t0

    assert t_big < t_small_total * 1.01, \
        f"Vectorized ({t_big:.4f}s) should be faster than {n} scalar calls ({t_small_total:.4f}s)"