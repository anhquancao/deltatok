"""Demo: round-trip depth through Vision Banana encoding at multiple c values.

Compares 8-bit quantized round-trip error for different c parameters against
naive linear depth mapping (depth/max_depth).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from occrae.depth_rgb import depth_to_rgb, rgb_to_depth


def main() -> None:
    # 3D scatter plot of depth-to-RGB mapping
    depths_1d = np.linspace(0, 80, 1000)
    rgb_1d = depth_to_rgb(depths_1d.reshape(1, -1)).squeeze(0)
    r, g, b = rgb_1d[:, 0], rgb_1d[:, 1], rgb_1d[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(depths_1d) - 1):
        ax.plot(r[i:i+2], g[i:i+2], b[i:i+2], color=rgb_1d[i], linewidth=6)

    label_indices = np.linspace(0, len(depths_1d) - 1, 10, dtype=int)
    for i in label_indices:
        ax.text(r[i], g[i], b[i], f"{depths_1d[i]:.0f}m", fontsize=12, fontweight="bold", ha="center")

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title("Depth → RGB mapping (0–80 m)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.savefig("demo_depth_rgb_3d.png", dpi=150)
    print("Saved demo_depth_rgb_3d.png")

    H, W = 100, 300
    row_depths = np.linspace(80, 0, H)
    depth = np.tile(row_depths[:, None], (1, W))
    mid_row = depth[:, W // 2]

    # --- Linear mapping baseline ---
    max_depth = 80.0
    linear_f = np.clip(depth / max_depth, 0.0, 1.0)

    linear_u8 = np.clip(np.round(linear_f * 255.0), 0, 255).astype(np.uint8)
    linear_recovered = linear_u8.astype(np.float64) / 255.0 * max_depth
    err_linear_1ch = np.abs(linear_recovered - depth)

    print("\n=== Linear 1-channel 8-bit (depth/80 → uint8 → depth) ===")
    print(f"  max  abs error: {err_linear_1ch.max():.2e}")
    print(f"  mean abs error: {err_linear_1ch.mean():.2e}")
    print(f"  med  abs error: {np.median(err_linear_1ch):.2e}")
    print(f"  uniform step size: {max_depth / 255.0:.4f} m")

    # --- Vision Banana at multiple c values ---
    c_values = [5, 10, 20, 50, 100]

    for c in c_values:
        rgb = depth_to_rgb(depth, c=c)
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        rgb_q = rgb_u8.astype(np.float64) / 255.0
        recovered_q = rgb_to_depth(rgb_q, c=c)
        err_quant = np.abs(recovered_q - depth)
        print(f"\n=== Vision Banana 8-bit (c={c}) ===")
        print(f"  max  abs error: {err_quant.max():.2e}")
        print(f"  mean abs error: {err_quant.mean():.2e}")
        print(f"  med  abs error: {np.median(err_quant):.2e}")

    # --- Comparison plot: quantization error vs depth ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mid_row, err_linear_1ch[:, W // 2],
            label="Linear 1-ch (grayscale)", linewidth=2, color="gray")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(c_values)))
    for c, color in zip(c_values, colors):
        rgb = depth_to_rgb(depth, c=c)
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        rgb_q = rgb_u8.astype(np.float64) / 255.0
        recovered_q = rgb_to_depth(rgb_q, c=c)
        err_quant = np.abs(recovered_q - depth)
        ax.plot(mid_row, err_quant[:, W // 2],
                label=f"Vision Banana (c={c})", linewidth=2, color=color)

    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Absolute error after 8-bit quantization (m)")
    ax.set_title("Quantization error: Vision Banana (varying c) vs linear")
    ax.legend()
    ax.set_xlim(0, 80)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("demo_depth_error_comparison.png", dpi=150)
    print("\nSaved demo_depth_error_comparison.png")


if __name__ == "__main__":
    main()