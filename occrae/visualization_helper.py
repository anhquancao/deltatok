import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

from occany.utils.helpers import depth2rgb


_COLOR_CONTEXT = np.array([0, 200, 255], dtype=np.uint8)   # cyan
_COLOR_FORECAST = np.array([255, 100, 0], dtype=np.uint8)  # orange
_BORDER_WIDTH = 6


# ---------------------------------------------------------------------------
# Top-down (bird's-eye-view) camera-pose rendering.
# `cameras_bev` and the `_BEV_*` style constants are copied verbatim from
# ../VATIX/Utils/visualization.py (lines 57-229) so the BEV panels match the
# VATIX trajectory-preview style. Do not refactor — keep it 1:1 with the source.
# ---------------------------------------------------------------------------
_BEV_PANEL_BG = "#020712"
_BEV_PANEL_FRAME = "#26C6DA"
_BEV_PANEL_TEXT = "#D7FFFB"
_BEV_PANEL_GRID = "#204b6d"
_BEV_PANEL_TRAJ_OUTLINE = "#001322"
_BEV_PANEL_TRAJ_INNER = "#26C6DA"
_BEV_FRUSTUM_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bev_frustum",
    ["#26C6DA", "#9CCC65", "#FFA726", "#EF5350"],
)
_BEV_PANEL_START = "#9CCC65"
_BEV_PANEL_END = "#FFA726"


_BEV_PANEL_CURRENT = "#EF5350"
_BEV_PANEL_CURRENT_EDGE = "#FFFFFF"


def cameras_bev(
    R,
    T,
    H=256,
    W=256,
    frustum_len=0.12,
    fov_deg=60.0,
    margin=0.25,
    title=None,
    current_index=None,
):
    """Render a bird's-eye-view of camera frustums on the XZ plane.

    Styled to match the trajectory preview panel (dark navy background,
    cyan accents, neon gradient frustums, glowing trajectory line).

    Args:
        R: ``(T, 3, 3)`` c2w rotation matrices (numpy).
        T: ``(T, 3)`` c2w translations in metres (numpy).
        H, W: output image size in pixels.
        frustum_len: frustum length as fraction of plot span (0..1).
        fov_deg: full horizontal FOV used to draw the frustum triangle.
        margin: fractional padding around the camera positions.
        title: optional title drawn inside the panel.

    Returns:
        ``(H, W, 3)`` uint8 RGB array.
    """
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n_frames = T.shape[0]

    fwd = R[:, :, 2]  # (T, 3) — camera +Z in world
    px, pz = T[:, 0], T[:, 2]
    fx, fz = fwd[:, 0], fwd[:, 2]

    # Bounds (square, margin-padded).
    all_x = np.concatenate([px, px + frustum_len * fx])
    all_z = np.concatenate([pz, pz + frustum_len * fz])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    z_min, z_max = float(all_z.min()), float(all_z.max())
    pad = max(x_max - x_min, z_max - z_min, 1e-6) * margin
    x_min -= pad; x_max += pad
    z_min -= pad; z_max += pad
    span = max(x_max - x_min, z_max - z_min, 1e-6)
    cx = 0.5 * (x_min + x_max)
    cz = 0.5 * (z_min + z_max)
    x_min, x_max = cx - span / 2, cx + span / 2
    z_min, z_max = cz - span / 2, cz + span / 2

    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(_BEV_PANEL_BG)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.75, color=_BEV_PANEL_GRID, alpha=0.75)
    ax.tick_params(axis="both", colors=_BEV_PANEL_TEXT, labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_color(_BEV_PANEL_FRAME)
        spine.set_alpha(0.28)

    # Frustums.
    half_angle = math.radians(fov_deg / 2)
    flen_world = frustum_len * span
    cos_a, sin_a = math.cos(half_angle), math.sin(half_angle)

    cur_idx = (
        int(np.clip(current_index, 0, n_frames - 1))
        if (current_index is not None and n_frames > 0)
        else None
    )

    for t in range(n_frames):
        f_norm = math.hypot(fx[t], fz[t])
        if f_norm < 1e-8:
            continue
        dx_n, dz_n = fx[t] / f_norm, fz[t] / f_norm
        lx = dx_n * cos_a - dz_n * sin_a
        lz = dx_n * sin_a + dz_n * cos_a
        rx = dx_n * cos_a + dz_n * sin_a
        rz = -dx_n * sin_a + dz_n * cos_a

        is_current = (cur_idx is not None and t == cur_idx)
        apex = (px[t], pz[t])
        left = (px[t] + flen_world * lx, pz[t] + flen_world * lz)
        right = (px[t] + flen_world * rx, pz[t] + flen_world * rz)
        colour = _BEV_FRUSTUM_CMAP(t / max(n_frames - 1, 1))
        tri = plt.Polygon(
            [apex, left, right],
            closed=True,
            facecolor=colour,
            edgecolor=_BEV_PANEL_CURRENT_EDGE if is_current else "#020712",
            linewidth=1.4 if is_current else 0.8,
            alpha=1.0 if is_current else 0.78,
            zorder=5 if is_current else 4,
        )
        ax.add_patch(tri)

    # Start / end markers.
    if n_frames > 0:
        ax.scatter(
            [px[0]], [pz[0]],
            color=_BEV_PANEL_START, marker="s", s=92,
            linewidths=0.0, zorder=6,
        )
        ax.scatter(
            [px[-1]], [pz[-1]],
            color=_BEV_PANEL_END, marker="x", s=116,
            linewidths=2.2, zorder=6,
        )

    # Current-frame highlight: small marker at the active camera position.
    if cur_idx is not None:
        ax.scatter(
            [px[cur_idx]], [pz[cur_idx]],
            color=_BEV_PANEL_CURRENT, s=70,
            edgecolors=_BEV_PANEL_CURRENT_EDGE,
            linewidths=1.2, zorder=7,
        )

    # Bounds overlay (top-left).
    ax.text(
        0.02, 0.98,
        f"x: [{x_min:.1f}, {x_max:.1f}]\nz: [{z_min:.1f}, {z_max:.1f}]",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color=_BEV_PANEL_TEXT,
        bbox=dict(facecolor="#07101f", edgecolor=_BEV_PANEL_FRAME, boxstyle="round,pad=0.25", alpha=0.92),
        zorder=8,
    )

    if title is not None:
        ax.text(
            0.98, 0.02, str(title),
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=12, color=_BEV_PANEL_TEXT,
            bbox=dict(facecolor="#07101f", edgecolor=_BEV_PANEL_FRAME, boxstyle="round,pad=0.3", alpha=0.92),
            zorder=8,
        )

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
    return rgb.astype(np.uint8)


def _build_bev_panel(c2w, view_order, panel_h, panel_w, title=None):
    """Per-view BEV panel: each row is the full camera trajectory with the
    current (time-ordered) view highlighted.

    Plays the same role as the depth/RGB panels in ``_log_viz_sample``: returns
    a ``(V, panel_h, panel_w, 3)`` float tensor in ``[0, 255]`` whose row ``t``
    shows time-ordered camera ``t`` highlighted, so the existing horizontal
    concat lays the BEV column out next to the depth/RGB columns.

    Args:
        c2w: ``(V, 3, 4)`` (decoded ``pose_from_depth_ray``) or ``(V, 4, 4)``
            (GT ``batch["gt_c2w"]``) camera-to-world matrices, torch or numpy,
            all in a single shared coordinate frame. NOT pre-indexed by
            ``view_order`` — this helper reorders internally.
        view_order: list[int] time-sorted view indices (length V).
        panel_h, panel_w: per-view tile size in pixels. Pass the image height
            for both so the row height matches the depth/RGB panels (square BEV).
        title: optional title drawn inside every tile.

    Returns:
        torch.FloatTensor ``(V, panel_h, panel_w, 3)`` in ``[0, 255]``.
    """
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().float().cpu().numpy()
    c2w = np.asarray(c2w)[..., :3, :]              # (V, 3, 4) — handles 4x4 GT and 3x4 decoded
    idx = np.asarray(view_order, dtype=int)        # (V,) time-sorted order
    R_all = c2w[idx, :, :3]                        # (V, 3, 3) c2w rotations, time-ordered
    T_all = c2w[idx, :, 3]                         # (V, 3) c2w translations, time-ordered

    tiles = [
        cameras_bev(R_all, T_all, H=panel_h, W=panel_w, current_index=t, title=title)
        for t in range(R_all.shape[0])
    ]                                              # V x (panel_h, panel_w, 3) uint8
    return torch.from_numpy(np.stack(tiles, axis=0)).float()  # (V, panel_h, panel_w, 3) in [0, 255]


def _draw_left_borders(combined_np, num_views, frame_height, is_context):
    """Draw colored left border lines on each frame row to indicate context vs forecast."""
    for t in range(num_views):
        color = _COLOR_CONTEXT if is_context[t] else _COLOR_FORECAST
        y0 = t * frame_height
        y1 = y0 + frame_height
        combined_np[y0:y1, :_BORDER_WIDTH, :] = color


def _add_column_titles(img_np, col_widths, titles):
    """Prepend a title bar above each column of the concatenated image.

    Columns are NOT equal width (depth panels are wider than the RGB camera
    frames, and BEV panels are square at the view height), so each title is
    centered on its own panel's actual horizontal extent using the running
    offset of ``col_widths`` rather than a uniform ``W // num_cols`` stride.
    """
    title_h = 36
    H, W = img_np.shape[:2]
    header = np.full((title_h, W, 3), 30, dtype=np.uint8)
    pil_header = Image.fromarray(header)
    draw = ImageDraw.Draw(pil_header)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    x0 = 0  # left edge of the current column, accumulated across panels
    for width, title in zip(col_widths, titles):
        cx = x0 + width // 2  # center of this column's actual extent
        draw.text((cx, title_h // 2), str(title), fill=(255, 255, 255), anchor="mm", font=font)
        x0 += width
    titled = np.vstack([np.array(pil_header), img_np])
    return titled


def _log_viz_sample(
    batch,
    decoded,
    batch_idx,
    epoch,
    epoch_step,
    output_dir,
    log_writer,
    tb_prefix,
    extra_panels=None,
    view_order=None,
    max_depth=50.0,
    context_mask=None,
    col_titles=None,
    pred_blank_views=None,
    include_input_rgb=True,
):
    """Log a side-by-side validation sample for OccRAE reconstructions.

    include_input_rgb: when True (default) the first column is the GT input RGB
    image; set False to drop that column (the caller must then also omit the
    matching "RGB" entry from ``col_titles``).
    """
    pred_depth = decoded["depth"][batch_idx].detach().float().cpu()
    gt_img = (
        batch["imgs"][batch_idx].detach().cpu().permute(0, 2, 3, 1)
        if include_input_rgb else None
    )

    if view_order is not None:
        index = torch.as_tensor(view_order, dtype=torch.long)
        pred_depth = pred_depth[index]
        if gt_img is not None:
            gt_img = gt_img[index]
    num_views = pred_depth.shape[0]

    if gt_img is not None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        gt_img = (gt_img * std + mean).clamp(0, 1) * 255.0

    scene_name = str(batch["scene_name"][batch_idx]).replace(os.sep, "_")
    timesteps = batch["timesteps"][batch_idx]
    if isinstance(timesteps, torch.Tensor):
        timesteps = timesteps.tolist()
    has_timesteps = timesteps is not None and len(timesteps) > 0
    frame_stems = batch.get("frame_stems", [None])[batch_idx]
    if frame_stems and has_timesteps:
        anchor = str(frame_stems[0]).replace(os.sep, "_")
        frame_id = f"{scene_name}_{anchor}_t{timesteps[0]}-{timesteps[-1]}"
    elif has_timesteps:
        frame_id = f"{scene_name}_t{timesteps[0]}-{timesteps[-1]}"
    else:
        frame_id = f"{scene_name}_b{batch_idx}"

    pred_depth_color = torch.stack([
        torch.from_numpy(
            depth2rgb(
                pred_depth[t].clamp(0, max_depth).numpy(),
                valid_mask=pred_depth[t].numpy() > 0,
                min_depth=0.0,
                max_depth=max_depth,
            ).astype(np.float32)
        )
        for t in range(num_views)
    ])
    if pred_blank_views is not None:
        for t, blank in enumerate(pred_blank_views):
            if blank:
                pred_depth_color[t] = 30.0

    all_panels = ([gt_img] if gt_img is not None else []) + [pred_depth_color] + (extra_panels or [])
    cols = [torch.cat([panel[t] for t in range(num_views)], dim=0) for panel in all_panels]
    combined_np = torch.cat(cols, dim=1).numpy()

    if context_mask is not None:
        frame_height = combined_np.shape[0] // num_views
        _draw_left_borders(combined_np, num_views, frame_height, context_mask)

    if col_titles is not None:
        col_widths = [c.shape[1] for c in cols]  # actual width (px) of each concatenated column
        combined_np = _add_column_titles(combined_np, col_widths, col_titles)

    if log_writer is not None:
        log_writer.add_image(
            f"{tb_prefix}/{frame_id}",
            combined_np / 255.0,
            epoch_step,
            dataformats="HWC",
        )

    save_root = str(output_dir or "").strip()
    if save_root:
        save_dir = os.path.join(save_root, *tb_prefix.split("/"))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{frame_id}_epoch{epoch}_concat.jpg")
        Image.fromarray(combined_np.astype(np.uint8)).save(save_path)
        return save_path
    return None