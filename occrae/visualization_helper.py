import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from occany.utils.helpers import depth2rgb


_COLOR_CONTEXT = np.array([0, 200, 255], dtype=np.uint8)   # cyan
_COLOR_FORECAST = np.array([255, 100, 0], dtype=np.uint8)  # orange
_BORDER_WIDTH = 6


def _draw_left_borders(combined_np, num_views, frame_height, is_context):
    """Draw colored left border lines on each frame row to indicate context vs forecast."""
    for t in range(num_views):
        color = _COLOR_CONTEXT if is_context[t] else _COLOR_FORECAST
        y0 = t * frame_height
        y1 = y0 + frame_height
        combined_np[y0:y1, :_BORDER_WIDTH, :] = color


def _add_column_titles(img_np, num_cols, titles):
    """Prepend a title bar above each column of the concatenated image."""
    title_h = 36
    H, W = img_np.shape[:2]
    panel_w = W // num_cols
    header = np.full((title_h, W, 3), 30, dtype=np.uint8)
    pil_header = Image.fromarray(header)
    draw = ImageDraw.Draw(pil_header)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    for i, title in enumerate(titles):
        x = i * panel_w + panel_w // 2
        draw.text((x, title_h // 2), str(title), fill=(255, 255, 255), anchor="mm", font=font)
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
):
    """Log a side-by-side validation sample for OccRAE reconstructions."""
    gt_img = batch["imgs"][batch_idx].detach().cpu().permute(0, 2, 3, 1)
    pred_depth = decoded["depth"][batch_idx].detach().float().cpu()

    if view_order is not None:
        index = torch.as_tensor(view_order, dtype=torch.long)
        gt_img = gt_img[index]
        pred_depth = pred_depth[index]
    num_views = gt_img.shape[0]

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

    all_panels = [gt_img, pred_depth_color] + (extra_panels or [])
    cols = [torch.cat([panel[t] for t in range(num_views)], dim=0) for panel in all_panels]
    combined_np = torch.cat(cols, dim=1).numpy()

    if context_mask is not None:
        frame_height = combined_np.shape[0] // num_views
        _draw_left_borders(combined_np, num_views, frame_height, context_mask)

    if col_titles is not None:
        combined_np = _add_column_titles(combined_np, len(all_panels), col_titles)

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