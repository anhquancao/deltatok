"""Shared helpers for OccRAE token workflows."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def build_condition_latents(tokens_5d: torch.Tensor, cond_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
	batch_size, num_views, channels, seq_len, last_dim = tokens_5d.shape
	if cond_num != 2:
		raise ValueError(f"This pipeline is fixed to cond_num=2, got {cond_num}")
	if num_views <= cond_num:
		raise ValueError(f"Need num_views > cond_num, got num_views={num_views}, cond_num={cond_num}")

	x1_all = tokens_5d.reshape(batch_size * num_views, channels, seq_len, last_dim)
	x1_cond_5d = torch.zeros_like(tokens_5d)
	x1_cond_5d[:, :cond_num] = tokens_5d[:, :cond_num]
	x1_cond = x1_cond_5d.reshape(batch_size * num_views, channels, seq_len, last_dim)
	return x1_cond, x1_all


def build_zero_camera_embedding(
	*,
	batch_size: int,
	num_views: int,
	output_resolution: Tuple[int, int],
	channels: int,
	device: torch.device,
	dtype: torch.dtype,
) -> torch.Tensor:
	width, height = output_resolution
	return torch.zeros(
		batch_size * num_views,
		channels,
		height,
		width,
		device=device,
		dtype=dtype,
	)


def _normalize_resolution(value: Sequence[int]) -> Tuple[int, int]:
	if len(value) != 2:
		raise ValueError(f"Expected output_resolution to have length 2, got {value!r}")
	width, height = int(value[0]), int(value[1])
	if width <= 0 or height <= 0:
		raise ValueError(f"Invalid output_resolution={value!r}")
	return width, height


def prepare_batch_inputs(
	samples: List[Dict[str, object]],
	*,
	cond_num: int,
	camera_channels: int,
	device: torch.device,
	latent_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], int, int]:
	if not samples:
		raise ValueError("Received an empty batch")

	first_resolution = _normalize_resolution(samples[0]["output_resolution"])
	for sample in samples[1:]:
		resolution = _normalize_resolution(sample["output_resolution"])
		if resolution != first_resolution:
			raise ValueError(
				f"Mixed output_resolution inside one batch is not supported: "
				f"got {first_resolution} and {resolution}"
			)

	tokens = torch.stack([sample["tokens"] for sample in samples], dim=0)
	tokens = tokens.to(device=device, dtype=latent_dtype, non_blocking=True)
	batch_size, num_views = int(tokens.shape[0]), int(tokens.shape[1])
	x1_cond, x1_all = build_condition_latents(tokens, cond_num=cond_num)
	camera_embedding = build_zero_camera_embedding(
		batch_size=batch_size,
		num_views=num_views,
		output_resolution=first_resolution,
		channels=camera_channels,
		device=device,
		dtype=latent_dtype,
	)
	model_kwargs = {
		"camera_embedding": camera_embedding,
		"total_view": num_views,
		"cond_num": cond_num,
		"is_concat_mode": True,
		"ref_cond": x1_cond,
		"x1_global": x1_all,
		"freeze_cond": False,
	}
	return x1_cond, x1_all, model_kwargs, batch_size, num_views


# Backward-compatible alias for older imports.
prepare_group_inputs = prepare_batch_inputs


def resize_positional_embeddings(model, checkpoint_state):
	new_state = checkpoint_state.copy()

	# ---- Temporal embedding ----
	if "temporal_pos.weight" in checkpoint_state:
		old_temporal = checkpoint_state["temporal_pos.weight"]   # [T_old, D]
		new_temporal = model.temporal_pos.weight                  # [T_new, D]

		if old_temporal.shape[0] != new_temporal.shape[0]:
			print(f"Resizing temporal_pos: {old_temporal.shape[0]} -> {new_temporal.shape[0]}")

			old = old_temporal.unsqueeze(0).permute(0, 2, 1)  # [1, D, T_old]
			new = F.interpolate(old, size=new_temporal.shape[0], mode="linear", align_corners=False)
			new = new.permute(0, 2, 1).squeeze(0)             # [T_new, D]

			new_state["temporal_pos.weight"] = new

	# ---- Spatial embedding ----
	if "spatial_pos.weight" in checkpoint_state:
		old_spatial = checkpoint_state["spatial_pos.weight"]    # [N_old, D]
		new_spatial = model.spatial_pos.weight                  # [N_new, D]

		if old_spatial.shape[0] != new_spatial.shape[0]:
			print(f"Resizing spatial_pos: {old_spatial.shape[0]} -> {new_spatial.shape[0]}")
			print(old_spatial.shape, new_spatial.shape)
			D = old_spatial.shape[1]

			old_h, old_w = 20, 26
			new_h, new_w = 20, 26*4

			old = old_spatial.reshape(old_h, old_w, D)
			new = old.repeat(1, 4, 1).reshape(new_h * new_w, D)

			# The interpolation-based resizing is commented out as it may cause performance drop.
			# The tile-based resizing preserves the original values without interpolation artifacts.
			# old = old_spatial.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)  # [1,D,H,W]
			# new = F.interpolate(old, size=(new_h, new_w), mode="bicubic", align_corners=False)
			# new = new.permute(0, 2, 3, 1).reshape(new_h * new_w, D)

			new_state["spatial_pos.weight"] = new

	return new_state
