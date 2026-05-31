"""Loss helpers for OccRAE token training."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from occrae.flow_matching import flow_loss
from occrae.util import prepare_batch_inputs


def compute_batch_losses(
	batch: List[Dict[str, object]],
	*,
	model,
	transport,
	cond_num: int,
	camera_channels: int,
	device: torch.device,
	latent_dtype: torch.dtype,
	t_override: Optional[float] = None,
	algorithm: str = "transport",
) -> Tuple[torch.Tensor, Dict[str, float]]:
	x1_cond, x1_all, model_kwargs, batch_size, num_views = prepare_batch_inputs(
		batch,
		cond_num=cond_num,
		camera_channels=camera_channels,
		device=device,
		latent_dtype=latent_dtype,
	)

	if algorithm == "flow_matching":
		total_loss, loss_dict = flow_loss(
			model,
			x1_all,
			model_kwargs,
			cond_num=cond_num,
			num_views=num_views,
			device=device,
			latent_dtype=latent_dtype,
			t_override=t_override,
		)
	else:
		# Default to transport
		loss_dict = transport.training_multiview_losses(
			model,
			x1_cond,
			num_views,
			cond_num,
			model_kwargs=model_kwargs,
			t_override=t_override,
		)
		total_loss = loss_dict["loss"].mean()

	ref_loss = loss_dict.get("ref_loss", loss_dict["loss"]).mean()
	tgt_loss = loss_dict.get("tgt_loss", loss_dict["loss"]).mean()

	total_view_items = batch_size * num_views
	total_ref_items = batch_size * cond_num
	total_tgt_items = batch_size * (num_views - cond_num)
	metric_sums = {
		"loss": total_loss.detach().float().item() * total_view_items,
		"ref_loss": ref_loss.detach().float().item() * total_ref_items,
		"tgt_loss": tgt_loss.detach().float().item() * total_tgt_items,
	}
	metric_means = {
		"loss": metric_sums["loss"] / total_view_items,
		"ref_loss": metric_sums["ref_loss"] / total_ref_items,
		"tgt_loss": metric_sums["tgt_loss"] / total_tgt_items,
	}
	return total_loss, metric_means
