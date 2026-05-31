"""Metric helpers for OccRAE token workflows."""

from __future__ import annotations

from typing import Dict, List

import torch
import torchmetrics

from occrae.util import prepare_batch_inputs


_EVAL_LOSS_KEYS = (
	"LossRecon",
	"LossPointmap_PredVsGT", "LossDepth_PredVsGT", "LossRaymap_PredVsGT",
	"LossPointmap_OrigVsGT", "LossDepth_OrigVsGT", "LossRaymap_OrigVsGT",
	# Autoregressive rollout: encoder still uses GT pairs (z_t = encode(GT_{t-1}, GT_t))
	# but the decoder feeds back its own previous prediction. For surround mode the
	# N cameras at each timestep are concatenated into one joint sequence.
	"LossRecon_AR",
	"LossPointmap_PredVsGT_AR", "LossDepth_PredVsGT_AR", "LossRaymap_PredVsGT_AR",
)


class DeltaTokEvalMetric(torchmetrics.Metric):
	"""Weighted-mean eval losses with automatic distributed reduction."""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		for key in _EVAL_LOSS_KEYS:
			self.add_state(f"sum_{key}", default=torch.tensor(0.0), dist_reduce_fx="sum")
		self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

	def update(self, batch_size: int, **losses: float) -> None:
		for key, value in losses.items():
			getattr(self, f"sum_{key}").add_(value * batch_size)
		self.count += batch_size

	def compute(self) -> dict[str, torch.Tensor]:
		n = self.count.float().clamp(min=1)
		return {key: getattr(self, f"sum_{key}") / n for key in _EVAL_LOSS_KEYS}


@torch.no_grad()
def sample_batch_tokens(
	batch: List[Dict[str, object]],
	*,
	model,
	sampler_fn,
	cond_num: int,
	camera_channels: int,
	device: torch.device,
	latent_dtype: torch.dtype,
) -> Dict[str, float]:
	x1_cond, x1_all, model_kwargs, batch_size, num_views = prepare_batch_inputs(
		batch,
		cond_num=cond_num,
		camera_channels=camera_channels,
		device=device,
		latent_dtype=latent_dtype,
	)
	init_noise = torch.randn_like(x1_all)
	init_state = torch.cat([x1_cond, init_noise], dim=1)
	sampled = sampler_fn(init_state, model, **model_kwargs)[-1]
	channels = x1_all.shape[1]
	pred = sampled[:, channels:]

	pred_5d = pred.reshape(batch_size, num_views, *pred.shape[1:])
	tgt_5d = x1_all.reshape(batch_size, num_views, *x1_all.shape[1:])
	metric_sums = {"sample_mse": 0.0, "sample_ref_mse": 0.0, "sample_tgt_mse": 0.0}
	metric_counts = {"sample_mse": 0.0, "sample_ref_mse": 0.0, "sample_tgt_mse": 0.0}
	diff_sq = (pred_5d.float() - tgt_5d.float()).pow(2)
	view_mse = diff_sq.mean(dim=(2, 3, 4))

	metric_sums["sample_mse"] += float(view_mse.sum().item())
	metric_counts["sample_mse"] += float(view_mse.numel())

	ref_mse = view_mse[:, :cond_num]
	tgt_mse = view_mse[:, cond_num:]
	metric_sums["sample_ref_mse"] += float(ref_mse.sum().item())
	metric_counts["sample_ref_mse"] += float(ref_mse.numel())
	metric_sums["sample_tgt_mse"] += float(tgt_mse.sum().item())
	metric_counts["sample_tgt_mse"] += float(tgt_mse.numel())

	metrics = {
		key: metric_sums[key] / metric_counts[key]
		for key in metric_sums
		if metric_counts[key] > 0
	}
	return pred_5d, tgt_5d, metrics


@torch.no_grad()
def compute_sampling_metrics(
	batch: List[Dict[str, object]],
	*,
	model,
	sampler_fn,
	cond_num: int,
	camera_channels: int,
	device: torch.device,
	latent_dtype: torch.dtype,
) -> Dict[str, float]:
	_sampled_tokens, _target_tokens, metrics = sample_batch_tokens(
		batch,
		model=model,
		sampler_fn=sampler_fn,
		cond_num=cond_num,
		camera_channels=camera_channels,
		device=device,
		latent_dtype=latent_dtype,
	)
	return metrics
