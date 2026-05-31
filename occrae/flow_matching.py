"""Flow-matching objective for OccRAE token training."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _sampling_progress(
	step_idx: int,
	*,
	num_steps: int,
	scheduler_mode: str,
	device: torch.device,
	dtype: torch.dtype,
) -> torch.Tensor:
	# VATIX does not have to march in equally spaced t values.  The schedule controls
	# how aggressively we move early vs late in the rollout.
	progress = torch.tensor(step_idx / num_steps, device=device, dtype=dtype)
	if scheduler_mode == "cosine":
		return 1.0 - torch.cos(progress * torch.pi / 2)
	if scheduler_mode == "square":
		return torch.sqrt(progress)
	if scheduler_mode == "linear":
		return progress
	raise ValueError(f"Unsupported flow-matching scheduler_mode={scheduler_mode!r}")


def flow_noising(
	x1_all: torch.Tensor,
	*,
	cond_num: int,
	num_views: int,
	device: torch.device,
	latent_dtype: torch.dtype,
	t_override: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Apply flow-matching noising to a batch of tokens.
	
	Args:
		x1_all: GT tokens (B * V, C, S, 1)
		cond_num: Number of reference frames.
		num_views: Total number of frames in the sequence.
		device: Device to use.
		latent_dtype: Target dtype for latents.
		t_override: Optional fixed timestep for validation.
		
	Returns:
		t: Timesteps (B * V,)
		xt: Noisy tokens (B * V, C, S, 1)
		ut: Target velocity (B * V, C, S, 1)
	"""
	BV, C, S, L = x1_all.shape
	B = BV // num_views
	
	# 1. Sample time t
	if t_override is not None:
		t_b = torch.full((B,), t_override, device=device, dtype=latent_dtype)
	else:
		t_b = torch.rand((B,), device=device, dtype=latent_dtype)
	
	# Broadcast t to all views
	# Training uses one sampled time per sequence, then repeats it across views.
	t = t_b.unsqueeze(1).expand(B, num_views).reshape(BV)
	
	# 2. Sample noise x0
	x0 = torch.randn_like(x1_all)
	
	# 3. Compute linear interpolation: xt = (1-t)x0 + t*x1
	t_exp = t.view(BV, 1, 1, 1)
	xt = (1.0 - t_exp) * x0 + t_exp * x1_all
	
	# 4. Target velocity: ut = x1 - x0
	ut = x1_all - x0
	
	# 5. Fixed context for reference frames
	# For v < cond_num, xt should be x1 (clean) and ut should be 0.
	# We reshape to (B, V, ...) to apply the mask.
	xt_5d = xt.view(B, num_views, C, S, L)
	ut_5d = ut.view(B, num_views, C, S, L)
	x1_5d = x1_all.view(B, num_views, C, S, L)
	
	# Condition frames: clean tokens, zero velocity target
	xt_5d[:, :cond_num] = x1_5d[:, :cond_num]
	ut_5d[:, :cond_num] = 0.0
	
	return t, xt, ut


def flow_loss(
	model,
	x1_all: torch.Tensor,
	model_kwargs: Dict[str, torch.Tensor],
	*,
	cond_num: int,
	num_views: int,
	device: torch.device,
	latent_dtype: torch.dtype,
	t_override: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
	"""
	Compute flow-matching loss for OccRAE tokens.
	"""
	t, xt, ut = flow_noising(
		x1_all,
		cond_num=cond_num,
		num_views=num_views,
		device=device,
		latent_dtype=latent_dtype,
		t_override=t_override,
	)
	
	# In concat mode the model input is split into two channel groups:
	#   [:C]   = clean conditioning features for reference views
	#   [C:2C] = current noised state that the model should denoise / transport
	# The target remains the velocity on the second half only.
	ref_cond = model_kwargs.get("ref_cond")
	if ref_cond is None:
		raise ValueError("ref_cond must be provided in model_kwargs for flow_loss")
	
	model_input = torch.cat([ref_cond, xt], dim=1)  # (BV, 2C, S, 1)
	
	# The DiT receives total_view as a positional argument.  Remove the duplicate
	# copy from kwargs so the call site matches the transport path contract.
	m_kwargs = {k: v for k, v in model_kwargs.items() if k != "total_view"}
	
	model_output = model(model_input, t, num_views, **m_kwargs)
	
	# Loss computation
	diff_sq = (model_output - ut) ** 2
	
	# Reshape for separate ref/tgt metrics
	diff_sq_5d = diff_sq.view(-1, num_views, *diff_sq.shape[1:])
	
	ref_loss = diff_sq_5d[:, :cond_num].mean()
	tgt_loss = diff_sq_5d[:, cond_num:].mean()
	total_loss = diff_sq.mean()
	
	terms = {
		"loss": total_loss,
		"ref_loss": ref_loss,
		"tgt_loss": tgt_loss,
		"pred": model_output,
	}
	
	return total_loss, terms


@torch.no_grad()
def flow_sampler_euler(
	model,
	init_state: torch.Tensor,
	num_views: int | None = None,
	num_steps: int = 50,
	cond_num: int | None = None,
	scheduler_mode: str = "cosine",
	alpha: float = 0.5,
	model_kwargs: Dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor]:
	"""
	VATIX-style Euler sampler for flow-matching.

	The state layout is the same as training concat mode:
	  - first C channels: fixed clean conditioning features
	  - last  C channels: evolving target state

	Only the second half is integrated.  Reference views are clamped to the clean
	conditioning latents at every step so validation sampling matches the training
	assumption that context views are fixed inputs, not generated outputs.
	
	Args:
		model: Model that predicts velocity.
		init_state: Initial state (BV, 2C, S, 1) - [ref_cond | noise].
		num_views: Total number of frames. If None, tries to get from model_kwargs['total_view'].
		num_steps: Number of Euler steps.
		cond_num: Number of context views to keep fixed.
		scheduler_mode: Integration schedule, one of linear, square, or cosine.
		alpha: Per-view timestep offset strength.
		model_kwargs: Additional model arguments.
		
	Returns:
		List of states during integration.
	"""
	if num_steps < 1:
		raise ValueError(f"num_steps must be >= 1, got {num_steps}")
	if model_kwargs is None:
		m_kwargs = {}
	else:
		m_kwargs = {k: v for k, v in model_kwargs.items() if k != "total_view"}
	
	if num_views is None:
		if model_kwargs is not None and "total_view" in model_kwargs:
			num_views = int(model_kwargs["total_view"])
		else:
			# Fallback if possible, but num_views is usually needed for positional embedding in DiT
			raise ValueError("num_views must be provided either as an argument or in model_kwargs['total_view']")
	if cond_num is None:
		if model_kwargs is not None and "cond_num" in model_kwargs:
			cond_num = int(model_kwargs["cond_num"])
		else:
			cond_num = 0
	if not 0 <= cond_num < int(num_views):
		raise ValueError(f"cond_num must satisfy 0 <= cond_num < num_views, got cond_num={cond_num}, num_views={num_views}")
	
	device = init_state.device
	dtype = init_state.dtype
	BV, C2, S, L = init_state.shape
	C = C2 // 2
	B = BV // num_views
	# VATIX offsets the timestep per frame/view so some views progress slightly
	# later than others during rollout.  alpha=0 collapses back to a shared schedule.
	view_offsets = 1.0 + torch.linspace(1.0, 0.0, int(num_views), device=device, dtype=dtype) * alpha
	xt = init_state.clone()
	xt_5d = xt.view(B, int(num_views), C2, S, L)
	# Save the clean reference targets once, then restore them after every step.
	ref_target = xt_5d[:, :cond_num, :C].clone() if cond_num > 0 else None
	states = [xt.clone()]
	
	for i in range(num_steps):
		progress = _sampling_progress(
			i,
			num_steps=num_steps,
			scheduler_mode=scheduler_mode,
			device=device,
			dtype=dtype,
		)
		progress_next = _sampling_progress(
			i + 1,
			num_steps=num_steps,
			scheduler_mode=scheduler_mode,
			device=device,
			dtype=dtype,
		)
		t_curr = torch.clamp(progress * view_offsets, 0.0, 1.0).expand(B, int(num_views)).clone()
		t_next = torch.clamp(progress_next * view_offsets, 0.0, 1.0).expand(B, int(num_views)).clone()
		if cond_num > 0:
			# Context views should stay at t=1 because they are already clean.
			t_curr[:, :cond_num] = 1.0
			t_next[:, :cond_num] = 1.0
			# Keep the evolving half for reference views identical to the clean input.
			xt_5d[:, :cond_num, C:] = ref_target
		t = t_curr.reshape(BV)
		
		v = model(xt, t, num_views, **m_kwargs)
		if cond_num > 0:
			# Even if the model predicts something on reference views, we do not use it.
			v.view(B, int(num_views), C, S, L)[:, :cond_num] = 0.0
		
		# Euler update: x_{t_next} = x_t + (t_next - t_curr) * v.
		delta_t = (t_next - t_curr).reshape(BV, 1, 1, 1)
		xt[:, C:] = xt[:, C:] + delta_t * v
		if cond_num > 0:
			# Re-apply the reference clamp to avoid numerical drift in context views.
			xt_5d[:, :cond_num, C:] = ref_target
		states.append(xt.clone())
		
	return states
