"""Training orchestration for OccRAE token flow."""

from __future__ import annotations

import argparse
import contextlib
import logging
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from occrae.dataset.occrae_tokens import (
	OccRAETokenDataset,
	ProcessedRootBatchSampler,
	collate_identity,
)
from occrae.flow_matching import flow_sampler_euler
from occrae.loss import compute_batch_losses
from occrae.metric import sample_batch_tokens
from stage2.transport import Sampler
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.train_utils import create_transport, requires_grad, update_ema


def _to_dict(config_section) -> Dict[str, object]:
	if config_section is None:
		return {}
	if OmegaConf.is_config(config_section):
		return OmegaConf.to_container(config_section, resolve=True)
	return dict(config_section)


def _setup_logger(log_path: Path, *, is_main_process: bool) -> logging.Logger:
	logger = logging.getLogger("occrae_token_flow")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	if is_main_process:
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)
		logger.addHandler(stream_handler)
		file_handler = logging.FileHandler(log_path)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	else:
		logger.addHandler(logging.NullHandler())
	logger.propagate = False
	return logger


class OccRAETrainerOld:
	def __init__(
		self,
		*,
		args: argparse.Namespace,
		cfg,
		device: torch.device,
		rank: int,
		world_size: int,
		distributed: bool,
	) -> None:
		self.args = args
		self.cfg = cfg
		self.device = device
		self.rank = rank
		self.world_size = world_size
		self.distributed = distributed
		self.is_main_process = rank == 0

		self.model_config = cfg.get("stage_2")
		if self.model_config is None:
			raise ValueError("Config must define a stage_2 section.")
		self.transport_config = _to_dict(cfg.get("transport"))
		self.sampler_config = _to_dict(cfg.get("sampler"))
		self.training_cfg = _to_dict(cfg.get("training"))
		self.validation_cfg = _to_dict(cfg.get("validation"))
		self.dataset_cfg = _to_dict(cfg.get("dataset"))
		self.model_params = _to_dict(self.model_config.get("params", {}))
		self.algorithm = str(self.training_cfg.get("algorithm", "transport")).lower()

		self._configure_runtime()
		self._set_seed()
		self._setup_experiment()

		self.train_loader: Optional[DataLoader] = None
		self.train_sampler = None
		self.val_loader: Optional[DataLoader] = None
		self.steps_per_epoch = 0

		self.model = None
		self.ema = None
		self.opt = None
		self.sched = None
		self.transport = None
		self.eval_sampler = None

		self.train_steps = 0
		self.start_epoch = 0
		self.micro_batches_to_skip = 0

		self.running_loss = 0.0
		self.running_ref_loss = 0.0
		self.running_tgt_loss = 0.0
		self.log_steps = 0
		self.start_time = time()

	def _configure_runtime(self) -> None:
		if bool(self.model_params.get("use_prope", False)):
			raise ValueError("Token-only training does not support use_prope=true.")
		if not bool(self.model_params.get("predict_cls", False)):
			raise ValueError("stage_2.params.predict_cls must be true for packed layer-18 tokens.")
		if not bool(self.model_params.get("is_concat_mode", False)):
			raise ValueError("stage_2.params.is_concat_mode must be true for token conditioning.")

		self.cond_num = int(self.dataset_cfg.get("cond_num", 2))
		if self.cond_num != 2:
			raise ValueError(f"This pipeline is fixed to cond_num=2, got {self.cond_num}")
		self.camera_channels = int(self.model_params.get("cam_in_channels", 1))
		if self.camera_channels < 1:
			raise ValueError(f"cam_in_channels must be >= 1, got {self.camera_channels}")

		self.global_seed = int(self.training_cfg.get("global_seed", 0))
		self.seed = self.global_seed + self.rank

		self.global_batch_size = int(self.training_cfg.get("global_batch_size", 1))
		self.grad_accum_steps = int(self.training_cfg.get("grad_accum_steps", 1))
		self.num_epochs = int(self.training_cfg.get("epochs", 100))
		self.num_workers = int(self.training_cfg.get("num_workers", 4))
		self.ema_decay = float(self.training_cfg.get("ema_decay", 0.9999))
		self.log_every = int(self.training_cfg.get("log_every", 100))
		self.clip_grad = float(self.training_cfg.get("clip_grad", 1.0))
		if self.grad_accum_steps < 1:
			raise ValueError("grad_accum_steps must be >= 1")
		if self.global_batch_size % (self.world_size * self.grad_accum_steps) != 0:
			raise ValueError(
				f"global_batch_size={self.global_batch_size} must be divisible by "
				f"world_size*grad_accum_steps={self.world_size * self.grad_accum_steps}"
			)
		self.micro_batch_size = self.global_batch_size // (self.world_size * self.grad_accum_steps)
		if self.micro_batch_size < 1:
			raise ValueError("micro_batch_size must be >= 1")

		self.use_bf16 = self.args.precision == "bf16"
		if self.device.type == "cuda" and self.use_bf16 and not torch.cuda.is_bf16_supported():
			raise ValueError("bf16 precision requested, but the current CUDA device does not support it.")
		self.latent_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
		if self.device.type == "cuda":
			self.autocast_context = lambda: torch.autocast(  # noqa: E731
				device_type="cuda",
				dtype=torch.bfloat16,
				enabled=self.use_bf16,
			)
		else:
			self.autocast_context = contextlib.nullcontext

		self.train_feature_root = self.resolve_feature_root(self.dataset_cfg.get("train_feature_path"))
		if self.train_feature_root is None:
			raise ValueError("dataset.train_feature_path must be set in the YAML config")
		self.val_feature_root = self.resolve_feature_root(self.dataset_cfg.get("val_feature_path"))

		self.ckpt_every = int(self.validation_cfg.get("ckpt_every", self.training_cfg.get("ckpt_every", 0)))
		self.val_every = int(self.validation_cfg.get("sample_every", self.training_cfg.get("sample_every", 0)))
		self.val_num_batches = self.validation_cfg.get("val_num_batches")
		self.val_num_batches = None if self.val_num_batches is None else int(self.val_num_batches)
		self.val_t_override = self.validation_cfg.get("t_override", 0.5)
		self.val_t_override = None if self.val_t_override is None else float(self.val_t_override)
		self.save_latents = bool(self.validation_cfg.get("save_latents", False))

	def _set_seed(self) -> None:
		random.seed(self.seed)
		torch.manual_seed(self.seed)
		if self.device.type == "cuda":
			torch.cuda.manual_seed(self.seed)

	def _setup_experiment(self) -> None:
		experiment_name = f"{self.args.run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
		self.experiment_dir = Path(self.args.results_dir).expanduser().resolve() / experiment_name
		self.checkpoint_dir = self.experiment_dir / "checkpoints"
		self.validation_latents_dir = self.experiment_dir / "validation_latents"
		if self.is_main_process:
			self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
			if self.save_latents:
				self.validation_latents_dir.mkdir(parents=True, exist_ok=True)
		if self.distributed:
			dist.barrier()
		self.logger = _setup_logger(self.experiment_dir / "log.txt", is_main_process=self.is_main_process)
		self.logger.info(f"Experiment directory: {self.experiment_dir}")
		self.logger.info(
			f"Seed={self.seed}, world_size={self.world_size}, micro_batch_size={self.micro_batch_size}"
		)
		if self.save_latents:
			self.logger.info(f"Validation latents enabled: dir={self.validation_latents_dir}")

	@staticmethod
	def resolve_feature_root(config_value: Optional[str]) -> Optional[str]:
		if config_value in (None, ""):
			return None
		return str(Path(config_value).expanduser().resolve())

	def unwrap_model(self):
		return self.model.module if isinstance(self.model, DDP) else self.model

	def reduce_mean(self, value: float) -> float:
		tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
		if self.distributed:
			dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
			tensor /= self.world_size
		return float(tensor.item())

	def reduce_weighted_metrics(
		self,
		metric_sums: Dict[str, float],
		metric_counts: Dict[str, float],
	) -> Dict[str, float]:
		reduced: Dict[str, float] = {}
		for key, value in metric_sums.items():
			value_tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
			count_tensor = torch.tensor(metric_counts[key], device=self.device, dtype=torch.float64)
			if self.distributed:
				dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
				dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
			reduced[key] = float((value_tensor / count_tensor).item()) if count_tensor.item() > 0 else 0.0
		return reduced

	def build_dataloader(
		self,
		dataset: OccRAETokenDataset,
		*,
		batch_size: int,
		shuffle: bool,
		drop_last: bool,
	) -> Tuple[DataLoader, object]:
		sampler = ProcessedRootBatchSampler(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			drop_last=drop_last,
			seed=self.global_seed,
			rank=self.rank,
			world_size=self.world_size,
		)
		loader = DataLoader(
			dataset,
			batch_sampler=sampler,
			num_workers=self.num_workers,
			pin_memory=True,
			collate_fn=collate_identity,
		)
		return loader, sampler

	def setup_data(self) -> None:
		train_dataset = OccRAETokenDataset(
			self.train_feature_root,
			cond_num=self.cond_num,
			processed_roots=self.dataset_cfg.get("processed_roots"),
			max_samples=int(self.dataset_cfg.get("max_train_samples", -1)),
		)
		self.train_loader, self.train_sampler = self.build_dataloader(
			train_dataset,
			batch_size=self.micro_batch_size,
			shuffle=True,
			drop_last=True,
		)
		self.logger.info(f"Loaded training dataset with {len(train_dataset)} samples")

		if self.val_feature_root is not None:
			val_dataset = OccRAETokenDataset(
				self.val_feature_root,
				cond_num=self.cond_num,
				processed_roots=self.dataset_cfg.get("val_processed_roots", self.dataset_cfg.get("processed_roots")),
				max_samples=int(self.dataset_cfg.get("max_val_samples", -1)),
			)
			self.val_loader, _ = self.build_dataloader(
				val_dataset,
				batch_size=int(self.validation_cfg.get("batch_size", 1)),
				shuffle=False,
				drop_last=False,
			)
			self.logger.info(f"Loaded validation dataset with {len(val_dataset)} samples")

		self.steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
		if self.steps_per_epoch <= 0:
			raise ValueError("Training configuration results in zero optimizer steps per epoch.")

	def setup_model(self) -> None:
		self.model = instantiate_from_config(self.model_config).to(self.device)
		self.ema = deepcopy(self.model).to(self.device)
		requires_grad(self.ema, False)
		update_ema(self.ema, self.model, decay=0.0)
		param_count = sum(param.numel() for param in self.model.parameters())
		trainable_param_count = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
		self.logger.info(
			f"Model parameters: total={param_count:,}, trainable={trainable_param_count:,}"
		)

		opt_state = None
		sched_state = None
		if self.args.ckpt is not None:
			checkpoint = torch.load(self.args.ckpt, map_location="cpu", weights_only=False)
			self.model.load_state_dict(checkpoint["model"])
			self.ema.load_state_dict(checkpoint["ema"])
			opt_state = checkpoint.get("opt")
			sched_state = checkpoint.get("scheduler")
			self.train_steps = int(checkpoint.get("train_steps", 0))
			self.start_epoch = self.train_steps // self.steps_per_epoch
			self.micro_batches_to_skip = int((self.train_steps % self.steps_per_epoch) * self.grad_accum_steps)
			self.logger.info(
				f"Resuming from {self.args.ckpt}: train_steps={self.train_steps}, "
				f"start_epoch={self.start_epoch}, skip_micro_batches={self.micro_batches_to_skip}"
			)

		if self.distributed:
			self.model = DDP(self.model, device_ids=[self.device.index], gradient_as_bucket_view=False)

		self.opt, opt_msg = build_optimizer(self.unwrap_model().parameters(), self.training_cfg)
		if opt_state is not None:
			self.opt.load_state_dict(opt_state)
		self.sched, sched_msg = build_scheduler(self.opt, self.steps_per_epoch, self.training_cfg, sched_state)
		self.logger.info(opt_msg)
		self.logger.info(sched_msg)

	def setup_transport(self) -> None:
		if self.algorithm == "flow_matching":
			self.logger.info("Using flow-matching algorithm (VATIX-style)")
			sampler_params = dict(self.sampler_config.get("params", {}))
			num_steps = int(sampler_params.get("num_steps", 50))
			# We keep the sampler config surface under sampler.params so validation
			# can swap between transport and flow-matching without changing the rest
			# of the trainer wiring.
			sampling_method = str(sampler_params.get("sampling_method", "euler")).lower()
			scheduler_mode = str(sampler_params.get("scheduler_mode", "cosine")).lower()
			alpha = float(sampler_params.get("alpha", 0.5))
			if sampling_method != "euler":
				raise ValueError(
					f"Flow-matching sampler only supports sampling_method='euler', got {sampling_method!r}"
				)
			if bool(sampler_params.get("reverse", False)):
				raise ValueError("Flow-matching sampler does not support reverse=True")

			def fm_sampler(init_x, m, **kw):
				# sample_batch_tokens initializes the second half with Gaussian noise.
				# For reference views we do not want generated content, so before the
				# rollout starts we copy the clean conditioning latents into the noisy
				# half as the fixed target for those views.
				BV, C2, S, L = init_x.shape
				C = C2 // 2
				num_views = kw.get("total_view")
				if num_views is None:
					raise ValueError("total_view missing from model_kwargs during flow-matching sampling")
				B = BV // int(num_views)
				init_x_res = init_x.clone()
				init_x_5d = init_x_res.view(B, int(num_views), C2, S, L)
				init_x_5d[:, :self.cond_num, C:] = init_x_5d[:, :self.cond_num, :C]
				return flow_sampler_euler(
					m,
					init_x_res,
					num_views=num_views,
					num_steps=num_steps,
					cond_num=self.cond_num,
					scheduler_mode=scheduler_mode,
					alpha=alpha,
					model_kwargs=kw,
				)

			self.eval_sampler = fm_sampler
			self.transport = None  # Not used for FM
			return

		transport_params = dict(self.transport_config.get("params", {}))
		time_dist_shift = float(transport_params.pop("time_dist_shift", 1.0))
		self.transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
		sampler_params = dict(self.sampler_config.get("params", {}))
		sampler_mode = str(self.sampler_config.get("mode", "ODE")).upper()
		transport_sampler = Sampler(self.transport)
		if sampler_mode != "ODE":
			raise ValueError(f"Only ODE sampling is supported for this token trainer, got {sampler_mode}")
		self.eval_sampler = transport_sampler.sample_ode(**sampler_params)

	def _validation_step_dir(self) -> Path:
		return self.validation_latents_dir / f"step_{self.train_steps:07d}"

	def _unpack_tokens_for_payload(self, packed_tokens: torch.Tensor) -> torch.Tensor:
		if packed_tokens.ndim != 4 or packed_tokens.shape[-1] != 1:
			raise ValueError(
				f"Expected packed tokens with shape (V, C, S, 1), got {tuple(packed_tokens.shape)}"
			)
		return packed_tokens.detach().cpu().squeeze(-1).permute(0, 2, 1).contiguous()

	def _save_validation_latent_family(
		self,
		batch: list[Dict[str, object]],
		packed_tokens: torch.Tensor,
	) -> list[Path]:
		if not self.is_main_process:
			return []
		family_dir = self._validation_step_dir()
		saved_paths: list[Path] = []
		for sample, sample_tokens in zip(batch, packed_tokens):
			relpath = Path(str(sample["first_frame_relpath"]))
			out_path = family_dir / relpath.parent / f"{relpath.stem}.pth"
			out_path.parent.mkdir(parents=True, exist_ok=True)
			payload = {
				"1st_frame_relpath": str(sample["first_frame_relpath"]),
				"timesteps": list(sample["timesteps"]),
				"output_resolution": tuple(sample["output_resolution"]),
				"tokens": self._unpack_tokens_for_payload(sample_tokens).to(torch.float32),
			}
			torch.save(payload, out_path)
			saved_paths.append(out_path)
		return saved_paths

	def _save_validation_latents(
		self,
		batch: list[Dict[str, object]],
		sampled_tokens: torch.Tensor,
	) -> list[Path]:
		if not self.save_latents:
			return []
		return self._save_validation_latent_family(batch, sampled_tokens)

	@torch.no_grad()
	def eval_one_epoch(self) -> Dict[str, float]:
		if self.val_loader is None:
			return {}

		self.ema.eval()
		metric_sums = {
			"val_loss": 0.0,
			"val_ref_loss": 0.0,
			"val_tgt_loss": 0.0,
			"val_sample_mse": 0.0,
			"val_sample_ref_mse": 0.0,
			"val_sample_tgt_mse": 0.0,
		}
		metric_counts = {key: 0.0 for key in metric_sums}
		saved_paths: list[Path] = []

		for batch_idx, batch in enumerate(self.val_loader):
			if self.val_num_batches is not None and batch_idx >= self.val_num_batches:
				break
			with self.autocast_context():
				_, loss_metrics = compute_batch_losses(
					batch,
					model=self.ema,
					transport=self.transport,
					cond_num=self.cond_num,
					camera_channels=self.camera_channels,
					device=self.device,
					latent_dtype=self.latent_dtype,
					t_override=self.val_t_override,
					algorithm=self.algorithm,
				)
				sampled_tokens, _target_tokens, sampling_metrics = sample_batch_tokens(
					batch,
					model=self.ema,
					sampler_fn=self.eval_sampler,
					cond_num=self.cond_num,
					camera_channels=self.camera_channels,
					device=self.device,
					latent_dtype=self.latent_dtype,
				)
			for key, value in loss_metrics.items():
				metric_sums[f"val_{key}"] += value
				metric_counts[f"val_{key}"] += 1.0
			for key, value in sampling_metrics.items():
				metric_sums[f"val_{key}"] += value
				metric_counts[f"val_{key}"] += 1.0
			if self.save_latents:
				saved_paths.extend(self._save_validation_latents(batch, sampled_tokens))

		reduced = self.reduce_weighted_metrics(metric_sums, metric_counts)
		if self.save_latents and saved_paths and self.is_main_process:
			self.logger.info(
				f"Saved {len(saved_paths)} validation latent files under {self._validation_step_dir()} "
				f"(example: {saved_paths[0]})"
			)
		self.model.train()
		return reduced

	def save_checkpoint(self, epoch: int) -> None:
		if self.is_main_process:
			checkpoint = {
				"model": self.unwrap_model().state_dict(),
				"ema": self.ema.state_dict(),
				"opt": self.opt.state_dict(),
				"scheduler": self.sched.state_dict(),
				"train_steps": self.train_steps,
				"epoch": epoch,
				"config_name": self.args.config_name,
				"cli_args": vars(self.args),
			}
			checkpoint_path = self.checkpoint_dir / f"{self.train_steps:07d}.pt"
			torch.save(checkpoint, checkpoint_path)
			self.logger.info(f"Saved checkpoint to {checkpoint_path}")
		if self.distributed:
			dist.barrier()

	def train_one_epoch(self, epoch: int) -> None:
		if hasattr(self.train_sampler, "set_epoch"):
			self.train_sampler.set_epoch(epoch)

		self.model.train()
		self.ema.eval()
		self.opt.zero_grad(set_to_none=True)
		accum_counter = 0
		step_loss_accum = 0.0
		step_ref_loss_accum = 0.0
		step_tgt_loss_accum = 0.0

		train_iter = iter(self.train_loader)
		batches_to_skip = self.micro_batches_to_skip if epoch == self.start_epoch else 0
		for _ in range(batches_to_skip):
			next(train_iter)
		self.micro_batches_to_skip = 0

		for batch in train_iter:
			with self.autocast_context():
				loss_tensor, loss_metrics = compute_batch_losses(
					batch,
					model=self.model,
					transport=self.transport,
					cond_num=self.cond_num,
					camera_channels=self.camera_channels,
					device=self.device,
					latent_dtype=self.latent_dtype,
					algorithm=self.algorithm,
				)

			bad_flag = torch.tensor(
				0 if torch.isfinite(loss_tensor.detach()).all() else 1,
				device=self.device,
				dtype=torch.int32,
			)
			if self.distributed:
				dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
			if int(bad_flag.item()) != 0:
				raise RuntimeError(f"Encountered non-finite loss at train_steps={self.train_steps}")

			step_loss_accum += loss_metrics["loss"]
			step_ref_loss_accum += loss_metrics["ref_loss"]
			step_tgt_loss_accum += loss_metrics["tgt_loss"]

			sync_step = (accum_counter + 1) % self.grad_accum_steps == 0
			backward_context = (
				contextlib.nullcontext()
				if sync_step or not isinstance(self.model, DDP)
				else self.model.no_sync()
			)
			with backward_context:
				(loss_tensor / self.grad_accum_steps).backward()

			accum_counter += 1
			if accum_counter < self.grad_accum_steps:
				continue

			if self.clip_grad > 0:
				torch.nn.utils.clip_grad_norm_(self.unwrap_model().parameters(), self.clip_grad)
			self.opt.step()
			self.sched.step()
			update_ema(self.ema, self.unwrap_model(), decay=self.ema_decay)
			self.opt.zero_grad(set_to_none=True)

			self.running_loss += step_loss_accum / self.grad_accum_steps
			self.running_ref_loss += step_ref_loss_accum / self.grad_accum_steps
			self.running_tgt_loss += step_tgt_loss_accum / self.grad_accum_steps
			self.log_steps += 1
			self.train_steps += 1
			accum_counter = 0
			step_loss_accum = 0.0
			step_ref_loss_accum = 0.0
			step_tgt_loss_accum = 0.0

			if self.log_every > 0 and self.train_steps % self.log_every == 0:
				if self.device.type == "cuda":
					torch.cuda.synchronize(self.device)
				elapsed = time() - self.start_time
				steps_per_sec = self.log_steps / elapsed if elapsed > 0 else 0.0
				avg_loss = self.reduce_mean(self.running_loss / self.log_steps)
				avg_ref_loss = self.reduce_mean(self.running_ref_loss / self.log_steps)
				avg_tgt_loss = self.reduce_mean(self.running_tgt_loss / self.log_steps)
				if self.is_main_process:
					self.logger.info(
						f"(step={self.train_steps:07d}) train_loss={avg_loss:.6f}, "
						f"ref_loss={avg_ref_loss:.6f}, tgt_loss={avg_tgt_loss:.6f}, "
						f"steps_per_sec={steps_per_sec:.2f}"
					)
				self.running_loss = 0.0
				self.running_ref_loss = 0.0
				self.running_tgt_loss = 0.0
				self.log_steps = 0
				self.start_time = time()

			if self.ckpt_every > 0 and self.train_steps % self.ckpt_every == 0:
				self.save_checkpoint(epoch)

			if self.val_loader is not None and self.val_every > 0 and self.train_steps % self.val_every == 0:
				val_metrics = self.eval_one_epoch()
				if self.is_main_process:
					self.logger.info(
						f"[validation step={self.train_steps:07d}] "
						+ ", ".join(f"{key}={value:.6f}" for key, value in val_metrics.items())
					)
				if self.distributed:
					dist.barrier()

		if accum_counter != 0:
			raise RuntimeError("Gradient accumulation counter is not zero at epoch end.")

	def run(self) -> None:
		self.setup_data()
		self.setup_model()
		self.setup_transport()

		self.logger.info(f"Training for {self.num_epochs} epochs")
		self.start_time = time()
		for epoch in range(self.start_epoch, self.num_epochs):
			self.train_one_epoch(epoch)

		if self.is_main_process:
			self.logger.info("Done!")
