# Trainer for DeltaTok delta-token Flow Matching
import datetime
import os
import time
from collections import deque
from contextlib import nullcontext, contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from occrae.metric_logger import (  # SLURM-friendly line-per-print logging (no tqdm carriage returns)
    MetricLogger, SmoothedValue, rank_cpu_mem_mb, rank_cpu_budget_mb, total_gpu_mem_mb,
)

from occrae.abstract_trainer import Trainer
from occrae.network.efficient_transformer import Transformer
from occrae.network.ema import EMA
from occrae.deltatok_shared import DeltaTokSharedMixin
from occrae.visualization_helper import _log_viz_sample
from occany.datasets import get_data_loader
from occany.utils.helpers import depth2rgb
from occany.loss import PointmapLoss, DepthLosses, RaymapLoss
from occrae.generation_helper import flow_euler_sample


# Fixed eval-loss key order so every rank reduces the same-sized vector even
# when some ranks see batches without GT (missing keys contribute 0).
_EVAL_KEYS = (
    "LossPointmap", "LossDepth", "LossRaymap",                # sampled-delta rollout vs GT (forecast frames)
    "LossPointmap_tok", "LossDepth_tok", "LossRaymap_tok",    # GT-delta rollout (tokenizer upper bound)
)


class DeltaTokFlowMatchingTrainer(DeltaTokSharedMixin, Trainer):
    def __init__(self, args, cfg, device, rank, world_size, distributed):
        """ Initialize model, optimizer, loss function, and data loaders."""
        args.is_master = rank == 0
        args.writer_log = str(cfg.training.get("writer_log", ""))
        super().__init__(args)
        print(f"Init DeltaTok Flow Matching on [GPU{rank}]")

        self.args = args
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.is_master = (self.rank == 0)

        # Gradient accumulation is derived, not configured:
        # effective_bsize = world_size * bsize * grad_cum.
        bsize = int(self.cfg.training.bsize)
        effective_bsize = int(self.cfg.training.effective_bsize)
        denom = self.world_size * bsize
        assert effective_bsize % denom == 0, (
            f"training.effective_bsize={effective_bsize} must be divisible by "
            f"world_size * bsize = {self.world_size} * {bsize} = {denom}"
        )
        self.grad_cum = effective_bsize // denom
        if self.is_master:
            print(f"effective_bsize={effective_bsize} = {self.world_size} rank(s) "
                  f"x bsize {bsize} x grad_cum {self.grad_cum}")

        self._ema_state = None

        # Set up automatic mixed precision BEFORE the frozen models: the mixin
        # helpers (_extract_pair_feats, _build_occ_rae) read it / cfg dtype.
        if self.device.type != 'cpu' and self.cfg.training.dtype == "bfloat16":
            self.autocast = autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        # Temporal slot capacity of the flow transformer: max T-1 frame transitions.
        self.num_views = self.cfg.model["num_views"]
        print(f"Number of temporal slots (max T-1): {self.num_views}")

        # Frozen OccRAE encoder/decoder (mixin; never DDP-wrapped, no optimizer).
        self._build_occ_rae()
        backbone = self.occ_rae.model._get_pretrained_backbone()
        # Strip CLS + register tokens from OccRAE features before DeltaTok.
        self._num_prefix_tokens = 1 + int(backbone.num_register_tokens)
        self._patch_size = int(backbone.patch_size)

        # Frozen DeltaTok tokenizer (never DDP-wrapped, no optimizer).
        self.deltatok = self._build_deltatok(backbone)

        # Load transformer (the only trainable network)
        self.vit = self.get_network("vit")
        print(f"Number of parameters: {sum(p.numel() for p in self.vit.parameters())/1e6:.2f}M")

        # Define optimizer
        self.optim = self.get_optim(
            self.vit, self.cfg.training.lr, betas=(0.9, 0.999),
            weight_decay=self.cfg.training.weight_decay, mode=self.cfg.training.optimizer
        )

        # Set up Exponential Moving Average (EMA) for model parameters
        self.ema = EMA(self._ema_model()) if self.cfg.training.use_ema else None
        if self.cfg.training.use_ema and self._ema_state is not None:
            self.ema.load_state_dict(self._ema_state, self._ema_model())

        # print logs
        if self.is_master:
            self.full_training_bar = None
            print("Config: ", self.cfg)
            print(f"TensorBoard log dir: {self.args.writer_log}")
            cfg_str = str(self.cfg)
            self.log_add_txt("Parameters", cfg_str, self.cfg.training.iter)

        self._train_iter = None
        self._train_sampler_epoch = int(self.cfg.training.global_epoch)

        # Initialize evaluation criteria for sampled spatial tokens
        self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
        self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.raymap_criterion = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")

    def _ema_model(self):
        return self.vit.module if self.distributed else self.vit

    @contextmanager
    def ema_scope(self):
        if not self.cfg.training.use_ema:
            yield
            return

        model = self._ema_model()
        self.ema.store(model)
        self.ema.copy_to(model)
        try:
            yield
        finally:
            self.ema.restore(model)

    def _reset_train_iterator(self):
        self._set_train_loader_epoch(self._train_sampler_epoch)
        self._train_iter = iter(self.train_loader)
        self._train_sampler_epoch += 1

    def _next_train_batch(self):
        if self._train_iter is None:
            self._reset_train_iterator()

        while True:
            try:
                return next(self._train_iter)
            except StopIteration:
                self._reset_train_iterator()

    def _build_deltatok(self, backbone):
        """Build the frozen DeltaTok tokenizer and load its checkpoint.

        Mirrors `DeltaTokTrainer.get_network("deltatok")` minus DDP: the module
        is frozen here, so it must never be DDP-wrapped nor enter the optimizer.
        """
        model = self._make_deltatok_module(backbone)  # shared factory (mixin)

        ckpt_path = str(self.cfg.model.deltatok_ckpt)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"DeltaTok checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
        model.load_state_dict(state_dict, strict=True)
        if self.is_master:
            print(f"[INFO] Loaded frozen DeltaTok from: {ckpt_path}")

        model = model.to(self.device).eval()
        model.requires_grad_(False)
        if self.cfg.training.dtype == "bfloat16":
            model.to(torch.bfloat16)
        return model

    def get_network(self, archi):
        """ return the network, load checkpoint if resuming or using pretrained weights """
        if archi == "vit":
            if self.rank == 0:
                if not os.path.exists(self.cfg.training.vit_folder):
                    os.makedirs(self.cfg.training.vit_folder)
                    print(f"Folder created: {self.cfg.training.vit_folder}")

            # Define transformer architecture parameters
            hidden_dim, depth, heads = self.transformer_size(self.cfg.model.get("vit_size", "base"))

            model = Transformer(
                out_dim=1536,
                num_views=self.num_views,
                cross_dim=1536,  # enable per-camera cross-attention on frame-0 patch tokens
                hidden_dim=hidden_dim,
                proj=1,
                depth=depth,
                heads=heads,
                mlp_dim=hidden_dim * 4,
                dropout=self.cfg.model.get("dropout", 0.0),
                is_causal=self.cfg.model.get("is_causal", False),
                use_trajectory_cond=False,
                trajectory_length=0,
                # Spatial slots = cameras (one delta token per camera). A (1, 1)
                # reference grid broadcasts ONE shared embedding to every slot:
                # no camera identity -> camera-permutation equivariance holds.
                ref_spatial_size=(1, 1),
            )

            # Load model checkpoint for resume or pretrained initialization.
            ckpt = self.get_model_checkpoint_path()
            if ckpt is not None:
                checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)

                if self.rank == 0:
                    print("Load ckpt from:", ckpt)
                if getattr(self.args, "resume", False):
                    # Update the current epoch and iteration only for true resume.
                    self.cfg.training.iter = checkpoint['iter']
                    self.cfg.training.global_epoch = checkpoint['global_epoch']
                    self._ema_state = checkpoint.get("ema_state")
                    if self.rank == 0:
                        print("Number of iteration(s):", self.cfg.training.iter)
        else:
            model = None

        if model is not None:
            model = model.to(self.device)
            if getattr(self.args, "compile", False):  # Enable model compilation if using PyTorch 2.0+
                model = torch.compile(model)

            if self.distributed:  # Enable multi-GPU training if available
                model = DDP(model, device_ids=[self.device])

            if self.rank == 0:
                print(f"Size of model {archi}: "
                      f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    def get_resume_checkpoint_path(self):
        if not getattr(self.args, "resume", False):
            return None
        ckpt = os.path.join(self.cfg.training.vit_folder, "current.pth")
        if os.path.isfile(ckpt):
            return ckpt
        return None

    def get_pretrained_checkpoint_path(self):
        ckpt = getattr(self.args, "ckpt", None)
        if not ckpt:
            return None
        ckpt = os.path.expanduser(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt}")
        return ckpt

    def get_model_checkpoint_path(self):
        ckpt = self.get_resume_checkpoint_path()
        if ckpt is not None:
            return ckpt
        return self.get_pretrained_checkpoint_path()

    def adapt_learning_rate(self, mode="cosine"):
        if self.cfg.training.iter < self.cfg.training.warm_up:  # linear warmup updates
            if self.cfg.training.warm_up > 0:
                scale = self.cfg.training.iter / self.cfg.training.warm_up
            else:
                scale = 1.0
        elif mode == "cosine":
            import math
            progress = (self.cfg.training.iter - self.cfg.training.warm_up) / (self.cfg.training.max_iter - self.cfg.training.warm_up)
            progress = min(max(progress, 0.0), 1.0)
            scale = max(0.5 * (1 + math.cos(math.pi * progress)), 0.01)
        else:
            scale = 1.0

        for group in self.optim.param_groups:
            base_lr = group.get('initial_lr', group['lr'])
            group['lr'] = base_lr * scale

    def _encode_deltas(self, feats, H, W):
        """Frozen-DeltaTok encode of all GT frame pairs (the flow target).

        Thin wrapper over the shared ``_encode_pair_deltas``: adds the temporal
        slot-capacity check and the no-grad/autocast guards for the frozen net.

        Args:
            feats: spatial-only patch features (B, T, N, P, C).

        Returns:
            z: delta tokens (B, T-1, N, C).
        """
        T = feats.shape[1]
        assert T - 1 <= self.num_views, (
            f"T-1={T - 1} transitions exceed temporal slot capacity model.num_views={self.num_views}"
        )
        with torch.no_grad(), self.autocast:
            return self._encode_pair_deltas(self.deltatok, feats, H, W)  # (B, T-1, N, C)

    def _build_cross_cond(self, x0, H, W):
        """Build the frame-0 conditioning grids fed to the flow transformer.

        Delta tokens exit DeltaTok through a LayerNorm (~unit scale); raw DA3
        feats are not — `model.ctx_norm: layernorm` applies a parameter-free
        layer norm so both streams live at comparable scale. Callers keep the
        un-normalized first frame for the DeltaTok decode rollout.

        Args:
            x0: GT first-frame patch features (B, N, P, C).

        Returns:
            ctx: (B, N, Hp, Wp, C) — per-camera frame-0 patch grids; camera i's
                 grid is only seen by delta slot i (per-slot cross-attention).
        """
        B, N, P, C = x0.shape
        Hp, Wp = H // self._patch_size, W // self._patch_size   # patch grid size; P = Hp*Wp
        ctx = x0                                                # (B, N, P, C) frame-0 patch feats
        ctx_norm = str(self.cfg.model.get("ctx_norm", "layernorm"))
        if ctx_norm not in ("layernorm", "none"):
            # Fail loudly: an unknown value (e.g. a typo) silently skipping the
            # norm would change the conditioning scale with no error signal.
            raise ValueError(f"model.ctx_norm must be 'layernorm' or 'none', got {ctx_norm!r}")
        if ctx_norm == "layernorm":
            ctx = F.layer_norm(ctx.float(), (C,)).to(x0.dtype)  # (B, N, P, C) unit-scale per token
        return ctx.view(B, N, Hp, Wp, C)                        # (B, N, Hp, Wp, C) unflatten patch grid

    def flow_noising(self, x, context=None, mu=-0.6, sigma=1):
        device = x.device
        b, c, t_dim, h, w = x.shape

        # Sample timestep from shifted distribution
        s = sigma * torch.randn(b, t_dim, device=device) + mu
        t = torch.sigmoid(s)                     # (b, t)
        t_view = t.view(b, 1, t_dim, 1, 1)       # broadcast over C,H,W

        # Sample noise
        e = torch.randn_like(x)

        # Compute noised latent
        z_t = t_view * x + (1.0 - t_view) * e

        # Apply context masking if specified
        if context is not None:
            if isinstance(context, int):
                if context > 0:
                    z_t[:, :, :context] = x[:, :, :context].clone()
                    t[:, :context] = 1  # Set timesteps of context frames to 1 (no noise)
            elif isinstance(context, (list, tuple)):
                for idx, ctx in enumerate(context):
                    z_t[idx, :, :ctx] = x[idx, :, :ctx].clone()
                    t[idx, :ctx] = 1  # Set timesteps of context frames to 1 (no noise)
            
        return z_t, e, t 

    def flow_loss(self, pred, x, z_t, e, t, context=None):
        mask = torch.ones_like(x)        # 0→don't compute the loss, 1→compute the loss
        if isinstance(context, int):
            if context > 0:
                mask[:, :, :context] = 0
        elif isinstance(context, (list, tuple)):
            for idx, ctx in enumerate(context):
                mask[idx, :, :ctx] = 0
        
        mask = mask.bool() 
        # Expand t for spatial broadcasting: (B, 1, 1, 1, 1)
        t_v = t.view(t.shape[0], 1, t.shape[1], 1, 1)
        
        pred_mode = self.cfg.model.get("pred_mode", "v")
        loss_mode = self.cfg.model.get("loss_mode", "v")

        if pred_mode == "x":
            x_pred = pred
            v_pred = (x_pred - z_t) / torch.clamp(1 - t_v, min=0.05)
            e_pred = (z_t - t_v * x_pred) / torch.clamp(1 - t_v, min=0.05)
        elif pred_mode == "v":
            v_pred = pred
            x_pred = z_t + (1 - t_v) * v_pred
            e_pred = (z_t - t_v * x_pred) / torch.clamp(1 - t_v, min=0.05)
        elif pred_mode == "e":
            e_pred = pred
            x_pred = (z_t - (1 - t_v) * e_pred) / torch.clamp(t_v, min=1e-5)
            v_pred = (x_pred - z_t) / torch.clamp(1 - t_v, min=0.05)

        if loss_mode == "x":
            loss = ((x_pred - x) ** 2)[mask].mean()
        elif loss_mode == "v":
            v = (x - z_t) / torch.clamp(1 - t_v, min=0.05)
            loss = ((v_pred - v) ** 2)[mask].mean()
        elif loss_mode == "e":
            loss = ((e_pred - e) ** 2)[mask].mean()
            
        return loss

    def train_one_epoch(self, log_iter=1000, virtual_epoch=5_000):
        self.vit.train()
        num_batches = 0
        iter_start = int(self.cfg.training.iter)
        last_update_time = time.time()
        self.optim.zero_grad(set_to_none=True)

        # SLURM-friendly progress: one flushed print line every `print_freq`
        # optimizer updates (tqdm carriage returns garble batch logs).
        print_freq = int(self.cfg.training.get("print_freq", 20))
        # Loss bookkeeping stays on-GPU between prints: .cpu()/.item() every
        # micro-batch forces a GPU->CPU sync, so scalars are materialized only
        # every `print_freq` optimizer updates. The window spans everything
        # since the last print.
        cum_loss = torch.zeros((), device=self.device)            # () running sum of un-scaled losses
        window_loss = deque(maxlen=self.grad_cum * print_freq)    # detached () GPU scalars
        window_grad = deque(maxlen=print_freq)                    # () pre-clip grad norms, on GPU
        header = f"Training (Epoch {self.cfg.training.global_epoch})"
        update_time = SmoothedValue(fmt='{avg:.4f}')  # seconds per optimizer update
        updates_done = 0

        while True:
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                break
            if (self.cfg.training.iter - iter_start) >= virtual_epoch:
                break

            batch = self._normalize_batch(self._next_train_batch())
            num_batches += 1
            update_grad = (num_batches % self.grad_cum) == 0

            self.adapt_learning_rate("cosine")

            imgs = batch["imgs"].to(self.device, non_blocking=True)  # (B, V, 3, H, W) with V = T*N views
            num_cameras = int(batch.get("num_cameras", 1))

            # Frozen OccRAE encode -> spatial-only patch feats per timestep/camera.
            _, feats, _, _, H, W = self._extract_pair_feats(imgs, num_cameras=num_cameras, return_pairs=False)  # feats (B, T, N, P, C)

            # Frozen DeltaTok encode -> the flow target.
            z = self._encode_deltas(feats, H, W)                          # (B, T-1, N, C) delta tokens
            x_spatial = z.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()  # (B, C, T-1, N, 1) flow latent layout
            cross_cond = self._build_cross_cond(feats[:, 0], H, W)        # (B, N, Hp, Wp, C) frame-0 conditioning

            # No in-sequence context slots: conditioning is purely cross-attention.
            z_t, e, timestep = self.flow_noising(x_spatial, context=0, mu=self.cfg.model.mu, sigma=self.cfg.model.sigma)

            with self.autocast:
                pred = self.vit(
                    x=z_t,
                    ada_cond=timestep,
                    cross_cond=cross_cond,
                    return_feat=False,
                )
                loss_flow_total = self.flow_loss(pred=pred, x=x_spatial, z_t=z_t, e=e, t=timestep, context=0)

            loss = loss_flow_total / self.grad_cum
            loss.backward()

            if update_grad:
                # clip_grad_norm_ returns the pre-clip total norm — the direct
                # gradient-explosion signal; keep it on GPU for the log window.
                grad_norm = nn.utils.clip_grad_norm_(self.vit.parameters(), self.cfg.training.grad_clip)
                window_grad.append(grad_norm.detach())
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
                if self.cfg.training.use_ema:
                    self.ema.update(self._ema_model())

            loss_detached = loss_flow_total.detach()  # () un-scaled flow loss, kept on GPU (no sync)
            cum_loss = cum_loss + loss_detached
            window_loss.append(loss_detached)

            if update_grad:
                updates_done += 1
                if self.is_master:
                    now = time.time()
                    elapsed = max(now - last_update_time, 1e-6)
                    last_update_time = now
                    update_time.update(elapsed)

                # Same condition on every rank: the all_reduce below is a
                # collective and must be entered by all ranks together.
                if updates_done % print_freq == 0 or updates_done == virtual_epoch:
                    window = torch.stack(tuple(window_loss)).float()  # (W,) losses since last print, W = grad_cum*print_freq
                    loss_mean = window.mean()                         # () convergence signal: cross-rank mean below
                    # (2,) [window-max loss, window-max pre-clip grad norm]:
                    # spike detectors, so reduced with MAX (worst rank wins).
                    peaks = torch.stack((window.max(), torch.stack(tuple(window_grad)).float().max()))
                    if self.distributed:
                        # On-GPU NCCL reduces; the old all_gather_object pickled
                        # tensors through the CPU every optimizer update.
                        dist.all_reduce(loss_mean, op=dist.ReduceOp.SUM)
                        loss_mean = loss_mean / self.world_size
                        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)

                    if self.is_master:
                        loss_val = loss_mean.item()        # the GPU->CPU syncs: once per print_freq updates
                        loss_max, grad_max = peaks.tolist()
                        speed_samples_per_sec = self.cfg.training.bsize / max(update_time.avg, 1e-6)

                        self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.cfg.training.iter)
                        self.log_add_scalar('Train/LossMean', loss_val, self.cfg.training.iter)
                        self.log_add_scalar('Train/LossMax', loss_max, self.cfg.training.iter)
                        self.log_add_scalar('Train/GradNorm', grad_max, self.cfg.training.iter)
                        self.log_add_scalar('Train/SpeedSamplesPerSec', speed_samples_per_sec, self.cfg.training.iter)

                        eta_seconds = update_time.avg * (virtual_epoch - updates_done)
                        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
                        print(
                            f"{header}  [{updates_done}/{virtual_epoch}]  eta: {eta}  "
                            f"loss: {loss_val:.4f} (max {loss_max:.4f})  "
                            f"grad: {grad_max:.2f}  "
                            f"lr: {self.optim.param_groups[0]['lr']:.6f}  "
                            f"time: {update_time.avg:.4f}  "
                            f"max gpu mem: {gpu_mem_mb:.0f} ({total_gpu_mem_mb():.0f})  "
                            f"cpu mem: {rank_cpu_mem_mb():.0f} ({rank_cpu_budget_mb():.0f})",
                            flush=True,
                        )

                if self.cfg.training.iter % log_iter == 0 and self.is_master:
                    if self.cfg.training.iter > 0:
                        self.save_network(
                            model=self.vit, path=os.path.join(self.cfg.training.vit_folder, "current.pth"),
                            optimizer=self.optim, iter=self.cfg.training.iter, global_epoch=self.cfg.training.global_epoch,
                            ema_state=self.ema.state_dict() if self.cfg.training.use_ema else None
                        )
                
                self.cfg.training.iter += 1

                if self.is_master and self.cfg.training.iter % 50_000 == 0 and self.cfg.training.iter > 0:
                    self.save_network(
                        model=self.vit,
                        path=os.path.join(self.cfg.training.vit_folder, f"iter_{self.cfg.training.iter:06d}.pth"),
                        optimizer=self.optim,
                        iter=self.cfg.training.iter,
                        global_epoch=self.cfg.training.global_epoch,
                        ema_state=self.ema.state_dict() if self.cfg.training.use_ema else None,
                    )
        
        return (cum_loss / max(1, num_batches)).item()

    # _compute_frame_losses comes from DeltaTokSharedMixin.

    @torch.no_grad()
    def eval_one_epoch(self, sanity_check: bool = False):
        if not getattr(self, "test_loaders", None):
            return torch.tensor(0.0, device=self.device)

        self.vit.eval()

        if sanity_check:
            eval_num_items_global = int(self.cfg.training.get("sanity_check_num_items", 4))
            eval_num_visualizations = 2
        else:
            eval_num_items_global = int(self.cfg.training.get("eval_num_items", 256))
            eval_num_visualizations = int(self.cfg.training.get("eval_num_visualizations", 8))
        eval_num_items = max(1, eval_num_items_global // self.world_size)
        eval_num_steps = self.cfg.training.get("eval_num_steps", 50)
        eval_viz_dir = str(
            self.cfg.training.get(
                "eval_viz_dir",
                os.path.join(self.cfg.training.vit_folder, "eval_viz"),
            )
        )

        # Overall eval scalar = sampled-rollout pointmap loss across loaders.
        overall_loss = 0.0
        overall_n = 0

        with self.ema_scope():
            # Run eval on each test loader independently; per-loader metrics are
            # logged under `Eval/<test_name>/...` and viz under `eval_depth/<test_name>`.
            for test_name, loader in self.test_loaders.items():
                sums = {k: 0.0 for k in _EVAL_KEYS}
                items_seen = 0  # all items (drives the eval-item budget)
                gt_items = 0    # items with GT only (metric denominator)
                num_vis = 0

                # Pin the eval loader to epoch 0 so each eval pass sees the same
                # samples in the same order.
                sampler = getattr(loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(0)
                ds = getattr(loader, "dataset", None)
                if ds is not None and hasattr(ds, "set_epoch"):
                    ds.set_epoch(0)

                # SLURM-friendly progress: MetricLogger prints one flushed line
                # per print_freq batches (master only, matching tqdm's disable).
                if self.is_master:
                    metric_logger = MetricLogger(delimiter="  ")
                    batch_iter = metric_logger.log_every(
                        loader,
                        int(self.cfg.training.get("print_freq", 20)),
                        header=f"Evaluating {test_name} (Epoch {self.cfg.training.global_epoch})",
                    )
                else:
                    batch_iter = loader

                for batch in batch_iter:
                    if items_seen >= eval_num_items:
                        break
                    batch = self._normalize_batch(batch)

                    imgs = batch["imgs"].to(self.device, non_blocking=True)  # (B, V, 3, H, W)
                    B, V = imgs.shape[:2]
                    num_cameras = int(batch.get("num_cameras", 1))

                    tokens, feats, _, _, H, W = self._extract_pair_feats(imgs, num_cameras=num_cameras, return_pairs=False)  # tokens (B, V, N_tok, C); feats (B, T, N, P, C)

                    z = self._encode_deltas(feats, H, W)                          # (B, T-1, N, C) GT delta tokens
                    x_spatial = z.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()  # (B, C, T-1, N, 1) flow latent layout
                    cross_cond = self._build_cross_cond(feats[:, 0], H, W)        # (B, N, Hp, Wp, C) frame-0 conditioning

                    batch_losses = {}

                    # 1. Sample delta tokens from pure noise (conditioning only via cross_cond).
                    zr = torch.randn_like(x_spatial)                              # (B, C, T-1, N, 1) init noise
                    gen = flow_euler_sample(
                        self._ema_model(), zr,
                        pred_mode=self.cfg.model.pred_mode,
                        context=0,
                        num_steps=eval_num_steps,
                        cross_cond=cross_cond,
                        autocast_ctx=self.autocast,
                    )
                    z_hat = gen.squeeze(-1).permute(0, 2, 3, 1).contiguous()      # (B, T-1, N, C) sampled deltas

                    if "gt_mask" in batch:
                        height, width = batch["output_resolution_hw"]

                        # 2. Autoregressive DeltaTok decode from the GT first frame:
                        #    sampled deltas + GT-delta upper bound (tokenizer-only error).
                        with self.autocast:
                            x_hat = self._rollout_from_z(self.deltatok, feats[:, 0], z_hat, H, W, num_cameras)     # (B*(T-1), N, P, C)
                            x_hat_tok = self._rollout_from_z(self.deltatok, feats[:, 0], z, H, W, num_cameras)     # (B*(T-1), N, P, C)

                        # 3. Stitch into full token tensors (GT timestep 0 + prefix tokens), decode with OccRAE.
                        full_tokens = self._reconstruct_full_tokens(tokens, x_hat, B, V, num_cameras=num_cameras)          # (B, V, N_tok, C)
                        full_tokens_tok = self._reconstruct_full_tokens(tokens, x_hat_tok, B, V, num_cameras=num_cameras)  # (B, V, N_tok, C)
                        with self.autocast:
                            decoded = self._decode_tokens(full_tokens, height, width, num_cameras=num_cameras)
                            decoded_tok = self._decode_tokens(full_tokens_tok, height, width, num_cameras=num_cameras)

                        ray_conf = decoded.get("ray_conf")
                        ray_conf_tok = decoded_tok.get("ray_conf")

                        # 4. Losses on forecast views only: timestep 0 (num_cameras
                        # views) carries GT tokens (the rollout seed), not predictions.
                        pred_slice = slice(num_cameras, V)
                        loss_pm, loss_d, loss_ray = self._compute_frame_losses(
                            decoded, batch, pred_slice, ray_conf, B, height, width
                        )
                        loss_pm_tok, loss_d_tok, loss_ray_tok = self._compute_frame_losses(
                            decoded_tok, batch, pred_slice, ray_conf_tok, B, height, width
                        )
                        batch_losses.update({
                            "LossPointmap": loss_pm.item(),
                            "LossDepth": loss_d.item(),
                            "LossRaymap": loss_ray.item(),
                            "LossPointmap_tok": loss_pm_tok.item(),
                            "LossDepth_tok": loss_d_tok.item(),
                            "LossRaymap_tok": loss_ray_tok.item(),
                        })

                        if self.is_master and num_vis < eval_num_visualizations:
                            for batch_idx in range(B):
                                if num_vis >= eval_num_visualizations:
                                    break

                                view_order = sorted(
                                    range(V),
                                    key=lambda idx: batch["timesteps"][batch_idx][idx],
                                )
                                # Timestep-0 views carry GT tokens (rollout seed); mark them
                                # with a colored border instead of blanking them.
                                ctx_mask = [v < num_cameras for v in view_order]
                                view_index = torch.as_tensor(view_order, dtype=torch.long)
                                gt_depth = batch["gt_depth"][batch_idx].detach().cpu()[view_index]   # (V, H, W) time-ordered
                                gt_mask = batch["gt_mask"][batch_idx].detach().cpu()[view_index]     # (V, H, W)
                                gt_depth_color = torch.stack([
                                    torch.from_numpy(
                                        depth2rgb(
                                            gt_depth[t].clamp(0, 50).numpy(),
                                            valid_mask=gt_mask[t].numpy().astype(bool),
                                            min_depth=0.0,
                                            max_depth=50.0,
                                        ).astype(np.float32)
                                    )
                                    for t in range(V)
                                ])
                                tok_depth = decoded_tok["depth"][batch_idx].detach().float().cpu()[view_index]  # (V, H, W)
                                tok_depth_color = torch.stack([
                                    torch.from_numpy(
                                        depth2rgb(
                                            tok_depth[t].clamp(0, 50).numpy(),
                                            valid_mask=tok_depth[t].numpy() > 0,
                                            min_depth=0.0,
                                            max_depth=50.0,
                                        ).astype(np.float32)
                                    )
                                    for t in range(V)
                                ])

                                saved_path = _log_viz_sample(
                                    batch=batch,
                                    decoded=decoded,
                                    batch_idx=batch_idx,
                                    epoch=self.cfg.training.global_epoch,
                                    epoch_step=self.cfg.training.iter,
                                    output_dir=eval_viz_dir,
                                    log_writer=self.writer,
                                    tb_prefix=f"eval_depth/{test_name}",
                                    extra_panels=[tok_depth_color, gt_depth_color],
                                    view_order=view_order,
                                    max_depth=50.0,
                                    context_mask=ctx_mask,
                                    col_titles=[
                                        "RGB",
                                        "Pred Depth (Flow)",
                                        "Pred Depth (GT-z)",
                                        "GT Depth",
                                    ],
                                )
                                if saved_path is not None:
                                    print(f"Saved viz: {saved_path}")
                                num_vis += 1

                    # Only GT batches enter the metric sums AND the denominator;
                    # counting GT-less batches would silently dilute metrics to 0.
                    if batch_losses:
                        for key, val in batch_losses.items():
                            sums[key] += val * B
                        gt_items += B
                    items_seen += B
                    if self.is_master and "LossPointmap" in batch_losses:
                        metric_logger.update(loss=batch_losses["LossPointmap"])

                # Per-loader reduction over a fixed key vector (DDP-safe: every
                # rank reduces the same shape even if some saw no GT batches).
                vec = torch.tensor(
                    [sums[k] for k in _EVAL_KEYS] + [float(gt_items)],
                    device=self.device, dtype=torch.float64,
                )
                if self.distributed:
                    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
                n = vec[-1].item()
                if n == 0:
                    continue
                results = {k: (vec[i] / n).item() for i, k in enumerate(_EVAL_KEYS)}

                overall_loss += results["LossPointmap"] * n
                overall_n += n

                if self.is_master and not sanity_check:
                    for key, val in results.items():
                        self.log_add_scalar(f"Eval/{test_name}/{key}", val, self.cfg.training.iter)
                    if self.writer is None:
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in results.items())
                        print(f"[Eval/{test_name}] {metrics_str}")

        final_loss = overall_loss / overall_n if overall_n > 0 else 0.0

        self.vit.train()
        return final_loss

    # _build_occ_rae comes from DeltaTokSharedMixin (frozen + bf16 cast).

    def fit(self, log_iter=1000):
        # Build train/test loaders via `get_data_loader` (dust3r-style dataset
        # expression strings), mirroring DeltaTokTrainer.fit.
        train_dataset_str = str(self.cfg.dataset.train_dataset)
        test_dataset_str = self.cfg.dataset.get("test_dataset", None)
        per_dataset_sampling = bool(self.cfg.dataset.get("per_dataset_sampling", False))

        if self.is_master:
            print(f"Building train dataset: {train_dataset_str}")
            print(f"Per-dataset sampling: {per_dataset_sampling}")
        self.train_loader = get_data_loader(
            train_dataset_str,
            batch_size=self.cfg.training.bsize,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
            drop_last=True,
            per_dataset_sampling=per_dataset_sampling,
        )

        # One DataLoader per `+`-separated sub-dataset so each test set runs in
        # its own eval pass and logs under its own TensorBoard prefix.
        self.test_loaders: dict = {}
        if test_dataset_str:
            test_dataset_str = str(test_dataset_str)
            if self.is_master:
                print(f"Building test datasets: {test_dataset_str}")
            for sub in test_dataset_str.split("+"):
                sub = sub.strip()
                if not sub:
                    continue
                test_name = sub.split("(")[0].strip()
                try:
                    loader = get_data_loader(
                        sub,
                        batch_size=self.cfg.training.bsize,
                        num_workers=int(self.cfg.training.get("val_num_workers", 2)),
                        shuffle=False,
                        drop_last=False,
                    )
                    self.test_loaders[test_name] = loader
                    if self.is_master:
                        print(f"  - {test_name}: {len(loader)} batches")
                except Exception as e:
                    if self.is_master:
                        print(f"  - {test_name}: failed to build ({e})")

        if self.is_master:
            print("Start training:")

        # Initial evaluation for sanity check
        if self.test_loaders:
            if self.is_master:
                print("Running initial evaluation for sanity check...")
            self.eval_one_epoch(sanity_check=True)

        start = time.time()

        for e in range(self.cfg.training.global_epoch, self.cfg.training.epoch + 1):
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                print("End of training: reached max iterations")
                break

            # Train-sampler epochs are advanced by _reset_train_iterator.
            train_loss = self.train_one_epoch(log_iter=log_iter, virtual_epoch=self.cfg.training.virtual_epoch)
            test_loss = self.eval_one_epoch()

            if self.distributed:
                train_loss = self.all_gather(train_loss)
                test_loss = self.all_gather(test_loss)

            if self.is_master:
                clock_time = (time.time() - start)
                self.log_add_scalar('Train/Loss', train_loss, self.cfg.training.global_epoch)
                self.log_add_scalar('Eval/Loss', test_loss, self.cfg.training.global_epoch)
                now = os.popen('date').read().strip()
                print(f"\r\033[KEpoch {self.cfg.training.global_epoch},"
                      f" Iter {self.cfg.training.iter},"
                      f" Train: {train_loss:.4f}, Eval: {test_loss:.4f},"
                      f" Time: {int(clock_time // 3600):.0f}:{int((clock_time % 3600) // 60):02d}:{int(clock_time % 60):02d},"
                      f" Date: {now}")

            self.cfg.training.global_epoch += 1

    def run(self):
        self.fit()
