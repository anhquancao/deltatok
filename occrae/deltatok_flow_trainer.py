# Trainer for Token Flow Matching
import os
import time
import random
from collections import deque
from contextlib import nullcontext, contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from occrae.abstract_trainer import Trainer
from occrae.network.efficient_transformer import Transformer
from occrae.network.ema import EMA
from occrae.dataset.occrae_tokens import ProcessedRootBatchSampler
from occrae.dataset.preprocessed_sequence import (
    PreprocessedSequenceDataset,
    build_sequence_records,
    _normalize_resolutions,
)
from occrae.visualization_helper import _log_viz_sample
from occany.model.occ_rae import OccRAE
from depth_anything_3.utils.io.input_processor import InputProcessor
from occany.utils.helpers import crop_resize_if_necessary, intrinsics_c2w_to_raymap
from occany.utils.helpers import depth2rgb
from occany.loss import PointmapLoss, DepthLosses, RaymapLoss
from occrae.generation_helper import flow_euler_sample


def collate_preprocessed(batch):
    # Each item has: split, processed_root, scene_name, frame_stems, timesteps, output_resolution, imgs
    # Optional GT: depthmaps, intrinsics, cam2worlds
    # imgs is a List[np.ndarray]; output_resolution is a ResolutionList (List[Tuple[int, int]]).
    # ProcessedRootBatchSampler guarantees all items in a batch share the same processed_root,
    # so taking the first item's allowed-resolution list is safe.

    first = batch[0]
    allowed_resolutions_hw = _normalize_resolutions(first["output_resolution"])

    # Pick one (H, W) for the entire batch so all encoded inputs share dimensions.
    output_resolution_hw = random.choice(allowed_resolutions_hw)
    height, width = output_resolution_hw
    resolution_wh = (width, height)

    all_imgs = []
    all_gt_depth = []
    all_gt_intrinsics = []
    all_gt_c2w = []
    has_gt = "depthmaps" in first

    for item in batch:
        item_imgs = []
        item_depth = []
        item_intrinsics = []
        item_c2w = []
        
        for i in range(len(item["imgs"])):
            img = item["imgs"][i]
            if has_gt:
                depth = item["depthmaps"][i]
                intrinsics = item["intrinsics"][i]
                c2w = item["cam2worlds"][i]
            else:
                depth = np.zeros(img.shape[:2], dtype=np.float32)
                intrinsics = np.eye(3, dtype=np.float32)
                if isinstance(img, Image.Image):
                    W, H = img.size
                else:
                    H, W = img.shape[:2]
                intrinsics[0, 2] = W / 2
                intrinsics[1, 2] = H / 2
                c2w = np.eye(4, dtype=np.float32)

            cropped_img, cropped_depth, cropped_intrinsics = crop_resize_if_necessary(
                img,
                depth,
                intrinsics,
                resolution=resolution_wh,
                rng=None,
                info=f"{item['scene_name']}",
            )
            item_imgs.append(InputProcessor.NORMALIZE(to_tensor(cropped_img)))
            if has_gt:
                item_depth.append(torch.from_numpy(cropped_depth).float())
                item_intrinsics.append(torch.from_numpy(cropped_intrinsics).float())
                item_c2w.append(torch.from_numpy(c2w).float())

        all_imgs.append(torch.stack(item_imgs)) # (V, C, H, W)
        if has_gt:
            all_gt_depth.append(torch.stack(item_depth)) # (V, H, W)
            all_gt_intrinsics.append(torch.stack(item_intrinsics)) # (V, 3, 3)
            all_gt_c2w.append(torch.stack(item_c2w)) # (V, 4, 4)

    imgs = torch.stack(all_imgs) # (B, V, C, H, W)
    assert imgs.shape[-2:] == (height, width)

    out = {
        "imgs": imgs,
        "output_resolution_hw": output_resolution_hw,
        "processed_root": [item["processed_root"] for item in batch],
        "timesteps": [item["timesteps"] for item in batch],
        "scene_name": [item["scene_name"] for item in batch],
        "frame_stems": [item["frame_stems"] for item in batch],
        "num_cameras": first.get("num_cameras", 1),
    }

    if has_gt:
        gt_depth = torch.stack(all_gt_depth) # (B, V, H, W)
        gt_intrinsics = torch.stack(all_gt_intrinsics) # (B, V, 3, 3)
        gt_c2w = torch.stack(all_gt_c2w) # (B, V, 4, 4)
        
        # Compute raymap and pointmap
        B, V, H, W = gt_depth.shape
        gt_raymap = intrinsics_c2w_to_raymap(gt_intrinsics, gt_c2w, H, W) # (B, V, H, W, 6)
        gt_pointmap = gt_depth.unsqueeze(-1) * gt_raymap[..., :3] + gt_raymap[..., 3:]
        gt_mask = gt_depth > 0

        out.update({
            "gt_depth": gt_depth,
            "gt_intrinsics": gt_intrinsics,
            "gt_c2w": gt_c2w,
            "gt_raymap": gt_raymap,
            "gt_pointmap": gt_pointmap,
            "gt_mask": gt_mask,
        })

    return out
class OccRAEFlowMatchingTrainer(Trainer):
    def __init__(self, args, cfg, device, rank, world_size, distributed):
        """ Initialize model, optimizer, loss function, and data loaders."""
        args.is_master = rank == 0
        args.writer_log = str(cfg.training.get("writer_log", ""))
        super().__init__(args)
        print(f"Init Token Flow Matching on [GPU{rank}]")

        self.args = args
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed

        self._ema_state = None
        self.occ_rae = None

        # Determine architecture params from config
        self.num_views = self.cfg.model["num_views"]
        print(f"Number of views: {self.num_views}")
        self.patch_h = 518 // 14
        self.patch_w = 518 // 14

        # Load transformer (Bidirectional Transformer)
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

        # Set up automatic mixed precision for training efficiency
        if self.device.type != 'cpu' and self.cfg.training.dtype == "bfloat16":
            self.autocast = autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        self.is_master = (self.rank == 0)

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
        if self.distributed and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(self._train_sampler_epoch)
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
                hidden_dim=hidden_dim, 
                proj=1,
                depth=depth, 
                heads=heads, 
                mlp_dim=hidden_dim * 4,
                dropout=self.cfg.model.get("dropout", 0.0), 
                is_causal=self.cfg.model.get("is_causal", False),
                use_trajectory_cond=False,
                trajectory_length=0,
                ref_spatial_size=(self.patch_h, self.patch_w)
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

    def _tokens_to_spatial(self, x):
        """ (B, V, C, S, 1) -> (B, C, V, S, 1) """
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def _spatial_to_tokens(self, x):
        """ (B, C, V, S, 1) -> (B, V, C, S, 1) """
        return x.permute(0, 2, 1, 3, 4).contiguous()

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
        cum_loss = 0.
        num_batches = 0
        iter_start = int(self.cfg.training.iter)
        last_update_time = time.time()
        window_loss = deque(maxlen=self.cfg.training.grad_cum)
        self.optim.zero_grad(set_to_none=True)

        pbar = tqdm(total=virtual_epoch, desc=f"Training (Epoch {self.cfg.training.global_epoch})", disable=not self.is_master)

        while True:
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                break
            if (self.cfg.training.iter - iter_start) >= virtual_epoch:
                break

            batch = self._next_train_batch()
            num_batches += 1
            update_grad = (num_batches % self.cfg.training.grad_cum) == 0

            self.adapt_learning_rate("cosine")

            # Online encoding using the shared frozen OccRAE.
            imgs = batch["imgs"].to(self.device)

            with torch.no_grad():
                latents = self.occ_rae.encode(imgs)
                # Convert (B, V, S, C) -> (B, V, C, S, 1)
                x_tokens = latents["tokens"].permute(0, 1, 3, 2).unsqueeze(-1).contiguous()
                if self.cfg.training.dtype == "bfloat16":
                    x_tokens = x_tokens.to(torch.bfloat16)

            B = x_tokens.shape[0]

            x_spatial = self._tokens_to_spatial(x_tokens) # (B, C, V, S, 1)

            cond_num = int(self.cfg.dataset.cond_num)
            z_t, e, timestep = self.flow_noising(x_spatial, context=cond_num, mu=self.cfg.model.mu, sigma=self.cfg.model.sigma)

            with self.autocast:
                pred = self.vit(
                    x=z_t,
                    ada_cond=timestep,
                    cross_cond=None,
                    return_feat=False,
                )
                loss_flow_total = self.flow_loss(pred=pred, x=x_spatial, z_t=z_t, e=e, t=timestep, context=cond_num)

            loss = loss_flow_total / self.cfg.training.grad_cum
            loss.backward()

            if update_grad:
                nn.utils.clip_grad_norm_(self.vit.parameters(), self.cfg.training.grad_clip)
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
                if self.cfg.training.use_ema:
                    self.ema.update(self._ema_model())

            cum_loss += loss.cpu().item() * self.cfg.training.grad_cum
            window_loss.append(loss.cpu().item() * self.cfg.training.grad_cum)
            
            if update_grad:
                if self.distributed:
                    mini_batch_loss = self.all_gather(torch.tensor(window_loss).mean())
                else:
                    mini_batch_loss = torch.tensor(window_loss).mean()

                if self.is_master:
                    now = time.time()
                    elapsed = max(now - last_update_time, 1e-6)
                    speed_samples_per_sec = self.cfg.training.bsize / elapsed
                    last_update_time = now

                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.cfg.training.iter)
                    self.log_add_scalar('Train/LossFlow', loss_flow_total, self.cfg.training.iter)
                    self.log_add_scalar('Train/LossTot', mini_batch_loss, self.cfg.training.iter)
                    self.log_add_scalar('Train/SpeedSamplesPerSec', speed_samples_per_sec, self.cfg.training.iter)
                
                    pbar.update(1)
                    pbar.set_postfix(loss=mini_batch_loss.item())

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
        
        pbar.close()
        return cum_loss / max(1, num_batches)

    @torch.no_grad()
    def _compute_frame_losses(self, decoded, batch, view_slice, ray_conf, B, height, width):
        mask = batch["gt_mask"][:, view_slice].to(self.device)
        num_views = mask.shape[1]

        loss_pm, _ = self.pointmap_criterion(
            decoded["pointmap"][:, view_slice].float(),
            batch["gt_pointmap"][:, view_slice].to(self.device).float(),
            mask=mask
        )

        pred_d = decoded["depth"][:, view_slice].float().reshape(B * num_views, 1, height, width)
        gt_d = batch["gt_depth"][:, view_slice].to(self.device).float().reshape(B * num_views, 1, height, width)
        d_mask = mask.reshape(B * num_views, 1, height, width).float()
        loss_d, _ = self.depth_criterion(pred_d, gt_d, confidence=None, mask=d_mask)

        rc = ray_conf[:, view_slice] if ray_conf is not None else None
        loss_ray, _ = self.raymap_criterion(
            decoded["ray"][:, view_slice].float(),
            rc,
            batch["gt_c2w"][:, view_slice].to(self.device).float(),
            batch["gt_intrinsics"][:, view_slice].to(self.device).float(),
            batch["gt_raymap"][:, view_slice].to(self.device).float()
        )

        return loss_pm, loss_d, loss_ray

    def eval_one_epoch(self, sanity_check: bool = False):
        if not hasattr(self, "test_loader") or self.test_loader is None:
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
        
        items_seen = 0
        num_visualizations = 0
        total_loss_flow = 0.0
        total_loss_pm = 0.0
        total_loss_d = 0.0
        total_loss_ray = 0.0
        total_loss_pm_ctx = 0.0
        total_loss_d_ctx = 0.0
        total_loss_ray_ctx = 0.0
        
        B_val = self.cfg.training.bsize # Assume bsize is used for val
        total_batches = min(len(self.test_loader), (eval_num_items + B_val - 1) // B_val)
        pbar = tqdm(self.test_loader, total=total_batches, desc=f"Evaluating (Epoch {self.cfg.training.global_epoch})", disable=not self.is_master)

        with self.ema_scope():
            for batch in pbar:
                if items_seen >= eval_num_items:
                    break
                    
                imgs = batch["imgs"].to(self.device)
                B, V, C, H, W = imgs.shape

                with torch.no_grad():
                    latents = self.occ_rae.encode(imgs)
                    # Convert (B, V, S, C) -> (B, V, C, S, 1)
                    x_tokens = latents["tokens"].permute(0, 1, 3, 2).unsqueeze(-1).contiguous()
                    if self.cfg.training.dtype == "bfloat16":
                        x_tokens = x_tokens.to(torch.bfloat16)

                x_spatial = self._tokens_to_spatial(x_tokens)

                # 1. Existing flow loss
                cond_num = int(self.cfg.dataset.cond_num)
                z_t, e, timestep = self.flow_noising(x_spatial, context=cond_num, mu=self.cfg.model.mu, sigma=self.cfg.model.sigma)

                with self.autocast:
                    pred = self.vit(x=z_t, ada_cond=timestep, cross_cond=None, return_feat=False)
                    loss_flow = self.flow_loss(pred=pred, x=x_spatial, z_t=z_t, e=e, t=timestep, context=cond_num)

                total_loss_flow += loss_flow.detach().float().item() * B

                # 2. Sample spatial tokens (condition on first cond_num frames)
                z = torch.randn_like(x_spatial)
                z[:, :, :cond_num] = x_spatial[:, :, :cond_num]
                gen_spatial = flow_euler_sample(
                    self._ema_model(), z,
                    pred_mode=self.cfg.model.pred_mode,
                    context=cond_num,
                    num_steps=eval_num_steps,
                    autocast_ctx=self.autocast
                )
                
                # 3. Convert back to tokens
                gen = self._spatial_to_tokens(gen_spatial)
                gen = gen.squeeze(-1).transpose(-1, -2).contiguous() # (B, V, S, C)
                
                # 4. Decode
                height, width = batch["output_resolution_hw"]
                decoded = self.occ_rae.decode({"tokens": gen, "H": height, "W": width})
                
                # 5. Compute metrics (forecast frames only)
                if "gt_mask" in batch:
                    ray_conf = decoded.get("ray_conf")

                    loss_pm, loss_d, loss_ray = self._compute_frame_losses(
                        decoded, batch, slice(cond_num, None), ray_conf, B, height, width
                    )
                    total_loss_pm += loss_pm.item() * B
                    total_loss_d += loss_d.item() * B
                    total_loss_ray += loss_ray.item() * B

                    # Context frame losses
                    if cond_num > 0:
                        loss_pm_ctx, loss_d_ctx, loss_ray_ctx = self._compute_frame_losses(
                            decoded, batch, slice(0, cond_num), ray_conf, B, height, width
                        )
                        total_loss_pm_ctx += loss_pm_ctx.item() * B
                        total_loss_d_ctx += loss_d_ctx.item() * B
                        total_loss_ray_ctx += loss_ray_ctx.item() * B

                    if self.is_master and num_visualizations < eval_num_visualizations:
                        for batch_idx in range(B):
                            if num_visualizations >= eval_num_visualizations:
                                break

                            view_order = sorted(
                                range(V),
                                key=lambda idx: batch["timesteps"][batch_idx][idx],
                            )
                            view_index = torch.as_tensor(view_order, dtype=torch.long)
                            gt_depth = batch["gt_depth"][batch_idx].detach().cpu()[view_index]
                            gt_mask = batch["gt_mask"][batch_idx].detach().cpu()[view_index]
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
                            context_set = set(range(cond_num))
                            ctx_mask = [v in context_set for v in view_order]
                            _log_viz_sample(
                                batch=batch,
                                decoded=decoded,
                                batch_idx=batch_idx,
                                epoch=self.cfg.training.global_epoch,
                                epoch_step=self.cfg.training.iter,
                                output_dir=eval_viz_dir,
                                log_writer=self.writer,
                                tb_prefix="eval_depth",
                                extra_panels=[gt_depth_color],
                                view_order=view_order,
                                max_depth=50.0,
                                context_mask=ctx_mask,
                            )
                            num_visualizations += 1

                items_seen += B
                if self.is_master:
                    pbar.set_postfix(loss=loss_flow.item())

        # Final reduction
        metrics_sum = torch.tensor([
            total_loss_flow, total_loss_pm, total_loss_d, total_loss_ray,
            total_loss_pm_ctx, total_loss_d_ctx, total_loss_ray_ctx,
            float(items_seen),
        ], device=self.device)
        if self.distributed:
            dist.all_reduce(metrics_sum, op=dist.ReduceOp.SUM)

        n = metrics_sum[7]
        final_loss_flow = (metrics_sum[0] / n).item()
        final_loss_pm = (metrics_sum[1] / n).item()
        final_loss_d = (metrics_sum[2] / n).item()
        final_loss_ray = (metrics_sum[3] / n).item()
        final_loss_pm_ctx = (metrics_sum[4] / n).item()
        final_loss_d_ctx = (metrics_sum[5] / n).item()
        final_loss_ray_ctx = (metrics_sum[6] / n).item()

        if self.is_master and not sanity_check:
            self.log_add_scalar('Eval/LossFlow', final_loss_flow, self.cfg.training.iter)
            self.log_add_scalar('Eval/LossPointmap', final_loss_pm, self.cfg.training.iter)
            self.log_add_scalar('Eval/LossDepth', final_loss_d, self.cfg.training.iter)
            self.log_add_scalar('Eval/LossRaymap', final_loss_ray, self.cfg.training.iter)
            self.log_add_scalar('Eval/LossPointmap_ctx', final_loss_pm_ctx, self.cfg.training.iter)
            self.log_add_scalar('Eval/LossDepth_ctx', final_loss_d_ctx, self.cfg.training.iter)
            self.log_add_scalar('Eval/LossRaymap_ctx', final_loss_ray_ctx, self.cfg.training.iter)

        self.vit.train()
        return final_loss_flow

    def _build_occ_rae(self):
        if self.occ_rae is not None:
            return
        self.occ_rae = OccRAE(
            weights_path=self.cfg.model.occany_recon_ckpt,
            device=str(self.device),
            encode_layer=int(self.cfg.model.encode_layer),
        )
        self.occ_rae.eval()
        if self.is_master:
            print(f"[INFO] Built shared OccRAE encoder at 518x518, "
                  f"encode_layer={int(self.cfg.model.encode_layer)}")

    def fit(self, log_iter=1000):
        # Build sequence records and load PreprocessedSequenceDataset
        preprocess_data_root = Path(self.cfg.dataset.preprocess_data_root).expanduser().resolve()
        
        train_processed_roots = self.cfg.dataset.get("processed_roots")
        if not train_processed_roots:
            from occrae.dataset.preprocessed_sequence import DEFAULT_TRAIN_PROCESSED_ROOTS
            train_processed_roots = list(DEFAULT_TRAIN_PROCESSED_ROOTS)
            
        use_seq_cache = True
        train_records, train_stats = build_sequence_records(
            preprocess_data_root=preprocess_data_root,
            processed_roots=train_processed_roots,
            subsampling_rate=self.cfg.dataset.subsampling_rate,
            max_stride=self.cfg.dataset.max_stride,
            frame_stride=self.cfg.dataset.get("frame_stride"),
            use_cache=use_seq_cache,
        )
        if self.is_master:
            print(f"Train dataset stats: {train_stats}")

        max_train = int(self.cfg.dataset.get("max_train_samples", -1))
        if max_train > 0:
            train_records = train_records[:max_train]
            repeat_count = max(1, self.cfg.training.virtual_epoch * self.world_size)
            train_records = train_records * repeat_count
            if self.is_master:
                print(f"[OVERFIT] Limiting to {max_train} train sample(s), repeated {repeat_count}x -> {len(train_records)} records")

        self.train_data = PreprocessedSequenceDataset(
            preprocess_data_root=preprocess_data_root,
            records=train_records,
            load_gt=True,
        )
        self.train_sampler = ProcessedRootBatchSampler(
            self.train_data,
            batch_size=self.cfg.training.bsize,
            shuffle=True,
            drop_last=True,
            seed=self.cfg.training.seed,
            rank=self.rank,
            world_size=self.world_size
        )
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            collate_fn=collate_preprocessed,
        )

        try:
            val_processed_roots = self.cfg.dataset.get("val_processed_roots", self.cfg.dataset.get("processed_roots"))
            if not val_processed_roots:
                from occrae.dataset.preprocessed_sequence import DEFAULT_VAL_PROCESSED_ROOTS
                val_processed_roots = list(DEFAULT_VAL_PROCESSED_ROOTS)
                
            val_records, val_stats = build_sequence_records(
                preprocess_data_root=preprocess_data_root,
                processed_roots=val_processed_roots,
                subsampling_rate=self.cfg.dataset.subsampling_rate,
                max_stride=self.cfg.dataset.max_stride,
                frame_stride=self.cfg.dataset.get("frame_stride"),
                use_cache=use_seq_cache,
            )
            if self.is_master:
                print(f"Val dataset stats: {val_stats}")

            max_val = int(self.cfg.dataset.get("max_val_samples", -1))
            if max_val > 0:
                val_records = val_records[:max_val]
                repeat_count = max(1, self.world_size)
                val_records = val_records * repeat_count
                if self.is_master:
                    print(f"[OVERFIT] Limiting to {max_val} val sample(s), repeated {repeat_count}x -> {len(val_records)} records")

            self.test_data = PreprocessedSequenceDataset(
                preprocess_data_root=preprocess_data_root,
                records=val_records,
                load_gt=True,
            )
            self.test_sampler = ProcessedRootBatchSampler(
                self.test_data,
                batch_size=self.cfg.training.bsize,
                shuffle=False,
                drop_last=False,
                seed=self.cfg.training.seed,
                rank=self.rank,
                world_size=self.world_size,
            )
            self.test_loader = DataLoader(
                self.test_data,
                batch_sampler=self.test_sampler,
                num_workers=self.cfg.training.num_workers,
                pin_memory=True,
                collate_fn=collate_preprocessed,
            )
        except Exception as e:
            print(f"Eval dataset not loaded or error: {e}")
            self.test_loader = None

        # Build the shared frozen OccRAE encoder once for online token extraction.
        self._build_occ_rae()

        if self.is_master:
            print("Start training:")

        # Initial evaluation for sanity check
        if self.test_loader is not None:
            if self.is_master:
                print("Running initial evaluation for sanity check...")
            self.eval_one_epoch(sanity_check=True)

        start = time.time()

        for e in range(self.cfg.training.global_epoch, self.cfg.training.epoch + 1):
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                print("End of training: reached max iterations")
                break
            
            if self.distributed:
                self.train_sampler.set_epoch(e)

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
