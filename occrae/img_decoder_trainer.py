"""OccRAE Image Decoder Trainer.

Trains a GeneralDecoder_Variable (MAE-style) to reconstruct RGB images from
frozen OccRAE 3-level backbone features (layers 19, 27, 33 of DA3-Giant-1.1).
Losses: L1 + LPIPS + GAN (DINO discriminator), following GLD's stage-1 recipe.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy

import numpy as np

from omegaconf import OmegaConf
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoConfig

from disc import (
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1.decoders import GeneralDecoder_Variable
from utils.optim_utils import build_optimizer, build_scheduler

from occany.model.occ_rae import OccRAE
from occrae.abstract_trainer import Trainer
from occany.datasets import get_data_loader


def _select_gan_losses(disc_kind: str, gen_kind: str):
    d_fn = {"hinge": hinge_d_loss, "vanilla": vanilla_d_loss}[disc_kind]
    g_fn = {"vanilla": vanilla_g_loss}[gen_kind]
    return d_fn, g_fn


@torch.no_grad()
def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def _calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


class ImgDecoderTrainer(Trainer):
    def __init__(self, args, cfg, device, rank, world_size, distributed):
        args.is_master = rank == 0
        args.writer_log = str(cfg.get("writer_log", ""))
        super().__init__(args)

        self.args = args
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.is_master = rank == 0

        if self.is_master:
            print(f"Init OccRAE Image Decoder Trainer on [GPU{rank}]")

        self.view_chunk_size = int(cfg.training.get("view_chunk_size", 0))

        self._build_occ_rae()
        self._build_decoder()
        self._build_discriminator()
        self._build_losses()

        if self.device.type != "cpu" and self.cfg.training.get("dtype", "bfloat16") == "bfloat16":
            self.autocast_ctx = autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast_ctx = nullcontext()

        if self.is_master:
            n_dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
            n_disc = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            print(f"MAE decoder trainable params: {n_dec / 1e6:.2f}M")
            print(f"Discriminator trainable params: {n_disc / 1e6:.2f}M")
            print("Config: ", self.cfg)

    def _build_occ_rae(self):
        occrae_cfg = self.cfg.occrae
        self.occ_rae = OccRAE(
            weights_path=str(occrae_cfg.weights_path),
            device=str(self.device),
            encode_layer=int(occrae_cfg.encode_layer),
        )
        self.occ_rae.eval()
        self.occ_rae.requires_grad_(False)
        self.occ_rae.to(torch.bfloat16)
        if self.is_master:
            print(f"[INFO] Built frozen OccRAE encoder at 518x518 (bfloat16), "
                  f"encode_layer={occrae_cfg.encode_layer}")

    def _build_decoder(self):
        decoder_cfg = self.cfg.decoder
        dec_config = AutoConfig.from_pretrained(str(decoder_cfg.config_path))
        dec_config.hidden_size = int(decoder_cfg.hidden_size)
        dec_config.patch_size = int(decoder_cfg.patch_size)

        self.decoder = GeneralDecoder_Variable(dec_config, base_image_size=(518, 518)).to(self.device)
        self.decoder.train()

        self.ema_decoder = deepcopy(self.decoder).to(self.device).eval()
        self.ema_decoder.requires_grad_(False)

        if self.distributed:
            self.ddp_decoder = DDP(self.decoder, device_ids=[self.device.index], broadcast_buffers=False)
        else:
            self.ddp_decoder = self.decoder
        self.decoder_module = self.ddp_decoder.module if isinstance(self.ddp_decoder, DDP) else self.ddp_decoder

        training_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        self.dec_optimizer, _ = build_optimizer(
            [p for p in self.decoder.parameters() if p.requires_grad], training_cfg
        )
        self._training_cfg = training_cfg

    def _build_discriminator(self):
        gan_cfg = OmegaConf.to_container(self.cfg.gan, resolve=True)
        disc_cfg = gan_cfg.get("disc", {})

        self.discriminator, self.disc_aug = build_discriminator(disc_cfg, self.device)
        if self.distributed:
            self.ddp_disc = DDP(self.discriminator, device_ids=[self.device.index], broadcast_buffers=False)
        else:
            self.ddp_disc = self.discriminator
        self.ddp_disc.train()

        disc_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        self.disc_optimizer, _ = build_optimizer(disc_params, disc_cfg)

        loss_cfg = gan_cfg.get("loss", {})
        self.disc_loss_fn, self.gen_loss_fn = _select_gan_losses(
            str(loss_cfg.get("disc_loss", "hinge")),
            str(loss_cfg.get("gen_loss", "vanilla")),
        )

        self.perceptual_weight = float(loss_cfg.get("perceptual_weight", 1.0))
        self.recon_weight = float(loss_cfg.get("recon_weight", 1.0))
        self.disc_weight = float(loss_cfg.get("disc_weight", 0.75))
        self.gan_start_epoch = int(loss_cfg.get("disc_start", 6))
        self.disc_update_epoch = int(loss_cfg.get("disc_upd_start", 4))
        self.lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
        self.disc_updates = int(loss_cfg.get("disc_updates", 1))
        self.max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))

        self._disc_cfg = disc_cfg

    def _build_losses(self):
        lpips_root = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "checkpoints", "GLD_pretrained_models")
        self.lpips = LPIPS(ckpt_root=lpips_root).to(self.device)
        self.lpips.eval()

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.decoder_module.load_state_dict(ckpt["decoder"])
        self.ema_decoder.load_state_dict(ckpt["ema_decoder"])
        self.dec_optimizer.load_state_dict(ckpt["optimizer"])
        if "disc" in ckpt:
            disc_mod = self.ddp_disc.module if isinstance(self.ddp_disc, DDP) else self.ddp_disc
            disc_mod.load_state_dict(ckpt["disc"])
            self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        self._resume_epoch = ckpt.get("epoch", 0)
        self._resume_step = ckpt.get("step", 0)
        self._resume_dec_scheduler = ckpt.get("dec_scheduler", None)
        self._resume_disc_scheduler = ckpt.get("disc_scheduler", None)
        if self.is_master:
            print(f"Resumed from {path} (epoch={self._resume_epoch}, step={self._resume_step})")

    def _save_checkpoint(self, path, step, epoch, dec_scheduler=None, disc_scheduler=None):
        state = {
            "step": step,
            "epoch": epoch,
            "decoder": self.decoder_module.state_dict(),
            "ema_decoder": self.ema_decoder.state_dict(),
            "optimizer": self.dec_optimizer.state_dict(),
            "disc": (self.ddp_disc.module if isinstance(self.ddp_disc, DDP) else self.ddp_disc).state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
        }
        if dec_scheduler is not None:
            state["dec_scheduler"] = dec_scheduler.state_dict()
        if disc_scheduler is not None:
            state["disc_scheduler"] = disc_scheduler.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            backup_path = path.replace(".pt", "_backup.pt")
            os.replace(path, backup_path)
        torch.save(state, path)

    def _build_data_loaders(self):
        train_dataset_str = str(self.cfg.dataset.train_dataset)
        test_dataset_str = self.cfg.dataset.get("test_dataset", None)
        batch_size = int(self.cfg.training.batch_size)
        num_workers = int(self.cfg.training.num_workers)

        if self.is_master:
            print(f"Building train dataset: {train_dataset_str}")
        self.train_loader = get_data_loader(
            train_dataset_str,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        if test_dataset_str:
            try:
                if self.is_master:
                    print(f"Building test dataset: {test_dataset_str}")
                self.test_loader = get_data_loader(
                    test_dataset_str,
                    batch_size=batch_size,
                    num_workers=self.cfg.training.get("val_num_workers", 2),
                    shuffle=False,
                    drop_last=False,
                )
            except Exception as e:
                if self.is_master:
                    print(f"Eval dataset not loaded: {e}")
                self.test_loader = None
        else:
            self.test_loader = None

    @torch.no_grad()
    def _run_validation(self, step, use_lpips=True):
        self.ema_decoder.eval()

        # Ensure the test dataset and sampler use a fixed epoch for reproducibility
        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            self.test_loader.dataset.set_epoch(0)
        if hasattr(self.test_loader, 'sampler') and hasattr(self.test_loader.sampler, 'set_epoch'):
            self.test_loader.sampler.set_epoch(0)

        val_rec = 0.0
        val_lpips = 0.0
        n_batches = 0

        encoder_mean = self.occ_rae.encoder_mean
        encoder_std = self.occ_rae.encoder_std

        n_vis_samples = 8
        vis_samples = []
        n_collected = 0

        for batch in tqdm(self.test_loader, desc="Validation", leave=False, disable=not self.is_master):
            images_enc = torch.stack([view['img'] for view in batch], dim=1).to(self.device, non_blocking=True)
            b, v, c, h, w = images_enc.shape

            images_01 = images_enc * encoder_std[None] + encoder_mean[None]
            images_01_flat = images_01.view(b * v, c, h, w)

            with self.autocast_ctx:
                latents = self.occ_rae.encode(images_enc)
                multi_feats = self.occ_rae.decode_to_features(latents, num_levels=3)
                z = torch.cat(multi_feats, dim=-1)
                z = z.view(b * v, z.shape[-2], z.shape[-1])
            del images_enc, latents, multi_feats

            vcs = self.view_chunk_size
            z_chunks = z.split(vcs, dim=0) if vcs > 0 else [z]
            img_chunks = images_01_flat.split(vcs, dim=0) if vcs > 0 else [images_01_flat]

            rec_parts = []
            for z_c, img_c in zip(z_chunks, img_chunks):
                with self.autocast_ctx:
                    out_c = self.ema_decoder(z_c, input_size=(h, w), drop_cls_token=False).logits
                    x_rec_c = self.ema_decoder.unpatchify(out_c, (h, w))
                    x_rec_c = x_rec_c * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
                val_rec += F.l1_loss(x_rec_c, img_c).item()
                if use_lpips:
                    real_c = img_c * 2.0 - 1.0
                    recon_c = x_rec_c * 2.0 - 1.0
                    val_lpips += self.lpips(real_c, recon_c).mean().item()
                rec_parts.append(x_rec_c)
            n_batches += len(z_chunks)

            if n_collected < n_vis_samples and self.is_master:
                x_rec = torch.cat(rec_parts, dim=0)
                x_rec_bv = x_rec.view(b, v, c, h, w)
                labels = [batch[0]['label'][i] for i in range(b)]
                take = min(n_vis_samples - n_collected, b)
                for i in range(take):
                    vis_samples.append((labels[i], images_01[i], x_rec_bv[i]))
                n_collected += take
            del z, z_chunks, rec_parts

        metrics = torch.tensor([val_rec, val_lpips, float(n_batches)], device=self.device)
        if self.distributed:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_rec, total_lpips, total_batches = metrics.tolist()

        if self.is_master and vis_samples and self.writer is not None:
            for label, gt_views, rec_views in vis_samples:
                pairs = torch.cat([rec_views, gt_views], dim=3).clamp(0, 1)
                grid = vutils.make_grid(pairs, nrow=1, padding=2, normalize=False)
                grid = torch.clip(grid * 255, 0, 255).to(torch.uint8)
                self.writer.add_image(f"Val/{label}", grid, step)

        return {
            "val/recon": total_rec / max(total_batches, 1),
            "val/lpips": total_lpips / max(total_batches, 1),
        }

    def fit(self):
        seed = int(self.cfg.training.get("seed", 0)) + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self._build_data_loaders()

        cfg = self.cfg
        training_cfg = self._training_cfg

        num_epochs = int(cfg.training.epochs)
        ema_decay = float(cfg.training.ema_decay)
        log_interval = int(cfg.training.get("log_interval", 100))

        checkpoint_dir = str(cfg.get("checkpoint_dir", "results/occrae_img_decoder/ckpts/"))
        if self.is_master:
            os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_path = os.path.join(checkpoint_dir, "current.pt")
        if os.path.isfile(ckpt_path):
            self._load_checkpoint(ckpt_path)

        steps_per_epoch = len(self.train_loader)
        if steps_per_epoch == 0:
            raise RuntimeError("Dataloader returned zero batches.")

        dec_scheduler = None
        if training_cfg.get("scheduler"):
            dec_scheduler, _ = build_scheduler(self.dec_optimizer, steps_per_epoch, training_cfg)
        disc_scheduler = None
        if self._disc_cfg.get("scheduler"):
            disc_scheduler, _ = build_scheduler(self.disc_optimizer, steps_per_epoch, self._disc_cfg)

        start_epoch = getattr(self, "_resume_epoch", 0)
        global_step = getattr(self, "_resume_step", 0)

        if dec_scheduler is not None and getattr(self, "_resume_dec_scheduler", None):
            dec_scheduler.load_state_dict(self._resume_dec_scheduler)
        if disc_scheduler is not None and getattr(self, "_resume_disc_scheduler", None):
            disc_scheduler.load_state_dict(self._resume_disc_scheduler)

        encoder_mean = self.occ_rae.encoder_mean
        encoder_std = self.occ_rae.encoder_std
        last_layer = self.decoder_module.decoder_pred.weight

        gan_start_step = self.gan_start_epoch * steps_per_epoch
        disc_update_step = self.disc_update_epoch * steps_per_epoch
        lpips_start_step = self.lpips_start_epoch * steps_per_epoch

        if self.is_master:
            print(f"Training {num_epochs} epochs, bs={cfg.training.batch_size}/GPU, "
                  f"{steps_per_epoch} steps/epoch, world_size={self.world_size}")

        for epoch in range(start_epoch, num_epochs):
            self.ddp_decoder.train()

            if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_metrics: Dict[str, float] = defaultdict(float)
            num_batches = 0

            pbar = tqdm(self.train_loader, total=steps_per_epoch,
                        desc=f"Epoch {epoch}/{num_epochs}") if self.is_master else self.train_loader

            for step, batch in enumerate(pbar):
                use_gan = global_step >= gan_start_step and self.disc_weight > 0.0
                train_disc = global_step >= disc_update_step and self.disc_weight > 0.0
                use_lpips = global_step >= lpips_start_step and self.perceptual_weight > 0.0

                images_enc = torch.stack([view['img'] for view in batch], dim=1).to(self.device, non_blocking=True)
                b, v, c, h, w = images_enc.shape

                images_01 = (images_enc * encoder_std[None] + encoder_mean[None]).view(b * v, c, h, w).contiguous()
                real_normed = (images_01 * 2.0 - 1.0).contiguous()

                # --- Frozen OccRAE feature extraction ---
                with torch.no_grad():
                    with self.autocast_ctx:
                        latents = self.occ_rae.encode(images_enc)
                        multi_feats = self.occ_rae.decode_to_features(latents, num_levels=3)
                        z = torch.cat(multi_feats, dim=-1)
                        z = z.view(b * v, z.shape[-2], z.shape[-1])
                del images_enc, latents, multi_feats

                # --- Chunked generator update ---
                self.dec_optimizer.zero_grad(set_to_none=True)
                self.ddp_disc.eval()
                disc_module = self.ddp_disc.module if isinstance(self.ddp_disc, DDP) else self.ddp_disc

                vcs = self.view_chunk_size
                z_chunks = z.split(vcs, dim=0) if vcs > 0 else [z]
                img_chunks = images_01.split(vcs, dim=0) if vcs > 0 else [images_01]
                real_chunks = real_normed.split(vcs, dim=0) if vcs > 0 else [real_normed]
                n_chunks = len(z_chunks)
                chunk_scale = 1.0 / n_chunks

                acc_rec = 0.0
                acc_lpips = 0.0
                acc_gan = 0.0
                acc_total = 0.0

                for i, (z_c, img_c, real_c) in enumerate(zip(z_chunks, img_chunks, real_chunks)):
                    is_last = (i == n_chunks - 1)
                    sync_ctx = nullcontext() if (is_last or not self.distributed) else self.ddp_decoder.no_sync()
                    with sync_ctx:
                        with self.autocast_ctx:
                            logits = self.ddp_decoder(z_c, input_size=(h, w), drop_cls_token=False).logits
                            recon = self.decoder_module.unpatchify(logits, (h, w))
                            recon_01 = recon * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
                            recon_pm1 = recon_01 * 2.0 - 1.0

                            rec_loss = F.l1_loss(recon_01, img_c)

                            if use_lpips:
                                lpips_loss = self.lpips(real_c, recon_pm1)
                            else:
                                lpips_loss = rec_loss.new_zeros(())

                            recon_total = self.recon_weight * rec_loss + self.perceptual_weight * lpips_loss

                            if use_gan:
                                crop_h = min(h, 224) // 8 * 8
                                crop_w = min(w, 224) // 8 * 8
                                i_crop, j_crop, h_crop, w_crop = T.RandomCrop.get_params(
                                    recon_pm1, output_size=(crop_h, crop_w)
                                )
                                recon_crop = TF.crop(recon_pm1, i_crop, j_crop, h_crop, w_crop)
                                fake_aug = self.disc_aug.aug(recon_crop)
                                logits_fake, _ = disc_module(fake_aug, None)
                                gan_loss = self.gen_loss_fn(logits_fake)
                            else:
                                gan_loss = torch.zeros_like(recon_total)

                        if use_gan:
                            adaptive_weight = _calculate_adaptive_weight(
                                recon_total, gan_loss, last_layer, self.max_d_weight
                            )
                            chunk_loss = (recon_total + self.disc_weight * adaptive_weight * gan_loss) * chunk_scale
                        else:
                            chunk_loss = recon_total * chunk_scale

                        chunk_loss.backward()

                    acc_rec += rec_loss.detach().item() * chunk_scale
                    acc_lpips += lpips_loss.detach().item() * chunk_scale
                    acc_gan += gan_loss.detach().item() * chunk_scale
                    acc_total += chunk_loss.detach().item()
                    del logits, recon, recon_01, recon_pm1, chunk_loss

                rec_loss = acc_rec
                lpips_loss = acc_lpips
                gan_loss = acc_gan
                total_loss = acc_total

                self.dec_optimizer.step()

                if dec_scheduler is not None:
                    dec_scheduler.step()

                _update_ema(self.ema_decoder, self.decoder_module, ema_decay)

                # --- Chunked discriminator update ---
                disc_metrics: Dict[str, torch.Tensor] = {}
                if train_disc:
                    self.ddp_decoder.eval()
                    self.ddp_disc.train()

                    for _ in range(self.disc_updates):
                        self.disc_optimizer.zero_grad(set_to_none=True)

                        crop_h = min(h, 224) // 8 * 8
                        crop_w = min(w, 224) // 8 * 8
                        i_crop, j_crop, h_crop, w_crop = T.RandomCrop.get_params(
                            real_normed, output_size=(crop_h, crop_w)
                        )

                        fake_crops = []
                        with self.autocast_ctx:
                            for z_c in z_chunks:
                                with torch.no_grad():
                                    logits_d = self.ddp_decoder(z_c, input_size=(h, w), drop_cls_token=False).logits
                                    recon_d = self.decoder_module.unpatchify(logits_d, (h, w))
                                    recon_d_01 = recon_d * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
                                    recon_d_pm1 = recon_d_01 * 2.0 - 1.0
                                    fake_det = recon_d_pm1.clamp(-1.0, 1.0)
                                    fake_det = torch.round((fake_det + 1.0) * 127.5) / 127.5 - 1.0
                                    fake_crops.append(TF.crop(fake_det, i_crop, j_crop, h_crop, w_crop))
                                    del logits_d, recon_d, recon_d_01, recon_d_pm1, fake_det

                            fake_crop = torch.cat(fake_crops, dim=0)
                            del fake_crops
                            real_crop = TF.crop(real_normed, i_crop, j_crop, h_crop, w_crop)
                            fake_input = self.disc_aug.aug(fake_crop)
                            real_input = self.disc_aug.aug(real_crop)
                            logits_fake_d, logits_real_d = self.ddp_disc(fake_input, real_input)
                            d_loss = self.disc_loss_fn(logits_real_d, logits_fake_d)
                            accuracy = (logits_real_d > logits_fake_d).float().mean()

                        d_loss.backward()
                        self.disc_optimizer.step()

                        disc_metrics = {
                            "disc_loss": d_loss.detach(),
                            "logits_real": logits_real_d.detach().mean(),
                            "logits_fake": logits_fake_d.detach().mean(),
                            "disc_accuracy": accuracy.detach(),
                        }
                        if disc_scheduler is not None:
                            disc_scheduler.step()

                    self.ddp_disc.eval()
                    self.ddp_decoder.train()

                del z, z_chunks, images_01, real_normed, img_chunks, real_chunks

                epoch_metrics["recon"] += rec_loss
                epoch_metrics["lpips"] += lpips_loss
                epoch_metrics["gan"] += gan_loss
                epoch_metrics["total"] += total_loss
                num_batches += 1

                if log_interval > 0 and global_step % log_interval == 0 and self.is_master:
                    self.log_add_scalar("Train/LossRecon", rec_loss, global_step)
                    self.log_add_scalar("Train/LossLPIPS", lpips_loss, global_step)
                    self.log_add_scalar("Train/LossGAN", gan_loss, global_step)
                    self.log_add_scalar("Train/LossTotal", total_loss, global_step)
                    self.log_add_scalar("Train/LR", self.dec_optimizer.param_groups[0]["lr"], global_step)
                    if disc_metrics:
                        self.log_add_scalar("Train/DiscLoss", disc_metrics["disc_loss"].item(), global_step)
                        self.log_add_scalar("Train/DiscAccuracy", disc_metrics["disc_accuracy"].item(), global_step)

                    print(
                        f"[Epoch {epoch} | Step {global_step}] "
                        f"recon={rec_loss:.4f} lpips={lpips_loss:.4f} "
                        f"gan={gan_loss:.4f} total={total_loss:.4f}"
                    )

                global_step += 1

            if self.is_master and num_batches > 0:
                epoch_stats = {k: v / num_batches for k, v in epoch_metrics.items()}
                print(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
                for k, v_ in epoch_stats.items():
                    self.log_add_scalar(f"Epoch/{k}", v_, global_step)

            if self.test_loader is not None:
                if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
                    self.test_loader.dataset.set_epoch(epoch)
                if self.is_master:
                    print(f"Running end-of-epoch validation (epoch {epoch})...")
                torch.cuda.empty_cache()
                val_stats = self._run_validation(step=global_step, use_lpips=use_lpips)
                if self.is_master:
                    for k, v_ in val_stats.items():
                        self.log_add_scalar(k, v_, global_step)
                    print(f"[Val @ epoch {epoch}] "
                          + ", ".join(f"{k}: {v_:.4f}" for k, v_ in val_stats.items()))

            if self.is_master:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, "current.pt"),
                    global_step, epoch + 1,
                    dec_scheduler=dec_scheduler,
                    disc_scheduler=disc_scheduler,
                )

    def run(self):
        self.fit()
