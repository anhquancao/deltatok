"""Trainer for the DeltaTok delta-token autoencoder.

Trains DeltaTok (Yusuf Dalva et al., CVPR 2026) over features from the frozen
OccRAE/DA3 backbone. Each item is a V-frame sequence; we compress every
consecutive pair (frame_t, frame_{t+1}) into a single delta token z and decode
(z, frame_t) back to predicted features for frame_{t+1}.
"""
import os
import math
import time
from collections import deque
from contextlib import nullcontext
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import croco.utils.misc as misc  # MetricLogger for line-by-line, SLURM-friendly logs (matches training_da3.py)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTLayer,
    DINOv3ViTRopePositionEmbedding,
)

# We avoid `from models.deltatok import DeltaTok` because that pulls in `training.base`,
# which depends on Lightning and dataset modules we don't have. The launcher
# (`train_deltatok.py`) prepends `third_party/deltatok` to sys.path.
from models.gated_attn import enable_gated_attn
from models.qk_norm import enable_dinov3_qk_norm

from occrae.abstract_trainer import Trainer
from occrae.metric import DeltaTokEvalMetric
from occrae.visualization_helper import _log_viz_sample
from occany.datasets import get_data_loader
from occany.model.occ_rae import OccRAE
from occany.utils.helpers import depth2rgb
from occany.loss import PointmapLoss, DepthLosses, RaymapLoss


class DeltaTokModule(nn.Module):
    """DeltaTok tokenizer: encode (x_prev_feat, x_feat) -> z, decode (z, x_prev_feat) -> x_hat.

    Differs from third_party/deltatok/models/deltatok.py by skipping the upstream
    Lightning-based ``training.base`` dependency and scaling ``intermediate_size``
    by ``mlp_ratio * hidden_size`` so the MLP width matches the backbone (upstream
    pins it to the DINOv3-ViT-B template, which underprovisions the MLP for a
    1536-d DA3-Giant backbone).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        patch_size: int,
        initializer_range: float = 0.02,
        num_hidden_layers: int = 12,
        layer_scale_init: float = 1e-5,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        use_swiglu: bool = True,
        use_rope_aug: bool = False,
        mlp_ratio: int = 4,
        alt_start: int = 4,
    ):
        super().__init__()
        self._use_rope_aug = use_rope_aug
        # DA3-style local/global alternation. Blocks at index i are global iff
        # ``i >= alt_start and i % 2 == 1``; otherwise they run per-camera.
        # Set ``alt_start = num_hidden_layers`` (or any value >= layer count) to
        # disable cross-camera attention entirely.
        self.alt_start = int(alt_start)

        # Construct DINOv3ViTConfig directly to avoid depending on the gated
        # facebook/dinov3-vitb16-pretrain-lvd1689m HF repo.
        cfg = DINOv3ViTConfig()
        cfg._attn_implementation = "sdpa"
        cfg.hidden_size = int(hidden_size)
        cfg.num_attention_heads = int(num_heads)
        cfg.patch_size = int(patch_size)
        cfg.initializer_range = float(initializer_range)
        cfg.intermediate_size = int(mlp_ratio * cfg.hidden_size)

        if use_swiglu:
            cfg.use_gated_mlp = True
            cfg.intermediate_size = max(1, (2 * cfg.intermediate_size) // 3)
            cfg.hidden_act = "silu"

        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(cfg)

        self.z_embed = nn.Embedding(1, cfg.hidden_size)
        nn.init.trunc_normal_(self.z_embed.weight, std=cfg.initializer_range)
        self.xy_embed = nn.Embedding(2, cfg.hidden_size)
        nn.init.trunc_normal_(self.xy_embed.weight, std=cfg.initializer_range)

        self.encoder_blocks = nn.ModuleList(
            [DINOv3ViTLayer(cfg) for _ in range(num_hidden_layers)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DINOv3ViTLayer(cfg) for _ in range(num_hidden_layers)]
        )
        for blk in chain(self.encoder_blocks, self.decoder_blocks):
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=cfg.initializer_range)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            blk.layer_scale1.lambda1.data.fill_(layer_scale_init)
            blk.layer_scale2.lambda1.data.fill_(layer_scale_init)
            if use_qk_norm:
                enable_dinov3_qk_norm(blk)
            if use_gated_attn:
                enable_gated_attn(blk)

        self.norm = nn.LayerNorm(cfg.hidden_size, cfg.layer_norm_eps)
        self._rope_cache: dict = {}

    def train(self, mode: bool = True):
        super().train(mode)
        # DINOv3ViTRopePositionEmbedding only augments coords while .training; keep
        # it frozen unless we explicitly opted into rope augmentation.
        if not self._use_rope_aug:
            self.rope_embeddings.eval()
        return self

    def _compute_rope(self, height: int, width: int, device, dtype):
        key = (int(height), int(width), device, dtype, self.rope_embeddings.training)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        dummy = torch.zeros(1, 3, int(height), int(width), device=device, dtype=dtype)
        rope = self.rope_embeddings(dummy)
        if not self.rope_embeddings.training:
            self._rope_cache[key] = rope
        return rope

    def _compute_global_rope(self, height: int, width: int, num_cameras: int, device, dtype):
        """DA3-style ``pos_nodiff`` rope for global (cross-camera) attention.

        Mirrors third_party/Depth-Anything-3/.../vision_transformer.py
        ``_prepare_rope``: on global layers DA3 sets every spatial patch to the
        same position (``pos_nodiff = 1``) so RoPE between any two patches is
        identity — cross-camera attention sees no spurious geometry from
        camera ordering, and camera identity is carried by separate tokens
        (DA3 uses ``cam_token``; here, the per-camera ``z`` tokens).

        Z tokens stay at the implicit "position 0" / identity rotation: HF's
        ``apply_rotary_pos_emb`` skips the leading ``num_tokens - num_patches``
        tokens, which matches DA3's prefix-at-0, patches-at-1 split. We pick
        rope at patch coordinate ``(1, 1)`` (index ``Wp + 1``) rather than
        ``(0, 0)`` so the patch rope is non-identity and z↔patch attention
        keeps a non-trivial relative rotation.
        """
        Hp = int(height) // self.rope_embeddings.config.patch_size
        Wp = int(width) // self.rope_embeddings.config.patch_size
        num_positions = int(num_cameras) * Hp * Wp
        cos, sin = self._compute_rope(height, width, device, dtype)
        # Index Wp + 1 = patch coordinate (1, 1) in row-major order, matching
        # DA3's pos_nodiff = 1. Index 0 would be (0, 0) / identity rotation,
        # which would also collapse z↔patch rope.
        idx = Wp + 1
        return (
            cos[idx : idx + 1].expand(num_positions, -1).contiguous(),
            sin[idx : idx + 1].expand(num_positions, -1).contiguous(),
        )

    def _is_global_layer(self, i: int) -> bool:
        return i >= self.alt_start and (i % 2 == 1)

    @staticmethod
    def _double_rope(rope):
        """Tile rope along the position dim (-2) so it covers a [prev, next] sequence."""
        return (torch.cat((rope[0], rope[0]), -2), torch.cat((rope[1], rope[1]), -2))

    def encode(
        self,
        x_prev: torch.Tensor,
        x: torch.Tensor,
        rope_local,
        rope_global,
    ) -> torch.Tensor:
        """Encode (x_prev, x) into N delta tokens (one per camera) with DA3-style alternation.

        Inputs
        ------
        x_prev, x : (M, N, P, C) where M is the pair-batch dim, N = num_cameras,
                    P = patches per camera. For single-camera datasets ``N=1``
                    and the alternation degenerates to a per-pair pass.
        rope_local  : single-camera rope of length ``P`` (un-doubled).
        rope_global : DA3-style ``pos_nodiff`` rope of length ``N*P`` (un-doubled),
                      built by ``_compute_global_rope``. Every position is
                      identical, so RoPE between any two patches collapses to
                      identity on global layers.

        Returns
        -------
        z : (M, N, 1, C) — one delta token per camera, all initialized from a
            shared ``z_embed`` parameter. Each camera's z is updated at every
            block: on local blocks via per-camera self-attention with that
            camera's ``[prev, next]`` strip, on global blocks via cross-camera
            attention over all N z's plus all N cameras' spatial tokens. This
            keeps the prefix-token structure compatible with HF's
            ``apply_rotary_pos_emb`` (which skips the leading
            ``num_tokens - num_patches`` tokens).
        """
        M, N, P, C = x_prev.shape

        # N z tokens, one per camera; all initialized from the shared z_embed.
        z = self.z_embed.weight[None, None].expand(M, N, 1, C).contiguous()

        prev_spatials = x_prev + self.xy_embed.weight[0]   # (M, N, P, C)
        next_spatials = x + self.xy_embed.weight[1]        # (M, N, P, C)
        # Local rope of length 2P, layout [pos, pos] — prev and next at the
        # same camera share positions.
        local_rope_full = self._double_rope(rope_local)
        # Global rope of length 2*N*P. All positions are identical
        # (DA3 pos_nodiff), so layout doesn't matter — patch↔patch rope
        # is identity regardless of camera or prev/next ordering.
        global_rope_full = self._double_rope(rope_global)

        for i, blk in enumerate(self.encoder_blocks):
            if self._is_global_layer(i) and N > 1:
                # Global layout: [z_0..z_{N-1}, all_prev (cam-major), all_next (cam-major)]
                # so prefix = N (the z's) and rope length = 2 * N * P.
                z_flat = z.reshape(M, N, C)
                prev_flat = prev_spatials.reshape(M, N * P, C)
                next_flat = next_spatials.reshape(M, N * P, C)
                hidden = torch.cat([z_flat, prev_flat, next_flat], dim=1)
                hidden = blk(hidden, position_embeddings=global_rope_full)
                z = hidden[:, :N].reshape(M, N, 1, C)
                prev_spatials = hidden[:, N : N + N * P].reshape(M, N, P, C)
                next_spatials = hidden[:, N + N * P :].reshape(M, N, P, C)
            else:
                # Local: per-camera [z_n, prev_n, next_n]. Each camera's z is
                # updated independently here.
                hidden = torch.cat([z, prev_spatials, next_spatials], dim=2)
                seq_len = hidden.shape[2]
                hidden = hidden.reshape(M * N, seq_len, C)
                hidden = blk(hidden, position_embeddings=local_rope_full)
                hidden = hidden.reshape(M, N, seq_len, C)
                z = hidden[:, :, :1]
                prev_spatials = hidden[:, :, 1 : 1 + P]
                next_spatials = hidden[:, :, 1 + P :]

        return self.norm(z)

    def decode(
        self,
        z: torch.Tensor,
        x_prev: torch.Tensor,
        rope_local,
        rope_global,
    ) -> torch.Tensor:
        """Decode predicted features for x given (z, x_prev), with DA3-style alternation.

        Inputs
        ------
        z      : (M, N, 1, C) — one delta token per camera (also accepts
                  (M, N, C) for convenience).
        x_prev : (M, N, P, C)
        rope_local  : length-P single-camera rope.
        rope_global : length-(N*P) DA3-style ``pos_nodiff`` rope (uniform
                      position across all patches).

        Returns x_hat : (M, N, P, C).

        Each camera's z is threaded through every block (local + global), so
        z is updated at all layers. On local blocks each camera attends to its
        own ``[z_n, x_prev_n]`` sequence; on global blocks all N z's attend
        across all cameras' spatial tokens.

        At init, layer_scale_init=1e-5 makes every block ~identity, so x_hat ≈ x_prev
        (frame_t copied into the frame_{t+1} slot). With adjacent frames sharing
        most of the scene, OccRAE decodes plausible depth before any DeltaTok
        training, so a low recon loss / reasonable pred depth at iter 0 is
        expected, not a sign that the encoder/decoder were initialized to mirror
        each other.
        """
        M, N, P, C = x_prev.shape

        if z.dim() == 3:
            # Accept (M, N, C) for convenience; broadcast to (M, N, 1, C).
            z = z.unsqueeze(2)
        z = z.contiguous()
        spatials = x_prev  # (M, N, P, C)

        for i, blk in enumerate(self.decoder_blocks):
            if self._is_global_layer(i) and N > 1:
                # Global: [z_0..z_{N-1}, all_spatials (cam-major)]; prefix = N, rope length = N*P.
                z_flat = z.reshape(M, N, C)
                hidden = torch.cat([z_flat, spatials.reshape(M, N * P, C)], dim=1)
                hidden = blk(hidden, position_embeddings=rope_global)
                z = hidden[:, :N].reshape(M, N, 1, C)
                spatials = hidden[:, N:].reshape(M, N, P, C)
            else:
                # Local: per-camera [z_n, x_prev_n]; prefix = 1, rope length = P.
                hidden = torch.cat([z, spatials], dim=2)  # (M, N, 1+P, C)
                hidden = hidden.reshape(M * N, 1 + P, C)
                hidden = blk(hidden, position_embeddings=rope_local)
                hidden = hidden.reshape(M, N, 1 + P, C)
                z = hidden[:, :, :1]
                spatials = hidden[:, :, 1:]

        return spatials

    def forward(
        self,
        x_prev: torch.Tensor,
        x: torch.Tensor,
        height: int,
        width: int,
        num_cameras: int = 1,
    ) -> torch.Tensor:
        """Reconstruct ``x`` from the (x_prev, x) pair via N delta tokens (one per camera).

        x_prev, x : (M, N, P, C) where N = num_cameras. For backward compatibility,
                    if x_prev is 3-D ``(M*N, P, C)`` it is interpreted as a single
                    -camera pair-batch (caller must then pass ``num_cameras=1``).
        Returns x_hat of the same shape as x.
        """
        if x_prev.dim() == 3:
            x_prev = x_prev.unsqueeze(1)
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        rope_local = self._compute_rope(height, width, x_prev.device, x_prev.dtype)
        rope_global = self._compute_global_rope(
            height, width, int(num_cameras), x_prev.device, x_prev.dtype
        )
        z = self.encode(x_prev, x, rope_local, rope_global)
        x_hat = self.decode(z, x_prev, rope_local, rope_global)

        if squeeze:
            x_hat = x_hat.squeeze(1)
        return x_hat


def _log_cosh(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Log-cosh loss (numerically-stable). Matches third_party/deltatok default.

    Smooth regression loss that behaves like L2 near zero and like L1 for large
    errors. The mathematical form is ``log(cosh(pred - tgt))``, computed via the
    identity ``log(cosh(x)) = |x| + log(1 + e^{-2|x|}) - log(2)
                           = |x| + softplus(-2|x|) - log(2)``
    which avoids cosh/exp overflow on large residuals.

    Why use it (vs MSE / L1):
      - Near zero: ``log(cosh(x)) ≈ x^2 / 2`` — smooth, well-behaved gradients
        like MSE.
      - Far from zero: ``log(cosh(x)) ≈ |x| - log(2)`` — grows linearly, so it is
        robust to outliers like L1/Huber.
      - Unlike Huber, it has no threshold hyperparameter and is C^∞ (smooth
        everywhere), which plays nicely with optimizers.

    In this trainer it operates on continuous token / latent residuals, where
    targets can have heavy-tailed errors during early training; log-cosh gives
    robustness without sacrificing the smooth quadratic regime at convergence.
    """
    diff = (pred - tgt).abs()
    return diff + F.softplus(-2.0 * diff) - math.log(2.0)


def _print_param_breakdown(model: nn.Module, archi: str) -> None:
    """Print per-component parameter counts for a DeltaTokModule."""
    def _count(module: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, train

    components = [
        ("encoder_blocks", model.encoder_blocks),
        ("decoder_blocks", model.decoder_blocks),
        ("rope_embeddings", model.rope_embeddings),
        ("z_embed", model.z_embed),
        ("xy_embed", model.xy_embed),
        ("norm", model.norm),
    ]

    grand_total, grand_train = _count(model)
    print(f"Parameter breakdown for {archi}:")
    for name, sub in components:
        total, train = _count(sub)
        share = (total / grand_total * 100.0) if grand_total else 0.0
        print(
            f"  {name:<16s} total={total / 1e6:8.3f}M  trainable={train / 1e6:8.3f}M  "
            f"({share:5.2f}% of total)"
        )
    print(
        f"  {'TOTAL':<16s} total={grand_total / 1e6:8.3f}M  "
        f"trainable={grand_train / 1e6:8.3f}M"
    )


class DeltaTokTrainer(Trainer):
    def __init__(self, args, cfg, device, rank, world_size, distributed):
        args.is_master = rank == 0
        args.writer_log = str(cfg.training.get("writer_log", ""))
        super().__init__(args)

        self.args = args
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.is_master = rank == 0

        if self.is_master:
            print(f"Init DeltaTok tokenizer on [GPU{rank}]")

        # DeltaTok needs the OccRAE backbone's hidden_size / num_heads / patch_size
        # at construction time, so build the shared frozen encoder first.
        self._build_occ_rae()
        backbone = self.occ_rae.model._get_pretrained_backbone()
        # Strip CLS + register tokens from OccRAE features before feeding spatial-only
        # patch tokens to DeltaTok.
        self._num_prefix_tokens = 1 + int(backbone.num_register_tokens)

        self.tokenizer = self.get_network("deltatok")
        if self.is_master:
            print(
                f"Number of parameters: "
                f"{sum(p.numel() for p in self.tokenizer.parameters()) / 1e6:.2f}M"
            )

        self.optim = self._build_optimizer()

        if self.device.type != 'cpu' and self.cfg.training.dtype == "bfloat16":
            self.autocast = autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        if self.is_master:
            print("Config: ", self.cfg)
            print(f"TensorBoard log dir: {self.args.writer_log}")
            self.log_add_txt("Parameters", str(self.cfg), self.cfg.training.iter)

        # Eval criteria — keep at L2 / lambda_c=0 so eval scalars stay comparable
        # to past TB runs. Do NOT touch these without coordinating an eval rerun.
        self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
        self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.raymap_criterion = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")

        # Train criteria — mirror sh/train_occany_plus_recon_1B.sh
        # (--lambda_pointmap 1.0 --lambda_depth 1.0). DepthLosses is L1 internally
        # regardless of any switch (losses_da3.py:415), so the train depth
        # criterion is functionally identical to the eval one; the duplicate
        # exists for symmetry and so the train path doesn't reach across to
        # "eval criteria".
        self.train_pointmap_criterion = PointmapLoss(lambda_c=1.0, gt_scale=True, loss_type="L1")
        self.train_depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.train_raymap_criterion = RaymapLoss(lambda_c=1.0, gt_scale=True, loss_type="L1")

    def _build_optimizer(self):
        """AdamW with DeltaTok-style param groups: weight decay applies only to
        ``weight`` of Linear/Conv layers; biases and norm params get zero decay.
        Matches third_party/deltatok/training/base.Base.configure_optimizers.
        """
        decay_types = (
            nn.Linear,
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        )
        net = self.tokenizer.module if self.distributed else self.tokenizer
        net = getattr(net, "_orig_mod", net)

        decay = {}
        for n, m in net.named_modules():
            if isinstance(m, decay_types):
                w = getattr(m, "weight", None)
                if isinstance(w, nn.Parameter) and w.requires_grad:
                    decay[f"{n}.weight" if n else "weight"] = w
        no_decay = {
            n: p for n, p in net.named_parameters()
            if p.requires_grad and n not in decay
        }

        opt = torch.optim.AdamW(
            [
                {"params": [decay[k] for k in sorted(decay)],
                 "weight_decay": self.cfg.training.weight_decay},
                {"params": [no_decay[k] for k in sorted(no_decay)],
                 "weight_decay": 0.0},
            ],
            lr=self.cfg.training.lr,
            betas=(0.9, 0.999),
        )
        for group in opt.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return opt

    def _set_train_loader_epoch(self, epoch: int):
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ds = getattr(self.train_loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

    def _normalize_batch(self, batch):
        """Convert a list-of-views batch (dust3r convention) to the dict format the
        rest of this trainer expects (with ``imgs``, ``gt_*``, ``num_cameras``...).

        ``get_data_loader`` over ``BaseSeqDatasetMultiView`` returns one collated
        view-dict per timestep/camera; ``num_cameras`` is recovered from the per-
        item ``timestep`` field (views are camera-major within each timestep).
        """
        if isinstance(batch, dict):
            return batch

        imgs = torch.stack([v["img"] for v in batch], dim=1).contiguous()  # (B, V, C, H, W)
        B, V = imgs.shape[:2]
        H, W = imgs.shape[-2:]

        timesteps_t = torch.stack([v["timestep"] for v in batch], dim=1)  # (B, V)
        item0_t = timesteps_t[0].tolist()
        num_unique_t = len(set(item0_t))
        num_cameras = V // num_unique_t if num_unique_t > 0 else 1

        out = {
            "imgs": imgs,
            "output_resolution_hw": (int(H), int(W)),
            "processed_root": list(batch[0]["dataset"]),
            "timesteps": timesteps_t,
            "num_cameras": num_cameras,
            "scene_name": list(batch[0]["scene_name"]),
            "frame_stems": list(zip(*[v["frame_id"] for v in batch])),
        }

        if "depthmap" in batch[0]:
            gt_depth = torch.stack([v["depthmap"] for v in batch], dim=1).float()
            gt_c2w = torch.stack([v["camera_pose"] for v in batch], dim=1).float()
            gt_intrinsics = torch.stack([v["camera_intrinsics"] for v in batch], dim=1).float()
            gt_raymap = torch.stack([v["gt_raymap"] for v in batch], dim=1).float()
            gt_pointmap = torch.stack([v["pts3d"] for v in batch], dim=1).float()
            if "valid_mask" in batch[0]:
                gt_mask = torch.stack([v["valid_mask"] for v in batch], dim=1)
            else:
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

    def get_network(self, archi):
        """Build the DeltaTok tokenizer and (optionally) load a checkpoint."""
        if archi != "deltatok":
            return None

        if self.is_master:
            os.makedirs(self.cfg.training.vit_folder, exist_ok=True)

        backbone = self.occ_rae.model._get_pretrained_backbone()
        deltatok_cfg = self.cfg.model.get("deltatok", {})
        model = DeltaTokModule(
            hidden_size=int(backbone.embed_dim),
            num_heads=int(backbone.num_heads),
            patch_size=int(backbone.patch_size),
            initializer_range=float(deltatok_cfg.get("initializer_range", 0.02)),
            num_hidden_layers=int(deltatok_cfg.get("num_hidden_layers", 12)),
            layer_scale_init=float(deltatok_cfg.get("layer_scale_init", 1e-5)),
            use_qk_norm=bool(deltatok_cfg.get("use_qk_norm", True)),
            use_gated_attn=bool(deltatok_cfg.get("use_gated_attn", True)),
            use_swiglu=bool(deltatok_cfg.get("use_swiglu", True)),
            use_rope_aug=bool(deltatok_cfg.get("use_rope_aug", False)),
            mlp_ratio=int(deltatok_cfg.get("mlp_ratio", 4)),
            alt_start=int(deltatok_cfg.get("alt_start", 4)),
        )

        if self.is_master:
            _print_param_breakdown(model, archi)

        model = model.to(self.device)
        if self.distributed:
            model = DDP(model, device_ids=[self.device], gradient_as_bucket_view=True)

        if self.is_master:
            print(
                f"Size of model {archi}: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M"
            )

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

    def _unwrapped_tokenizer(self):
        """Return the underlying DeltaTokModule, peeling DDP and torch.compile wrappers."""
        net = self.tokenizer.module if self.distributed else self.tokenizer
        return getattr(net, "_orig_mod", net)

    def _save_checkpoint(self, path: str) -> None:
        """Save trainer state to ``path``. If a file already exists at ``path``,
        atomically rename it to ``<stem>_backup<ext>`` first so a crash mid-
        ``torch.save`` cannot truncate the live checkpoint.
        """
        net = self._unwrapped_tokenizer()
        state = {
            "iter": self.cfg.training.iter,
            "global_epoch": self.cfg.training.global_epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            root, ext = os.path.splitext(path)
            os.replace(path, f"{root}_backup{ext}")
        torch.save(state, path)

    def _load_checkpoint(self, path: str, *, restore_train_state: bool) -> None:
        """Load model weights from ``path``. If ``restore_train_state`` is True,
        also restore optimizer state and ``iter``/``global_epoch`` counters
        (resume case); otherwise load weights only (pretrained-init case). The
        optimizer load is additionally skipped under ``--test_only`` since no
        further training will run.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        new_state_dict = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
        self._unwrapped_tokenizer().load_state_dict(new_state_dict, strict=False)
        if self.is_master:
            print(f"Load ckpt from: {path}")

        if not restore_train_state:
            return

        if not getattr(self.args, "test_only", False) and ckpt.get("optimizer_state_dict") is not None:
            self.optim.load_state_dict(ckpt["optimizer_state_dict"])
            for group in self.optim.param_groups:
                group.setdefault("initial_lr", group["lr"])
        self.cfg.training.iter = ckpt["iter"]
        self.cfg.training.global_epoch = ckpt["global_epoch"]
        if self.is_master:
            print(f"Number of iteration(s): {self.cfg.training.iter}")

    def adapt_learning_rate(self):
        # Linear warmup, then constant — matches third_party/deltatok base.lr_lambda.
        warm_up = self.cfg.training.warm_up
        step = self.cfg.training.iter
        if warm_up > 0 and step < warm_up:
            scale = step / warm_up
        else:
            scale = 1.0
        for group in self.optim.param_groups:
            base_lr = group.get('initial_lr', group['lr'])
            group['lr'] = base_lr * scale

    def _extract_pair_feats(self, imgs, num_cameras=1):
        """Run OccRAE.encode and split into (x_prev, x) pair tensors plus shape metadata.

        Returns
        -------
        tokens : Tensor (B, V, N, C) — full OccRAE tokens (CLS + registers + patches).
        feats  : Tensor (B, T, num_cameras, P, C) — spatial-only patch features
                 indexed by timestep and camera. Used by the autoregressive eval
                 path.
        x_prev : Tensor (M, num_cameras, P, C) where M = B*(T-1) — spatial-only
                 features for frame_t, with the camera axis kept so DeltaTok can
                 alternate per-camera local and cross-camera global blocks.
        x      : Tensor (M, num_cameras, P, C) — spatial-only features for frame_{t+1}.
        H, W   : ints — image height / width (used to compute the rope grid).
        """
        prefix = self._num_prefix_tokens

        B, V_total, C_img, H, W = imgs.shape
        T = V_total // num_cameras

        with torch.no_grad(), self.autocast:
            latents = self.occ_rae.encode(imgs)
        tokens = latents["tokens"]  # (B, T*N_cam, N_tok, C)
        H_out, W_out = int(latents["H"]), int(latents["W"])

        feats = tokens[:, :, prefix:].contiguous()
        if self.cfg.training.dtype == "bfloat16":
            feats = feats.to(torch.bfloat16)

        B, _, P, C = feats.shape
        feats = feats.view(B, T, num_cameras, P, C)
        assert T >= 2, "DeltaTok pair training requires at least 2 timesteps per item"
        x_prev = feats[:, :-1].reshape(B * (T - 1), num_cameras, P, C)
        x = feats[:, 1:].reshape(B * (T - 1), num_cameras, P, C)
        return tokens, feats, x_prev, x, H_out, W_out

    def _autoregressive_rollout(self, feats, height, width, num_cameras):
        """Joint multi-camera autoregressive rollout for the eval pass.

        The encoder always sees GT pairs (``z_t = encode(feats[:, t-1], feats[:, t])``),
        but the decoder rolls out predictions:
        ``x_hat[:, t] = decode(z_t, x_hat[:, t-1])`` with ``x_hat[:, 0] = feats[:, 0]``
        as GT context. The DA3-style local/global alternation inside
        ``DeltaTokModule`` lets cameras attend to each other on the global
        layers (i >= alt_start, i % 2 == 1) while staying per-camera elsewhere.

        Returns
        -------
        x_hat : Tensor (B*(T-1), num_cameras, P, C) — predictions for t=1..T-1,
                in the same (M, N, P, C) layout as the teacher-forced ``x``
                returned by ``_extract_pair_feats``.
        """
        B, T, N, P, C = feats.shape
        assert N == num_cameras
        assert T >= 2

        net = self.tokenizer.module if self.distributed else self.tokenizer
        net = getattr(net, "_orig_mod", net)

        rope_local = net._compute_rope(height, width, feats.device, feats.dtype)
        rope_global = net._compute_global_rope(
            height, width, N, feats.device, feats.dtype
        )

        # x_hat_prev tracks the per-camera "previous frame" the decoder feeds back.
        x_hat_prev = feats[:, 0]  # (B, N, P, C)

        x_hats = []
        for t in range(1, T):
            x_prev_gt = feats[:, t - 1]  # encoder GT prev (B, N, P, C)
            x_t_gt = feats[:, t]         # encoder GT next (B, N, P, C)
            z_t = net.encode(x_prev_gt, x_t_gt, rope_local, rope_global)
            x_hat_t = net.decode(z_t, x_hat_prev, rope_local, rope_global)
            x_hats.append(x_hat_t)
            x_hat_prev = x_hat_t

        return torch.stack(x_hats, dim=1).reshape(B * (T - 1), N, P, C)



    def train_one_epoch(self):
        self.tokenizer.train()
        cum_loss = 0.
        num_batches = 0
        last_update_time = time.time()
        window_loss = deque(maxlen=self.cfg.training.grad_cum)
        self.optim.zero_grad(set_to_none=True)

        epoch = int(self.cfg.training.global_epoch)
        self._set_train_loader_epoch(epoch)

        # Line-by-line logger (matches training_da3.py) — prints one full
        # message every `print_freq` iters with `flush=True`, plus
        # `max mem: ...` from torch.cuda.max_memory_allocated().
        print_freq = int(self.cfg.training.get("print_freq", 20))
        if self.is_master:
            metric_logger = misc.MetricLogger(delimiter="  ")
            iterable = metric_logger.log_every(
                self.train_loader, print_freq, header=f"Epoch: [{epoch}]"
            )
        else:
            metric_logger = None
            iterable = self.train_loader

        # Geometric supervision (pointmap/depth/raymap) is opt-in via
        # training.loss_weights. When absent we run the original token-only
        # training path; with it set we add weighted geom losses computed
        # through OccRAE.decode_grad. See configs/train_deltatok_geom.yaml.
        w = self.cfg.training.get("loss_weights", None)
        use_geom = w is not None
        # Gradient-checkpoint the DA3 decoder inside decode_grad. Default ON
        # because the un-checkpointed path OOMs on A100-40GB at the current
        # token shapes (~15 decoder layers of DA3-Giant with bf16 activations).
        use_decode_grad_checkpoint = self.cfg.training.get("use_decode_grad_checkpoint", True)

        for batch in iterable:
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                break

            batch = self._normalize_batch(batch)
            num_batches += 1
            update_grad = (num_batches % self.cfg.training.grad_cum) == 0

            self.adapt_learning_rate()

            imgs = batch["imgs"].to(self.device, non_blocking=True)
            num_cameras = batch.get("num_cameras", 1)
            B_cur, V_cur = imgs.shape[:2]
            T_cur = V_cur // num_cameras
            if self.is_master and num_batches <= 10:
                print(f"  [batch {num_batches}] B={B_cur}, V={V_cur}, num_cameras={num_cameras}, T={T_cur}, resolution={imgs.shape[-2:]}")
            tokens, _feats, x_prev, x, H, W = self._extract_pair_feats(
                imgs, num_cameras=num_cameras
            )

            # Pair-index tracking. After pairs_per_batch subsampling, each row i
            # of (x_prev, x) corresponds to batch item pair_batch_idx[i] and
            # timestep transition pair_t[i]-1 -> pair_t[i]. Needed to gather
            # GT views matching x.
            T = imgs.shape[1] // num_cameras
            num_pairs = x_prev.shape[0]
            pair_batch_idx = torch.arange(num_pairs, device=imgs.device) // (T - 1)
            pair_t = torch.arange(num_pairs, device=imgs.device) % (T - 1) + 1

            # Subsample pairs: keep `pairs_per_batch` pairs per item (not global).
            # With B items each having (T-1) pairs, this gives B * pairs_per_batch total.
            pairs_per_item = int(self.cfg.training.get("pairs_per_batch", 0))
            B_cur = imgs.shape[0]
            if pairs_per_item > 0 and (T - 1) > pairs_per_item:
                offsets = torch.stack([
                    torch.randperm(T - 1, device=imgs.device)[:pairs_per_item]
                    for _ in range(B_cur)
                ])  # (B, pairs_per_item)
                base = torch.arange(B_cur, device=imgs.device)[:, None] * (T - 1)
                keep = (base + offsets).reshape(-1)
                x_prev = x_prev[keep]
                x = x[keep]
                pair_batch_idx = pair_batch_idx[keep]
                pair_t = pair_t[keep]

            # 1. DeltaTok forward + token-residual loss.
            with self.autocast:
                x_hat = self.tokenizer(x_prev, x, H, W, num_cameras=num_cameras)

            with torch.autocast(device_type="cuda", enabled=False):
                L_token = _log_cosh(x_hat.float(), x.detach().float()).mean()

            if use_geom:
                # 2. Gather GT for the predicted views. Views are camera-major
                # within each timestep (see _normalize_batch): view index =
                # timestep * num_cameras + camera. GT tensors live on CPU
                # (straight from the DataLoader), so move to device BEFORE
                # advanced-indexing with CUDA index tensors.
                cam_offsets = torch.arange(num_cameras, device=pair_t.device)
                pair_view_start_x = pair_t * num_cameras
                pair_views_x = pair_view_start_x[:, None] + cam_offsets[None, :]

                def gather_x(tensor_BV):
                    return tensor_BV.to(self.device, non_blocking=True)[
                        pair_batch_idx[:, None], pair_views_x
                    ]

                gt_d = gather_x(batch["gt_depth"])
                gt_pm = gather_x(batch["gt_pointmap"])
                gt_ray = gather_x(batch["gt_raymap"])
                gt_m = gather_x(batch["gt_mask"])
                c2w = gather_x(batch["gt_c2w"])
                intr = gather_x(batch["gt_intrinsics"])

                # 3. Stitch GT CLS onto x_hat and decode through frozen DA3
                # with grad. Anchor frames excluded — single decode over the
                # predicted views only (spec D4). _num_prefix_tokens == 1 for
                # DA3-Giant (CLS only).
                prefix = self._num_prefix_tokens
                B_full, V_full, N_tok, C_tok = tokens.shape
                tokens_btnc = tokens.view(B_full, T, num_cameras, N_tok, C_tok)
                gt_cls_x = tokens_btnc[
                    pair_batch_idx[:, None],
                    pair_t[:, None],
                    cam_offsets[None, :],
                    :prefix,
                ]

                full_tokens_x = torch.cat([gt_cls_x, x_hat.to(tokens.dtype)], dim=2)

                with self.autocast:
                    decoded = self.occ_rae.decode_grad(
                        {"tokens": full_tokens_x, "H": H, "W": W},
                        use_checkpoint=use_decode_grad_checkpoint,
                    )
                # decoded: depth (M, N, H, W), depth_conf (M, N, H, W),
                #          pointmap (M, N, H, W, 3), ray (M, N, H, W, 6),
                #          ray_conf (M, N, H, W).

                # 4. Apply train criteria + weighted sum. Following the recon
                # training convention (da3_inference.py:1055), depth_conf is
                # reused as the pointmap confidence — DA3 does not produce a
                # separate pointmap_conf head.
                pointmap_conf = decoded.get("depth_conf")
                with torch.autocast(device_type="cuda", enabled=False):
                    L_pm, _ = self.train_pointmap_criterion(
                        decoded["pointmap"].float(),
                        gt_pm.float(),
                        mask=gt_m,
                        confidence=pointmap_conf.float() if pointmap_conf is not None else None,
                    )
                    L_d, _ = self.train_depth_criterion(
                        decoded["depth"].float().reshape(-1, 1, H, W),
                        gt_d.float().reshape(-1, 1, H, W),
                        confidence=None,
                        mask=gt_m.float().reshape(-1, 1, H, W),
                    )
                    L_ray, _ = self.train_raymap_criterion(
                        decoded["ray"].float(),
                        decoded.get("ray_conf"),
                        c2w.float(),
                        intr.float(),
                        gt_ray.float(),
                    )

                    L_total = (
                        w.token    * L_token
                        + w.pointmap * L_pm
                        + w.depth    * L_d
                        + w.raymap   * L_ray
                    )
            else:
                L_total = L_token

            (L_total / self.cfg.training.grad_cum).backward()

            if update_grad:
                nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.cfg.training.grad_clip)
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

            loss_val = L_total.detach().item()
            cum_loss += loss_val
            window_loss.append(loss_val)

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
                    # Train/LossRecon is the token loss alias kept for TB-run
                    # continuity with the baseline (token-only) trainer.
                    self.log_add_scalar('Train/LossRecon', L_token.detach(), self.cfg.training.iter)
                    if use_geom:
                        self.log_add_scalar('Train/LossToken',    L_token.detach(), self.cfg.training.iter)
                        self.log_add_scalar('Train/LossPointmap', L_pm.detach(),    self.cfg.training.iter)
                        self.log_add_scalar('Train/LossDepth',    L_d.detach(),     self.cfg.training.iter)
                        self.log_add_scalar('Train/LossRaymap',   L_ray.detach(),   self.cfg.training.iter)
                    self.log_add_scalar('Train/LossTot', mini_batch_loss, self.cfg.training.iter)
                    self.log_add_scalar('Train/SpeedSamplesPerSec', speed_samples_per_sec, self.cfg.training.iter)

                    if use_geom:
                        metric_logger.update(
                            loss=mini_batch_loss.item(),
                            tok=L_token.detach().item(),
                            pm=L_pm.detach().item(),
                            d=L_d.detach().item(),
                            ray=L_ray.detach().item(),
                        )
                    else:
                        metric_logger.update(loss=mini_batch_loss.item())

                self.cfg.training.iter += 1

        return cum_loss / max(1, num_batches)

    @torch.no_grad()
    def _compute_frame_losses(self, decoded, gt_dev, view_slice, ray_conf, B, height, width):
        # gt_dev: pre-sliced, on-device GT tensors (see _stage_gt_for_views).
        mask = gt_dev["mask"]
        num_views = mask.shape[1]

        loss_pm, _ = self.pointmap_criterion(
            decoded["pointmap"][:, view_slice].float(),
            gt_dev["pointmap"],
            mask=mask,
        )

        pred_d = decoded["depth"][:, view_slice].float().reshape(B * num_views, 1, height, width)
        gt_d = gt_dev["depth"].reshape(B * num_views, 1, height, width)
        d_mask = mask.reshape(B * num_views, 1, height, width).float()
        loss_d, _ = self.depth_criterion(pred_d, gt_d, confidence=None, mask=d_mask)

        rc = ray_conf[:, view_slice] if ray_conf is not None else None
        loss_ray, _ = self.raymap_criterion(
            decoded["ray"][:, view_slice].float(),
            rc,
            gt_dev["c2w"],
            gt_dev["intrinsics"],
            gt_dev["raymap"],
        )

        return loss_pm, loss_d, loss_ray

    def _decode_tokens(self, tokens, height, width, num_cameras=1):
        if num_cameras <= 1:
            return self.occ_rae.decode({"tokens": tokens, "H": height, "W": width})

        B, V, N_tok, C = tokens.shape
        T = V // num_cameras
        tokens_by_t = tokens.view(B, T, num_cameras, N_tok, C)
        decoded_parts = []
        for t in range(T):
            decoded_parts.append(
                self.occ_rae.decode({"tokens": tokens_by_t[:, t], "H": height, "W": width})
            )
        decoded = {}
        for key in decoded_parts[0]:
            vals = [d[key] for d in decoded_parts]
            if isinstance(vals[0], torch.Tensor) and vals[0].dim() >= 2:
                decoded[key] = torch.cat(vals, dim=1)
            else:
                decoded[key] = vals[0]
        return decoded

    def _reconstruct_full_tokens(self, tokens, x_hat, B, V, num_cameras=1):
        """Stitch DeltaTok-predicted spatial features back into full-token tensors.

        Timestep 0 keeps its ground-truth tokens (DeltaTok has no prediction for it);
        timesteps 1..T-1 use predicted spatial tokens with the original prefix tokens copied.
        For surround mode, timestep 0's N cameras are all GT context.
        """
        P, C = x_hat.shape[-2:]
        prefix = self._num_prefix_tokens

        if num_cameras > 1:
            T = V // num_cameras
            x_hat = x_hat.reshape(B, T - 1, num_cameras, P, C).to(tokens.dtype)
            tokens_5d = tokens.view(B, T, num_cameras, tokens.shape[2], tokens.shape[3])
            prefix_tokens = tokens_5d[:, 1:, :, :prefix]  # (B, T-1, N, prefix, C)
            predicted_full = torch.cat([prefix_tokens, x_hat], dim=3)  # (B, T-1, N, N_tok, C)
            first_timestep = tokens_5d[:, :1]  # (B, 1, N, N_tok, C)
            result = torch.cat([first_timestep, predicted_full], dim=1)  # (B, T, N, N_tok, C)
            return result.view(B, V, tokens.shape[2], tokens.shape[3])
        else:
            x_hat = x_hat.reshape(B, V - 1, P, C).to(tokens.dtype)
            prefix_tokens = tokens[:, 1:, :prefix]
            predicted_full = torch.cat([prefix_tokens, x_hat], dim=2)  # (B, V-1, N, C)
            first_frame = tokens[:, :1]  # (B, 1, N, C)
            return torch.cat([first_frame, predicted_full], dim=1)  # (B, V, N, C)

    @torch.no_grad()
    def eval_one_epoch(self, sanity_check: bool = False):
        if not getattr(self, "test_loaders", None):
            return torch.tensor(0.0, device=self.device)

        self.tokenizer.eval()

        if sanity_check:
            eval_num_items_global = int(self.cfg.training.get("sanity_check_num_items", 4))
            eval_num_visualizations = 2
        else:
            eval_num_items_global = int(self.cfg.training.get("eval_num_items", 256))
            eval_num_visualizations = int(self.cfg.training.get("eval_num_visualizations", 8))
        eval_num_items = max(1, eval_num_items_global // self.world_size)

        eval_viz_dir = str(
            self.cfg.training.get(
                "eval_viz_dir",
                os.path.join(self.cfg.training.vit_folder, "eval_viz"),
            )
        )

        for m in self.eval_metrics.values():
            m.reset()

        overall_loss_recon = 0.0
        overall_n = 0

        # Run eval on each test loader independently, mirroring the per-test-name
        # loop in occany/training_da3.py:train(). Per-loader metrics are logged
        # under `Eval/<test_name>/...`, and `_log_viz_sample` is namespaced under
        # `eval_depth/<test_name>` so visualisations don't collide across roots.
        for test_name, loader in self.test_loaders.items():
            metric = self.eval_metrics[test_name]
            items_seen = 0
            num_vis = 0

            # Pin the eval loader to epoch 0 so each eval pass sees the same
            # samples in the same order — same rationale as training_da3.py.
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(0)
            ds = getattr(loader, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(0)

            print_freq = int(self.cfg.training.get("print_freq", 20))
            if self.is_master:
                eval_logger = misc.MetricLogger(delimiter="  ")
                iterable = eval_logger.log_every(
                    loader, print_freq,
                    header=f"Eval {test_name} [{self.cfg.training.global_epoch}]",
                )
            else:
                eval_logger = None
                iterable = loader

            for batch in iterable:
                if items_seen >= eval_num_items:
                    break
                batch = self._normalize_batch(batch)

                imgs = batch["imgs"].to(self.device, non_blocking=True)
                B, V, _, _, _ = imgs.shape
                num_cameras = batch.get("num_cameras", 1)

                tokens, feats, x_prev, x, H, W = self._extract_pair_feats(imgs, num_cameras=num_cameras)
                with self.autocast:
                    x_hat = self.tokenizer(x_prev, x, H, W, num_cameras=num_cameras)

                with torch.autocast(device_type="cuda", enabled=False):
                    loss_recon = _log_cosh(x_hat.float(), x.float()).mean()

                # Autoregressive rollout: shares the same z_t as the teacher-forced
                # path (encoder sees GT pairs), but feeds predicted x_hat back as the
                # decoder's "previous frame" so error compounds across timesteps.
                with self.autocast:
                    x_hat_ar = self._autoregressive_rollout(feats, H, W, num_cameras)

                with torch.autocast(device_type="cuda", enabled=False):
                    loss_recon_ar = _log_cosh(x_hat_ar.float(), x.float()).mean()

                batch_losses = {
                    "LossRecon": loss_recon.detach().float().item(),
                    "LossRecon_AR": loss_recon_ar.detach().float().item(),
                }

                if "gt_mask" in batch:
                    full_tokens = self._reconstruct_full_tokens(tokens, x_hat, B, V, num_cameras=num_cameras)
                    full_tokens_ar = self._reconstruct_full_tokens(tokens, x_hat_ar, B, V, num_cameras=num_cameras)
                    height, width = batch["output_resolution_hw"]
                    with torch.no_grad(), self.autocast:
                        decoded = self._decode_tokens(full_tokens, height, width, num_cameras=num_cameras)
                        decoded_ar = self._decode_tokens(full_tokens_ar, height, width, num_cameras=num_cameras)

                    # For surround: timestep 0 (num_cameras views) is context; rest are predicted.
                    context_views = num_cameras if num_cameras > 1 else 1
                    pred_slice = slice(context_views, V)

                    with torch.no_grad(), self.autocast:
                        decoded_gt = self._decode_tokens(tokens, height, width, num_cameras=num_cameras)

                    # Stage predicted-view GT on device once; reused by all three
                    # _compute_frame_losses calls below.
                    gt_dev = {
                        "mask":       batch["gt_mask"][:, pred_slice].to(self.device),
                        "pointmap":   batch["gt_pointmap"][:, pred_slice].to(self.device).float(),
                        "depth":      batch["gt_depth"][:, pred_slice].to(self.device).float(),
                        "c2w":        batch["gt_c2w"][:, pred_slice].to(self.device).float(),
                        "intrinsics": batch["gt_intrinsics"][:, pred_slice].to(self.device).float(),
                        "raymap":     batch["gt_raymap"][:, pred_slice].to(self.device).float(),
                    }

                    for tag, dec in [("PredVsGT", decoded), ("PredVsGT_AR", decoded_ar), ("OrigVsGT", decoded_gt)]:
                        L_pm, L_d, L_ray = self._compute_frame_losses(
                            dec, gt_dev, pred_slice, dec.get("ray_conf"), B, height, width,
                        )
                        batch_losses[f"LossPointmap_{tag}"] = L_pm.item()
                        batch_losses[f"LossDepth_{tag}"]    = L_d.item()
                        batch_losses[f"LossRaymap_{tag}"]   = L_ray.item()

                    # Throttle TB viz to every Nth epoch (disk JPEGs still written
                    # every epoch via eval_viz_dir). Sanity-check pass runs at
                    # global_epoch=0 which always passes the modulo gate.
                    viz_every_n = max(1, int(self.cfg.training.get("viz_every_n_epochs", 1)))
                    viz_epoch_ok = (int(self.cfg.training.global_epoch) % viz_every_n) == 0
                    need_viz = (
                        self.is_master
                        and num_vis < eval_num_visualizations
                        and viz_epoch_ok
                    )

                    if need_viz:
                        have_img_dec = getattr(self.occ_rae, "img_decoder", None) is not None
                        if have_img_dec:
                            vcs = getattr(self, "_img_decoder_view_chunk", 0)
                            with torch.no_grad(), self.autocast:
                                rgb_tf_full, rgb_ar_full, rgb_gt_full = [
                                    self.occ_rae.decode_to_image(
                                        {"tokens": t, "H": height, "W": width},
                                        view_chunk_size=vcs,
                                    )
                                    for t in (full_tokens, full_tokens_ar, tokens)
                                ]
                            # Pull RGB tensors to CPU once; per-sample reindexing is free.
                            rgb_tf_cpu = rgb_tf_full.detach().float().cpu()
                            rgb_ar_cpu = rgb_ar_full.detach().float().cpu()
                            rgb_gt_cpu = rgb_gt_full.detach().float().cpu()

                        # One CUDA sync per tensor for the whole batch; view-order
                        # reindexing for each sample happens on CPU below.
                        ar_depth_cpu = decoded_ar["depth"].detach().float().cpu()
                        gt_token_depth_cpu = decoded_gt["depth"].detach().float().cpu()

                        def _depth_panel(depth_v, mask_np):
                            return torch.stack([
                                torch.from_numpy(depth2rgb(
                                    depth_v[t].clamp(0, 50).numpy(),
                                    valid_mask=mask_np[t],
                                    min_depth=0.0,
                                    max_depth=50.0,
                                ).astype(np.float32))
                                for t in range(V)
                            ])

                        for batch_idx in range(B):
                            if num_vis >= eval_num_visualizations:
                                break

                            view_order = sorted(
                                range(V),
                                key=lambda idx: batch["timesteps"][batch_idx][idx],
                            )
                            # Context views (timestep 0) carry GT tokens — blank their pred cells.
                            pred_blank_views = [v < context_views for v in view_order]
                            view_index = torch.as_tensor(view_order, dtype=torch.long)

                            gt_token_depth = gt_token_depth_cpu[batch_idx][view_index]
                            ar_pred_depth = ar_depth_cpu[batch_idx][view_index]

                            gt_token_depth_color = _depth_panel(gt_token_depth, gt_token_depth.numpy() > 0)
                            ar_pred_depth_color = _depth_panel(ar_pred_depth, ar_pred_depth.numpy() > 0)
                            for t, blank in enumerate(pred_blank_views):
                                if blank:
                                    ar_pred_depth_color[t] = 30.0

                            extra_panels = [
                                ar_pred_depth_color,
                                gt_token_depth_color,
                            ]
                            col_titles = [
                                "RGB",
                                "Pred Depth (TF)",
                                "Pred Depth (AR)",
                                "GT Token Depth",
                            ]
                            if have_img_dec:
                                def _rgb_panel(t_cpu):
                                    return t_cpu[batch_idx][view_index].permute(0, 2, 3, 1).contiguous() * 255.0
                                rgb_tf_b = _rgb_panel(rgb_tf_cpu)
                                rgb_ar_b = _rgb_panel(rgb_ar_cpu)
                                rgb_gt_b = _rgb_panel(rgb_gt_cpu)
                                for t, blank in enumerate(pred_blank_views):
                                    if blank:
                                        rgb_ar_b[t] = 255.0
                                # Order matches _log_viz_sample's fixed base layout
                                # (gt_img, pred_depth_TF) — depth panels come first.
                                extra_panels = extra_panels + [rgb_tf_b, rgb_ar_b, rgb_gt_b]
                                col_titles = col_titles + [
                                    "Pred RGB (TF)",
                                    "Pred RGB (AR)",
                                    "GT Token RGB",
                                ]

                            saved_path = _log_viz_sample(
                                batch=batch,
                                decoded=decoded,
                                batch_idx=batch_idx,
                                epoch=self.cfg.training.global_epoch,
                                epoch_step=self.cfg.training.iter,
                                output_dir=eval_viz_dir,
                                log_writer=self.writer,
                                tb_prefix=f"eval_depth/{test_name}",
                                extra_panels=extra_panels,
                                view_order=view_order,
                                max_depth=50.0,
                                pred_blank_views=pred_blank_views,
                                col_titles=col_titles,
                                resize_long_side=300,
                            )
                            if saved_path is not None:
                                print(f"Saved viz: {saved_path}")
                            num_vis += 1

                metric.update(batch_size=B, **batch_losses)
                items_seen += B
                if self.is_master:
                    eval_logger.update(loss=loss_recon.item())

            if metric.count == 0:
                continue

            results = metric.compute()
            overall_loss_recon += results["LossRecon"].item() * metric.count.item()
            overall_n += metric.count.item()

            if self.is_master and not sanity_check:
                for key, val in results.items():
                    self.log_add_scalar(f"Eval/{test_name}/{key}", val.item(), self.cfg.training.iter)
                # When TensorBoard is disabled (e.g. --eval-only), echo the
                # numbers to stdout so they aren't silently dropped.
                if self.writer is None:
                    metrics_str = ", ".join(f"{k}={v.item():.4f}" for k, v in results.items())
                    print(f"[Eval/{test_name}] {metrics_str}")

        final_loss_recon = overall_loss_recon / overall_n if overall_n > 0 else 0.0

        self.tokenizer.train()
        return final_loss_recon

    def _build_occ_rae(self):
        self.occ_rae = OccRAE(
            weights_path=self.cfg.model.occany_recon_ckpt,
            device=str(self.device),
            encode_layer=int(self.cfg.model.encode_layer),
        )
        self.occ_rae.eval()
        self.occ_rae.requires_grad_(False)
        if self.cfg.training.dtype == "bfloat16":
            self.occ_rae.to(torch.bfloat16)
        if self.is_master:
            print(f"[INFO] Built shared OccRAE encoder at 518x518 ({self.cfg.training.dtype}), "
                  f"encode_layer={int(self.cfg.model.encode_layer)}")

        # Optional pretrained MAE-style image decoder for eval-time RGB viz.
        img_dec_cfg = self.cfg.model.get("img_decoder", None)
        ckpt_path = img_dec_cfg.get("ckpt_path") if img_dec_cfg is not None else None
        if ckpt_path:
            self.occ_rae.load_image_decoder(
                ckpt_path=str(ckpt_path),
                config_path=str(img_dec_cfg.get("config_path", "third_party/GLD/configs/decoder/ViTXL")),
                hidden_size=int(img_dec_cfg.get("hidden_size", 9216)),
                patch_size=int(img_dec_cfg.get("patch_size", 14)),
                use_ema=bool(img_dec_cfg.get("use_ema", True)),
            )
            if self.cfg.training.dtype == "bfloat16":
                self.occ_rae.img_decoder.to(torch.bfloat16)
            self._img_decoder_view_chunk = int(img_dec_cfg.get("view_chunk_size", 0))
        else:
            self._img_decoder_view_chunk = 0

    def fit(self):
        # Build train/test loaders via `get_data_loader`, matching the
        # `train_occrae_img_decoder.py` data path. The dataset config is now a
        # Python expression string evaluated by `dust3r.datasets`, e.g.:
        #   "10000 @ WaymoSeqMultiView(...) + 5000 @ VKittiSeqMultiView(...)"
        train_dataset_str = str(self.cfg.dataset.train_dataset)
        test_dataset_str = self.cfg.dataset.get("test_dataset", None)
        per_dataset_sampling = bool(self.cfg.dataset.get("per_dataset_sampling", False))
        fixed_views_per_batch = bool(self.cfg.dataset.get("fixed_views_per_batch", False))

        # no_partial_views=True returns variable view counts per sample (rounded
        # down to a multiple of actual_vpt — see base_seq_dataset.py _get_views).
        # That breaks torch.stack at collate when bsize > 1, UNLESS the
        # fixed-views sampler is used (it guarantees uniform T*vpt per batch).
        uses_no_partial = "no_partial_views=True" in train_dataset_str or (
            test_dataset_str is not None and "no_partial_views=True" in str(test_dataset_str)
        )
        if uses_no_partial and int(self.cfg.training.bsize) != 1 and not fixed_views_per_batch:
            raise ValueError(
                f"no_partial_views=True yields variable view counts per sample; "
                f"per-GPU training.bsize must be 1 (got {self.cfg.training.bsize}). "
                f"Either set dataset.fixed_views_per_batch=true, lower bsize to 1, "
                f"or remove no_partial_views=True from the dataset string."
            )
        if self.is_master:
            print(f"Building train dataset: {train_dataset_str}")
            print(f"Per-dataset sampling: {per_dataset_sampling}")
            print(f"Fixed views per batch: {fixed_views_per_batch}")
        self.train_loader = get_data_loader(
            train_dataset_str,
            batch_size=self.cfg.training.bsize,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
            drop_last=True,
            per_dataset_sampling=per_dataset_sampling,
            fixed_views_per_batch=fixed_views_per_batch,
        )

        # One DataLoader per `+`-separated sub-dataset (mirrors the build/eval
        # loop in `occany/training_da3.py`), so each test set runs in its own
        # eval pass and logs under its own TensorBoard prefix.
        self.test_loaders: dict = {}
        self.eval_metrics: dict = {}
        if test_dataset_str:
            test_dataset_str = str(test_dataset_str)
            if self.is_master:
                print(f"Building test datasets: {test_dataset_str}")
            for sub in test_dataset_str.split("+"):
                sub = sub.strip()
                if not sub:
                    continue
                # Match training_da3.py naming: keep the "<count> @ <Class>" prefix.
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
                    self.eval_metrics[test_name] = DeltaTokEvalMetric().to(self.device)
                    if self.is_master:
                        print(f"  - {test_name}: {len(loader)} batches")
                except Exception as e:
                    if self.is_master:
                        print(f"  - {test_name}: failed to build ({e})")

        if self.is_master:
            print("Start training:")

        # Resume from current.pth takes precedence over a one-off pretrained
        # init via --ckpt; pretrained-init only restores model weights.
        resume_ckpt = self.get_resume_checkpoint_path()
        if resume_ckpt is not None:
            self._load_checkpoint(resume_ckpt, restore_train_state=True)
        else:
            pretrained_ckpt = self.get_pretrained_checkpoint_path()
            if pretrained_ckpt is not None:
                self._load_checkpoint(pretrained_ckpt, restore_train_state=False)

        # Initial evaluation for sanity check
        if self.test_loaders:
            if self.is_master:
                print("Running initial evaluation for sanity check...")
            self.eval_one_epoch(sanity_check=True)

        start = time.time()

        for e in range(self.cfg.training.global_epoch, self.cfg.training.epoch + 1):
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                if self.is_master:
                    print("End of training: reached max iterations")
                break

            train_loss = self.train_one_epoch()
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

            if self.is_master:
                self._save_checkpoint(
                    os.path.join(self.cfg.training.vit_folder, "current.pth")
                )

    def evaluate(self):
        """One-shot evaluation: build only test loaders, load checkpoint, run
        ``eval_one_epoch`` once, and print metrics. No optimizer / no train
        loader / no TensorBoard writer (writer is None when fired via
        ``--eval-only`` because train_deltatok.py blanks ``writer_log``).
        """
        test_dataset_str = self.cfg.dataset.get("test_dataset", None)
        if not test_dataset_str:
            raise RuntimeError("--eval-only requires dataset.test_dataset to be set.")

        self.test_loaders = {}
        self.eval_metrics = {}
        if self.is_master:
            print(f"Building test datasets: {test_dataset_str}")
        for sub in str(test_dataset_str).split("+"):
            sub = sub.strip()
            if not sub:
                continue
            test_name = sub.split("(")[0].strip()
            loader = get_data_loader(
                sub,
                batch_size=self.cfg.training.bsize,
                num_workers=int(self.cfg.training.get("val_num_workers", 2)),
                shuffle=False,
                drop_last=False,
            )
            self.test_loaders[test_name] = loader
            self.eval_metrics[test_name] = DeltaTokEvalMetric().to(self.device)
            if self.is_master:
                print(f"  - {test_name}: {len(loader)} batches")

        ckpt_path = os.path.join(self.cfg.training.vit_folder, "current.pth")
        pretrained_ckpt = self.get_pretrained_checkpoint_path()
        if os.path.isfile(ckpt_path):
            self._load_checkpoint(ckpt_path, restore_train_state=False)
        elif pretrained_ckpt is not None:
            self._load_checkpoint(pretrained_ckpt, restore_train_state=False)
        else:
            raise FileNotFoundError(
                f"--eval-only requires a checkpoint; none found at {ckpt_path} "
                f"and --ckpt was not provided."
            )

        if self.is_master:
            print("Running --eval-only pass...")
        eval_loss = self.eval_one_epoch(sanity_check=False)
        if self.is_master:
            print(f"[eval-only] final LossRecon = {float(eval_loss):.6f}")

    def run(self):
        if getattr(self.args, "eval_only", False):
            self.evaluate()
        else:
            self.fit()
