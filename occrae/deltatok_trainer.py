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
from occrae.deltatok_shared import DeltaTokSharedMixin
from occrae.network.rope_utils import compute_camera_rope  # shared 1xN camera-grid rope build
from occrae.metric_logger import MetricLogger, SmoothedValue  # SLURM-friendly line-per-print logging (no tqdm)
from occrae.metric import DeltaTokEvalMetric
from occrae.visualization_helper import _log_viz_sample
from occany.datasets import get_data_loader
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
        use_camera_rope: bool = False,
        mlp_ratio: int = 4,
        alt_start: int = 4,
        num_delta_tokens: int = 1,
    ):
        super().__init__()
        # Delta tokens per camera per transition (1 = max compression). K query
        # tokens compress a frame pair's P patches into K latents; raise it to
        # reduce compression.
        self.num_delta_tokens = int(num_delta_tokens)
        self._use_rope_aug = use_rope_aug
        # When True, global (cross-camera) layers give each camera one distinct
        # rope position (shared by its patches, identical in prev & next) so the
        # network can link the same physical camera across the two timesteps;
        # see ``_compute_global_rope``. Default False keeps the DA3 pos_nodiff
        # behavior (non-parametric flag — existing checkpoints load unchanged).
        self._use_camera_rope = bool(use_camera_rope)
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
        # Disable rope coord augmentation (rescale) defensively. It only fires when
        # the rope runs in train mode (use_rope_aug=true; else train() forces eval),
        # so None keeps the coords un-augmented even if rope aug is later enabled.
        cfg.pos_embed_rescale = None

        if use_swiglu:
            cfg.use_gated_mlp = True
            cfg.intermediate_size = max(1, (2 * cfg.intermediate_size) // 3)
            cfg.hidden_act = "silu"

        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(cfg)

        self.z_embed = nn.Embedding(self.num_delta_tokens, cfg.hidden_size)
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

    def _compute_camera_rope(self, max_cameras: int, device, dtype):
        """Resolution-independent rope over camera positions 0..max_cameras-1.

        Lays the camera slots along a synthetic ``1 x max_cameras`` patch grid so the
        coord normalization (DINOv3 divides patch coords by the grid) uses a FIXED
        ``max_cameras`` instead of the image's ``Hp x Wp``; a camera idx therefore maps
        to the same rotation at any image resolution. Only the width (x) coord varies —
        the row-0 y coord is constant, so the y-half of head_dim is identity here.
        Returns (cos, sin) of shape (max_cameras, head_dim).
        """
        key = ("camera", int(max_cameras), device, dtype, self.rope_embeddings.training)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        rope = compute_camera_rope(self.rope_embeddings, max_cameras, device, dtype)  # shared 1xN grid build
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

        With ``self._use_camera_rope``, each camera instead gets one distinct
        position (shared by its patches, identical in prev & next) so the model
        can link the same camera across timesteps (same-camera offset 0). The
        positions come from a dedicated, resolution-independent camera rope of a
        fixed maximum of 32 slots (``_compute_camera_rope``); camera idx maps to
        position idx ``0,1,2,...``. The camera↔position assignment is shuffled
        per forward while training to stay permutation-invariant; eval is
        deterministic.
        """
        Hp = int(height) // self.rope_embeddings.config.patch_size
        Wp = int(width) // self.rope_embeddings.config.patch_size
        num_grid = Hp * Wp                          # = P, patches per camera
        N = int(num_cameras)
        cos, sin = self._compute_rope(height, width, device, dtype)  # (num_grid, head_dim)

        if not self._use_camera_rope:
            # DA3 pos_nodiff: every patch at index Wp+1 = coord (1,1).
            idx = Wp + 1
            num_positions = N * num_grid
            return (
                cos[idx : idx + 1].expand(num_positions, -1).contiguous(),
                sin[idx : idx + 1].expand(num_positions, -1).contiguous(),
            )

        # Dedicated, resolution-independent camera rope: camera idx 0..N-1 over a
        # fixed max of MAX_CAMERAS positions (not the image grid).
        MAX_CAMERAS = 32
        assert N <= MAX_CAMERAS, f"camera rope supports <= {MAX_CAMERAS} cameras, got {N}"
        cam_cos, cam_sin = self._compute_camera_rope(MAX_CAMERAS, device, dtype)  # (MAX_CAMERAS, head_dim)
        base_idx = torch.arange(N, device=device)  # camera idx 0,1,2,...,N-1
        if self.training:
            base_idx = base_idx[torch.randperm(N, device=device)]  # shuffle camera↔position (perm-invariance)
        # cam-major repeat to match encode/decode reshape(M, N*P, C): [pos(c0)xP, pos(c1)xP, ...]
        cos_cam = cam_cos[base_idx].repeat_interleave(num_grid, dim=0).contiguous()  # (N*P, head_dim)
        sin_cam = cam_sin[base_idx].repeat_interleave(num_grid, dim=0).contiguous()  # (N*P, head_dim)
        return cos_cam, sin_cam

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
        z : (M, N, K, C) — K = num_delta_tokens delta tokens per camera, all
            initialized from the shared ``z_embed`` parameter. Each camera's z is updated at every
            block: on local blocks via per-camera self-attention with that
            camera's ``[prev, next]`` strip, on global blocks via cross-camera
            attention over all N z's plus all N cameras' spatial tokens. This
            keeps the prefix-token structure compatible with HF's
            ``apply_rotary_pos_emb`` (which skips the leading
            ``num_tokens - num_patches`` tokens).
        """
        M, N, P, C = x_prev.shape
        K = self.num_delta_tokens                          # delta tokens per camera

        # N*K z tokens (K per camera); all initialized from the shared z_embed.
        z = self.z_embed.weight[None, None].expand(M, N, K, C).contiguous()  # (M, N, K, C)

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
                # Global layout: [all z (cam-major, K each), all_prev (cam-major), all_next (cam-major)]
                # so prefix = N*K (the z's) and rope length = 2 * N * P.
                z_flat = z.reshape(M, N * K, C)
                prev_flat = prev_spatials.reshape(M, N * P, C)
                next_flat = next_spatials.reshape(M, N * P, C)
                hidden = torch.cat([z_flat, prev_flat, next_flat], dim=1)
                hidden = blk(hidden, position_embeddings=global_rope_full)
                z = hidden[:, : N * K].reshape(M, N, K, C)
                prev_spatials = hidden[:, N * K : N * K + N * P].reshape(M, N, P, C)
                next_spatials = hidden[:, N * K + N * P :].reshape(M, N, P, C)
            else:
                # Local: per-camera [z_n (K tokens), prev_n, next_n]. Each camera's z is
                # updated independently here.
                hidden = torch.cat([z, prev_spatials, next_spatials], dim=2)
                seq_len = hidden.shape[2]
                hidden = hidden.reshape(M * N, seq_len, C)
                hidden = blk(hidden, position_embeddings=local_rope_full)
                hidden = hidden.reshape(M, N, seq_len, C)
                z = hidden[:, :, :K]
                prev_spatials = hidden[:, :, K : K + P]
                next_spatials = hidden[:, :, K + P :]

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
        z      : (M, N, K, C) — K delta tokens per camera (also accepts
                  (M, N, C) for the K=1 convenience case).
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
            # Accept (M, N, C) for the K=1 convenience case; broadcast to (M, N, 1, C).
            z = z.unsqueeze(2)
        z = z.contiguous()
        K = z.shape[2]                     # delta tokens per camera
        spatials = x_prev  # (M, N, P, C)

        for i, blk in enumerate(self.decoder_blocks):
            if self._is_global_layer(i) and N > 1:
                # Global: [all z (cam-major, K each), all_spatials (cam-major)]; prefix = N*K, rope length = N*P.
                z_flat = z.reshape(M, N * K, C)
                hidden = torch.cat([z_flat, spatials.reshape(M, N * P, C)], dim=1)
                hidden = blk(hidden, position_embeddings=rope_global)
                z = hidden[:, : N * K].reshape(M, N, K, C)
                spatials = hidden[:, N * K :].reshape(M, N, P, C)
            else:
                # Local: per-camera [z_n (K tokens), x_prev_n]; prefix = K, rope length = P.
                hidden = torch.cat([z, spatials], dim=2)  # (M, N, K+P, C)
                hidden = hidden.reshape(M * N, K + P, C)
                hidden = blk(hidden, position_embeddings=rope_local)
                hidden = hidden.reshape(M, N, K + P, C)
                z = hidden[:, :, :K]
                spatials = hidden[:, :, K:]

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


class DeltaTokTrainer(DeltaTokSharedMixin, Trainer):
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

        # Gradient accumulation is derived, not configured:
        # effective_bsize = world_size * bsize * pairs_per_seq * grad_cum.
        bsize = int(self.cfg.training.bsize)
        pairs_per_seq = int(self.cfg.training.get("pairs_per_seq", 0))
        assert pairs_per_seq > 0, "training.pairs_per_seq must be > 0 to derive grad_cum"
        effective_bsize = int(self.cfg.training.effective_bsize)
        denom = self.world_size * bsize * pairs_per_seq
        assert effective_bsize % denom == 0, (
            f"training.effective_bsize={effective_bsize} must be divisible by "
            f"world_size * bsize * pairs_per_seq = "
            f"{self.world_size} * {bsize} * {pairs_per_seq} = {denom}"
        )
        self.grad_cum = effective_bsize // denom
        if self.is_master:
            print(f"effective_bsize={effective_bsize} = {self.world_size} rank(s) "
                  f"x bsize {bsize} x pairs_per_seq {pairs_per_seq} x grad_cum {self.grad_cum}")

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

        self.pointmap_criterion = PointmapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")
        self.depth_criterion = DepthLosses(lambda_c=0.0, gt_scale=True, alpha=0.0)
        self.raymap_criterion = RaymapLoss(lambda_c=0.0, gt_scale=True, loss_type="L2")

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
            # beta2=0.95 + eps=1e-6 cap Adam's 1/sqrt(v) amplification once the
            # model converges into the tiny-gradient regime (dtok32/64 blow-ups).
            betas=(0.9, 0.95),
            eps=1e-6,
        )
        for group in opt.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return opt

    # _set_train_loader_epoch / _normalize_batch moved verbatim to
    # occrae/deltatok_shared.py (DeltaTokSharedMixin).

    def get_network(self, archi):
        """Build the DeltaTok tokenizer and (optionally) load a checkpoint."""
        if archi != "deltatok":
            return None

        if self.is_master:
            os.makedirs(self.cfg.training.vit_folder, exist_ok=True)

        backbone = self.occ_rae.model._get_pretrained_backbone()
        model = self._make_deltatok_module(backbone)  # shared factory (mixin)

        if self.is_master:
            _print_param_breakdown(model, archi)

        model = model.to(self.device)
        if self.distributed:
            model = DDP(model, device_ids=[self.device])

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

    def _save_current_checkpoint(self) -> None:
        """current.pth save used by ``DeltaTokSharedMixin._end_of_epoch``."""
        self._save_checkpoint(os.path.join(self.cfg.training.vit_folder, "current.pth"))

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

    # _extract_pair_feats moved verbatim to occrae/deltatok_shared.py
    # (DeltaTokSharedMixin).

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
        # The per-step encode only ever saw GT pairs, so batching all T-1
        # encodes up front (shared `_encode_pair_deltas`) is the same compute;
        # the decode feedback loop lives in shared `_rollout_from_z`.
        net = self._unwrapped_tokenizer()
        z = self._encode_pair_deltas(net, feats, height, width)  # (B, T-1, N, C)
        return self._rollout_from_z(net, feats[:, 0], z, height, width, num_cameras)  # (B*(T-1), N, P, C)

    def _feature_loss(self, tokens, x_hat, B, T_minus_1, idx, num_cameras, height, width):
        """Downstream DA3 feature loss. Insert the predicted frames' patch features back
        among the GT OccAny tokens (every other frame stays GT), run ONE frozen forward
        (blocks 13->39) over all V=T*num_cameras views, and log-cosh-match every view's
        out_layer features against the same forward on the pure-GT tokens. Only the
        subsampled frames carry gradient (the tokenizer is run on those frames only); the
        signal still reaches them through cross-view attention to the other views.

        tokens: (B, V, N_tok, C) full GT tokens. x_hat: (n_pairs, N, P, C) predicted layer-12
        patch features for the subsampled frames (idx rows of the B*(T-1) t>=1 frames).
        """
        T = T_minus_1 + 1                                                 # timesteps per sequence
        V = tokens.shape[1]                                              # total views = T*num_cameras
        N_tok = tokens.shape[2]                                          # tokens/view (prefix + P)
        C_tok = tokens.shape[3]                                          # token channel dim
        prefix = self._num_prefix_tokens                                # CLS (+ register) token count

        # GT patch features for frames 1..T-1, row-aligned with x_hat's (B*(T-1)) layout.
        gt_t1 = tokens.view(B, T, num_cameras, N_tok, C_tok)[:, 1:, :, prefix:, :]    # (B, T-1, N, P, C)
        P = gt_t1.shape[-2]                                              # patches per view
        gt_flat = gt_t1.reshape(B * T_minus_1, num_cameras, P, C_tok)    # (B*(T-1), N, P, C)
        # Insert predicted patches at the subsampled rows; all other frames keep GT features.
        if idx is None:
            merged = x_hat.to(gt_flat.dtype)                            # every t>=1 frame predicted
        else:
            merged = gt_flat.index_copy(0, idx, x_hat.to(gt_flat.dtype))  # (B*(T-1), N, P, C), grad on idx rows
        # Rebuild the full sequence (GT prefix per view, GT timestep-0 context).
        full_pred = self._reconstruct_full_tokens(tokens, merged, B, V, num_cameras=num_cameras)  # (B, V, N_tok, C)

        with self.autocast:
            pred_feats = self.occ_rae.decode_to_features(
                {"tokens": full_pred, "H": height, "W": width}, num_levels=None,
                requires_grad=True, use_checkpoint=True,
            )                                                            # one tensor per out_layer (B, V, P, 3072), grad+ckpt
            gt_feats = self.occ_rae.decode_to_features(
                {"tokens": tokens, "H": height, "W": width}, num_levels=None, requires_grad=False
            )                                                            # one tensor per out_layer (B, V, P, 3072), detached

        with torch.autocast(device_type="cuda", enabled=False):
            loss_feat = sum(
                _log_cosh(p.float(), g.detach().float()).mean()
                for p, g in zip(pred_feats, gt_feats)
            ) / len(pred_feats)                                          # mean log-cosh over all out_layers and views
        return loss_feat

    def train_one_epoch(self):
        self.tokenizer.train()
        cum_loss = 0.
        num_batches = 0
        last_update_time = time.time()
        window_loss = deque(maxlen=self.grad_cum)
        self.optim.zero_grad(set_to_none=True)

        epoch = int(self.cfg.training.global_epoch)
        self._set_train_loader_epoch(epoch)

        # SLURM-friendly progress: one flushed print line every `print_freq`
        # batches (master only), instead of tqdm's carriage-return bar.
        print_freq = int(self.cfg.training.get("print_freq", 20))
        if self.is_master:
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            batch_iter = metric_logger.log_every(
                self.train_loader, print_freq, header=f"Training (Epoch {epoch})"
            )
        else:
            batch_iter = self.train_loader

        for batch in batch_iter:
            if self.cfg.training.iter >= self.cfg.training.max_iter:
                break

            batch = self._normalize_batch(batch)
            num_batches += 1
            update_grad = (num_batches % self.grad_cum) == 0

            self.adapt_learning_rate()

            imgs = batch["imgs"].to(self.device, non_blocking=True)
            num_cameras = batch.get("num_cameras", 1)
            tokens, _feats, x_prev, x, H, W = self._extract_pair_feats(imgs, num_cameras=num_cameras)

            # Subsample to `pairs_per_seq` transitions PER sequence (not per micro-batch
            # total), so the pairs actually used == bsize * pairs_per_seq and match the
            # grad_cum accounting (effective_bsize = world_size * bsize * pairs_per_seq *
            # grad_cum). x_prev is (B*(T-1), N, P, C), batch-major: row b*(T-1)+t is
            # sequence b, transition t. B=bsize, T=timesteps, N=cameras, P=patches/view.
            pairs_per_seq = int(self.cfg.training.get("pairs_per_seq", 0))
            B = imgs.shape[0]                                   # sequences in this micro-batch
            T_minus_1 = x_prev.shape[0] // B                    # transitions per seq (T-1), uniform across the batch
            idx = None                                          # subsample row index; None == all pairs kept
            if pairs_per_seq > 0 and T_minus_1 > pairs_per_seq:
                # argsort of per-row random keys = an independent random permutation of the
                # T-1 transitions for each of the B sequences (vectorized B-way randperm);
                # the prefix is then a without-replacement sample of pairs_per_seq per seq.
                perm = torch.argsort(
                    torch.rand(B, T_minus_1, device=x_prev.device), dim=1
                )[:, :pairs_per_seq]                           # (B, pairs_per_seq) transition idx within each seq
                base = (torch.arange(B, device=x_prev.device) * T_minus_1).unsqueeze(1)  # (B, 1) seq offsets
                idx = (base + perm).reshape(-1)               # (B*pairs_per_seq,) flat rows into x_prev
                x_prev = x_prev[idx]
                x = x[idx]

            with self.autocast:
                x_hat = self.tokenizer(x_prev, x, H, W, num_cameras=num_cameras)

            with torch.autocast(device_type="cuda", enabled=False):
                loss = _log_cosh(x_hat.float(), x.detach().float()).mean()

            # Optional downstream DA3 feature loss: insert the predicted frames back among the
            # GT OccAny tokens, run one forward over all V views (blocks 13->39), and match the
            # predicted frames' out_layer features vs the pure-GT decode. 0 weight == no-op.
            w_feat = float(self.cfg.training.get("feature_loss_weight", 0.0))
            if w_feat > 0:
                loss_feat = self._feature_loss(tokens, x_hat, B, T_minus_1, idx, num_cameras, H, W)
                loss_total = loss + w_feat * loss_feat
            else:
                loss_feat = None
                loss_total = loss

            (loss_total / self.grad_cum).backward()

            if update_grad:
                nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.cfg.training.grad_clip)
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

            loss_val = loss_total.detach().item()   # total (recon + feature) drives meters/LossTot
            cum_loss += loss_val
            window_loss.append(loss_val)

            # Update the console meters every batch (matches OccAny) so the first
            # log_every line at step 0 has populated `loss`/`lr` meters.
            if self.is_master:
                metric_logger.update(loss=loss_val, lr=self.optim.param_groups[0]['lr'])

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
                    self.log_add_scalar('Train/LossRecon', loss, self.cfg.training.iter)
                    self.log_add_scalar('Train/LossFeature', loss_feat if loss_feat is not None else 0.0, self.cfg.training.iter)
                    self.log_add_scalar('Train/LossTot', mini_batch_loss, self.cfg.training.iter)
                    self.log_add_scalar('Train/SpeedSamplesPerSec', speed_samples_per_sec, self.cfg.training.iter)

                self.cfg.training.iter += 1

        return cum_loss / max(1, num_batches)

    # _compute_frame_losses / _decode_tokens / _reconstruct_full_tokens moved
    # verbatim to occrae/deltatok_shared.py (DeltaTokSharedMixin).

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

            # SLURM-friendly progress: MetricLogger prints one flushed line per
            # print_freq batches (master only, matching tqdm's disable).
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

                    ray_conf = decoded.get("ray_conf")
                    ray_conf_ar = decoded_ar.get("ray_conf")

                    # For surround: timestep 0 (num_cameras views) is context; rest are predicted.
                    context_views = num_cameras if num_cameras > 1 else 1
                    pred_slice = slice(context_views, V)
                    loss_pm, loss_d, loss_ray = self._compute_frame_losses(
                        decoded, batch, pred_slice, ray_conf, B, height, width
                    )
                    loss_pm_ar, loss_d_ar, loss_ray_ar = self._compute_frame_losses(
                        decoded_ar, batch, pred_slice, ray_conf_ar, B, height, width
                    )
                    batch_losses.update({
                        "LossPointmap_PredVsGT": loss_pm.item(),
                        "LossDepth_PredVsGT": loss_d.item(),
                        "LossRaymap_PredVsGT": loss_ray.item(),
                        "LossPointmap_PredVsGT_AR": loss_pm_ar.item(),
                        "LossDepth_PredVsGT_AR": loss_d_ar.item(),
                        "LossRaymap_PredVsGT_AR": loss_ray_ar.item(),
                    })

                    need_viz = self.is_master and num_vis < eval_num_visualizations

                    with torch.no_grad(), self.autocast:
                        decoded_gt = self._decode_tokens(tokens, height, width, num_cameras=num_cameras)

                    ray_conf_gt = decoded_gt.get("ray_conf")
                    loss_pm_gt, loss_d_gt, loss_ray_gt = self._compute_frame_losses(
                        decoded_gt, batch, pred_slice, ray_conf_gt, B, height, width
                    )
                    batch_losses.update({
                        "LossPointmap_OrigVsGT": loss_pm_gt.item(),
                        "LossDepth_OrigVsGT": loss_d_gt.item(),
                        "LossRaymap_OrigVsGT": loss_ray_gt.item(),
                    })

                    mask_u = batch["gt_mask"][:, pred_slice].to(self.device)
                    nv = mask_u.shape[1]

                    def _pred_vs_orig(decoded_pred):
                        loss_pm_u, _ = self.pointmap_criterion(
                            decoded_pred["pointmap"][:, pred_slice].float(),
                            decoded_gt["pointmap"][:, pred_slice].float(),
                            mask=mask_u,
                        )
                        pred_du = decoded_pred["depth"][:, pred_slice].float().reshape(B * nv, 1, height, width)
                        tgt_du = decoded_gt["depth"][:, pred_slice].float().reshape(B * nv, 1, height, width)
                        d_mask_u = mask_u.reshape(B * nv, 1, height, width).float()
                        loss_d_u, _ = self.depth_criterion(pred_du, tgt_du, confidence=None, mask=d_mask_u)

                        pred_ray_u = decoded_pred["ray"][:, pred_slice].float()
                        tgt_ray_u = decoded_gt["ray"][:, pred_slice].float()
                        dir_l2 = torch.norm(pred_ray_u[..., :3] - tgt_ray_u[..., :3], dim=-1).mean()
                        org_l2 = torch.norm(pred_ray_u[..., 3:] - tgt_ray_u[..., 3:], dim=-1).mean()
                        loss_ray_u = 10.0 * dir_l2 + org_l2
                        return loss_pm_u, loss_d_u, loss_ray_u

                    loss_pm_u, loss_d_u, loss_ray_u = _pred_vs_orig(decoded)
                    loss_pm_u_ar, loss_d_u_ar, loss_ray_u_ar = _pred_vs_orig(decoded_ar)

                    batch_losses.update({
                        "LossPointmap_PredVsOrig": loss_pm_u.item(),
                        "LossDepth_PredVsOrig": loss_d_u.item(),
                        "LossRaymap_PredVsOrig": loss_ray_u.item(),
                        "LossPointmap_PredVsOrig_AR": loss_pm_u_ar.item(),
                        "LossDepth_PredVsOrig_AR": loss_d_u_ar.item(),
                        "LossRaymap_PredVsOrig_AR": loss_ray_u_ar.item(),
                    })

                    if need_viz:
                        # Optional RGB decode via the pretrained MAE-style image decoder.
                        # decode_to_image returns (B, V, 3, H, W) in [0, 1]; we permute to
                        # HWC and scale to 0-255 to match the depth panels in _log_viz_sample.
                        have_img_dec = getattr(self.occ_rae, "img_decoder", None) is not None
                        if have_img_dec:
                            vcs = getattr(self, "_img_decoder_view_chunk", 0)
                            with torch.no_grad(), self.autocast:
                                rgb_tf_full = self.occ_rae.decode_to_image(
                                    {"tokens": full_tokens, "H": height, "W": width},
                                    view_chunk_size=vcs,
                                )
                                rgb_ar_full = self.occ_rae.decode_to_image(
                                    {"tokens": full_tokens_ar, "H": height, "W": width},
                                    view_chunk_size=vcs,
                                )
                                rgb_gt_full = self.occ_rae.decode_to_image(
                                    {"tokens": tokens, "H": height, "W": width},
                                    view_chunk_size=vcs,
                                )

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
                            gt_token_depth = decoded_gt["depth"][batch_idx].detach().float().cpu()[view_index]
                            gt_token_depth_color = torch.stack([
                                torch.from_numpy(
                                    depth2rgb(
                                        gt_token_depth[t].clamp(0, 50).numpy(),
                                        valid_mask=gt_token_depth[t].numpy() > 0,
                                        min_depth=0.0,
                                        max_depth=50.0,
                                    ).astype(np.float32)
                                )
                                for t in range(V)
                            ])
                            ar_pred_depth = decoded_ar["depth"][batch_idx].detach().float().cpu()[view_index]
                            ar_pred_depth_color = torch.stack([
                                torch.from_numpy(
                                    depth2rgb(
                                        ar_pred_depth[t].clamp(0, 50).numpy(),
                                        valid_mask=ar_pred_depth[t].numpy() > 0,
                                        min_depth=0.0,
                                        max_depth=50.0,
                                    ).astype(np.float32)
                                )
                                for t in range(V)
                            ])
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
                                # (V, 3, H, W) -> (V, H, W, 3) in [0, 255] float32.
                                rgb_tf_b = (
                                    rgb_tf_full[batch_idx].detach().float().cpu()[view_index]
                                    .permute(0, 2, 3, 1).contiguous() * 255.0
                                )
                                rgb_ar_b = (
                                    rgb_ar_full[batch_idx].detach().float().cpu()[view_index]
                                    .permute(0, 2, 3, 1).contiguous() * 255.0
                                )
                                rgb_gt_b = (
                                    rgb_gt_full[batch_idx].detach().float().cpu()[view_index]
                                    .permute(0, 2, 3, 1).contiguous() * 255.0
                                )
                                # Blank context-view AR cells, matching ar_pred_depth_color.
                                for t, blank in enumerate(pred_blank_views):
                                    if blank:
                                        rgb_ar_b[t] = 255.0
                                # Append RGB panels after depth so col_titles align with the
                                # fixed base layout in _log_viz_sample (gt_img, pred_depth_TF).
                                extra_panels = extra_panels + [
                                    rgb_tf_b,
                                    rgb_ar_b,
                                    rgb_gt_b,
                                ]
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
                            )
                            if saved_path is not None:
                                print(f"Saved viz: {saved_path}")
                            num_vis += 1

                metric.update(batch_size=B, **batch_losses)
                items_seen += B
                if self.is_master:
                    metric_logger.update(loss=loss_recon.item())

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

    # _build_occ_rae moved verbatim to occrae/deltatok_shared.py
    # (DeltaTokSharedMixin).

    def fit(self):
        # Build train/test loaders via `get_data_loader`, matching the
        # `train_occrae_img_decoder.py` data path. The dataset config is now a
        # Python expression string evaluated by `dust3r.datasets`, e.g.:
        #   "10000 @ WaymoSeqMultiView(...) + 5000 @ VKittiSeqMultiView(...)"
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
            epoch_wall_t0 = time.time()  # full-epoch wall (train + eval + save) for the time-limit guard
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

            self._end_of_epoch(epoch_wall_t0)

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
