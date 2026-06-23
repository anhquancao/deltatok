import math 
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

# from transformer_block import RMSNorm, Attention, FeedForward, TimestepEmbedder
from occrae.network.transformer_block import RMSNorm, Attention, CrossAttention, FeedForward, TimestepEmbedder

# Reuse DINOv3's rope embedding so the flow ViT's optional camera rope is built from
# the exact same axial-rope frequencies / 1xN camera-grid as the DeltaTok tokenizer.
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding
from occrae.network.rope_utils import compute_camera_rope  # shared 1xN camera-grid rope build

def modulate(x, gamma):
    return x * (1 + gamma)


def gem_timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
    return embedding



class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0., use_cross_attn=False):
        super().__init__()

        self.spatial_attn = Attention(dim, heads, dropout=dropout)
        self.ln_spatial = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

        self.temporal_attn = Attention(dim, heads, dropout=dropout)
        self.ln_temporal = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

        if use_cross_attn:
            self.cross_attn = CrossAttention(dim, heads, dropout=dropout)
            self.ln_cross = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        else:
            self.cross_attn = None

        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.ln_mlp = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

    def forward(self, x, ada_cond, cross_cond, t, s, temporal_mask=None, cross_mask=None, spatial_rope=None):
        """
        Args:
            x: main hidden states [B, N, D] with N = n_reg + t*s video tokens
            ada_cond: 8-tuple of per-token AdaLN modulation tensors [B, N, D]
            cross_cond: optional key/values for per-slot cross-attention [(B*s), L, D]
            t: number of temporal slots (frames; for delta-token flow: T-1 transitions)
            s: number of spatial tokens per temporal slot (h_p*w_p; for delta-token
               flow: N_cam — one delta token per camera)
            spatial_rope: optional (cos, sin), each (s, head_dim), rotary over the
               camera (spatial) axis. Applied in spatial (cross-camera) attention only.
        """

        # Generate all modulation parameters from style
        (
            gamma_spatial, alpha_spatial,
            gamma_temporal, alpha_temporal,
            gamma_cross, alpha_cross,
            gamma_mlp, alpha_mlp,
        ) = ada_cond

        b, n, d = x.shape
        n_video = t * s
        n_reg = n - n_video

        if n_reg < 0:
            raise ValueError(f"Invalid token layout: got N={n}, expected at least T*S={n_video}")

        x_reg = x[:, :n_reg] if n_reg > 0 else None
        x_video = x[:, n_reg:]

        # --- Spatial self-attention (within frame; = cross-camera for delta-token flow) ---
        x_spatial = modulate(self.ln_spatial(x_video), gamma_spatial[:, n_reg:])
        x_spatial = rearrange(x_spatial, 'b (t s) d -> (b t) s d', t=t, s=s)
        x_spatial = self.spatial_attn(x_spatial, rope=spatial_rope)  # camera rope over the s axis
        x_spatial = rearrange(x_spatial, '(b t) s d -> b (t s) d', b=b, t=t, s=s)
        x_video = x_video + alpha_spatial[:, n_reg:] * x_spatial

        # --- Temporal self-attention (across time, per spatial location) ---
        x_temporal = modulate(self.ln_temporal(x_video), gamma_temporal[:, n_reg:])
        x_temporal = rearrange(x_temporal, 'b (t s) d -> (b s) t d', t=t, s=s)
        x_temporal = self.temporal_attn(x_temporal, mask=temporal_mask)
        x_temporal = rearrange(x_temporal, '(b s) t d -> b (t s) d', b=b, t=t, s=s)
        x_video = x_video + alpha_temporal[:, n_reg:] * x_temporal

        # --- Cross-attention (per spatial slot, camera-permutation-equivariant) ---
        # cross_cond is (B*s, L, D) with L context tokens per slot: slot i (camera i)
        # only attends to its own context, mirroring DeltaTok's structural
        # z_i <-> camera_i binding (no camera IDs needed).
        if self.cross_attn is not None and cross_cond is not None:
            x_cross = modulate(self.ln_cross(x_video), gamma_cross[:, n_reg:])      # (B, t*s, D) normalized + AdaLN-scaled queries
            x_cross = rearrange(x_cross, 'b (t s) d -> (b s) t d', t=t, s=s)        # (B*s, t, D) fold slots into batch
            x_cross = self.cross_attn(x_cross, cross_cond, cross_mask)              # (B*s, t, D) q=x_cross, kv=cross_cond (B*s, L, D)
            x_cross = rearrange(x_cross, '(b s) t d -> b (t s) d', b=b, t=t, s=s)   # (B, t*s, D) back to token layout
            x_video = x_video + alpha_cross[:, n_reg:] * x_cross                    # (B, t*s, D) gated residual (alpha_cross starts at 0)

        x = torch.cat([x_reg, x_video], dim=1) if n_reg > 0 else x_video

        # --- Feed-forward with AdaLN modulation ---
        x = x + alpha_mlp * self.ff(modulate(self.ln_mlp(x), gamma_mlp))

        return x
        

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_cross_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout, use_cross_attn=use_cross_attn))

    def forward(self, x, ada_cond, cross_cond, t, s, temporal_mask=None, cross_mask=None, spatial_rope=None):
        feat = None
        for i, block in enumerate(self.layers):
            x = block(x, ada_cond=ada_cond, cross_cond=cross_cond, t=t, s=s, temporal_mask=temporal_mask, cross_mask=cross_mask, spatial_rope=spatial_rope)
            if i == 8:
                feat = x
        return x, feat
    

class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
    def __init__(self, out_dim=1536, num_views=10, cross_dim=None, hidden_dim=768,
                 depth=12, heads=16, mlp_dim=3072, dropout=0.,
                 register=1, proj=1, is_causal=False,
                 use_trajectory_cond=False, trajectory_length=25,
                 ref_spatial_size=(16, 16), use_camera_rope=False, max_cameras=32,
                 use_camera_embed=False):
        super().__init__()

        self.c = out_dim                                                # Number of channels as input
        self.t = num_views                                              # Number of views/timesteps as input
        self.hidden_dim = hidden_dim                                    # Hidden dimension of the transformer
        self.proj = proj                                                # Projection
        self.register = register                                        # add register token
        self.is_causal = is_causal                                      # use temporal causal mask in the transformer
        self.use_trajectory_cond = use_trajectory_cond                  # fuse trajectory into ada conditioning
        self.trajectory_length = trajectory_length                      # expected trajectory temporal length
        self.ref_spatial_size = ref_spatial_size                        # reference spatial size for positional embeddings
        self.use_camera_rope = use_camera_rope                          # rotary over the camera (spatial) axis in spatial attn
        self.max_cameras = max_cameras                                  # rope/embed slot cap (matches tokenizer MAX_CAMERAS=32)
        self.use_camera_embed = use_camera_embed                        # absolute learned per-camera identity (additive, ungated)

        if self.use_camera_rope:
            head_dim = hidden_dim // heads
            assert head_dim % 2 == 0, f"camera rope needs even head_dim, got {head_dim}"
            # Same DINOv3 rope module the tokenizer uses, sized to THIS ViT's head_dim
            # (hidden_dim // heads). Kept in eval (no coord augmentation) — the
            # per-forward camera<->position shuffle gives permutation invariance.
            rope_cfg = DINOv3ViTConfig()
            rope_cfg.hidden_size = hidden_dim
            rope_cfg.num_attention_heads = heads
            rope_cfg.patch_size = 16                                    # only sizes the synthetic 1xN grid; cancels out
            self.rope_embeddings = DINOv3ViTRopePositionEmbedding(rope_cfg)
            self.rope_embeddings.eval()
            self._rope_cache = {}                                       # (device, dtype) -> (cos, sin) over max_cameras

        # Separate temporal and spatial positional embeddings
        # Temporal: Learned embedding (nn.Embedding)
        # Spatial: Learnable absolute pos_embed + bicubic interpolation
        ref_h, ref_w = self.ref_spatial_size
        self.temporal_pos = nn.Embedding(self.t, hidden_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, ref_h * ref_w + 1, hidden_dim))

        # Absolute per-camera identity over the spatial (camera) axis: one learned
        # vector per slot, added ungated so each generated delta binds to its camera.
        if self.use_camera_embed:
            self.camera_embed = nn.Embedding(max_cameras, hidden_dim)

        # number of spatial tokens (including CLS if present in spatial_pos, but here we use it for reference)
        self.num_spatial = ref_h * ref_w + 1

        self.time_embed = TimestepEmbedder(in_dim=hidden_dim, out_dim=hidden_dim*8)
        if self.use_trajectory_cond:
            self.trajectory_embed = GemTrajectoryEmbedder(
                trajectory_length=trajectory_length,
                coord_dim=2,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim * 8,
            )

        if cross_dim is not None:
            self.cross_cond_emb = nn.Sequential(
                nn.Linear(cross_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            # Interpolated spatial grid pos-embed for token-sequence (5-D) cross
            # conditioning. Per-slot only — no camera embedding: the slot<->camera
            # binding is structural (per-slot cross-attention in Block).
            self.cross_ref_spatial_size = (37, 37)                                            # reference (Hp, Wp) patch grid
            ref_ch, ref_cw = self.cross_ref_spatial_size
            self.cross_spatial_pos = nn.Parameter(torch.zeros(1, ref_ch * ref_cw, hidden_dim))  # (1, Hp*Wp, D), no CLS slot

        # project the input to a smaller space
        self.in_proj = nn.Conv2d(self.c, hidden_dim, kernel_size=self.proj, stride=self.proj)
        self.out_proj = nn.Linear(hidden_dim, self.c*proj**2)

        # The Transformer Encoder a la BERT :)
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                              dropout=dropout, use_cross_attn=cross_dim is not None)

        self.last_norm = RMSNorm(dim=hidden_dim, linear=True, bias=True)

        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.initialize_weights()  # Init weight

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.temporal_pos.weight, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        if self.use_camera_embed:
            nn.init.normal_(self.camera_embed.weight, std=0.02)
        if hasattr(self, "cross_spatial_pos"):
            nn.init.trunc_normal_(self.cross_spatial_pos, std=0.02)

        # Init proj layer
        if self.proj > 1:
            nn.init.xavier_uniform_(self.in_proj.weight)
            # nn.init.xavier_uniform_(self.out_proj.weight)

        # Init register
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

        # AdaLN-Zero style init: start modulation close to identity / zero residual scaling
        nn.init.constant_(self.time_embed.mlp[-1].weight, 0)
        nn.init.constant_(self.time_embed.mlp[-1].bias, 0)
        if self.use_trajectory_cond:
            nn.init.constant_(self.trajectory_embed.mlp[-1].weight, 0)
            nn.init.constant_(self.trajectory_embed.mlp[-1].bias, 0)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep DINOv3 rope coords un-augmented (deterministic) regardless of train
        # mode; the per-forward camera<->position shuffle in _camera_rope is what
        # provides permutation invariance.
        if self.use_camera_rope:
            self.rope_embeddings.eval()
        return self

    def _camera_rope(self, num_cameras, device, dtype):
        """Rotary cos/sin over camera slots, mirroring DeltaTokModule camera rope.

        Lays cameras on a synthetic 1 x max_cameras DINOv3 patch grid (resolution
        -independent), then maps camera slot c -> position ``base_idx[c]``. base_idx
        is 0..N-1, shuffled per forward while training (permutation invariance) and
        identity at eval. Returns (cos, sin) of shape (num_cameras, head_dim).
        """
        assert num_cameras <= self.max_cameras, (
            f"camera rope supports <= {self.max_cameras} cameras, got {num_cameras}")
        cache_key = (device, dtype)
        cam = self._rope_cache.get(cache_key)
        if cam is None:
            # eval mode => deterministic coords (no jitter), so cache per (device, dtype).
            cam = compute_camera_rope(self.rope_embeddings, self.max_cameras, device, dtype)
            self._rope_cache[cache_key] = cam
        cam_cos, cam_sin = cam
        base_idx = torch.arange(num_cameras, device=device)            # camera slot c -> position c
        if self.training:
            base_idx = base_idx[torch.randperm(num_cameras, device=device)]  # shuffle (perm-invariance)
        return cam_cos[base_idx], cam_sin[base_idx]                     # (num_cameras, head_dim) each

    def interpolate_pos_encoding(self, h, w, device):
        """
        Bicubic interpolation of positional embeddings to match input grid size.
        Follows DA3's interpolate_pos_encoding.
        """
        ref_h, ref_w = self.ref_spatial_size
        if h == ref_h and w == ref_w:
            return self.spatial_pos

        cls_pos_embed = self.spatial_pos[:, :1]
        patch_pos_embed = self.spatial_pos[:, 1:]
        dim = self.spatial_pos.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, ref_h, ref_w, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h, w),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

        return torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

    def interpolate_cross_pos_encoding(self, h, w):
        """Same as interpolate_pos_encoding but for cross_spatial_pos (no CLS slot)."""
        ref_h, ref_w = self.cross_ref_spatial_size
        if h == ref_h and w == ref_w:
            return self.cross_spatial_pos                                           # (1, ref_h*ref_w, D) no resize needed

        dim = self.cross_spatial_pos.shape[-1]
        pos = self.cross_spatial_pos.reshape(1, ref_h, ref_w, dim).permute(0, 3, 1, 2)  # (1, D, ref_h, ref_w) channels-first for interpolate
        pos = F.interpolate(pos, size=(h, w), mode='bicubic', align_corners=False)      # (1, D, h, w) bicubic resize to target grid
        return pos.permute(0, 2, 3, 1).flatten(1, 2)                                    # (1, h*w, D) back to token sequence

    def forward(self, x, ada_cond, cross_cond=None, return_feat=False, trajectory_cond=None, trajectory_keep_mask=None):
        b, c, t, h, w = x.size()
        
        h_p, w_p = h // self.proj, w // self.proj
        S = h_p * w_p # spatial token per frames

        if self.is_causal:
            temporal_mask = torch.full((t, t), float("-inf"), device=x.device)
            temporal_mask = torch.triu(temporal_mask, diagonal=1)
        else:
            temporal_mask = None
        
        x = rearrange(x, 'b c t h w -> (b t) c h w', b=b, t=t, c=c, h=h, w=w).contiguous()
        x = self.in_proj(x)
        _, c, h_p, w_p = x.shape
        x = rearrange(x, '(b t) c h w -> b (t h w) c', b=b, t=t, c=c, h=h_p, w=w_p).contiguous()

        # Positional embeddings
        # 1. Temporal: Learned Embedding
        t_pos = torch.arange(t, device=x.device).repeat_interleave(S)
        t_pos_embed = self.temporal_pos(t_pos) # (t*S, D)

        # 2. Spatial: Interpolated learnable grid
        s_pos_embed = self.interpolate_pos_encoding(h_p, w_p, x.device) # (1, h_p*w_p + 1, D)
        s_pos_embed = s_pos_embed[:, 1:] # Drop CLS slot since input doesn't have it (1, h_p*w_p, D)
        s_pos_embed = repeat(s_pos_embed, '1 s d -> (t s) d', t=t) # (t*s, D)

        pos = t_pos_embed + s_pos_embed

        # 3. Absolute per-camera identity (S = N_cam for delta-token flow), shared
        # across t and laid out in the same (t s) token order as the embeds above.
        if self.use_camera_embed:
            assert S <= self.max_cameras, f"camera embed supports <= {self.max_cameras} cameras, got S={S}"
            cam_idx = torch.arange(S, device=x.device)                            # camera slot 0..S-1
            cam_embed = repeat(self.camera_embed(cam_idx), 's d -> (t s) d', t=t)  # (t*S, D) per-camera, tiled over t
            pos = pos + cam_embed

        x = x + pos

        # timestep embeddings
        if ada_cond.dim() == 1:
            ada_cond = ada_cond[:, None]
        if ada_cond.shape[1] == 1 and t > 1:
            ada_cond = ada_cond.expand(-1, t)
        elif ada_cond.shape[1] != t:
            raise ValueError(f"ada_cond has shape {ada_cond.shape}, expected (B, T) with T={t}")
        
        t_emb = self.time_embed(ada_cond).chunk(8, dim=-1)
        if self.use_trajectory_cond:
            if trajectory_cond is None:
                raise ValueError("trajectory conditioning is enabled but trajectory_cond is missing")
            if trajectory_cond.shape[0] != b:
                raise ValueError(
                    f"trajectory_cond batch size {trajectory_cond.shape[0]} does not match input batch size {b}"
                )
            traj_emb = self.trajectory_embed(trajectory_cond).to(dtype=t_emb[0].dtype)
            if trajectory_keep_mask is not None:
                if trajectory_keep_mask.shape[0] != b:
                    raise ValueError(
                        f"trajectory_keep_mask batch size {trajectory_keep_mask.shape[0]} does not match input batch size {b}"
                    )
                traj_keep = trajectory_keep_mask.to(device=traj_emb.device, dtype=traj_emb.dtype).view(b, 1)
                traj_emb = traj_emb * traj_keep
            traj_emb = traj_emb[:, None, :].expand(-1, t, -1).contiguous()
            traj_emb = traj_emb.chunk(8, dim=-1)
            t_emb = [
                time_chunk + traj_chunk
                for time_chunk, traj_chunk in zip(t_emb, traj_emb)
            ]
        elif trajectory_cond is not None:
            raise ValueError("trajectory_cond was provided but trajectory conditioning is disabled for this model")
        elif trajectory_keep_mask is not None:
            raise ValueError("trajectory_keep_mask was provided but trajectory conditioning is disabled for this model")

        t_emb_expanded = [e.repeat_interleave(S, dim=1) for e in t_emb]  # (B, N, D)

        # Cross-attention conditioning
        if cross_cond is not None:
            if cross_cond.dim() == 5:
                # Per-slot token conditioning: one (Hp, Wp) context grid per spatial
                # slot (S slots per frame; for delta-token flow S = N_cam). Slot i
                # only sees its own grid in Block -> camera-permutation-equivariant.
                bc, n_cond, ch, cw, _ = cross_cond.shape                                       # (B, S, Hp, Wp, cross_dim)
                if n_cond != S:
                    raise ValueError(f"cross_cond has {n_cond} slots, expected S={S}")
                cc = self.cross_cond_emb(cross_cond)                                           # (B, S, Hp, Wp, D) project to hidden dim
                cc = cc + self.interpolate_cross_pos_encoding(ch, cw).view(1, 1, ch, cw, -1)   # (B, S, Hp, Wp, D) add spatial pos-embed
                cross_cond = rearrange(cc, 'b n h w d -> (b n) (h w) d')                       # (B*S, Hp*Wp, D) per-slot kv sequence
            else:
                # Legacy single-vector conditioning (e.g. CLIP embedding).
                cc = self.cross_cond_emb(cross_cond)                                           # (B, D)
                cross_cond = cc[:, None, None, :].expand(b, S, 1, -1).reshape(b * S, 1, -1)    # (B*S, 1, D) broadcast same vector to every slot
        
        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([self.reg_tokens(reg).expand(b, self.register, self.hidden_dim), x], dim=1)
            t_emb_expanded = [torch.cat([torch.zeros(b, self.register, self.hidden_dim, dtype=x.dtype, device=x.device), e], dim=1)
                                for e in t_emb_expanded]

        # Camera rope over the spatial axis (= cameras for delta-token flow, S = N_cam);
        # gives each camera a distinct rotation in the cross-camera (spatial) attention.
        spatial_rope = self._camera_rope(S, x.device, x.dtype) if self.use_camera_rope else None

        x, feat = self.transformer(x=x, ada_cond=t_emb_expanded, cross_cond=cross_cond, t=t, s=S, temporal_mask=temporal_mask, spatial_rope=spatial_rope)

        # drop the register(s)
        x = x[:, self.register:].contiguous()

        x = self.last_norm(x)
        x = self.out_proj(x)
        x = rearrange(x, 'b (t h w) (c s1 s2) -> b c t (h s1) (w s2)', s1=self.proj, s2=self.proj, b=b, c=self.c, h=h, w=w).contiguous()

        if return_feat:
            return x, feat
        
        return x 


if __name__ == "__main__":
    import math
    import time
    from collections import deque
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs") / "efficient_transformer_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import psutil
        process = psutil.Process()
    except Exception:
        process = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    def get_memory_stats():
        if device.type == "cuda":
            mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            return f"mem_alloc={mem_alloc_mb:.1f}MB | mem_peak={mem_peak_mb:.1f}MB"

        if process is not None:
            rss_mb = process.memory_info().rss / (1024 ** 2)
            return f"rss={rss_mb:.1f}MB"

        return "rss=n/a"

    def make_synthetic_video(B=2, C=3, T=8, H=64, W=64, device="cpu"):
        x = torch.zeros(B, C, T, H, W, device=device)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        for b in range(B):
            for t in range(T):
                phase = 2 * math.pi * t / T

                # --- Channel 0: moving square ---
                cx = int((W//2) + (W//4) * math.sin(phase))
                cy = int((H//2) + (H//4) * math.cos(phase))
                size = H // 6
                x[b, 0, t, cy-size:cy+size, cx-size:cx+size] = 1.0

                # --- Channel 1: moving circle ---
                radius = H // 5
                mask = (xx - math.sin(phase)*0.5)**2 + (yy - math.cos(phase)*0.5)**2 < (radius/H)**2
                x[b, 1, t][mask] = 1.0

                # --- Channel 2: temporal gradient ---
                grad = (xx + 1) / 2
                grad = grad * (0.5 + 0.5 * math.sin(phase))
                x[b, 2, t] = grad

        return x

    B, C, T, H, W = 32, 3, 7, 40, 52
    x1 = make_synthetic_video(B, C, T, H, W, device=device)
    def show_video(x, b=0, c=0, save_path=None):
        T = x.shape[2]
        fig, axes = plt.subplots(1, T, figsize=(3*T, 3))
        for t in range(T):
            axes[t].imshow(x[b, c, t].cpu(), cmap="gray")
            axes[t].set_title(f"t={t}")
            axes[t].axis("off")
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # show_video(x1, b=0, c=0)  # moving square
    # show_video(x1, b=0, c=1)  # moving circle
    # show_video(x1, b=0, c=2)  # temporal gradient

    clip_emb = torch.zeros(B, 768, device=device)  # constant conditioning

    x0_fixed = torch.randn_like(x1)
    t_fixed = torch.ones(B, T, device=device) * 0.5

    def flow_matching_loss(model, x1, clip_emb):
        B = x1.size(0)
        x0 = torch.randn_like(x1)

        t = torch.rand(B, 1, device=x1.device)

        xt = (1 - t.view(B,1,1,1,1)) * x0 + t.view(B,1,1,1,1) * x1
        v_target = x1 - x0

        v_pred = model(x=xt, ada_cond=t, cross_cond=clip_emb)

        return ((v_pred - v_target) ** 2).mean()

    @torch.no_grad()
    def sample(model, clip_emb, shape, steps=50, device="cpu"):
        B = shape[0]
        x = torch.randn(shape, device=device)

        t_vals = torch.linspace(0.0, 1.0, steps+1, device=device)

        for i in range(steps):
            t = t_vals[i]
            t_batch = torch.full((B,1), t, device=device)

            v = model(x=x, ada_cond=t_batch, cross_cond=clip_emb)

            dt = t_vals[i+1] - t_vals[i]

            x = x + v * dt

        return x

    def visualize_gt_vs_gen_rgb(gt, gen, batch_idx=0, save_path=None):
        gt_vid = gt[batch_idx].permute(1,2,3,0)   # (T,H,W,C)
        gen_vid = gen[batch_idx].permute(1,2,3,0)

        T = gt_vid.shape[0]
        fig, axes = plt.subplots(T, 2, figsize=(6, 3*T))

        for t in range(T):
            gt_frame = gt_vid[t]
            gen_frame = gen_vid[t]

            # normalize for display
            gt_frame = (gt_frame - gt_frame.min()) / (gt_frame.max() - gt_frame.min() + 1e-8)
            gen_frame = (gen_frame - gen_frame.min()) / (gen_frame.max() - gen_frame.min() + 1e-8)

            axes[t,0].imshow(gt_frame.cpu())
            axes[t,0].set_title(f"GT {t}")
            axes[t,0].axis("off")

            axes[t,1].imshow(gen_frame.cpu())
            axes[t,1].set_title(f"GEN {t}")
            axes[t,1].axis("off")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Keep this mapping aligned with Trainer.abstract_trainer.transformer_size.
    def transformer_size(size):
        if size == "tiny":
            return 384, 6, 6
        if size == "small":
            return 384, 12, 6
        if size == "base":
            return 768, 12, 12
        if size == "large":
            return 1024, 24, 16
        if size == "xlarge":
            return 1152, 28, 16
        if size == "xxlarge":
            return 1536, 28, 16
        if str(size).lower() in ["2b", "giant", "xxxl"]:
            return 2048, 32, 16
        return 768, 12, 12


    def estimate_flops_one_iteration(model, batch_size, c, t, h, w, device):
        # Preferred path: THOP library profiling for one forward pass.
        try:
            from thop import profile

            x = torch.randn(batch_size, c, t, h, w, device=device)
            ada_cond = torch.full((batch_size, t), 0.5, device=device)

            model.eval()
            macs, _ = profile(model, inputs=(x, ada_cond), verbose=False)
            flops_forward = 2.0 * float(macs)  # convention: 1 MAC ~= 2 FLOPs
            flops_train_step = 3.0 * flops_forward  # approximation: fwd + bwd
            return flops_forward, flops_train_step, "thop"
        except Exception:
            # Fallback: analytical approximation.
            d = model.hidden_dim
            p = model.proj
            r = model.register

            mlp_dim = model.transformer.layers[0].ff.net[0].out_features
            depth = len(model.transformer.layers)

            h_p = h // p
            w_p = w // p
            n_video = t * h_p * w_p
            n = n_video + r

            flops_in_proj = 2.0 * n_video * (c * p * p) * d
            flops_out_proj = 2.0 * n_video * d * (c * p * p)
            flops_attn_linear = 4.0 * n * d * d
            flops_attn_matmul = 2.0 * n * n * d
            flops_mlp = 2.0 * n * d * mlp_dim
            flops_per_layer = flops_attn_linear + flops_attn_matmul + flops_mlp

            flops_forward_per_sample = flops_in_proj + depth * flops_per_layer + flops_out_proj
            flops_forward = flops_forward_per_sample * batch_size
            flops_train_step = 3.0 * flops_forward
            return flops_forward, flops_train_step, "analytical-fallback"

    def format_flops(v):
        if v >= 1e15:
            return f"{v / 1e15:.3f} PFLOPs"
        if v >= 1e12:
            return f"{v / 1e12:.3f} TFLOPs"
        if v >= 1e9:
            return f"{v / 1e9:.3f} GFLOPs"
        if v >= 1e6:
            return f"{v / 1e6:.3f} MFLOPs"
        return f"{v:.3f} FLOPs"
    
    def param_count(archi, model):
        print(f"Size of model {archi}: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

    for vit_size in ["tiny", "small", "base", "large", "xlarge", "xxlarge"]:
        hidden_dim, depth, heads = transformer_size(vit_size)

        model = Transformer(
            out_dim=C,
            num_views=T,
            cross_dim=None,
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=hidden_dim * 4,
            dropout=0.0,
            register=1,
            proj=2, 
            is_causal=False
        ).to(device)

        print(f"Using device: {device}")
        param_count(vit_size, model)
        if vit_size in ["xlarge", "xxlarge"]:
            flops_fwd, flops_train, flop_source = estimate_flops_one_iteration(model, B, C, T, H, W, device)
            print(f"FLOPs source: {flop_source}")
            print(f"Estimated FLOPs/iteration (forward only): {format_flops(flops_fwd)}")
            print(f"Estimated FLOPs/iteration (train step): {format_flops(flops_train)}")
    exit()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    max_step = 10_000
    loss_window = deque(maxlen=100)
    step_time_window = deque(maxlen=100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_step, eta_min=1e-6)
    train_start_time = time.perf_counter()

    for step in range(max_step):
        step_start_time = time.perf_counter()
        opt.zero_grad()
        loss = flow_matching_loss(model, x1, clip_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        scheduler.step()

        loss_window.append(loss.item())
        step_time_window.append(time.perf_counter() - step_start_time)

        if step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            loss_avg = sum(loss_window) / len(loss_window)
            step_time_avg_ms = (sum(step_time_window) / len(step_time_window)) * 1000
            elapsed_s = time.perf_counter() - train_start_time
            mem_stats = get_memory_stats()
            print(f"step {step} | loss = {loss_avg:.6f} | step_ms = {step_time_avg_ms:.2f} | elapsed = {elapsed_s:.1f}s | {mem_stats}")

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                gen_vid = sample(model=model, clip_emb=clip_emb.to(device), shape=(2, C, T, H, W), steps=50, device=device)
            show_video(gen_vid, b=0, c=0, save_path=output_dir / f"sample_step_{step:05d}_ch0.png")
            model.train()
   
    model.eval()
    with torch.no_grad():
        gen_vid = sample(model=model, clip_emb=clip_emb.to(device), shape=(2, C, T, H, W), steps=50, device=device)
    show_video(gen_vid, b=0, c=0, save_path=output_dir / "final_ch0_square.png")
    show_video(gen_vid, b=0, c=1, save_path=output_dir / "final_ch1_circle.png")
    show_video(gen_vid, b=0, c=2, save_path=output_dir / "final_ch2_gradient.png")
    total_train_s = time.perf_counter() - train_start_time
    avg_step_ms = (total_train_s / max_step) * 1000
    if device.type == "cuda":
        final_mem = f"mem_alloc={torch.cuda.memory_allocated(device)/(1024**2):.1f}MB | mem_peak={torch.cuda.max_memory_allocated(device)/(1024**2):.1f}MB"
    elif process is not None:
        final_mem = f"rss={process.memory_info().rss/(1024**2):.1f}MB"
    else:
        final_mem = "rss=n/a"
    print(f"Training done | total_time = {total_train_s:.1f}s | avg_step = {avg_step_ms:.2f}ms | {final_mem}")
    print(f"Saved plots to: {output_dir.resolve()}")

 

