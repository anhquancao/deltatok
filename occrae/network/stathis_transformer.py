# BERT architecture for the Masked Bidirectional Encoder Transformer
# Updated with: RMSNorm, QK Normalization, and SwiGLU activation
import torch
import einops
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union
from einops import repeat, rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        """ RMSNorm normalization layer
            :param:
                dim    -> int: Dimension of the input
                eps    -> float: Small value for numerical stability
                linear -> bool: Whether to use learnable weight parameter
                bias   -> bool: Whether to use learnable bias parameter
        """
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """ Apply RMS normalization """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """ Forward pass through RMSNorm
            :param:
                x -> torch.Tensor: Input tensor
            :return:
                torch.Tensor: Normalized output
        """
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class PreNorm_Modulate(nn.Module):

    def __init__(self, dim, fn, use_rmsnorm=True):
        """ PreNorm module to apply normalization before a given function
            :param:
                dim         -> int: Dimension of the input
                fn          -> nn.Module: The function to apply after normalization
                use_rmsnorm -> bool: Whether to use RMSNorm (True) or LayerNorm (False)
            """
        super().__init__()
        if use_rmsnorm:
            self.norm = RMSNorm(dim, linear=True, bias=False)
        else:
            self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, shift, scale, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                shift    -> torch.Tensor: Shift tensor
                scale    -> torch.Tensor: Scale tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after normalization
        """
        # return self.fn(self.norm(x), **kwargs)
        return self.fn(modulate(self.norm(x), shift, scale), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., use_swiglu=True, multiple_of=256, bias=True):
        """ Initialize the Multi-Layer Perceptron (MLP) with optional SwiGLU activation.
            :param:
                dim         -> int: Dimension of the input
                hidden_dim  -> int: Dimension of the hidden layer
                dropout     -> float: Dropout rate
                use_swiglu  -> bool: Whether to use SwiGLU activation (True) or standard GELU (False)
                multiple_of -> int: Make hidden dimension a multiple of this value (for SwiGLU)
                bias        -> bool: Whether to use bias in linear layers
        """
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout = dropout
        
        if use_swiglu:
            # SwiGLU activation: requires 3 weight matrices
            hidden_dim = int(2 * hidden_dim / 3)
            # Make sure it is a multiple of 'multiple_of' for efficiency
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
            
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
            self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        else:
            # Standard GELU activation
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim, bias=bias),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        if self.use_swiglu:
            # SwiGLU: SiLU(W1(x)) * W3(x), then W2
            x = F.silu(self.w1(x)) * self.w3(x)
            if self.dropout > 0. and self.training:
                x = F.dropout(x, self.dropout)
            return self.w2(x)
        else:
            return self.net(x)


class QKNorm(nn.Module):
    def __init__(self, dim: int, use_rmsnorm=True):
        """ QK Normalization layer for normalizing queries and keys
            :param:
                dim -> int: Dimension of the query/key vectors
        """
        super().__init__()
        if use_rmsnorm:
            self.query_norm = RMSNorm(dim, linear=False, bias=False)
            self.key_norm = RMSNorm(dim, linear=False, bias=False)
        else:
            self.query_norm = nn.LayerNorm(dim)
            self.key_norm = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        """ Normalize queries and keys
            :param:
                q -> torch.Tensor: Query tensor
                k -> torch.Tensor: Key tensor
                v -> torch.Tensor: Value tensor (used for dtype casting)
            :return:
                Tuple[torch.Tensor, torch.Tensor]: Normalized query and key tensors
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_qk_norm=True, use_rmsnorm=True):
        super().__init__()
        self.dim = embed_dim
        self.h = num_heads
        self.use_qk_norm = use_qk_norm
        self.use_rmsnorm = use_rmsnorm
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.o = nn.Linear(embed_dim, embed_dim, bias=True)
        self.drop = nn.Dropout(dropout)
        if use_qk_norm:
            self.qk_norm = QKNorm(embed_dim, use_rmsnorm=use_rmsnorm)

    def forward(self, x, mask=None, rope_freqs: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)  # 3, B, H, N, Dh
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, H, N, Dh

        if self.use_qk_norm:
            # Normalize per head dimension
            q = q.reshape(B, N, C)
            k = k.reshape(B, N, C)
            v_flat = v.reshape(B, N, C)
            q, k = self.qk_norm(q, k, v_flat)
            q = q.reshape(B, self.h, N, C // self.h)
            k = k.reshape(B, self.h, N, C // self.h)
            v = v  # v unchanged

        # Apply RoPE to q,k if provided
        if rope_freqs is not None:
            # Expect rope_freqs: [N, 1, 1, Dh]
            q_bshd = q.transpose(1, 2)  # [B, N, H, Dh]
            k_bshd = k.transpose(1, 2)  # [B, N, H, Dh]
            # print(f"Applying RoPE with rope_freqs shape: {rope_freqs.shape}, q shape: {q_bshd.shape}")
            q_bshd = apply_rotary_pos_emb(q_bshd, rope_freqs, tensor_format="bshd")
            k_bshd = apply_rotary_pos_emb(k_bshd, rope_freqs, tensor_format="bshd")
            q = q_bshd.transpose(1, 2)
            k = k_bshd.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.o(attn_out)
        return attn_out, None


class TransformerEncoderSeperableAttention(nn.Module):
        def __init__(self, dim, depth, heads, mlp_dim, dropout=0., window_size=1,
                use_rmsnorm=True, use_swiglu=True, use_qk_norm=True):
            """ Initialize the Attention module.
                :param:
                    dim          -> int: number of hidden dimension of attention
                    depth        -> int: number of layer for the transformer
                    heads        -> int: Number of heads
                    mlp_dim      -> int: number of hidden dimension for mlp
                    dropout      -> float: Dropout rate
                    window_size  -> int: Window size for spatial attention
                    use_rmsnorm  -> bool: Whether to use RMSNorm (True) or LayerNorm (False)
                    use_swiglu   -> bool: Whether to use SwiGLU activation in FFN
                    use_qk_norm  -> bool: Whether to use QK normalization in attention
            """
            super().__init__()
            self.window_size = window_size
            self.layers = nn.ModuleList([])
            self.adaLN_mods = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))
            # self.adaLN_mods = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm_Modulate(dim, Attention(dim, heads, dropout=dropout, use_qk_norm=use_qk_norm, use_rmsnorm=use_rmsnorm), use_rmsnorm=use_rmsnorm),
                    PreNorm_Modulate(dim, Attention(dim, heads, dropout=dropout, use_qk_norm=use_qk_norm, use_rmsnorm=use_rmsnorm), use_rmsnorm=use_rmsnorm),
                    PreNorm_Modulate(dim, FeedForward(dim, mlp_dim, dropout=dropout, use_swiglu=use_swiglu), use_rmsnorm=use_rmsnorm)
                ]))
                # mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))
                # self.adaLN_mods.append(mod)

        def forward(self, x, full_shape, rope_freqs: Optional[torch.Tensor] = None, c=None):
            b, t, h, w, d = full_shape
            rope_freqs_spatial, rope_freqs_temporal = rope_freqs
            l_attn = []
            (shift_t,  scale_t,  gate_t,
            shift_s,  scale_s,  gate_s,
            shift_ff, scale_ff, gate_ff) = self.adaLN_mods(c).chunk(9, dim=-1)
            shift_s = repeat(shift_s, 'b 1 d -> (b t) 1 d', t=t)
            scale_s = repeat(scale_s, 'b 1 d -> (b t) 1 d', t=t)
            gate_s = repeat(gate_s, 'b 1 d -> (b t) 1 d', t=t)
            n_repeats = (h // self.window_size) * (w // self.window_size)
            shift_t = repeat(shift_t, 'b 1 d -> (b n) 1 d', n=n_repeats)
            scale_t = repeat(scale_t, 'b 1 d -> (b n) 1 d', n=n_repeats)
            gate_t = repeat(gate_t, 'b 1 d -> (b n) 1 d', n=n_repeats)
            # n_repeats_ff = h * w
            # shift_ff = repeat(shift_ff, 'b t d -> b (t s) d', s=n_repeats_ff)
            # scale_ff = repeat(scale_ff, 'b t d -> b (t s) d', s=n_repeats_ff)
            # gate_ff = repeat(gate_ff, 'b t d -> b (t s) d', s=n_repeats_ff)
            for (attn_temporal, attn_spatial, ff) in self.layers:
            # for (attn_temporal, attn_spatial, ff), adaln in zip(self.layers, self.adaLN_mods):
            #     (shift_t,  scale_t,  gate_t,
            #      shift_s,  scale_s,  gate_s,
            #      shift_ff, scale_ff, gate_ff) = adaln(c).chunk(9, dim=-1)
                
                if self.window_size == 1:
                    x = einops.rearrange(x, 'b (t h w) c -> (b h w) t c', b=b, t=t, h=h, w=w)
                    n_repeats = h * w
                else:
                    x = einops.rearrange(x, 'b (t h w) c -> b t h w c', b=b, t=t, h=h, w=w)
                    x = einops.rearrange(
                        x, 'b t (h k1) (w k2) c -> (b h w) (t k1 k2) c',
                        k1=self.window_size, k2=self.window_size)
                    n_repeats = (h // self.window_size) * (w // self.window_size)

                # shift_t = repeat(shift_t, 'b t d -> (b n) t d', n=n_repeats)
                # scale_t = repeat(scale_t, 'b t d -> (b n) t d', n=n_repeats)
                # gate_t = repeat(gate_t, 'b t d -> (b n) t d', n=n_repeats)
                attention_value, attention_weight = attn_temporal(x, rope_freqs=rope_freqs_temporal, shift=shift_t, scale=scale_t)
                # x = x + attention_value
                x = x + gate_t * attention_value
                l_attn.append(attention_weight)

                if self.window_size == 1:
                    x = einops.rearrange(x, '(b h w) t c -> (b t) (h w) c', b=b, t=t, h=h, w=w)
                else:
                    x = einops.rearrange(
                        x, '(b h w) (t k1 k2) c -> (b t) (h k1) (w k2) c',
                        b=b, t=t, h=h//self.window_size, w=w//self.window_size, k1=self.window_size, k2=self.window_size)
                    x = x.flatten(1,2)
                # shift_s = repeat(shift_s, 'b t d -> (b t) 1 d')
                # scale_s = repeat(scale_s, 'b t d -> (b t) 1 d')
                # gate_s = repeat(gate_s, 'b t d -> (b t) 1 d')
                attention_value, attention_weight = attn_spatial(x, rope_freqs=rope_freqs_spatial, shift=shift_s, scale=scale_s)
                # x = x + attention_value
                x = x + gate_s * attention_value
                l_attn.append(attention_weight)

                x = einops.rearrange(x, '(b t) (h w) c -> b (t h w) c', b=b, t=t, h=h, w=w)
                # n_repeats = h * w
                # shift_ff = repeat(shift_ff, 'b t d -> b (t s) d', s=n_repeats)
                # scale_ff = repeat(scale_ff, 'b t d -> b (t s) d', s=n_repeats)
                # gate_ff = repeat(gate_ff, 'b t d -> b (t s) d', s=n_repeats)
                # x = x + ff(x)
                x = x + gate_ff * ff(x, shift=shift_ff, scale=scale_ff)
            return x, l_attn

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    # def forward(self, t):
    #     t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    #     t_emb = self.mlp(t_freq)
    #     return t_emb

    # def forward(self, t):
    #     # t: (B, T, 1, 1, 1)
    #     t_flat = einops.rearrange(t, 'b t 1 1 1 -> (b t)')
    #     t_freq = self.timestep_embedding(t_flat, self.frequency_embedding_size)
    #     t_emb = self.mlp(t_freq)
    #     t_emb = einops.rearrange(t_emb, '(b t) d -> b t d', b=t.shape[0], t=t.shape[1])
    #     return t_emb

    def forward(self, t):
        # t: (B, T, 1, 1, 1)
        t_flat = einops.rearrange(t, 'b 1 1 1 1 -> (b 1)')
        t_freq = self.timestep_embedding(t_flat, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        t_emb = einops.rearrange(t_emb, '(b 1) d -> b 1 d', b=t.shape[0])
        return t_emb

        

class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, out_channels, use_bias=True):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=use_bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift = repeat(shift, 'b 1 d -> b (1 s) d', s=x.shape[1] // shift.shape[1])
        scale = repeat(scale, 'b 1 d -> b (1 s) d', s=x.shape[1] // scale.shape[1])
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class MaskTransformer(nn.Module):
    def __init__(self, shape, img_size=256, embedding_dim=768, hidden_dim=768, codebook_size=1024, depth=24, heads=8, mlp_dim=3072, 
                dropout=0.1, one_var=False, use_fc_bias=False, use_first_last=False, 
                seperable_attention=False, seperable_window_size=1, train_aux=False,
                use_rmsnorm=True, use_swiglu=True, use_qk_norm=True):
        """ Initialize the Transformer model with modern features.
            :param:
                shape                  -> tuple: Shape of input [T, H, W]
                img_size               -> int: Input image size (default: 256)
                embedding_dim          -> int: Embedding dimension
                hidden_dim             -> int: Hidden dimension for the transformer (default: 768)
                codebook_size          -> int: Size of the codebook (default: 1024)
                depth                  -> int: Depth of the transformer (default: 24)
                heads                  -> int: Number of attention heads (default: 8)
                mlp_dim                -> int: MLP dimension (default: 3072)
                dropout                -> float: Dropout rate (default: 0.1)
                use_fc_bias            -> bool: Whether to use bias in fc layers
                use_first_last         -> bool: Whether to use first/last projection layers
                seperable_attention    -> bool: Whether to use separable space-time attention
                seperable_window_size  -> int: Window size for separable attention
                use_rmsnorm            -> bool: Whether to use RMSNorm (True) or LayerNorm (False)
                use_swiglu             -> bool: Whether to use SwiGLU activation in FFN
                use_qk_norm            -> bool: Whether to use QK normalization in attention
        """
        super().__init__()
        
        # RoPE frequencies generator (per-head dim, must be even)
        per_head_dim = hidden_dim // heads
        assert per_head_dim % 2 == 0, f"Per-head dim must be even for RoPE, got {per_head_dim}"
        self.pos_embd_spat = VideoRopePosition3DEmb(
            head_dim=per_head_dim,
            len_h=32,
            len_w=64,
            len_t=1,
            enable_fps_modulation=False,
        )
        self.pos_embd_spat.reset_parameters()

        self.pos_embd_temp = VideoRopePosition3DEmb(
            head_dim=per_head_dim,
            len_h=1,
            len_w=1,
            len_t=20,
            enable_fps_modulation=False,
        )
        self.pos_embd_temp.reset_parameters()
        

        self.t_embedder = TimestepEmbedder(hidden_size=hidden_dim)
    
        # First layer before the Transformer block
        self.first_layer = nn.Identity() 
        if use_first_last:
            if use_rmsnorm:
                self.first_layer = nn.Sequential(
                    RMSNorm(hidden_dim, linear=True, bias=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    RMSNorm(hidden_dim, linear=True, bias=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                )
            else:
                self.first_layer = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                )

        self.seperable_attention = seperable_attention
        if seperable_attention:
            self.transformer = TransformerEncoderSeperableAttention(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
                window_size=seperable_window_size, use_rmsnorm=use_rmsnorm, 
                use_swiglu=use_swiglu, use_qk_norm=use_qk_norm)
        else:
            assert "Not implemented for non-separable attention"

        # Last layer after the Transformer block
        self.last_layer = nn.Identity()
        if use_first_last:
            if use_rmsnorm:
                self.last_layer = nn.Sequential(
                    RMSNorm(hidden_dim, linear=True, bias=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    RMSNorm(hidden_dim, linear=True, bias=True),
                )
            else:
                self.last_layer = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                )

        # Bias for the last linear output
        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=use_fc_bias)
        self.fc_in.weight.data.normal_(std=0.02) 

        
        self.fc_out = FinalLayer(hidden_dim, embedding_dim, use_bias=use_fc_bias)

        
        # adaLN-Zero init: zero out the final linear of each adaLN modulator
        # for layer in self.transformer.adaLN_mods:
        #     for mod in layer:
        #         if isinstance(mod, nn.Linear):
        #             nn.init.constant_(mod.weight, 0)
        #             if mod.bias is not None:
        #                 nn.init.constant_(mod.bias, 0)
        nn.init.constant_(self.transformer.adaLN_mods[-1].weight, 0)
        nn.init.constant_(self.transformer.adaLN_mods[-1].bias, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].bias, 0)
