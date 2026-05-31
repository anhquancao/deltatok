import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class TimestepEmbedder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.SiLU(),
            nn.Linear(in_dim * 4, out_dim)
        )

    def forward(self, t):
        half = self.mlp[0].in_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        freqs = freqs[None, None, :]  # (1, 1, half)
        t_ = t[:, :, None]            # (B, T, 1)
        freqs = t_ * freqs            # (B, T, half)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return self.mlp(emb)
    

class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, multiple_of=256, bias=False, dropout=0.):
        super().__init__()
        self.dropout = dropout
        # swinGLU
        h_dim = int(2 * h_dim / 3)
        # make sure it is a power of 256
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, h_dim, bias=bias)

    def forward(self, x):
        # SwiGLU activation
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.dropout > 0. and self.training:
            x = F.dropout(x, self.dropout)

        return self.w2(x)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.flash = use_flash # use flash attention?
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, mask=None, return_attn=False):
        b, h_w, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # normalize queries and keys
        xq, xk = self.qk_norm(xq, xk, xv)
        xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
        xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
        xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)

        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention
        if self.flash:
            if mask is not None:
                mask = mask.view(b, 1, 1, h_w)
            output = F.scaled_dot_product_attention(xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(b, h_w, -1)
        # output projection
        proj = self.wo(output)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)

        if return_attn and not self.flash:
            return proj, scores

        return proj

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, use_flash=True):
        super().__init__()
        self.flash = use_flash
        self.n_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)  # used on text
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)  # used on text
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, q_x, kv_text, text_mask=None):
        """
        q_x: (B, Nq, D)  -- image tokens (queries)
        kv_text: (B, Nt, D) -- text embeddings (keys/values)
        text_mask: optional boolean mask (B, Nt) where True = keep, False = mask out
        """
        b, nq, _ = q_x.shape
        _, nt, _ = kv_text.shape

        q = self.wq(q_x).view(b, nq, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, Hd)
        k = self.wk(kv_text).view(b, nt, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nt, Hd)
        v = self.wv(kv_text).view(b, nt, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nt, Hd)

        # If flash scaled_dot_product_attention available, use it
        if self.flash:
            # flash expects (B, H, Lq, D) etc. but PyTorch's scaled_dot_product_attention uses (B, H, Lq, D)
            attn_mask = None
            if text_mask is not None:
                # convert (B, Nt) -> (B, 1, 1, Nt) with True=keep -> we need False where masked -> use boolean mask
                # scaled_dot_product_attention expects Bool mask with True indicating keep (or is it the opposite)? using this shape is safe for torch>=2.1
                attn_mask = text_mask.view(b, 1, 1, nt)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout if self.training else 0.)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Nq,Nt)
            if text_mask is not None:
                # text_mask: (B, Nt) -> expand to (B,1,1,Nt); masked positions set to -inf
                mask = (~text_mask).view(b, 1, 1, nt)  # True where masked
                scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores.float(), dim=-1).type_as(q)
            out = torch.matmul(attn, v)  # (B,H,Nq,Hd)
        out = out.transpose(1, 2).contiguous().view(b, nq, -1)  # (B, Nq, H*Hd)
        proj = self.wo(out)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj


class GroupFlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, group_size, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.group_size = group_size
        self.dropout = dropout
        self.use_flash = use_flash

        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

    def forward(self, x, mask=None, return_attn=False):
        b, n_tokens, _ = x.shape

        # Linear projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.qk_norm(xq, xk, xv)

        # Split heads
        xq = xq.view(b, n_tokens, self.n_local_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(b, n_tokens, self.n_local_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(b, n_tokens, self.n_local_heads, self.head_dim).transpose(1, 2)

        n_groups = n_tokens // self.group_size
        # print(x.size(), xq.size(), n_groups, self.group_size, n_tokens)
        # Reshape so groups become “batch elements”
        xq = xq.reshape(b, self.n_local_heads, n_groups, self.group_size, self.head_dim)
        xk = xk.reshape(b, self.n_local_heads, n_groups, self.group_size, self.head_dim)
        xv = xv.reshape(b, self.n_local_heads, n_groups, self.group_size, self.head_dim)

        # Merge batch and group dimension for flash attention
        xq = xq.permute(0, 2, 1, 3, 4).reshape(b * n_groups, self.n_local_heads, self.group_size, self.head_dim)
        xk = xk.permute(0, 2, 1, 3, 4).reshape(b * n_groups, self.n_local_heads, self.group_size, self.head_dim)
        xv = xv.permute(0, 2, 1, 3, 4).reshape(b * n_groups, self.n_local_heads, self.group_size, self.head_dim)

        if mask is not None:
            mask = mask.view(b, n_groups, 1, self.group_size)
            mask = mask.repeat(1, 1, self.n_local_heads, 1).reshape(b * n_groups, 1, 1, self.group_size)

        if self.use_flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.
            )
        else:
            scores = torch.matmul(xq, xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)

        # Reshape back to original
        output = output.view(b, n_groups, self.n_local_heads, self.group_size, self.head_dim)
        output = output.permute(0, 2, 1, 3, 4).reshape(b, self.n_local_heads, n_groups*self.group_size, self.head_dim)
        output = output[:, :, :n_tokens, :]  # remove padding

        proj = self.wo(output.transpose(1,2).contiguous().view(b, n_tokens, -1))
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)

        return proj


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class AdaNorm(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.norm_final = RMSNorm(x_dim, linear=True, bias=True, eps=1e-5)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, x_dim * 2))

    def forward(self, x, y):
        shift, scale = self.mlp(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0., use_cond=True):
        super().__init__()

        if use_cond:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

        self.attn = Attention(dim, heads, dropout=dropout) 

        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond=None, mask=None):
        if cond:
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
            x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), mask=mask)
            x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        else:
            x = x + self.attn(self.ln1(x), mask=mask)
            x = x + self.ff(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_cond=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.l_to_extract = 4 if depth <= 12 else 8
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout, use_cond=use_cond))

    def forward(self, x, cond=None, mask=None):
        hidden_feat = None
        for idx, block in enumerate(self.layers):
            x = block(x, cond, mask=mask)
            if idx == self.l_to_extract:
                hidden_feat = x
        return x, hidden_feat
    

def build_temporal_causal_mask(t, n_spatial, device):
    """
    t: number of frames
    n_spatial: tokens per frame (h*w)
    returns: (T*n_spatial, T*n_spatial) mask
             True = masked (not allowed)
    """
    T = t
    S = n_spatial
    L = T * S

    mask = torch.zeros(L, L, device=device, dtype=torch.bool)

    for ti in range(T):
        start_i = ti * S
        end_i = (ti + 1) * S

        # future frames
        future_start = (ti + 1) * S
        mask[start_i:end_i, future_start:] = True

    return mask