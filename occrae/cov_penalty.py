"""Direct second-moment penalty on the DeltaTok z bottleneck.

``||E[z z^T] - I||_F^2 / Cz`` on the same live+pooled rows SIGReg already sees. SIGReg's
sliced characteristic-function statistic reaches spectrum shape only second-hand: at Cz=512
its variance term carries 1/257 the weight of its scale term, so the usable rank saturates
near 100/512 across the whole sigreg_weight axis. This term is all shape. It rides on top of
SIGReg, never instead of it -- SIGReg keeps the higher-moment (tail) term this one lacks.
See docs/research/plan/2026-09-02_sigreg_cov_penalty.md.
"""
import torch
import torch.distributed as dist
from torch.distributed.nn import all_reduce as autograd_all_reduce  # differentiable collective


def cov_penalty(live: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """``||E[z z^T] - I||_F^2 / Cz`` over live + pooled rows; 0 iff the second moment is I.

    Same pooling and collective contract as ``SIGReg.forward``: only ``live`` carries grad,
    the Gram is a MEAN so live+pool is exact, and the caller's ``scale`` undoes the 1/pooled
    grad attenuation. Uncentered by design -- the off-center energy is ~0.01% of the trace
    on these arms, and the SIGReg target is N(0, I) anyway.
    """
    C = live.shape[-1]                                              # feature dim (Cz)
    s = live.reshape(-1, C).float()                                 # (L, C) the only rows with grad
    gram, n = s.T @ s, s.shape[0]                                   # (C, C) differentiable
    if pool is not None and pool.numel():
        p = pool.reshape(-1, C).float()                             # (P, C) detached FIFO rows
        gram = gram + p.T @ p                                       # (C, C) out-of-place: keeps the live graph
        n += p.shape[0]
    gram = gram / n                                                 # (C, C) second-moment estimate
    if dist.is_available() and dist.is_initialized():
        # Equal-weight rank average, differentiable; DDP's 1/W grad average keeps the scale (as SIGReg).
        gram = autograd_all_reduce(gram, op=dist.ReduceOp.SUM) / dist.get_world_size()   # (C, C)
    eye = torch.eye(C, device=gram.device, dtype=gram.dtype)        # (C, C)
    return (gram - eye).square().sum() / C                          # scalar, 0 iff E[z z^T] = I
