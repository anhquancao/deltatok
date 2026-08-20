"""SIGReg: Sketched Isometric Gaussian Regularization (LeJEPA, Balestriero & LeCun 2025).

Random 1D slices + the Epps-Pulley characteristic-function statistic push an embedding
toward isotropic N(0, sigma^2 I). Cramer-Wold: every 1D marginal being N(0, sigma^2)
<=> the code is isotropic Gaussian. Used as an anti-collapse regularizer on the DeltaTok
z bottleneck, which under-uses its budget (tc768 -> participation rank ~290), and to hand
downstream flow matching an easier target. Self-contained reimpl of the paper's
MINIMAL.md snippet; no dependency on the lejepa package.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.nn import all_reduce as autograd_all_reduce  # differentiable collective


class SIGReg(nn.Module):
    """Mean over random slices of the Epps-Pulley discrepancy against N(0, sigma^2).

    ``forward(live, pool, seed, sigma)`` flattens leading dims into samples, treats the last
    dim as features. Drops the classical ``* N * world_size`` test-statistic scaling so the
    value is batch-size-independent -- our ``sigreg_weight`` is not on the paper's ``lamb``
    scale. ``sigma`` is the target marginal std, per call so it can follow a schedule; it
    enters only as a projection rescale.

    ``pool`` is separate from ``live`` because only ``live`` carries grad: the CF is a MEAN,
    so ``(sum_pool + sum_live) / (n_pool + n_live)`` is exact, and running the pool under
    no_grad keeps the retained ``(S, K, knots)`` down to the live rows.

    The statistic needs S >> C: at S=128, C=768 the finite-sample floor is ~0.0085 while a
    rank-290 collapse adds only ~0.0002. ``seed`` (the rank-synced train iter) makes every
    rank draw the SAME directions, so the all-reduced CF is estimated over world_size * S
    samples. Shared directions forfeit the free per-rank world_size multiplier, so
    ``num_slices`` should be >= 2*C (the reference uses 1000-4096 at C=512).

    ``seed`` MUST advance every call and be checkpointed. Frozen directions let the model
    satisfy the measured slices while drifting non-Gaussian in the rest: the loss falls,
    the code stays collapsed.
    """

    def __init__(self, num_slices: int = 256, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_slices = int(num_slices)
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)      # (knots,) freqs, positive half-line
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)   # trapezoid, doubled for [-t,t] symmetry
        weights[[0, -1]] = dt                                           # half-weight endpoints
        window = torch.exp(-t.square() / 2.0)                           # exp(-t^2/2): N(0,1) CF and gaussian window
        self.register_buffer("t", t)                                    # (knots,)
        self.register_buffer("phi", window)                             # (knots,) target real CF (imag is 0)
        self.register_buffer("weights", weights * window)               # (knots,) quadrature weight * window

    def _directions(self, C: int, device, dtype, seed: int) -> torch.Tensor:
        """(C, K) unit-norm slice directions, identical on every rank for this call."""
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))                                        # caller's rank-synced counter
        A = torch.randn(C, self.num_slices, device=device, dtype=dtype, generator=g)  # (C, K) K=num_slices
        return A / A.norm(p=2, dim=0, keepdim=True)                     # unit-norm columns -> sphere

    def forward(self, live: torch.Tensor, pool: Optional[torch.Tensor], seed: int,
                sigma: float = 1.0) -> torch.Tensor:
        assert sigma > 0, f"sigma must be positive, got {sigma}"        # target std N(0, sigma^2 I)
        C = live.shape[-1]                                              # feature dim (Cz)
        s = live.reshape(-1, C).float()                                 # (L, C) L=live rows, the only ones with grad
        A = self._directions(C, s.device, s.dtype, seed)                # (C, K) shared across ranks
        x_t = ((s @ A) / sigma).unsqueeze(-1) * self.t                  # (L, K, Q) Q=knots; t * <z,a>/sigma
        cos_sum, sin_sum = x_t.cos().sum(0), x_t.sin().sum(0)           # (K, Q) each, differentiable
        n = s.shape[0]                                                  # local rows behind the CF
        if pool is not None and pool.numel():
            p = pool.reshape(-1, C).float()                             # (P, C) P=detached queue rows
            with torch.no_grad():
                proj = (p @ A) / sigma                                  # (P, K) reused across knots
                p_cos, p_sin = torch.zeros_like(cos_sum), torch.zeros_like(sin_sum)
                for q in range(self.t.numel()):                         # per knot: (P, K, Q) never lands
                    x = proj * self.t[q]                                # (P, K)
                    p_cos[:, q], p_sin[:, q] = x.cos().sum(0), x.sin().sum(0)
            cos_sum, sin_sum = cos_sum + p_cos, sin_sum + p_sin         # out-of-place: keeps the live graph
            n += p.shape[0]
        cos_mean, sin_mean = cos_sum / n, sin_sum / n                    # (K, Q) Re/Im of empirical CF
        if dist.is_available() and dist.is_initialized():
            # Same directions on every rank, so the CF means pool: S -> world_size * S.
            # Differentiable, so each rank still backprops into its own rows and DDP's 1/W
            # grad average keeps the usual scale. This is an equal-weight rank AVERAGE, not
            # the true pooled mean when S differs across ranks: unbiased, <=~11% effective-
            # sample loss, same weighting DDP already applies to the recon loss.
            # no_sync() TRAP: this 1/W cancels DDP's own. Micro-batches accumulated under
            # no_sync() lose that cancellation and their sigreg grad is W x too big.
            world = dist.get_world_size()
            cos_mean = autograd_all_reduce(cos_mean, op=dist.ReduceOp.SUM) / world  # (K, knots)
            sin_mean = autograd_all_reduce(sin_mean, op=dist.ReduceOp.SUM) / world  # (K, knots)
        err = (cos_mean - self.phi).square() + sin_mean.square()        # (K, knots) |CF_emp - CF_N(0,1)|^2
        stat = err @ self.weights                                       # (K,) weighted CF distance per slice
        return stat.mean()                                             # scalar: mean over slices
