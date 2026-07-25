"""SIGReg: Sketched Isometric Gaussian Regularization (LeJEPA, Balestriero & LeCun 2025).

Pushes an embedding distribution toward isotropic N(0, I) via random 1D slicing +
the Epps-Pulley characteristic-function statistic per slice. By Cramer-Wold, every
1D marginal being standard normal <=> the multivariate code is isotropic Gaussian.

Used here as an anti-collapse regularizer on the DeltaTok z bottleneck: the tc768
channel squeeze under-uses its 768-d budget (participation rank ~290 << 768), and
SIGReg is the pressure that spreads the code across all dims and Gaussianizes it
(also an easier target for downstream flow matching). Self-contained reimpl of the
paper's minimal snippet (LeJEPA MINIMAL.md); no dependency on the lejepa package.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.nn import all_reduce as autograd_all_reduce  # differentiable collective


class SIGReg(nn.Module):
    """Sketched Isometric Gaussian Regularization loss.

    ``forward(z, seed)`` flattens all leading dims of ``z`` into samples, treats the
    last dim as features, projects onto ``num_slices`` random unit directions, and
    returns the mean Epps-Pulley discrepancy of each 1D marginal against N(0,1).
    The classical test-statistic ``* N`` (sample-count) scaling is dropped so the
    value is batch-size-independent and weights cleanly against the recon loss.

    The statistic needs many samples relative to the feature dim: at S=128, C=768
    its finite-sample floor is ~0.0085 while a rank-290 collapse only adds ~0.0002,
    i.e. below the run-to-run noise, so it cannot see the collapse it exists to fix.
    The reference (lejepa/univariate/epps_pulley.py) reaches S/C ~ 8 partly by
    averaging the empirical CF across ranks, which is what ``forward`` does here:
    ``seed`` (the train iter -- equal on every rank by construction) drives the
    direction draw so every rank projects onto the SAME directions, then cos/sin
    means are all-reduced through a differentiable collective -> the statistic is
    estimated over world_size * S samples. Because directions are now shared,
    ``num_slices`` should be >= 2*C (the reference uses 1000-4096 for C=512); it no
    longer gets a free world_size multiplier from per-rank draws.

    ``seed`` MUST advance every call: directions frozen across many optimizer steps
    let the model satisfy those fixed slices while drifting non-Gaussian in the
    unmeasured ones -- a failure that LOOKS like success (the loss falls, the code
    stays collapsed). It must also be checkpointed, or a resume replays seed 0.
    """

    def __init__(self, num_slices: int = 256, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_slices = int(num_slices)
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)      # (knots,) positive half-line of freqs t
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)   # trapezoid rule, doubled for [-t,t] symmetry
        weights[[0, -1]] = dt                                           # half-weight at the two endpoints
        window = torch.exp(-t.square() / 2.0)                           # phi(t)=exp(-t^2/2): std-normal CF AND gaussian window
        self.register_buffer("t", t)                                    # (knots,)
        self.register_buffer("phi", window)                             # (knots,) target real CF (imag part is 0)
        self.register_buffer("weights", weights * window)               # (knots,) quadrature weight * window

    def _directions(self, C: int, device, dtype, seed: int) -> torch.Tensor:
        """(C, K) unit-norm slice directions, identical on every rank for this call."""
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))                                        # caller's rank-synced counter
        A = torch.randn(C, self.num_slices, device=device, dtype=dtype, generator=g)  # (C, K) K=num_slices
        return A / A.norm(p=2, dim=0, keepdim=True)                     # unit-norm columns -> points on the sphere

    def forward(self, z: torch.Tensor, seed: int) -> torch.Tensor:
        C = z.shape[-1]                                                 # feature dim (Cz)
        s = z.reshape(-1, C).float()                                    # (S, C) samples x features
        A = self._directions(C, s.device, s.dtype, seed)                # (C, K) shared across ranks; randn builds no graph
        proj = s @ A                                                    # (S, K) 1D slices <z, a>
        x_t = proj.unsqueeze(-1) * self.t                              # (S, K, knots) t * <z, a>
        cos_mean = x_t.cos().mean(0)                                    # (K, knots) Re of empirical CF
        sin_mean = x_t.sin().mean(0)                                    # (K, knots) Im of empirical CF
        if dist.is_available() and dist.is_initialized():
            # Pool the CF over all ranks (same directions, so the means are comparable):
            # S -> world_size * S effective samples. Differentiable, so each rank still
            # backprops into its own samples; DDP's grad averaging keeps the usual 1/W scale.
            # NOTE this is an equal-weight rank AVERAGE, not the true pooled mean when S
            # differs across ranks (N=1 vs 2 cameras): unbiased, <=~11% effective-sample
            # loss, and the same weighting DDP already applies to the recon loss.
            # no_sync() TRAP: the 1/W here cancels exactly against DDP's own 1/W grad
            # averaging. If micro-batches are ever accumulated under no_sync(), the
            # non-synced ones lose that cancellation and their sigreg grad is W x too big.
            world = dist.get_world_size()
            cos_mean = autograd_all_reduce(cos_mean, op=dist.ReduceOp.SUM) / world  # (K, knots)
            sin_mean = autograd_all_reduce(sin_mean, op=dist.ReduceOp.SUM) / world  # (K, knots)
        err = (cos_mean - self.phi).square() + sin_mean.square()        # (K, knots) |CF_emp - CF_N(0,1)|^2
        stat = err @ self.weights                                       # (K,) weighted CF-distance integral per slice
        return stat.mean()                                             # scalar: mean over slices
