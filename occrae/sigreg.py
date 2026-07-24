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


class SIGReg(nn.Module):
    """Sketched Isometric Gaussian Regularization loss.

    ``forward(z)`` flattens all leading dims of ``z`` into samples, treats the last
    dim as features, projects onto ``num_slices`` random unit directions, and
    returns the mean Epps-Pulley discrepancy of each 1D marginal against N(0,1).
    The classical test-statistic ``* N`` (sample-count) scaling is dropped so the
    value is batch-size-independent and weights cleanly against the recon loss.

    In DDP each rank uses its own random directions on its own shard; parameter
    gradients are averaged across ranks in backward, so the effective regularizer
    integrates over ``world_size * num_slices`` directions. No cross-rank
    characteristic-function reduction (that would require synchronized directions).
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        C = z.shape[-1]                                                 # feature dim (Cz)
        s = z.reshape(-1, C).float()                                    # (S, C) samples x features
        A = torch.randn(C, self.num_slices, device=s.device, dtype=s.dtype)  # (C, K) random directions, K=num_slices
        A = A / A.norm(p=2, dim=0, keepdim=True)                        # unit-norm columns -> points on the sphere
        proj = s @ A                                                    # (S, K) 1D slices <z, a>
        x_t = proj.unsqueeze(-1) * self.t                              # (S, K, knots) t * <z, a>
        cos_mean = x_t.cos().mean(0)                                    # (K, knots) Re of empirical CF
        sin_mean = x_t.sin().mean(0)                                    # (K, knots) Im of empirical CF
        err = (cos_mean - self.phi).square() + sin_mean.square()        # (K, knots) |CF_emp - CF_N(0,1)|^2
        stat = err @ self.weights                                       # (K,) weighted CF-distance integral per slice
        return stat.mean()                                             # scalar: mean over slices
