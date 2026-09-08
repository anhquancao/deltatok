"""RAE decoder-noise finetune (third_party/RAE/src/stage1/rae.py:74-86).

The decoder learns to map a ball around each code back to x. DeltaTok decodes a DETACHED
z + sigma*eps as a separate loss, so the encoder's gradients match a no-noise run and only
the decoder pays. Both helpers are pure; the trainer owns the live tau.
"""
import torch


def ramp_tau(tau: float, warmup: int, it: int) -> float:
    """Linear warmup of tau over `warmup` iters (0 = constant). Full noise from step 0
    fights the code before it forms. `it` must be equal on every rank, else ranks
    disagree on whether the noised decode runs at all and the all-reduce hangs."""
    if tau <= 0 or warmup <= 0:
        return max(tau, 0.0)
    return tau * min(1.0, it / warmup)


def noise_z(z: torch.Tensor, tau: float) -> torch.Tensor:
    """z + sigma*eps, sigma ~ U(0, tau) per sample. The caller decides whether z carries
    grad: pass z.detach() to keep the encoder out of it."""
    sigma = tau * torch.rand(
        z.shape[0], 1, 1, 1, device=z.device, dtype=z.dtype)          # (M,1,1,1) per-sample sigma
    return z + sigma * torch.randn_like(z)                            # (M, N, K, Cz)
