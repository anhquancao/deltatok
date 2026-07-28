"""Eigenspectrum spread of the DeltaTok z bottleneck: how much of Cz the code uses.

Companion diagnostic to ``occrae/sigreg.py``. SIGReg is the pressure that spreads z
across its budget; this measures whether it worked, and runs with or without SIGReg,
which is what makes a sigreg arm and its plain twin comparable.

The reported quantities all come from the eigenvalues of the z covariance -- the
rotation-invariant view of the variance, where per-channel std is not (nothing aligns
the code with the canonical basis). Measured offline by ``compute_deltatok_z_spread.py``
and per eval loader by ``DeltaTokTrainer`` under ``Eval/<test>/Z*``.
"""
import torch
import torch.distributed as dist


class ZSpreadStats:
    """Rolling float64 moments of z rows -> eigenspectrum spread scalars.

    Moments accumulate over MANY batches (and all ranks) before the eigendecomposition:
    one micro-batch is S = M*N*K = 128..256 rows against Cz=768, so its covariance is
    rank-deficient by construction and every effective-rank number would read as a
    collapse that isn't there. Nothing here touches the autograd graph.

    float64 because cov subtracts mean^2 from E[z^2], which cancels catastrophically in
    fp32 once z drifts off unit scale (as it does under z_norm=false).
    """

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.n = 0            # rows banked since the last summary
        self.total = None     # (Cz,) sum of z rows
        self.outer = None     # (Cz, Cz) sum of z z^T
        self.row_ms_sum = 0.0  # sum over rows of mean(z^2) -- the scale companion

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """Bank one batch's z, shape (M, N, K, Cz)."""
        m = z.detach().reshape(-1, z.shape[-1]).double()      # (S, Cz) token rows
        if self.total is None:
            Cz = m.shape[1]
            self.total = torch.zeros(Cz, dtype=torch.float64, device=self.device)
            self.outer = torch.zeros(Cz, Cz, dtype=torch.float64, device=self.device)
        self.total += m.sum(0)                                # (Cz,) per-channel sum
        self.outer += m.T @ m                                 # (Cz, Cz) second moment
        self.row_ms_sum += float(m.square().mean(-1).sum())
        self.n += m.shape[0]

    @torch.no_grad()
    def summary(self, distributed: bool, full: bool = False):
        """Pool across ranks and reduce to scalars; None if NO rank banked anything.

        Collective: every rank must call this, and on the same step. A rank whose shard
        was empty still runs every collective below -- returning early on a local n == 0
        would leave the others blocked in all_reduce until the NCCL watchdog kills the job.

        Returns (first four scale-invariant, so comparable across runs; the rest
        describe where z sits, which z_norm=false leaves free):
          part_rank        participation ratio 1/sum(p^2) -- effective dims used, weighted
                           toward the strong axes. Cz for a flat spectrum, 1 for rank-1.
          ent_rank         exp(H(p)) -- same range, credits the weak tail more, so
                           part_rank moving alone means only the top of the spectrum flattened.
          n90              axes holding 90% of the variance; the most literal "dims used".
          top1_share       variance share of the strongest axis alone.
          total_var        trace(cov). Scale, not spread -- read against itself.
          row_mean_square  mean per-row mean(z^2); 1.0 == the unit-RMS rows of N(0,I).
          mean_abs_max     largest per-channel mean; 0 == centered.
          rows, cz         pooled row count and channel count -- the caller must reject
                           rows < cz, where the covariance is singular by construction and
                           every effective rank is a finite-sample artifact, not a code
                           property (it caps at rows-1 and reads as collapse).
        With ``full``, also ``evals`` (Cz,) descending and ``std`` (Cz,) per-channel.
        """
        # Shapes must be known before the buffer collectives, and a rank that banked
        # nothing has no buffers -- so agree on Cz first (MAX, not SUM).
        meta = torch.tensor([float(self.n), self.row_ms_sum],
                            dtype=torch.float64, device=self.device)
        cz_t = torch.tensor([0.0 if self.total is None else float(self.total.numel())],
                            dtype=torch.float64, device=self.device)
        if distributed:
            dist.all_reduce(meta)
            dist.all_reduce(cz_t, op=dist.ReduceOp.MAX)
        n, row_ms_sum, Cz = float(meta[0]), float(meta[1]), int(cz_t[0])
        if n == 0 or Cz == 0:
            return None

        # Reduce COPIES: all_reduce is in-place, and folding the pooled totals back into
        # the accumulator would double-count every rank on the next summary.
        zeros = self.total is None
        total = torch.zeros(Cz, dtype=torch.float64, device=self.device) if zeros else self.total.clone()
        outer = torch.zeros(Cz, Cz, dtype=torch.float64, device=self.device) if zeros else self.outer.clone()
        if distributed:
            dist.all_reduce(total)
            dist.all_reduce(outer)

        mean = total / n                                               # (Cz,) channel means
        cov = outer / n - torch.outer(mean, mean)                      # (Cz, Cz) covariance
        cov = 0.5 * (cov + cov.T)                                      # eigvalsh needs exact symmetry
        evals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)        # (Cz,) descending
        tot = float(evals.sum())                                       # = trace(cov), total variance
        # tot == 0 means z is a constant -- the collapse this diagnostic exists to catch.
        # Every share is then undefined, and the cumsum would never cross 0.90 and hand
        # back n90 = Cz+1, i.e. "maximally spread". Report the floor instead.
        if tot <= 0.0:
            part_rank = ent_rank = top1 = 0.0
            n90 = 0
            p = torch.zeros_like(evals)
        else:
            p = evals / tot                                            # (Cz,) per-axis variance share
            part_rank = tot ** 2 / max(float(evals.square().sum()), 1e-30)
            ent_rank = float(torch.exp(-(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum()))
            n90 = int((p.cumsum(0) < 0.90).sum()) + 1
            top1 = float(p[0])

        out = {
            "part_rank": part_rank,
            "ent_rank": ent_rank,
            "n90": n90,
            "top1_share": top1,
            "total_var": tot,
            "row_mean_square": row_ms_sum / n,
            "mean_abs_max": float(mean.abs().max()),
            "rows": int(n),
            "cz": Cz,
        }
        if full:
            out["evals"] = evals.cpu()                                 # (Cz,) descending eigenvalues
            out["std"] = cov.diagonal().clamp_min(0).sqrt().cpu()      # (Cz,) per-channel std
            out["shares"] = p.cpu()                                    # (Cz,) variance shares
        return out
