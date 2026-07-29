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

# Log10 histogram of per-row mean(z^2), for the row-scale quantiles. Fixed grid so every
# rank owns the same buffer without agreeing on a range first (one all_reduce, no gather).
# [-8, 8] decades at 1/64 per bin: covers any z a live run can produce, ~2% quantile error.
_ROW_MS_LO, _ROW_MS_HI, _ROW_MS_BINS = -8.0, 8.0, 1024


def _row_ms_quantile(hist: torch.Tensor, n: float, q: float) -> float:
    """Inverse-CDF of the log10 row-mean-square histogram, interpolated within the bin."""
    cum = hist.cumsum(0)                                      # (_ROW_MS_BINS,) running count
    # searchsorted, not argmax: cum is sorted, and this pins the FIRST bin crossing q*n.
    b = int(torch.searchsorted(cum, torch.tensor(q * n, dtype=torch.float64, device=hist.device)))
    b = min(b, _ROW_MS_BINS - 1)
    before = float(cum[b - 1]) if b > 0 else 0.0              # count strictly below bin b
    inbin = float(hist[b])
    frac = (q * n - before) / inbin if inbin > 0 else 0.0     # position within bin b
    step = (_ROW_MS_HI - _ROW_MS_LO) / _ROW_MS_BINS           # decades per bin
    return float(10.0 ** (_ROW_MS_LO + (b + frac) * step))


def _row_ms_top_share(hist: torch.Tensor, n: float, frac: float) -> float:
    """Share of total row mean-square held by the largest ``frac`` of rows.

    This is the one that decides whether an MSE over these rows is tail-carried:
    frac == 0.01 returning 0.5 means the top 1% of tokens own half the objective.
    Bin centres approximate each row's value -- 1/64 decade wide, so <2% error.
    """
    step = (_ROW_MS_HI - _ROW_MS_LO) / _ROW_MS_BINS
    idx = torch.arange(_ROW_MS_BINS, dtype=torch.float64, device=hist.device)   # (_ROW_MS_BINS,)
    centre = 10.0 ** (_ROW_MS_LO + (idx + 0.5) * step)                          # (_ROW_MS_BINS,) bin value
    mass = hist * centre                                                        # (_ROW_MS_BINS,) mean-square per bin
    total = float(mass.sum())
    if total <= 0.0:
        return 0.0
    # Walk down from the top bin until `frac` of the ROWS are covered, taking a
    # PROPORTIONAL slice of the bin the boundary lands in -- whole bins only would
    # report ~0 whenever the rows are tight enough to share one bin.
    cnt_from_top = hist.flip(0).cumsum(0).flip(0)                               # (_ROW_MS_BINS,) rows at or above bin
    above = torch.cat([cnt_from_top[1:],                                        # (_ROW_MS_BINS,) rows STRICTLY above bin
                       torch.zeros(1, dtype=hist.dtype, device=hist.device)])
    # clamp_min on the divisor: an empty bin has mass 0, so its term drops out anyway.
    take = ((frac * n - above) / hist.clamp_min(1e-30)).clamp(0.0, 1.0)         # (_ROW_MS_BINS,) fraction of bin in the tail
    return float((mass * take).sum()) / total


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
        # Row-scale DISTRIBUTION, not just its mean: z_norm=false leaves the row scale free,
        # so the mean alone cannot tell a tight code from one with a heavy tail.
        self.row_ms_hist = torch.zeros(_ROW_MS_BINS, dtype=torch.float64, device=self.device)
        self.row_ms_max = 0.0

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
        row_ms = m.square().mean(-1)                          # (S,) per-row mean(z^2), >= 0
        self.row_ms_sum += float(row_ms.sum())
        self.row_ms_max = max(self.row_ms_max, float(row_ms.max()))
        # clamp_min before log10: an all-zero row is legitimate (collapse) and would
        # otherwise send -inf into the bin index and NaN the whole histogram.
        lg = row_ms.clamp_min(1e-300).log10()                 # (S,) decades
        idx = ((lg - _ROW_MS_LO) / (_ROW_MS_HI - _ROW_MS_LO) * _ROW_MS_BINS).floor()  # (S,) bin
        idx = idx.clamp(0, _ROW_MS_BINS - 1).long()           # (S,) out-of-range folds into the end bins
        self.row_ms_hist += torch.bincount(idx, minlength=_ROW_MS_BINS).double()  # (_ROW_MS_BINS,)
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
          row_ms_top1pct   share of the summed row mean-square held by the top 1% of rows.
                           1% under z_norm=true; the decisive number for whether an MSE
                           over these rows is carried by a few tokens.
          row_ms_p50/p90/p99/max, row_ms_tail
                           quantiles of that same per-row mean(z^2), and p99/p50. Under
                           z_norm=true every row is unit-RMS, so tail == 1 by construction;
                           _nozn leaves the scale free and a large tail means an MSE
                           objective over these rows is carried by a few outlier tokens.
                           Squared units: the row-NORM ratio is sqrt(row_ms_tail).
                           Quantiles floor at 1e-8 (the histogram's low edge) -- a fully
                           collapsed z reads as that, not 0; total_var == 0 is the tell.
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
        # MAX-reduced pair: Cz (a rank that banked nothing contributes 0) and the row-scale
        # max, which is a max not a sum -- riding along here costs no extra collective.
        cz_t = torch.tensor([0.0 if self.total is None else float(self.total.numel()),
                             self.row_ms_max], dtype=torch.float64, device=self.device)
        # Reduce a COPY: all_reduce is in-place and this buffer keeps accumulating.
        row_hist = self.row_ms_hist.clone()                            # (_ROW_MS_BINS,)
        if distributed:
            dist.all_reduce(meta)
            dist.all_reduce(cz_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(row_hist)
        n, row_ms_sum, Cz = float(meta[0]), float(meta[1]), int(cz_t[0])
        row_ms_max = float(cz_t[1])
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

        p50, p90, p99 = (_row_ms_quantile(row_hist, n, q) for q in (0.50, 0.90, 0.99))

        out = {
            "part_rank": part_rank,
            "ent_rank": ent_rank,
            "n90": n90,
            "top1_share": top1,
            "total_var": tot,
            "row_mean_square": row_ms_sum / n,
            "mean_abs_max": float(mean.abs().max()),
            "row_ms_p50": p50,
            "row_ms_p90": p90,
            "row_ms_p99": p99,
            "row_ms_max": row_ms_max,
            "row_ms_tail": p99 / max(p50, 1e-300),
            "row_ms_top1pct": _row_ms_top_share(row_hist, n, 0.01),
            "rows": int(n),
            "cz": Cz,
        }
        if full:
            out["evals"] = evals.cpu()                                 # (Cz,) descending eigenvalues
            out["std"] = cov.diagonal().clamp_min(0).sqrt().cpu()      # (Cz,) per-channel std
            out["shares"] = p.cpu()                                    # (Cz,) variance shares
        return out
