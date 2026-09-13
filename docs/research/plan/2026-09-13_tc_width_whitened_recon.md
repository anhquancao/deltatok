# Plan: per-direction whitened recon loss (tc1536 bneck arm)

**Date:** 2026-09-13 · **Thread:** tc_width · **prior cycle:** `analysis/2026-09-13_tc_width_tc1536_rank.md`
· **Control:** `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm`

Loss becomes `log_cosh((x_hat − x) @ W)` with `W = U diag((λ+ε)^(−α/2)) Uᵀ`, `(λ, U)` the eigendecomposition of
the covariance of the recon targets `x`. Σ is measured in-trainer: bank `x` for the first
`recon_whiten_warmup` optim steps under the raw loss, pool across ranks, build W once, freeze it, save it in
the checkpoint. `recon_whiten_alpha: 0` reproduces the current loss bit-for-bit.

## 1. Config — `configs/deltatok/train_deltatok.yaml`

Insert after line 133 (`cov_weight`):

```yaml
  recon_whiten_alpha: 0.0  # recon on (x_hat-x) @ W, W = U diag((λ+eps)^(-α/2)) Uᵀ of cov(x); 0 = raw log-cosh
  recon_whiten_eps: 0.05  # eigenvalue floor
  recon_whiten_warmup: 500  # optim steps banking cov(x) under the raw loss, then W is frozen
```

## 2. Trainer — `occrae/deltatok_trainer.py`

**2a. Split `_log_cosh` (lines 474-496)** so the whitened path reuses the identity:

```python
def _log_cosh_diff(diff: torch.Tensor) -> torch.Tensor:
    """log(cosh(diff)) via |x| + softplus(-2|x|) - log 2; see _log_cosh."""
    diff = diff.abs()
    return diff + F.softplus(-2.0 * diff) - math.log(2.0)


def _log_cosh(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """<docstring unchanged>"""
    return _log_cosh_diff(pred - tgt)
```

**2b. State in `__init__`**, after line 619 (`cov_weight` print):

```python
        # Per-direction whitened recon: cov(x) banked for recon_whiten_warmup steps, then W frozen.
        self._recon_whiten_alpha = float(self.cfg.training.get("recon_whiten_alpha", 0.0))
        self._recon_whiten_eps = float(self.cfg.training.get("recon_whiten_eps", 0.05))
        self._recon_whiten_warmup = int(self.cfg.training.get("recon_whiten_warmup", 500))
        self._recon_W = None                                            # (C, C) once frozen
        self._recon_acc = ZSpreadStats(self.device) if self._recon_whiten_alpha > 0 else None
        self._recon_acc_steps = 0                                       # optim steps banked so far
        if self.is_master:
            print(f"recon_whiten_alpha={self._recon_whiten_alpha} eps={self._recon_whiten_eps} "
                  f"warmup={self._recon_whiten_warmup}")                  # 0.0 on a stale trainer = raw loss
```

**2c. Three methods**, next to `_sigreg_pooled` (line 686):

```python
    @torch.no_grad()
    def _recon_whiten_bank(self, x: torch.Tensor) -> None:
        """Bank recon targets (..., C) until W is frozen. No-op when off or frozen."""
        if self._recon_acc is not None:
            self._recon_acc.update(x.float())                           # (S, C) rows, fp64 inside

    @torch.no_grad()
    def _recon_whiten_freeze(self) -> None:
        """Collective: pool cov(x) over ranks, build W, drop the accumulator."""
        s = self._recon_acc.summary(distributed=self.distributed, full=True)
        lam, U = s["evals"].double(), s["evecs"].double()               # (C,), (C, C) descending
        scale = (lam.clamp_min(0) + self._recon_whiten_eps).pow(-self._recon_whiten_alpha / 2)  # (C,)
        W = ((U * scale) @ U.T).float().to(self.device)                 # (C, C) U diag(scale) Uᵀ
        if self.distributed:
            dist.broadcast(W, src=0)                                    # bit-identical W on every rank
        self._recon_W, self._recon_acc = W, None
        if self.is_master:
            print(f"[recon_whiten] froze W at iter {self.cfg.training.iter}: rows={s['rows']} "
                  f"part_rank={s['part_rank']:.1f} eig[0,100,500,1000]="
                  f"{[round(float(lam[k]), 4) for k in (0, 100, 500, 1000)]} "
                  f"W_cond={float(scale.max() / scale.min()):.1f}", flush=True)

    def _recon_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """log-cosh of the residual, through W once frozen. fp32 caller."""
        diff = pred - tgt                                               # (..., C)
        if self._recon_W is not None:
            diff = diff @ self._recon_W                                 # (..., C) per-direction scaled
        return _log_cosh_diff(diff).mean()
```

**2d. Train call sites** → `self._recon_loss(...)`, args unchanged:

| line | before | after |
|---|---|---|
| 992 | `_log_cosh(x_hat.float(), x.detach().float()).mean()` | `self._recon_loss(x_hat.float(), x.detach().float())` |
| 993 | `_log_cosh(x_hat_comp.float(), feats[:, 2].detach().float()).mean()` | `self._recon_loss(x_hat_comp.float(), feats[:, 2].detach().float())` |
| 1008 | `_log_cosh(x_hat_dn.float(), x.detach().float()).mean()` | `self._recon_loss(x_hat_dn.float(), x.detach().float())` |
| 1137 | `_log_cosh(x_hat.float(), x.detach().float()).mean()` | `self._recon_loss(x_hat.float(), x.detach().float())` |

Bank the targets once per micro-batch, only at the two primary sites (993 and 1008 reuse the same frames):

- after line 992: `self._recon_whiten_bank(x)  # (B*2, N, P, C) targets`
- after line 1137: `self._recon_whiten_bank(x)  # (M, N, P, C) targets`

Line 1050 (`_feature_loss`) and every eval log-cosh stay raw.

**2e. Freeze** inside `if update_grad:`, immediately before line 1245 (`self.cfg.training.iter += 1`):

```python
                if self._recon_acc is not None:
                    self._recon_acc_steps += 1
                    if self._recon_acc_steps >= self._recon_whiten_warmup:
                        self._recon_whiten_freeze()                     # collective, same step on every rank
```

**2f. Checkpoint.** `_save_checkpoint` state dict (line 853-858), add:

```python
            "recon_W": self._recon_W,                                   # (C, C) or None
```

`_load_checkpoint`, inside the `restore_train_state` branch after line 905:

```python
        if ckpt.get("recon_W") is not None and self._recon_whiten_alpha > 0:
            self._recon_W, self._recon_acc = ckpt["recon_W"].to(self.device), None  # frozen W resumes as-is
```

A resume before the freeze, or from a ckpt without `recon_W`, re-banks from zero for `recon_whiten_warmup` steps.

**2g. Eval**: after line 1405 (`batch_losses = {...}`):

```python
            if self._recon_W is not None:
                with torch.autocast(device_type="cuda", enabled=False):
                    batch_losses["LossRecon_W"] = self._recon_loss(x_hat.float(), x.float()).item()
```

## 3. Training slurm — `slurm/deltatok/train_deltatok_tc1536_bneck_whitenrecon_bsc.slurm` (new)

`cp` the control script, then:

- `--job-name=dt1536_whitenrecon_bsc`; `--output/--error` → `slurm/output/train_deltatok_tc1536_bneck_whitenrecon_bsc_%j.{out,err}`.
- Header: replace the 2 description lines with `# bneck tc1536 twin + per-direction whitened recon.` and
  `# Control: ..._bneck_nozn_tc1536_bsc.slurm`; keep the sbatch line with the new name. Drop the eval-ladder lines.
- After the `DECNOISE_WEIGHT` export:

  ```bash
  export RECON_WHITEN_ALPHA=${RECON_WHITEN_ALPHA:-0.5}    # 0 = raw loss (control)
  export RECON_WHITEN_EPS=${RECON_WHITEN_EPS:-0.05}       # eigenvalue floor
  export RECON_WHITEN_WARMUP=${RECON_WHITEN_WARMUP:-500}  # steps banking cov(x) before W freezes
  ```
- `RUN_NAME=${RUN_NAME:-deltatok_tc1536_bneck_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_whitenrecon_alpha${RECON_WHITEN_ALPHA}_eps${RECON_WHITEN_EPS}_sw${SIGREG_WARMUP}}`.
  Compose 1.0 is not in the name; changing it needs a new RUN_NAME.
- Recipe vs the control: `SIGREG_WEIGHT` 0.02 and `SIGREG_POOL_SAMPLES` 24576 (as `BSC:45727710`); the three
  `DECNOISE_*` exports and cfg lines replaced by `model.deltatok.decode_noise_tau=0.0`.
- `EXTRA_CFG_ARGS`: add

  ```bash
      training.recon_whiten_alpha=${RECON_WHITEN_ALPHA}
      training.recon_whiten_eps=${RECON_WHITEN_EPS}
      training.recon_whiten_warmup=${RECON_WHITEN_WARMUP}
  ```
- Everything else identical to the control.

## 4. Pre-flight

1. User syncs. md5 local vs BSC: `configs/deltatok/train_deltatok.yaml`, `occrae/deltatok_trainer.py`, the new slurm.
2. Smoke: `RUN_NAME=smoke_wrecon RECON_WHITEN_WARMUP=20 sbatch --qos=acc_debug --time=00:40:00 <new slurm>`.
   Pass when the log shows `recon_whiten_alpha=0.5 eps=0.05 warmup=20`, a first loss line, then
   `[recon_whiten] froze W at iter 19` with `part_rank ≈ 100` and `eig[1000] ≈ 0.05` (`f` row of `BSC:45825393`:
   ZPartRank 97.6, eig[1000] 0.048), and training continues with finite loss after it. `scancel`.
3. Null: same smoke with `RECON_WHITEN_ALPHA=0`. Log prints `recon_whiten_alpha=0.0`, no `froze W` line.
4. Resume: relaunch the step-2 smoke after one ckpt save. No second `froze W` line; loss continuous.
5. Prod: `sbatch <new slurm>` (acc_ehpc, 40 h, `exit_before_time_limit=true`), chain with `chain-slurm-jobs`.
   Watch until `RUNNING` + first loss line, then until the `froze W` line at iter 499.
6. `todos.json` status board: add the queued arm; rebuild the index.
