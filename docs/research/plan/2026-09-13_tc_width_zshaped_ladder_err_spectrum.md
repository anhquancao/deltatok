# tc_width — Σ_z-shaped noise ladder at matched MSEToken, and the flow error's spectrum vs N

Created 2026-09-13 · thread `tc_width` · prior cycle:
[`../results/2026-09-13_tc_width_decnoise_detach_tc512_flow_read_slides.html`](../results/2026-09-13_tc_width_decnoise_detach_tc512_flow_read_slides.html)
· tokenizers: tc512 detached `epoch_100.pth` (`BSC:45665286`), tc128 source `epoch_100.pth` (`BSC:44846726`), tc128 ft10 `epoch_10.pth` (`BSC:45421190`)
· flows: tc512 `iter_100000.pth` (`BSC:45736861`), tc128 `current.pth` (`BSC:45111164`)

Eval-only. One pre-pass over the eval loader banks the Waymo GT-z covariance `Σ_z` (forecast slots), giving the
eigenbasis `U` and eigenvalues `λ`. Two uses: (a) the noise probe draws `ε ~ N(0, Σ_z·C/tr Σ_z)` instead of
`N(0, I)`, so `E[MSEToken] = σ²` exactly as on the isotropic ladder; (b) every pass projects `z_hat − z` on `U` and
reports the error's share in the top-k directions next to the signal's share.

## 1 `occrae/z_spread.py`

**1a line 87** `evals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)` →

```python
        if full:
            evals, evecs = torch.linalg.eigh(cov)                          # ascending; evecs columns
            evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)       # (Cz,), (Cz, Cz) descending
        else:
            evals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)        # (Cz,) descending
```

**1b after line 104** `out["std"] = ...`:

```python
            out["evecs"] = evecs.cpu()                                 # (Cz, Cz) eigenvectors, column i <-> evals[i]
```

`compute_deltatok_z_spread.py:369` is the only `full=True` caller; an extra key is inert there.

## 2 `configs/deltatok_flow/train_deltatok_flow.yaml`

**2a after line 133** `eval_noise_probe_sigma: null`:

```yaml
  # true: the probe's noise is N(0, Σ_z) rescaled to per-element variance 1 (same MSEToken as isotropic).
  # Needs the Σ_z basis the sampler script banks (eval_deltatok_flow_sampler.py --z_basis).
  eval_noise_probe_shaped: false
```

## 3 `occrae/deltatok_flow_trainer.py`

**3a after line 74** `self._eval_t_gen = None`:

```python
        self._z_basis = {}            # {test_name: (U (C,C) evecs desc, lam (C,))} of GT-z cov; set by the sampler script
        self._err_spectrum = {}       # {test_name: (err_dir (C,) mean sq error per eigen-dir, lam)} written by eval_one_epoch
```

**3b after line 732** `noise_probe_sigma = None if ...`:

```python
        noise_probe_shaped = bool(self.cfg.training.get("eval_noise_probe_shaped", False))
        assert not noise_probe_shaped or self._z_basis, "eval_noise_probe_shaped needs --z_basis (sampler script)"
```

**3c after line 755** `num_vis = 0`:

```python
                basis = self._z_basis.get(test_name)                        # (U, lam) or None
                err_dir = None                                              # (C,) sum of squared error per eigen-dir
                err_rows = 0
```

**3d lines 830–833** the `fc += noise_probe_sigma * torch.randn(...)` call →

```python
                        eps = torch.randn(fc.shape, generator=self._eval_noise_gen,
                                          device=z.device, dtype=torch.float32)          # (B, F, N, K, C) isotropic
                        if noise_probe_shaped:
                            U, lam = basis
                            # scale eigen-coords so per-element variance stays 1, rotate back: cov = Σ_z·C/tr Σ_z
                            eps = (eps * (lam * lam.numel() / lam.sum()).sqrt()) @ U.T   # (B, F, N, K, C)
                        fc += noise_probe_sigma * eps.to(fc.dtype)
```

**3e after line 900** (the `batch_losses["MSEToken"] = ...` statement):

```python
                        if basis is not None:
                            U, _ = basis
                            e = (z_hat[:, self.n_ctx:] - z[:, self.n_ctx:]).float()                  # (B, F, N, K, C)
                            e = e.reshape(-1, e.shape[-1]) @ U                                      # (rows, C) eigen-coords
                            sq = e.square().sum(0).double()                                         # (C,)
                            err_dir = sq if err_dir is None else err_dir + sq
                            err_rows += e.shape[0]
```

**3f after line 1055** (inside `if self.is_master:` … after the `log_add_scalar` loop, still in the loader loop, at the
`if self.is_master:` indent):

```python
                if basis is not None and err_dir is not None:
                    if self.distributed:
                        dist.all_reduce(err_dir)
                    U, lam = basis
                    err_dir = err_dir / max(err_rows * self.world_size, 1)                          # (C,) mean sq error per dir
                    self._err_spectrum[test_name] = (err_dir.float().cpu(), lam.float().cpu())
                    if self.is_master:
                        tot_e, tot_s = float(err_dir.sum()), float(lam.sum())
                        shares = "  ".join(
                            f"ErrShareTop{k}: {float(err_dir[:k].sum()) / tot_e:.4f} "
                            f"SigShareTop{k}: {float(lam[:k].sum()) / tot_s:.4f}"
                            for k in (16, 32, 64, 96, 128) if k < lam.numel())
                        print(f"[Eval/{test_name}] ErrSpectrum  {shares}", flush=True)
```

`err_rows * world_size` is exact only when every rank sees the same row count; the sampler script runs single-GPU, and
in DDP training `_z_basis` is empty so this block never runs.

## 4 `eval_deltatok_flow_sampler.py`

**4a after line 109** (the `--noise_sigmas` argument):

```python
    parser.add_argument(
        "--z_basis", action="store_true",
        help="Pre-pass: bank the GT-z covariance of each eval loader (forecast slots) and hand its "
             "eigenbasis to the trainer. Enables training.eval_noise_probe_shaped and the "
             "ErrSpectrum line (error share per eigen-direction) on every pass.",
    )
```

**4b after line 209** `trainer.test_loaders = _build_test_loaders(cfg)`:

```python
    if args.z_basis:
        _bank_z_basis(trainer, cfg)
```

**4c before `def main()`** (after `_build_test_loaders`, line 146):

```python
@torch.no_grad()
def _bank_z_basis(trainer, cfg):
    """Σ_z of the GT deltas on each eval loader's forecast slots -> trainer._z_basis[test] = (U, lam)."""
    from occrae.z_spread import ZSpreadStats
    n_items = int(cfg.training.get("eval_num_items", 256))
    for test_name, loader in trainer.test_loaders.items():
        # same pinning as eval_one_epoch so the basis comes from the items the passes score
        for obj in (getattr(loader, "sampler", None), getattr(loader, "dataset", None)):
            if obj is not None and hasattr(obj, "set_epoch"):
                obj.set_epoch(0)
        acc = ZSpreadStats(trainer.device)
        seen = 0
        for batch in loader:
            if seen >= n_items:
                break
            batch = trainer._normalize_batch(batch)
            imgs = batch["imgs"].to(trainer.device, non_blocking=True)            # (B, V, 3, H, W)
            _, _, z, _, _ = trainer._encode_inputs(batch, imgs, int(batch.get("num_cameras", 1)), want_tokens=False)
            acc.update(z[:, trainer.n_ctx:])                                      # (B, F, N, K, C) forecast slots only
            seen += imgs.shape[0]
        s = acc.summary(distributed=False, full=True)
        U, lam = s["evecs"].float().to(trainer.device), s["evals"].float().to(trainer.device)
        trainer._z_basis[test_name] = (U, lam)                                    # (C, C), (C,)
        print(f"[INFO] z_basis {test_name}: rows={s['rows']} Cz={s['cz']} ZPartRank={s['part_rank']:.1f} "
              f"ZTotalVar={s['total_var']:.2f}", flush=True)
```

**4d after line 234** (the `print(f"[INFO] steps=...` inside the loops):

```python
                if trainer._err_spectrum:
                    out = os.path.join(output_dir, f"err_spectrum_{mode}_steps{n_steps}_sigma{sigma}.pt")
                    torch.save({k: {"err_dir": e, "lam": l} for k, (e, l) in trainer._err_spectrum.items()}, out)
```

## 5 `slurm/eval_deltatok_flow_zshaped_bsc.slurm`

`cp slurm/eval_deltatok_flow_ladder_tc512detach_bsc.slurm slurm/eval_deltatok_flow_zshaped_bsc.slurm`, then:

- `--job-name=deltatok_flow_zshaped`; `--output` / `--error` → `eval_deltatok_flow_zshaped_bsc_%j`.
- Header (3 lines, replaces the current 3):
  ```
  # Σ_z-shaped noise ladder (matched MSEToken) + flow error spectrum per eigen-direction. --z_basis
  # banks the Waymo GT-z covariance first. TC/CKPT/DELTATOK_CKPT select the arm; SHAPED=false = plain pass.
  #   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/eval_deltatok_flow_zshaped_bsc.slurm'"
  ```
- `: "${OUTPUT_DIR:=results/deltatok_flow_zshaped/tc512detach_ep100}"`
- `: "${NOISE_SIGMAS:=0.32,0.55,0.82}"` (σ = 0 is the plain pass; covered by the N-step jobs)
- new lines after `NOISE_SIGMAS`: `: "${SHAPED:=true}"` and `: "${TC:=512}"`
- `model.deltatok.target_channels=512` → `model.deltatok.target_channels=${TC}`
- new CFG_ARGS entry after `training.val_bsize=4`: `training.eval_noise_probe_shaped=${SHAPED}   # Σ_z-shaped probe noise`
- `srun` line: add `--z_basis \` before `--noise_sigmas`.

## 6 Jobs (1 GPU each, `acc_ehpc`, 1 h; ~1 min pre-pass + ~5 min per pass)

| # | arm | env | passes |
|---|---|---|---|
| J1 | tc512 detached ep100 | defaults | shaped ladder σ 0.32/0.55/0.82 |
| J2 | tc512 detached ep100 | `SHAPED=false NOISE_SIGMAS=" " NUM_STEPS=1,2,20 OUTPUT_DIR=.../tc512detach_ep100_steps` | flow spectrum at N = 1, 2, 20 |
| J3 | tc128 source | `TC=128 CKPT=<tc128 flow>/ckpts/current.pth DELTATOK_CKPT=<tc128 compose sigreg0.005>/ckpts/epoch_100.pth OUTPUT_DIR=.../tc128src` | shaped ladder |
| J4 | tc128 source | J3 env + `SHAPED=false NOISE_SIGMAS=" " NUM_STEPS=1,2,20 OUTPUT_DIR=.../tc128src_steps` | flow spectrum |
| J5 | tc128 ft10 | J3 env with `DELTATOK_CKPT=<..._decnoise0.8_ft10>/ckpts/epoch_10.pth OUTPUT_DIR=.../tc128ft10` | shaped ladder |

Paths: tc128 flow `$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit`;
tokenizers `$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0[_decnoise0.8_ft10]`.
J4's spectrum is the same flow as J3's, so ft10 needs no N-step job.

## 7 Pre-flight

1. Cluster copy: `md5sum` of the four edited files and the new slurm script on BSC must match local before any `sbatch`.
2. Built-in checks in J1's log: the isotropic ladder passes (BSC:45818711) print no `ErrSpectrum`; a shaped pass must
   show `MSEToken` within a few % of σ² (0.1024 / 0.3025 / 0.6725) and `ErrShareTop{k} ≈ SigShareTop{k}`.
   In J2, a plain pass is the real test: `ErrShareTopk` vs `SigShareTopk` vs the flat `k/C`.
3. `[INFO] z_basis ...` prints rows = 128 × 2 × 64 = 16384 and Cz = TC.
4. Watch J1 until its first `[Eval/128 @ WaymoSeqMultiView]` line, then hand back the IDs.
