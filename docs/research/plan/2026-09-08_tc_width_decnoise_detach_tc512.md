# tc_width — decoder-noise with a detached extra decode, end-to-end tc512

Created 2026-09-08 · thread `tc_width` · prior cycle:
[`../plan/2026-09-07_tc_width_decnoise_e2e_tc512.md`](../plan/2026-09-07_tc_width_decnoise_e2e_tc512.md)
· arm: `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8_detach`
· control: `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0` (`BSC:45296347`)
· sibling: `..._compose1.0_decnoise0.8` (`BSC:45529827`, in-graph noise)

Same arm as the prior cycle, but the noised decode is a **second** decode of `z.detach() + σ·ε` with its own loss.
The clean decode, SIGReg, compose and the round-trip keep the prior-cycle path, so the encoder's gradients are
those of the twin. This **replaces** the in-graph noise: `decode_noise_tau` now always means the detached decode,
so the prior-cycle behaviour is gone and BSC:45529827 must not be resumed against this code. Three files edited, one new module,
one new script.

## 1 `configs/deltatok/train_deltatok.yaml`

Restate `decode_noise_tau`; add one key. No new mode flag — detached is the only mode.

```yaml
    decode_noise_tau: 0.0  # RAE noise finetune: a 2nd decode of z.detach()+σ·ε, σ ~ U(0, tau), trains the decoder alone. 0 = off
  ...
  decode_noise_warmup: 0  # linear ramp of decode_noise_tau over N iters (0 = constant)
  decode_noise_weight: 1.0  # loss weight of that detached noised decode
```

## 2 `occrae/decode_noise.py` (new)

Both pure, no trainer or module state: the trainer owns the live tau and the tokenizer call. Sits beside
`occrae/sigreg.py` and `occrae/cov_penalty.py` as a self-contained training-trick module.

```python
def ramp_tau(tau: float, warmup: int, it: int) -> float:
    if tau <= 0:
        return 0.0
    return tau * (1.0 if warmup <= 0 else min(1.0, it / max(1, warmup)))


def noise_z(z: torch.Tensor, tau: float) -> torch.Tensor:
    sigma = tau * torch.rand(
        z.shape[0], 1, 1, 1, device=z.device, dtype=z.dtype)          # (M,1,1,1) per-sample sigma
    return z + sigma * torch.randn_like(z)                            # (M, N, K, Cz)
```

## 3 `occrae/deltatok_shared.py`

Drop the `decode_noise_tau=` kwarg from `_make_deltatok_module` (`:505`); the module no longer takes it.

## 4 `occrae/deltatok_trainer.py`

Import beside the other regularisers (`:37`):

```python
from occrae.decode_noise import noise_z, ramp_tau  # RAE decoder-noise, applied off a detached z
```

### 4a Module — `__init__` and `forward` (`:79`, `:191`, `:483-489`)

Delete the `decode_noise_tau` ctor param, the `self._decode_noise_tau` attribute and its comment, and the
in-graph noise block. Nothing in the module reads tau any more; `forward` always decodes clean:

```python
        x_hat = self.decode(z, x_prev, rope_local, rope_global)
```

### 4b Trainer — helper, below `_compose_forward` (`:971`)

```python
    def _detached_noise_loss(self, x_prev, x, H, W, num_cameras, z):
        """decode(z.detach()+σ·ε) vs x: gradient reaches decoder_blocks + z_proj_up only. None when off."""
        if self._decode_noise_tau <= 0:
            return None
        assert z is not None, "decode_noise_tau > 0 needs the caller to pass return_z=True"
        with self.autocast:
            x_hat_dn = self.tokenizer(                                   # (M, N, P, C) same path as compose
                x_prev, None, H, W, num_cameras=num_cameras,
                z_input=noise_z(z.detach(), self._decode_noise_tau))
        with torch.autocast(device_type="cuda", enabled=False):
            return _log_cosh(x_hat_dn.float(), x.detach().float()).mean()
```

### 4c Trainer — `_compose_forward` (`:987-1005`)

After the hop encode and before `z = z.reshape(B, 2, ...)`:

```python
        loss_dn = self._detached_noise_loss(x_prev, x, H, W, num_cameras, z)   # (B*2 rows) or None
```

`forward` never noises (4a), so the composed decode needs the same explicit treatment as the
hops, weighted like the clean term it mirrors. Without it the arm would differ from the sibling in *which*
decodes see noise as well as in where the gradient stops.

```python
        loss_dn_comp = self._detached_noise_loss(feats[:, 0], feats[:, 2], H, W, num_cameras, z_comp)
        if loss_dn_comp is not None:
            loss_dn = loss_dn + self._compose_weight * loss_dn_comp
```

Return `loss_recon, loss_compose, loss_dn, z, step_t`. Cost is 2 encodes + 6 decodes a step, against the
sibling's 2 + 3: every decode now runs twice, clean and noised.

### 4d Trainer — `train_one_epoch`

Meters (`:1057-1058`): add `cum_dn = 0.` and `n_dn = 0`.

Compose branch (`:1103-1104`):

```python
                loss, loss_compose, loss_dn, z_compose, step_t = self._compose_forward(imgs, num_cameras)
                loss_total = loss + self._compose_weight * loss_compose
```

Pair branch: `z_bneck` is only bound on the `return_z=True` paths, so init it to `None` and request it
whenever tau is live, else the noise loss silently no-ops.

```python
                z_bneck = None                       # stays None on the no-sigreg, no-bneck path
                want_dn = self._decode_noise_tau > 0
                ...
                    elif self.sigreg is not None or want_dn:
                ...
                loss_dn = self._detached_noise_loss(x_prev, x, H, W, num_cameras, z_bneck)
```

One add-to-total for both branches, after the `if/else` and before the SIGReg block:

```python
            if loss_dn is not None:
                loss_total = loss_total + self._decode_noise_weight * loss_dn
```

Meter update beside `cum_compose` (`:1219`): `if loss_dn is not None: cum_dn += loss_dn.detach().item(); n_dn += 1`.
Scalar beside `Train/LossCompose` (`:1245`): `self.log_add_scalar('Train/LossDecNoise', loss_dn if loss_dn is not None else 0.0, ...)`.
Stats (`:1258`): `if n_dn: stats["decnoise"] = cum_dn / n_dn`, plus `("decnoise", "DecNoise")` in the epoch
summary tuple (`:1799`) or the stat never prints.

### 4e Trainer — `get_network` (`:817-823`)

The cfg target and the live value both live on the trainer now, so the ramp and the TB scalar drop their
`_unwrapped_tokenizer()` calls.

```python
        self._decode_noise_tau_cfg = float(self.cfg.model.deltatok.get("decode_noise_tau", 0.0))
        self._decode_noise_tau = 0.0
        self._decode_noise_weight = float(self.cfg.training.get("decode_noise_weight", 1.0))
        ...
            print(f"decode_noise_tau={self._decode_noise_tau_cfg} "
                  f"decode_noise_warmup={self._decode_noise_warmup} "
                  f"decode_noise_weight={self._decode_noise_weight}")
```

The ramp in `train_one_epoch` becomes one call, unconditional (`ramp_tau` returns 0 for tau 0):

```python
            self._decode_noise_tau = ramp_tau(
                self._decode_noise_tau_cfg, self._decode_noise_warmup, self.cfg.training.iter)
```

Eval is untouched: `_detached_noise_loss` is train-only and the ladder already uses the `z_input` path.

## 5 `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc512_bsc.slurm`

```bash
cp slurm/deltatok/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc.slurm \
   slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc512_bsc.slurm
```

Change only:

```
--job-name=deltatok_decnoise_dt_bsc
--output/--error = slurm/output/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc512_bsc_%j.{out,err}
header: detached-decode twin of BSC:45529827; hops and compose both get the 2nd decode, so the only
         difference is where the gradient stops + sbatch line with the new filename
training.bsize comment: 2 enc + 6 dec/step
export DECNOISE_WEIGHT=${DECNOISE_WEIGHT:-1.0}   # loss weight of the detached decode
RUN_NAME=...compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}_detach
+ training.decode_noise_weight=${DECNOISE_WEIGHT}
```

Everything else (tc512, sigreg 0.02, ns1024, pool 8192, compose 1.0, tau 0.8, warmup 2000, eval ladder
`[0.32,0.55,0.82]`, bsize 2 / effective 16, 40 h, `exit_before_time_limit`) is the sibling's.

## 6 Pre-flight on BSC (user syncs), then submit

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && ls occrae/decode_noise.py && grep -n \"noise_z\|_detached_noise_loss\|decode_noise_weight\" occrae/deltatok_trainer.py && grep -E \"RUN_NAME=|decode_noise|--time|--account\" slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc512_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc512_bsc.slurm'"
```

Watch until the first `[KEpoch` line. The `.out` must show `decode_noise_weight=1.0`,
`encoder_blocks … trainable= 340.465M`, `Train/LossDecNoise` > 0 once `DecodeNoiseTau` > 0, and a step time
above the sibling's (2 enc + 3 dec → 2 enc + 5 dec). Chain at 40 h with `chain-slurm-jobs`, as the sibling.
