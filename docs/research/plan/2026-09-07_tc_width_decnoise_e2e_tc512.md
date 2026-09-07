# tc_width — end-to-end decoder-noise training of the tc512 compose tokenizer

Created 2026-09-07 · thread `tc_width` · prior cycle:
[`../plan/2026-09-04_flow_decoder_noise_finetune.md`](../plan/2026-09-04_flow_decoder_noise_finetune.md)
· arm: `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8`
· control: `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0` (`BSC:45296347`)
· jobs: `BSC:45529827`

Encoder + decoder trained jointly from scratch with `decode_noise_tau=0.8`. The add `z_dec = z + σ·ε`
(`occrae/deltatok_trainer.py:470-497`) is differentiable, so the encoder trains through the noised decode.
SIGReg, cov and the compose sum keep reading the clean `z`. Four files, one new script.

## 1 `configs/deltatok/train_deltatok.yaml`

Two keys, default-off.

```yaml
  sigreg_warmup: 0
  decode_noise_warmup: 0  # linear ramp of decode_noise_tau over N iters (0 = constant)
  ...
  eval_z_spread: true
  eval_noise_sigmas: []  # eval decodes z + σ·N(0,I) per σ, logged as LossRecon_noise<σ>
```

## 2 `occrae/deltatok_trainer.py`

### 2a Tau ramp — `get_network`, beside the startup print (`:812-815`)

```python
self._decode_noise_tau_cfg = float(model._decode_noise_tau)
self._decode_noise_warmup = int(self.cfg.training.get("decode_noise_warmup", 0))
print(f"decode_noise_tau={self._decode_noise_tau_cfg} decode_noise_warmup={self._decode_noise_warmup}")
```

### 2b Tau ramp — `train_one_epoch`, top of each micro-batch (after `:1045`, before the forward)

Same ramp form as SIGReg at `:1155`, written onto the real module via `_unwrapped_tokenizer()` (`:844`).

```python
w = self._decode_noise_warmup
ramp = 1.0 if w <= 0 else min(1.0, self.cfg.training.iter / max(1, w))
self._unwrapped_tokenizer()._decode_noise_tau = self._decode_noise_tau_cfg * ramp
```

One line beside `Train/LossSIGReg` at `:1224`: `self.log_add_scalar('Train/DecodeNoiseTau', ..., self.cfg.training.iter)`.

### 2c Eval ladder — eval loop (`:1340-1386`)

The single-pair call always uses `return_z=True` (the `z_spread` branch is unchanged; the other branch discards
`z`). After `loss_recon`:

```python
for s in self._eval_noise_sigmas:                                   # tuple of floats from cfg
    eps = torch.randn(z_bneck.shape, generator=self._eval_noise_gen,
                      device=z_bneck.device, dtype=z_bneck.dtype)   # (M, N, K, Cz) seeded draw
    with self.autocast:
        x_hat_n = self.tokenizer(x_prev, None, H, W, num_cameras=num_cameras,
                                 z_input=z_bneck + s * eps)         # (M, N, P, C) same path as Comp at :1374
    batch_losses[f"LossRecon_noise{s:g}"] = _log_cosh(x_hat_n.float(), x.float()).mean().item()
```

`self._eval_noise_gen = torch.Generator(device).manual_seed(training.seed)` at the start of every
`eval_one_epoch` (`:1251`), as in `deltatok_flow_trainer.py:770`. The module is in eval mode there, so the
train-time hook cannot double-noise.

## 3 `occrae/metric.py`

`_EVAL_LOSS_KEYS` is a fixed tuple and `compute()` iterates it (`:35`, `:46`). `DeltaTokEvalMetric.__init__`
gains `extra_keys=()`; the state loop and `compute()` iterate `_EVAL_LOSS_KEYS + extra_keys`; the state attr
strips the `.` (`sum_LossRecon_noise0_32`). Both construction sites (`deltatok_trainer.py:1675`, `:1769`) pass
`extra_keys=tuple(f"LossRecon_noise{s:g}" for s in sigmas)`. The `[Eval/…]` stdout line at `:1414` and TB pick
the keys up unchanged.

## 4 `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc.slurm`

```bash
cp slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm \
   slurm/deltatok/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc.slurm
```

Change only:

```
--job-name=deltatok_decnoise_bsc   --account=ehpc1001   --time=40:00:00
--output/--error = slurm/output/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc_%j.{out,err}
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.02}      # twin BSC:45296347
export DECNOISE_TAU=${DECNOISE_TAU:-0.8}
export DECNOISE_WARMUP=${DECNOISE_WARMUP:-2000}  # = SIGREG_WARMUP
RUN_NAME=...sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}
+ model.deltatok.decode_noise_tau=${DECNOISE_TAU}
+ training.decode_noise_warmup=${DECNOISE_WARMUP}
+ training.eval_noise_sigmas=[0.32,0.55,0.82]
```

## 5 Control ladder — no code

`sh/train_deltatok.sh:26` maps `EVAL_ONLY=1` to `--eval-only`; `train_deltatok.py:136` runs one eval with
`test_only` set. Same script, env overrides, fresh `RUN_NAME` so it does not resume the twin:

```bash
EVAL_ONLY=1 DECNOISE_TAU=0 RUN_NAME=<fresh> \
INIT_CKPT=$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0/ckpts/epoch_80.pth \
sbatch --qos=acc_debug --time=01:00:00 slurm/deltatok/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc.slurm
```

## 6 Pre-flight on BSC (user syncs), then submit

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n decode_noise_warmup occrae/deltatok_trainer.py configs/deltatok/train_deltatok.yaml && grep -n eval_noise_sigmas occrae/deltatok_trainer.py configs/deltatok/train_deltatok.yaml && grep -n extra_keys occrae/metric.py && grep -E \"RUN_NAME=|decode_noise|eval_noise_sigmas|--time|--account\" slurm/deltatok/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_sigreg_decnoise_nozn_tc512_bsc.slurm'"
```

Watch until the first `[KEpoch` line; the `.out` must show `decode_noise_tau=0.8 decode_noise_warmup=2000`,
`encoder_blocks … trainable= 340.465M` (not `0.000M`), and `DecodeNoiseTau` reaching 0.8 by iter 2000.
