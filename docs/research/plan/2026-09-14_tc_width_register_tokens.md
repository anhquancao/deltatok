# Plan: register tokens on the raw-z tc1536 arm

**Date:** 2026-09-14 · **Thread:** tc_width · **prior cycle:** `analysis/2026-09-14_tc_width_forced_bneck_stall.md`
· **Control:** `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm` at
`SIGREG_WEIGHT=0.02 SIGREG_POOL_SAMPLES=24576` (`BSC:45727710`, raw z, no bottleneck)

`num_registers` R extra learned slots ride through the encoder blocks next to the K delta slots and are
dropped before `self.norm` (no bottleneck is built: `force_bottleneck=false`, `target_channels == hidden`);
R more ride through the decoder next to z. Neither set is read by any loss. `num_registers: 0` is
bit-identical to today and keeps every checkpoint key.

## 1. Config

**`configs/deltatok/train_deltatok.yaml`**, insert after line 69 (`force_bottleneck`):

```yaml
    num_registers: 0  # extra encoder/decoder slots never read as z: a free home for the attention sink
```

**`configs/deltatok_flow/train_deltatok_flow.yaml`**, insert after line 91 (`force_bottleneck`):

```yaml
    num_registers: 0  # must match the frozen ckpt
```

## 2. Module — `occrae/deltatok_trainer.py`

**2a. Constructor.** Line 80, after `force_bottleneck: bool = False,`:

```python
        num_registers: int = 0,
```

After line 86 (`self.num_delta_tokens = ...`):

```python
        self.num_registers = int(num_registers)              # R register slots per camera, never read as z
```

After line 128 (`nn.init.trunc_normal_(self.xy_embed.weight, ...)`):

```python
        if self.num_registers > 0:
            self.enc_reg_embed = nn.Embedding(self.num_registers, cfg.hidden_size)   # (R, C) encoder registers
            self.dec_reg_embed = nn.Embedding(self.num_registers, cfg.hidden_size)   # (R, C) decoder registers
            nn.init.trunc_normal_(self.enc_reg_embed.weight, std=cfg.initializer_range)
            nn.init.trunc_normal_(self.dec_reg_embed.weight, std=cfg.initializer_range)
        else:
            self.enc_reg_embed = None                         # absent at R=0 so old ckpts load strict
            self.dec_reg_embed = None
```

**2b. Encode.** Replace lines 324–327:

```python
        K = self.num_delta_tokens + self.num_registers     # slots per camera through the blocks: K z + R registers

        # N*K prefix tokens per camera: K z slots from z_embed, then R register slots from enc_reg_embed.
        z = self.z_embed.weight[None, None].expand(M, N, self.num_delta_tokens, C)  # (M, N, K, C)
        if self.enc_reg_embed is not None:
            reg = self.enc_reg_embed.weight[None, None].expand(M, N, self.num_registers, C)  # (M, N, R, C)
            z = torch.cat([z, reg], dim=2)                 # (M, N, K+R, C)
        z = z.contiguous()
```

The loop body (lines 340–361) is unchanged: `K` now counts both slot kinds, and both stay prefix tokens
for `apply_rotary_pos_emb`.

Insert before line 363 (`if self.z_proj_down is not None:`):

```python
        self._enc_row_absmax = z.detach().float().abs().amax(-1)   # (M, N, K+R) pre-LN |h| max per slot, sink probe
        z = z[:, :, : self.num_delta_tokens]               # (M, N, K, C) registers dropped, never read
```

**2c. Decode.** Insert after line 403 (the `z_proj_up` block, a no-op here), before line 404 (`z = z.contiguous()`):

```python
        if self.dec_reg_embed is not None:
            reg = self.dec_reg_embed.weight[None, None].expand(M, N, self.num_registers, C)  # (M, N, R, C)
            z = torch.cat([z, reg], dim=2)                 # (M, N, K+R, C) registers ride along, never read
```

Lines 404–425 unchanged: `K = z.shape[2]` already counts the appended slots, and `spatials = hidden[:, :, K:]`
still returns only patches.

**2d. Parameter breakdown.** After line 521 (`z_proj_up` components line):

```python
    if model.enc_reg_embed is not None:
        components += [("enc_reg_embed", model.enc_reg_embed), ("dec_reg_embed", model.dec_reg_embed)]
```

**2e. Sink probe scalars.** Replace line 1272:

```python
                rm = net._enc_row_absmax                            # (M, N, K+R) from the last encode this step
                zrow_max = float(rm[:, :, : net.num_delta_tokens].max())
                reg_max = float(rm[:, :, net.num_delta_tokens :].max()) if net.num_registers else 0.0
                metric_logger.update(loss=loss_val, lr=self.optim.param_groups[0]['lr'],
                                     zrow_max=zrow_max, reg_max=reg_max)
```

The loop does not bind `net` (verified, lines 1100–1320). Insert after line 1138 (`num_batches += 1`):

```python
            net = self._unwrapped_tokenizer()                       # sink probe reads net._enc_row_absmax
```

After line 1295 (`Train/SpeedSamplesPerSec`):

```python
                    self.log_add_scalar('Train/EncRowAbsMaxZ', zrow_max, self.cfg.training.iter)
                    self.log_add_scalar('Train/EncRowAbsMaxReg', reg_max, self.cfg.training.iter)
```

## 3. Factory — `occrae/deltatok_shared.py`

After line 504 (`force_bottleneck=...`):

```python
            num_registers=int(deltatok_cfg.get("num_registers", 0)),
```

Replace line 485 `return DeltaTokModule(` with `net = DeltaTokModule(`, and after the closing `)` of the call:

```python
        if self.is_master:
            print(f"[INFO] DeltaTok num_registers={net.num_registers}", flush=True)  # stale trainer prints 0
        return net
```

## 4. Slurm — `slurm/deltatok/train_deltatok_reg_nozn_tc1536_bsc.slurm` (new)

`cp slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm` → the new name, then:

- Line 2: `#SBATCH --job-name=deltatok_reg_dt1536_bsc`.
- Lines 13–14: `slurm/output/train_deltatok_reg_nozn_tc1536_bsc_%j.{out,err}`.
- Lines 16–21 → `# Register-token twin of BSC:45727710 (raw z, no bottleneck): R free slots for the attention sink.`
  and the sbatch line with the new name. Drop the eval-ladder lines 20–21.
- Line 24: `export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.02}    # as BSC:45727710`.
- After line 27: `export NUM_REGISTERS=${NUM_REGISTERS:-4}     # register slots per camera, encoder and decoder`.
- Line 46: `export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-24576}  # as BSC:45727710`.
- Line 50: `export RUN_NAME=${RUN_NAME:-deltatok_l12_dtok64_tc1536_nozn_reg${NUM_REGISTERS}_maxgap9_vpt1to2_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}_detach_sw${SIGREG_WARMUP}}`.
- After line 55 (`force_bottleneck=false`): `    model.deltatok.num_registers=${NUM_REGISTERS}   # dropped before self.norm, never read`.
- Everything else identical (ehpc880, 40 h, decnoise 0.8 / warmup 2000 / weight 1.0, compose 1.0).

## 5. Pre-flight

1. User syncs. md5 local vs BSC: `occrae/deltatok_trainer.py`, `occrae/deltatok_shared.py`,
   `configs/deltatok/train_deltatok.yaml`, the new slurm.
2. Smoke: `RUN_NAME=smoke_reg4 sbatch --qos=acc_debug --time=00:40:00 <new slurm>`. Pass when the log shows
   `[INFO] DeltaTok num_registers=4`, `enc_reg_embed` and `dec_reg_embed` in the parameter breakdown, a first
   loss line, and `zrow_max` / `reg_max` meters on the `log_every` lines. `scancel`.
3. Null: `NUM_REGISTERS=0 RUN_NAME=smoke_reg0` on the same script. Log prints `num_registers=0`, no
   `*_reg_embed` in the breakdown, `reg_max=0.0`. Loss at iter 0 matches the control's iter-0 value in
   `slurm/output/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc_45727710.out`.
4. Resume: relaunch the step-2 smoke after one ckpt save; loss continuous, no strict-load error.
5. Prod: `sbatch <new slurm>` (acc_ehpc, 40 h, `exit_before_time_limit=true`), chain with `chain-slurm-jobs`.
   Watch until `RUNNING` + first loss line.
6. Read at ep 1–2: `grep -o "zrow_max: [0-9.]*\|reg_max: [0-9.]*"` on the `.out`. Sink on the registers means
   `reg_max` in the hundreds and `zrow_max` under ~50; the control had slot 44 at 2000–7000. Eval `ZPartRank`
   should then read the non-tail rank (~250) from ep 1, not 1.0. At ep 10 the `BSC:45857234` probe on the new
   ckpt: no single channel with layer_scale2 max above ~0.3 in every encoder block.
7. `todos.json` status board: add the queued arm; ledger row; rebuild the index.
