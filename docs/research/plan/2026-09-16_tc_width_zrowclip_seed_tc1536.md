# Plan: seeded replicate of the pool-24576 raw-z arm, with and without a per-row RMS clip on z

**Date:** 2026-09-16 · **Thread:** tc_width · **prior cycle:** `results/2026-09-16_tc_width_tc1536_rank_collapse_fixes_slides.html`
· **Control:** `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm` at
`SIGREG_WEIGHT=0.02 SIGREG_POOL_SAMPLES=24576 RUN_NAME=..._sw0` (`BSC:45727710`, raw z, no bottleneck, ep 31)

Two knobs. `model.deltatok.z_row_clip` caps the per-row RMS of z at the end of `encode()`, straight-through
(forward clipped, backward identity), so SIGReg reads every row inside its restoring band while the encoder
gets the unclipped gradient. `training.init_seed` calls `torch.manual_seed(init_seed + rank)` at trainer init;
today nothing seeds torch, so every arm starts from the default generator seed. Both default off and are
bit-identical to today; neither adds a parameter. One new slurm script, launched twice: `Z_ROW_CLIP=0` and
`Z_ROW_CLIP=4`, same `INIT_SEED=1`, both with decnoise off (`DECNOISE_TAU=0`; the control ran 0.8). The same edit removes `num_registers` (register tokens, `BSC:45861170`),
so only that arm's checkpoints stop loading.

All line numbers below are today's files, before any edit.

## 1. Config — `configs/deltatok/train_deltatok.yaml`

Replace line 70 (`num_registers: 0 ...`) with:

```yaml
    z_row_clip: 0.0  # cap on per-row RMS of z at encode, straight-through; 0 = off
```

Insert after line 144 (`seed: 0`):

```yaml
  init_seed: -1  # >=0 seeds torch (init_seed + rank) at trainer init; -1 = torch default
```

**`configs/deltatok_flow/train_deltatok_flow.yaml`:** delete line 92 (`num_registers: 0  # must match the frozen ckpt`).

## 2. Module — `occrae/deltatok_trainer.py`

**2a. Constructor.** Replace line 81 (`num_registers: int = 0,`) with:

```python
        z_row_clip: float = 0.0,
```

Replace line 88 (`self.num_registers = ...`) with:

```python
        self.z_row_clip = float(z_row_clip)     # cap on per-row RMS of z (0 = off); straight-through
```

Delete lines 131–138 (the `if self.num_registers > 0:` block building `enc_reg_embed` / `dec_reg_embed`).

**2b. Encode.** Replace lines 334–341 (`K = self.num_delta_tokens + self.num_registers` through
`z = z.contiguous()`) with the pre-register form:

```python
        K = self.num_delta_tokens                          # delta tokens per camera

        # N*K z tokens (K per camera); all initialized from the shared z_embed.
        z = self.z_embed.weight[None, None].expand(M, N, K, C).contiguous()  # (M, N, K, C)
```

Line 377, the shape comment `(M, N, K+R)` becomes `(M, N, K)`. Delete line 378
(`z = z[:, :, : self.num_delta_tokens]`).

Replace line 383 (`return self.norm(z)`):

```python
        if self.z_row_clip > 0:
            rms = z.float().pow(2).mean(-1, keepdim=True).sqrt()                          # (M, N, K, 1) per-row RMS
            zc = z * (self.z_row_clip / rms.clamp_min(self.z_row_clip)).to(z.dtype)      # (M, N, K, Cz) rows over the cap rescaled onto it
            z = z + (zc - z).detach()                                                     # (M, N, K, Cz) STE: forward clipped, backward identity
        return self.norm(z)
```

Line 377 (`self._enc_row_absmax = ...`) stays before the clip, so `Train/EncRowAbsMaxZ` still reads the raw
residual. The clipped z is what `forward(return_z=True)` returns, so the decoder, `_sigreg_pooled`, the
noised decodes and the `Train/Z*` and `Eval/Z*` stats all see it.

**2c. Decode.** Delete lines 421–423 (`if self.dec_reg_embed is not None:` and its two lines).

**2d. Param breakdown.** Delete lines 542–543 (`if model.enc_reg_embed is not None:` and its line).

**2e. Seed.** Insert after line 573 (`self.is_master = rank == 0`):

```python
        init_seed = int(self.cfg.training.get("init_seed", -1))
        if init_seed >= 0:                          # -1 = torch default seed (every arm so far)
            torch.manual_seed(init_seed + rank)
        if self.is_master:
            print(f"[INFO] init_seed={init_seed}", flush=True)
```

**2f. Sink probe, train loop.** Keep line 1161 and `Train/EncRowAbsMaxZ`. Replace lines 1295–1299 (`rm = ...`
through `zrow_max=zrow_max, reg_max=reg_max)`) with:

```python
                zrow_max = float(net._enc_row_absmax.max())         # pre-LN |h| max over z slots, last encode this step
                metric_logger.update(loss=loss_val, lr=self.optim.param_groups[0]['lr'], zrow_max=zrow_max)
```

Delete line 1324 (`Train/EncRowAbsMaxReg`).

## 3. Factory — `occrae/deltatok_shared.py`

Replace line 505 (`num_registers=...`) with:

```python
            z_row_clip=float(deltatok_cfg.get("z_row_clip", 0.0)),
```

Line 508, the print becomes:

```python
            print(f"[INFO] DeltaTok z_row_clip={net.z_row_clip}", flush=True)
```

## 4. Slurm — `slurm/deltatok/train_deltatok_nozn_tc1536_pool24576_seed_bsc.slurm` (new)

`cp slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm` → the new name, then:

- Line 2: `#SBATCH --job-name=deltatok_dn1536_p24576_seed`.
- Line 10: `#SBATCH --time=48:00:00`.
- Lines 13–14: `slurm/output/train_deltatok_nozn_tc1536_pool24576_seed_bsc_%j.{out,err}`.
- Lines 16–21 → `# BSC:45727710 seeded, decnoise off (raw z, SIGReg 0.02, pool 24576, warmup 0); Z_ROW_CLIP=4 adds the per-row RMS cap.`
  and the sbatch line with the new name. Drop the eval-ladder lines 20–21.
- Line 24: `export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.02}    # as BSC:45727710`.
- Line 25: `export DECNOISE_TAU=${DECNOISE_TAU:-0}         # 0 = off; 45727710 ran 0.8`.
- After line 27:
  ```bash
  export INIT_SEED=${INIT_SEED:-1}       # torch.manual_seed(INIT_SEED + rank); 45727710 ran on torch's default
  export Z_ROW_CLIP=${Z_ROW_CLIP:-0}     # per-row RMS cap on z, straight-through; 0 = off, 4 = the clip arm
  ```
- Line 46: `export SIGREG_POOL_SAMPLES=${SIGREG_POOL_SAMPLES:-24576}  # as BSC:45727710`.
- Line 50: `export RUN_NAME=${RUN_NAME:-deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}_sw${SIGREG_WARMUP}_seed${INIT_SEED}_rclip${Z_ROW_CLIP}}`.
  No `_detach`: at tau 0 the detached noised decode never runs.
- After line 55 (`force_bottleneck=false`):
  ```bash
      model.deltatok.z_row_clip=${Z_ROW_CLIP}     # per-row RMS cap on z; 0 = off
      training.init_seed=${INIT_SEED}
  ```
- Everything else identical (ehpc880, compose 1.0, grad_clip 0.1). Decnoise warmup 2000 / weight 1.0 stay but are inert at tau 0.

Retire the register arm: `git mv slurm/deltatok/train_deltatok_reg_nozn_tc1536_bsc.slurm slurm/deltatok/archive/`.

## 5. Pre-flight

1. Local: `python3 -m py_compile occrae/deltatok_trainer.py occrae/deltatok_shared.py`, and
   `grep -rn "num_registers\|reg_embed\|reg_max\|EncRowAbsMaxReg" occrae configs slurm/deltatok/*.slurm` prints nothing.
2. User syncs. Grep the BSC copies for `z_row_clip` (trainer, factory, yaml, slurm) and `init_seed` (trainer,
   yaml, slurm), and confirm `num_registers` is gone from the trainer, factory and both yamls, before any `sbatch`.
3. Smoke, both arms, 30 min on `acc_debug`:
   `RUN_NAME=smoke_seed1_rclip0 sbatch --qos=acc_debug --time=00:30:00 <new slurm>` and the same with
   `Z_ROW_CLIP=4 RUN_NAME=smoke_seed1_rclip4`. Pass when each log shows `[INFO] init_seed=1`,
   `[INFO] DeltaTok z_row_clip=0.0` / `4.0`, and a first loss line. The two iter-0 losses match
   each other (same seed, no row over the cap yet) and differ from `BSC:45727710`'s iter-0 loss
   (`slurm/output/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc_45727710.out`). `scancel`.
4. Prod, from a login shell at the repo root:
   ```bash
   sbatch slurm/deltatok/train_deltatok_nozn_tc1536_pool24576_seed_bsc.slurm
   Z_ROW_CLIP=4 sbatch slurm/deltatok/train_deltatok_nozn_tc1536_pool24576_seed_bsc.slurm
   ```
   One 48 h job each, no chain. Watch until `RUNNING` + first loss line.
5. Read at ep 1–6 from the `[Train]` and `[Eval/… @ KittiSeqMultiView]` z lines: train `ZRowMeanSquare` (the
   control peaked at 20,684 at ep 1) and eval `ZPartRank` (control 1.0 at ep 1 and 3, 127 at ep 6). The clip arm
   cannot exceed RMS 16 by construction; its `Train/EncRowAbsMaxZ` says whether the sink still forms in the
   residual. Then KITTI / nuScenes `LossRecon` against the control's 0.0305 / 0.0211 at ep 31 (control ran tau 0.8).
6. `questions.json` status board: move the arms from queued to running; ledger row; rebuild the index.

## 6. Tracking

Code `7704517`, synced and verified on BSC 2026-09-16. Steps 1–2 done.

| Job | Arm | Queue | State (2026-09-16 17:45) |
|---|---|---|---|
| `BSC:45931033` | smoke, `Z_ROW_CLIP=0`, `RUN_NAME=smoke_seed1_rclip0` | `acc_debug`, 30 min | **passed**: `init_seed=1`, `z_row_clip=0.0`, tau 0.0, 681.032M params, iter-0 loss 0.4290, trained to iter 160+; cancelled at 26 min |
| `BSC:45931830` | smoke, `Z_ROW_CLIP=4`, `RUN_NAME=smoke_seed1_rclip4` | `acc_debug`, 30 min (moved from `acc_ehpc` once 45931033 left) | `PENDING`; pass needs `z_row_clip=4.0` and iter-0 loss 0.4290 |

Logs: `slurm/output/train_deltatok_nozn_tc1536_pool24576_seed_bsc_<jobid>.{out,err}`. Control iter-0 train loss 0.4546
(`../monitor_jobs/data/logs/BSC/deltatok_dn1536_pool24576_sw0_45727710.out:132`).

Resume at step 3: check both smoke logs against the pass line, `scancel` whichever is still running, then step 4 and step 6.
