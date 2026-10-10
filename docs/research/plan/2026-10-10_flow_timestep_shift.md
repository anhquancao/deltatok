# Plan: timestep shift toward noise on the multires z-cache flow

**Date:** 2026-10-10 · **Thread:** flow · **Cluster:** BSC
· **Reference:** control `BSC:47058660` (multires z-cache, 200k, `iter_050000` at 10 h 57 min) · prior cycle: —

## Overview

- **`model.t_shift = a`:** each train draw's noise fraction `s = 1 - t` becomes `a·s / (1 + (a-1)·s)`. `a > 1` moves mass toward noise; `1` is off.
- **`model.sampler_t_shift = a`:** the same map on the eval sampler's t-grid.
- **Eval/LossFlow keeps the unshifted t**, so it stays comparable with the control.
- **Runs:** a sampler-only probe on the control's `iter_200000` (a = 5, 13), then two 50k training arms (a = 5, 13), then a 1-GPU read at 50k.

| File | Change |
|---|---|
| `configs/deltatok_flow/train_deltatok_flow.yaml` | 2 keys: `t_shift`, `sampler_t_shift` |
| `occrae/deltatok_flow_trainer.py` | read + print `t_shift`; `flow_noising(t_shift=)`; train call passes it; eval sampler passes `sampler_t_shift` |
| `occrae/generation_helper.py` | `flow_euler_sample(t_shift=)` maps the grid |
| `visualize_deltatok_flow.py` | passes `sampler_t_shift` |
| `slurm/eval_deltatok_flow_tshift_multires_zcache_bsc.slurm` | new: probe and 50k read, one eval set per job |
| `slurm/deltatok_flow/train_deltatok_flow_alldata_xxl_tc1536mg9_sigreg002seed1_ctx2fwd8_multires_zcache_tshift{5,13}_bsc.slurm` | new: 2 training arms |

## 1. `configs/deltatok_flow/train_deltatok_flow.yaml`

After `t_shared` (:72):

```yaml
  t_shift: 1.0  # train: noise fraction s -> a*s/(1+(a-1)*s); >1 = toward noise, 1 = off
  sampler_t_shift: 1.0  # same map on the eval t-grid; set equal to t_shift
```

Every flow config used here inherits it: `train_..._multires_zcache_bsc` and `eval_deltatok_flow_alldata_ctx2fwd8_bsc` → `train_deltatok_flow_bsc` → this file.

## 2. `occrae/deltatok_flow_trainer.py`

**:110–113**, after `force_zero_t_ratio`:

```python
        self.force_zero_t_ratio = float(self.cfg.model.get("force_zero_t_ratio", 0.0))
        self.t_shift = float(self.cfg.model.get("t_shift", 1.0))  # train only: Eval/LossFlow keeps the control's t
        print(f"t schedule: t_dist={self.cfg.model.get('t_dist', 'logitnormal')} "
              f"mu={self.cfg.model.mu} sigma={self.cfg.model.sigma} "
              f"force_zero_t_ratio={self.force_zero_t_ratio} "
              f"t_shift={self.t_shift} sampler_t_shift={self.cfg.model.get('sampler_t_shift', 1.0)}")
```

**:543**, signature: add `t_shift=1.0` after `force_zero_t_ratio=0.0`.

**:566**, after the `t_dist` branch, before the `force_zero_t_ratio` block (:568):

```python
            if t_shift != 1.0:
                s = 1.0 - t                                          # (b, 1|t) noise fraction
                t = 1.0 - t_shift * s / (1.0 + (t_shift - 1.0) * s)  # (b, 1|t) shifted toward noise
```

- It runs after the draw, so the RNG stream is the control's draw for draw.
- `train_fixed_t` is untouched.

**:714–717**, train call: add `t_shift=self.t_shift,`. The eval call at :1059 stays as is, so it uses uniform t.

**:966**, eval sampler call: add `t_shift=float(self.cfg.model.get("sampler_t_shift", 1.0)),` after `alpha=`.

## 3. `occrae/generation_helper.py`

- **Signature (:12–28):** `t_shift=1.0` after `step_mode="ode"`.
- **Docstring:** `t_shift: noise-ward t-grid shift, as model.t_shift in training; 1 = off.`
- **After the schedule branch (:78), before `offsets` (:81):**

```python
        if t_shift != 1.0:
            # clamp first: progress_next passes 1 on the last step, and the map has a pole there
            s, s_next = 1 - min(progress, 1.0), 1 - min(progress_next, 1.0)  # noise fractions
            progress = 1 - t_shift * s / (1 + (t_shift - 1) * s)
            progress_next = 1 - t_shift * s_next / (1 + (t_shift - 1) * s_next)
```

- **Grid at 20 linear steps:** with a = 13 the last step before t = 1 sits at t = 0.594; with a = 5 it sits at 0.792. The final step lands on x̂.
- **Untouched:** `GenerationHelper` (:194+, legacy OccRAE). `probe_flow_motion.py` keeps the default.

## 4. `visualize_deltatok_flow.py`

At :297, after `alpha=`: `t_shift=float(cfg.model.get("sampler_t_shift", 1.0)),`.

## 5. `slurm/eval_deltatok_flow_tshift_multires_zcache_bsc.slurm` (new)

`cp slurm/eval_deltatok_flow_numsteps_alldata_ctx2fwd8_bsc.slurm`, then:

- **Job name:** `deltatok_flow_tshift_multires_zcache`. Output and error go to `slurm/output/eval_deltatok_flow_tshift_multires_zcache_bsc_%j.{out,err}`.
- **Wall:** `--time=06:00:00`.
- **Header:** `# Sampler t-shift eval of the multires z-cache flow (BSC:47058660), one eval set per job.` Point the "Arch flags mirror" line at the `_multires_zcache_bsc.slurm` arm and fix the sbatch usage path.
- **Env defaults:**
  - `CKPT` → `$SCRATCH/quan/deltatok_flow_log/deltatok_flow_alldata_consec10cam0_ctx2fwd8_tc1536mg9sigreg002compose_seed1_ep70tok_xxl_dit_multires_zcache/ckpts/iter_200000.pth`
  - `: "${T_SHIFT:=1}"`
  - `: "${TEST_FILTER:=}"  # Kitti | Nuscenes | Waymo; empty = all 3`
  - `OUTPUT_DIR` → `results/deltatok_flow_tshift_multires_zcache/tshift${T_SHIFT}`
  - `NUM_STEPS` → `20`
- **CFG_ARGS:** add `model.sampler_t_shift="$T_SHIFT"   # eval t-grid shift; 1 = the control's grid`.
- **srun line:** add `${TEST_FILTER:+--test_filter "$TEST_FILTER"}` before `--cfg`.
- **Unchanged:** 1 GPU, `--fvd`, ode, linear grid, `sampler_alpha=0`, `val_bsize=4`.

## 6. Training arms (new, 2 files)

`cp slurm/deltatok_flow/train_deltatok_flow_alldata_xxl_tc1536mg9_sigreg002seed1_ctx2fwd8_multires_zcache_bsc.slurm` to `..._multires_zcache_tshift5_bsc.slurm` and `..._tshift13_bsc.slurm`. Edits for a = 5 (a = 13 is analogous, median noise 0.93):

- `--job-name=deltatok_flow_alldata_ctx2fwd8_multires_zcache_tshift5`
- `--time=14:00:00`
- `--output` / `--error` → `..._multires_zcache_tshift5_bsc_%j.{out,err}`
- Header: `# t-shift 5 twin of BSC:47058660, stopped at 50k.`, plus the sbatch usage path.
- `RUN_NAME=deltatok_flow_alldata_consec10cam0_ctx2fwd8_tc1536mg9sigreg002compose_seed1_ep70tok_xxl_dit_multires_zcache_tshift5`
- `training.epoch=25                          # 50k updates, vs 47058660 iter_050000`
- `training.max_iter=50000                    # epoch=25 x 2000 updates lands exactly here`
- After `model.t_shared=true`: `model.t_shift=5                            # train t toward noise, median noise 0.83`
- After `model.sampler_alpha=0`: `model.sampler_t_shift=5                    # same shift on the eval t-grid`

`iter_050000.pth` is written at iter 50000 (:832), before the `max_iter` break. Everything else is the control's: tokenizer and z-cache, arch, constant lr 1e-4, warmup 1000, effective bsize 64, seed, 20 eval steps, 4 GPUs.

## 7. Pre-flight

1. Sync with `monitor-sync`. md5 local vs BSC for the 4 edited files and the 3 new slurm scripts.
2. **Map check on the BSC login node** (CPU, `source env_bsc.sh`). Call `flow_euler_sample` with a stub model that records `ada_cond` and returns `x`. Use z `(1, 1, 2, 1, 1)`, `context=1`, linear grid, `alpha=0`, 20 steps.
   - `t_shift=1` reproduces `i/20` exactly.
   - `t_shift=13` is monotone in [0, 1], with the last step before 1 at 0.594. `t_shift=5` gives 0.792.
   - The train map on 1e6 uniform draws gives a median noise fraction of 0.833 (a = 5) and 0.929 (a = 13).
3. **Probe jobs:** the log shows `Loaded flow ViT from: .../iter_200000.pth` and `sampler_t_shift=5.0` / `13.0` in the `t schedule:` line.
4. **Arms:** the log shows `t_shift=5.0 sampler_t_shift=5` (resp. 13), an `Eval (sanity)` line (the shifted sampler runs here), then the first loss line. Watch until that line appears.
5. **a = 1 control for the probe:** `BSC:47126870` / `47126878` / `47126886` (another session's 20-step passes on the same checkpoint, same wrapper flags). Confirm their `CKPT`, `NUM_STEPS` and set filter from the `.out`. If any is missing or differs, add `T_SHIFT=1` jobs.

## 8. Submit

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  for TS in 5 13; do for SET in Kitti Nuscenes Waymo; do \
    sbatch --job-name=eval_tshift\${TS}_\${SET} --export=ALL,T_SHIFT=\$TS,TEST_FILTER=\$SET \
      slurm/eval_deltatok_flow_tshift_multires_zcache_bsc.slurm; \
  done; done; \
  sbatch slurm/deltatok_flow/train_deltatok_flow_alldata_xxl_tc1536mg9_sigreg002seed1_ctx2fwd8_multires_zcache_tshift5_bsc.slurm; \
  sbatch slurm/deltatok_flow/train_deltatok_flow_alldata_xxl_tc1536mg9_sigreg002seed1_ctx2fwd8_multires_zcache_tshift13_bsc.slurm'"
```

## 9. Read at 50k (after both arms finish)

Same wrapper: 1 GPU, 20 steps, one job per set, with `OUTPUT_DIR` set per checkpoint. That makes 9 jobs.

| Checkpoint | `T_SHIFT` |
|---|---|
| `47058660` `iter_050000.pth` | 1 |
| `..._multires_zcache_tshift5/ckpts/iter_050000.pth` | 5 |
| `..._multires_zcache_tshift13/ckpts/iter_050000.pth` | 13 |
