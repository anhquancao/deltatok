# tc_width — tc512's bottleneck norm at tc1536, SIGReg from step 0

Created 2026-09-10 · thread `tc_width` · prior cycle:
[`../plan/2026-09-09_tc_width_decnoise_detach_tc1536.md`](../plan/2026-09-09_tc_width_decnoise_detach_tc1536.md)
· arm: `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.06_ns3072_pool8192_compose1.0_decnoise0.8_detach_bneck_sw0`
· control: the prior cycle's arm (no norm on z, warmup 2000), `BSC:45610897`

One new model flag, `force_bottleneck`, builds `pre_bottleneck_norm` + `z_proj_down` + `z_proj_up` even when
`z_dim == hidden_size`. At tc1536 that is `LayerNorm(1536) → Linear(1536,1536)` on encode and `Linear(1536,1536)`
on decode, the same three modules tc512 has. `sigreg_warmup=0`. Default `false` builds nothing: no new keys,
no numeric change on any other arm.

## 1 `occrae/deltatok_trainer.py`

**1a `DeltaTokModule.__init__` signature, after line 79 `bottleneck_mlp: bool = False,`**

```python
        force_bottleneck: bool = False,
```

**1b line 150, the comment line, replace**

```python
        # z_dim == hidden_size so existing checkpoints load unchanged.
```
with
```python
        # z_dim == hidden_size unless force_bottleneck, so old ckpts load.
```

**1c line 151, replace**

```python
        if self.z_dim != cfg.hidden_size:
```
with
```python
        if self.z_dim != cfg.hidden_size or force_bottleneck:
```

Nothing else moves. `_print_param_breakdown` (line 513) already lists the three modules when built, the
ckpt loader (line 879) already warns when their keys are missing, and the decode path (line 401) already
gates on `z_proj_up is not None`.

## 2 `occrae/deltatok_shared.py` line 503, after `bottleneck_mlp=...`

```python
            force_bottleneck=bool(deltatok_cfg.get("force_bottleneck", False)),
```

## 3 `configs/deltatok/train_deltatok.yaml` after line 68 (`bottleneck_mlp: false`)

```yaml
    force_bottleneck: false  # build pre-norm + proj down/up even at target_channels == hidden_size
```

## 4 `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm`

```bash
cp slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm \
   slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm
```

Change only:

```
--job-name=deltatok_bn_dt1536_bsc
--output/--error = slurm/output/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc_%j.{out,err}
header: tc512's LN+proj on z at tc1536 (force_bottleneck), SIGReg from step 0; twin of BSC:45610897 + sbatch line with the new filename
export DECNOISE_WARMUP=${DECNOISE_WARMUP:-2000}  # decoder-side ramp
export SIGREG_WARMUP=${SIGREG_WARMUP:-0}                 # anchor on from step 0
RUN_NAME=...compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}_detach_bneck_sw${SIGREG_WARMUP}
model.deltatok.target_channels=1536        # == hidden_size; bottleneck forced below
model.deltatok.force_bottleneck=true       # LN(1536) + Linear down/up, as tc512
```

Everything else (sigreg 0.06, ns 3072, pool 8192, compose 1.0, tau 0.8, decnoise weight 1.0, eval ladder,
max_gap 9, bsize 2 / effective 16, lr 1e-3, grad_clip 0.1, 40 h, ehpc880) is the twin's.

## 5 Pre-flight (local), then BSC (user syncs), then submit

```bash
python3 -m py_compile occrae/deltatok_trainer.py occrae/deltatok_shared.py
bash -n slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n \"force_bottleneck\" occrae/deltatok_trainer.py occrae/deltatok_shared.py configs/deltatok/train_deltatok.yaml | wc -l && grep -E \"RUN_NAME=|SIGREG_WARMUP=|force_bottleneck|--job-name|--output|--time|--account\" slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm'"
```

The grep must count 3. Watch until the first `[KEpoch` line. The `.out` must show `pre_bottleneck_norm`,
`z_proj_down` and `z_proj_up` rows in the parameter table (2 × 1536·1536 + biases + LN ≈ 4.7M params),
`training.sigreg_warmup=0` in `EXTRA_CFG`, `Train/LossSIGReg` > 0 from the first print, and `ZRowMeanSquare`
near 1 on the first eval. Chain at 40 h with `chain-slurm-jobs`.
