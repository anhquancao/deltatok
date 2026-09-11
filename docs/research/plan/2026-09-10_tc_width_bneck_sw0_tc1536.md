# tc_width — tc512's bottleneck norm at tc1536, SIGReg from step 0

Created 2026-09-10 · thread `tc_width` · prior cycle:
[`../plan/2026-09-09_tc_width_decnoise_detach_tc1536.md`](../plan/2026-09-09_tc_width_decnoise_detach_tc1536.md)
· arm: `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.06_ns3072_pool8192_compose1.0_decnoise0.8_detach_bneck_sw0`
· control: the prior cycle's arm (no norm on z, warmup 2000), `BSC:45610897`
· jobs: `BSC:45678654` · deck: [`../results/2026-09-11_tc_width_bneck_sw0_tc1536_read_slides.html`](../results/2026-09-11_tc_width_bneck_sw0_tc1536_read_slides.html)

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
model.deltatok.bottleneck_mlp=false        # Linear down/up, not the SiLU MLP variant
```

The last line is a comment fix the list above missed: the twin reads `# inert at tc1536 (no bottleneck to
make nonlinear)`, and `force_bottleneck` routes through the same `if bottleneck_mlp:` branch, so the value
is no longer inert — it is what selects Linear over the SiLU MLP.

Everything else (sigreg 0.06, ns 3072, pool 8192, compose 1.0, tau 0.8, decnoise weight 1.0, eval ladder,
max_gap 9, bsize 2 / effective 16, lr 1e-3, grad_clip 0.1, 40 h, ehpc880) is the twin's.

## 5 Pre-flight (local), then BSC (user syncs), then submit

```bash
python3 -m py_compile occrae/deltatok_trainer.py occrae/deltatok_shared.py
bash -n slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm
F="occrae/deltatok_trainer.py occrae/deltatok_shared.py configs/deltatok/train_deltatok.yaml slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm"
md5sum $F
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && md5sum $F'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_bneck_nozn_tc1536_bsc.slurm'"
```

All four md5 pairs must match. Use md5, not a `grep force_bottleneck | wc -l` count — grep only checks the
lines you thought to name, and the count is not 3 anyway (the trainer holds the word three times: param,
comment, condition). Watch until the first `[KEpoch` line. The `.out` must show `pre_bottleneck_norm`,
`z_proj_down` and `z_proj_up` rows in the parameter table (2 × 1536·1536 + biases + LN ≈ 4.7M params),
`training.sigreg_warmup=0` in `EXTRA_CFG`, `Train/LossSIGReg` > 0 from the first print, and `ZRowMeanSquare`
near 1 on the first eval. Chain at 40 h with `chain-slurm-jobs`.

## 6 Status

Applied 2026-09-10. `py_compile` and `bash -n` clean; local and BSC copies md5-identical on all four files
(`deltatok_trainer.py`, `deltatok_shared.py`, `train_deltatok.yaml`, the new slurm). Submitted as
**`BSC:45678654`** at 14:15, PENDING (Priority), 40 h on `ehpc880`. Control `BSC:45610897` was cancelled
at ep 31/100 the same morning, so the two arms compare at matched epochs up to 31.

## 7 Outcome

Read at ep 22 in
[`../results/2026-09-11_tc_width_bneck_sw0_tc1536_read_slides.html`](../results/2026-09-11_tc_width_bneck_sw0_tc1536_read_slides.html).

**The code is cured and the decoder pays 3.2× for it.** Every mechanical check in §5 held: the param table shows
`pre_bottleneck_norm` 0.003 M + `z_proj_down` 2.361 M + `z_proj_up` 2.361 M = 4.725 M of 685.757 M,
`training.sigreg_warmup=0` in `EXTRA_CFG`, and `ZRowMeanSquare` 1.02 on the first eval. `ZPartRank` never touches
1.0, so the ep-4 gate passes and the ep-31 massive-activation diagnosis is confirmed by repair.

It is not enough. At matched ep 22 token `LossRecon` is ×3.24 / ×3.11 the control (0.1054 / 0.0722 against
0.0325 / 0.0232), ×3.72 / ×3.80 the plain tc1536 and ×1.56 the tc512 arm at half the width; pointmap `PredVsOrig`
is ×4.35 KITTI / ×6.62 nuScenes. The arm never has the ep-4 cliff the other three have. Rank lands at 2.3% of Cz
against the plain arm's 15.7%, and the raw SIGReg statistic is 4× *further* from the anchor than the control's
(0.0063 vs 0.0016) at the same weight and slice count — scale fixed, rank not. The σ ladder is flat (+0.3% at
σ=0.82 against the control's +55%) on a code of RMS 1.13, so the decoder is barely reading `z`.

**The missing norm was not the cause.** The plain tc1536 `BSC:44590128` has no norm on `z` either, sits at
`ZPartRank` 1.0 for epochs 1–3, and escapes on its own at ep 4 (44.2 → 112.8 → 172.4 → 240.7 by ep 22). What the
control has and the plain arm does not is 12× the sigreg weight and the decode-noise term. That is the next arm:
tc1536, no bottleneck, `sigreg_weight` 0.005, decode-noise on.

**Caveat.** The arm moved `force_bottleneck` and `sigreg_warmup` together, so this read cannot assign the recon
damage to either alone.

**Wall clock.** 37.2 min/epoch against 40 h from 2026-09-10 19:56 puts the wall at 2026-09-12 11:56 and the arm at
~ep 63, not 100. On these numbers the chained resume §5 calls for is not worth submitting.
