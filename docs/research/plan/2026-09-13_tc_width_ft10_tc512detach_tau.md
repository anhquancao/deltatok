# tc_width — ft10 recipe on the tc512 detached tokenizer at τ = 0.8 and τ = 1.6

Created 2026-09-13 · thread `tc_width` · prior cycle:
[`../results/2026-09-13_tc_width_zshaped_ladder_err_spectrum_read_slides.html`](../results/2026-09-13_tc_width_zshaped_ladder_err_spectrum_read_slides.html)
· source: `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8_detach/ckpts/epoch_100.pth` (`BSC:45665286`)
· recipe: ft10 `BSC:45421190` · flow read through: `iter_100000` (`BSC:45736861`)

Two 10-epoch decoder-only, noise-only finetunes from the same checkpoint; only `decode_noise_tau` differs.
The encoder is frozen, so the existing tc512 flow reads both without retraining.

## 1 `occrae/deltatok_trainer.py` — `clean_decode_weight`

Verbatim §1 of [`2026-09-13_tc_width_pool24576_ft20_flow_tc1536.md`](2026-09-13_tc_width_pool24576_ft20_flow_tc1536.md),
so that plan's step 2 needs no second edit.

**1a after line 792** `self._decode_noise_weight = ...`:

```python
        # Clean decode(s) weight; 0 with freeze_except_decoder = noise-only decoder finetune.
        self._clean_decode_weight = float(self.cfg.training.get("clean_decode_weight", 1.0))
        assert self._clean_decode_weight > 0 or bool(self.cfg.training.get("freeze_except_decoder", False)), \
            "clean_decode_weight=0 leaves a trainable encoder with no recon gradient"
```

**1b lines 796–798** startup print: append `f"clean_decode_weight={self._clean_decode_weight}"` (space-separated, as the
three fields before it).

**1c line 1094** `loss_total = loss + self._compose_weight * loss_compose` →
`loss_total = self._clean_decode_weight * (loss + self._compose_weight * loss_compose)`

**1d line 1131** `loss_total = loss` → `loss_total = self._clean_decode_weight * loss`

At the default 1.0 every existing arm is numerically identical.

## 2 `configs/deltatok/train_deltatok.yaml` after line 121 (`decode_noise_weight: 1.0`)

```yaml
  clean_decode_weight: 1.0  # weight of the clean decode(s); 0 = noise-only (needs freeze_except_decoder)
```

## 3 `slurm/deltatok/train_deltatok_compose_tc512detach_decnoise_ft10_bsc.slurm`

`cp slurm/deltatok/train_deltatok_compose_tc128_decnoise_ft_bsc.slurm slurm/deltatok/train_deltatok_compose_tc512detach_decnoise_ft10_bsc.slurm`, then:

| line | new |
|---|---|
| 2 | `#SBATCH --job-name=deltatok_dn512_ft10_bsc` |
| 10 | `#SBATCH --time=10:00:00` |
| 13 / 14 | `slurm/output/train_deltatok_compose_tc512detach_decnoise_ft10_bsc_%j.{out,err}` |
| 16–20 | 3-line header, below |
| after 22 (`export COMPOSE_WEIGHT=...`) | `export DECNOISE_TAU=${DECNOISE_TAU:-0.8}   # 0.8 or 1.6 (sigma^2 x4)` |
| 40 | `export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-1024}     # 2*Cz at tc512` |
| 45 | `export RUN_NAME=deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8_detach_ft10tau${DECNOISE_TAU}` |
| 48 | `export INIT_CKPT="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8_detach/ckpts/epoch_100.pth"` |
| 52 | `model.deltatok.target_channels=512         # < hidden_size -> linear z_proj_down/up bottleneck` |
| 57 | `model.deltatok.decode_noise_tau=${DECNOISE_TAU}   # decoder input z + sigma*N(0,I), sigma ~ U(0,tau)` |
| after 57 | `training.clean_decode_weight=0              # noise-only, as ft10 BSC:45421190` |
| after 57 | `"training.eval_noise_sigmas=[0.32,0.55,0.82]"  # token ladder, same grid as the source arm` |

Header:

```
# 10-epoch noise-only decoder finetune (ft10 recipe) of the detached tc512 arm's epoch_100; encoder frozen,
# so the tc512 flow iter_100000 stays valid. DECNOISE_TAU=0.8 (as ft10) or 1.6.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch --export=ALL,DECNOISE_TAU=0.8 slurm/deltatok/train_deltatok_compose_tc512detach_decnoise_ft10_bsc.slurm'"
```

Everything else stays as ft10: `freeze_except_decoder=true`, lr 1e-4 cosine to 1e-5, `warm_up=500`, 10 epochs /
11250 iters, `save_ckpt_every_n_epochs=5`, `grad_clip=0.1`, `sigreg_weight=0`, compose 1.0, max_gap 9, bsize 2 /
effective 16, 4 GPU, `ehpc1001`, no noise warmup. At tc512 `z_proj_up` trains with `decoder_blocks`
(`deltatok_trainer.py:784`). Wall: ft10 ran 30 min/epoch at tc128; the clean decodes still run forward and backward
at weight 0, so budget ≤ 40 min/epoch + evals ≈ 7.5 h.

## 4 Jobs

**Finetunes (together):**

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch --export=ALL,DECNOISE_TAU=0.8 slurm/deltatok/train_deltatok_compose_tc512detach_decnoise_ft10_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch --export=ALL,DECNOISE_TAU=1.6 slurm/deltatok/train_deltatok_compose_tc512detach_decnoise_ft10_bsc.slurm'"
```

**Reads, per arm, after `ls <run>/ckpts/epoch_10.pth`** (a time-limit exit returns 0, so `afterok` does not prove it).
All on `slurm/eval_deltatok_flow_zshaped_bsc.slurm`, `DELTATOK_CKPT=$SCRATCH/deltatok_log/<run>/ckpts/epoch_10.pth`,
`OUTPUT_DIR=results/deltatok_flow_ft10tc512/tau<τ>_<read>`:

| read | env |
|---|---|
| steps | `SHAPED=false NOISE_SIGMAS=" " NUM_STEPS=1,2,4,8,20` |
| shaped ladder | defaults (`SHAPED=true`, σ 0.32 / 0.55 / 0.82) |
| isotropic ladder | `SHAPED=false NOISE_SIGMAS=0.0,0.32,0.55,0.82` |

6 jobs, 1 GPU each. Baselines already measured: source step sweep (`BSC:45819224` / `45819762`), source shaped
(`BSC:45820899`) and isotropic (`BSC:45818711`) ladders, ft10 tc128 step sweep and ladders.

## 5 Pre-flight

1. `md5sum occrae/deltatok_trainer.py configs/deltatok/train_deltatok.yaml slurm/deltatok/train_deltatok_compose_tc512detach_decnoise_ft10_bsc.slurm` on BSC matches local before `sbatch`.
2. Startup print shows `decode_noise_tau=0.8` (or `1.6`) and `clean_decode_weight=0.0`; the param breakdown lists only
   `decoder_blocks` and `z_proj_up` as trainable; `INIT_CKPT` is the detached `epoch_100.pth` and no `current.pth`
   exists under either `RUN_NAME`.
3. The step-0 sanity eval reproduces the source arm's ep-100 KITTI / nuScenes `LossRecon` (BSC:45665286).
4. Watch both until the first `Training (Epoch 0)` loss line, then hand back the IDs.
5. Tell deltatok-5e that §1–§2 of its plan are implemented.
