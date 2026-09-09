# tc_width — decoder-noise detached decode at tc1536 (no channel bottleneck)

Created 2026-09-09 · thread `tc_width` · prior cycle:
[`../plan/2026-09-08_tc_width_decnoise_detach_tc512.md`](../plan/2026-09-08_tc_width_decnoise_detach_tc512.md)
· arm: `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.06_ns3072_pool8192_compose1.0_decnoise0.8_detach`
· control (tc512): `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_decnoise0.8_detach` (`BSC:45578805`)
· reference (tc1536, no noise): `deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.005_compose1.0` (`BSC:44590128`, ep 43)

No code change. One new slurm script, copied from the tc512 detach arm. Three knobs move with the width:
`target_channels` 512 → 1536, `sigreg_num_slices` 1024 → 3072 (2·Cz), `sigreg_weight` 0.02 → 0.06 (∝ Cz).
Pool stays 8192.

## 1 `slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm`

```bash
cp slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc512_bsc.slurm \
   slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm
```

Change only:

```
--job-name=deltatok_dn_dt1536_bsc
--account=ehpc880                                # the twin runs on ehpc1001; split the two allocations
--output/--error = slurm/output/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc_%j.{out,err}
header: no-bottleneck twin of BSC:45578805; tc 1536, ns 2*Cz, weight ∝ Cz + sbatch line with the new filename
export SIGREG_WEIGHT=${SIGREG_WEIGHT:-0.06}      # 3x the tc512 twin (weight ∝ Cz)
export SIGREG_NUM_SLICES=${SIGREG_NUM_SLICES:-3072}   # 2*Cz at tc1536
RUN_NAME=deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg${SIGREG_WEIGHT}_ns${SIGREG_NUM_SLICES}_pool${SIGREG_POOL_SAMPLES}_compose${COMPOSE_WEIGHT}_decnoise${DECNOISE_TAU}_detach
model.deltatok.target_channels=1536        # == hidden_size -> no z_proj_down/up built
model.deltatok.bottleneck_mlp=false        # inert at tc1536 (no bottleneck to make nonlinear)
```

Written, `bash -n` clean; the diff against the tc512 source is those ten lines and nothing else.

Everything else (compose 1.0, tau 0.8, decnoise warmup 2000, decnoise weight 1.0, eval ladder `[0.32,0.55,0.82]`,
pool 8192, sigreg warmup 2000, max_gap 9, bsize 2 / effective 16, lr 1e-3, grad_clip 0.1, 40 h,
`exit_before_time_limit`) is the tc512 twin's.

## 2 Pre-flight on BSC (user syncs), then submit

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -E \"RUN_NAME=|SIGREG_WEIGHT=|SIGREG_NUM_SLICES=|target_channels|--job-name|--output|--time|--account\" slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_sigreg_decnoise_detach_nozn_tc1536_bsc.slurm'"
```

Watch until the first `[KEpoch` line. The `.out` must show `sigreg_weight=0.06`, `sigreg_num_slices=3072`, no
`z_proj_down` / `z_proj_up` in the parameter table, `decode_noise_weight=1.0`, and `Train/LossDecNoise` > 0
once `DecodeNoiseTau` > 0. Chain at 40 h with `chain-slurm-jobs`, as the twin.

## 3 Early-stop guard (over-regularisation, the tc512 0.08 signature)

Read at ep 5 and ep 30 against the tc512 twin `BSC:45578805` at the same epoch. The 0.08 arm at tc512 showed
train recon 1.8× the plateau arms and a raw `SIGReg:` residual that never drops below theirs. If both hold here,
cancel and resubmit with `SIGREG_WEIGHT=0.02` under a fresh `RUN_NAME` (the weight enters the name, so no
`ckpts/` clash).
