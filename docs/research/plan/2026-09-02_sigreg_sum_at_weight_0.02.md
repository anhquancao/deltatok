# Does SIGReg-on-the-sum still win at weight 0.02?

Created 2026-09-02 · thread `sigreg` · prior cycle: `2026-08-27_sigreg_compose_z.md` · arm: tc512 compose 1.0
`sigreg_compose_z=true` @ `sigreg_weight=0.02` · job: BSC:45345063 · twin: BSC:45296347 (plain, same weight)

## 1 Hypothesis

**The sum arm beat its plain twin at 0.005. Does the win survive at 0.02, where the plain weight axis has
already flattened?**

At equal nominal weight the sum arm carries only ~1/3 the effective SIGReg pressure — 3× live rows at a fixed
pool 8192 drops `scale = (P+L)/L` from ~17 to ~6.3 (`2026-08-27_sigreg_compose_z.md`, "Pool sizing"). It won
anyway. Two readings, and this arm separates them:

- **Sum is a better statistic.** Regularising `z_a`, `z_b` and `z_a+z_b` as one population is the right
  objective, so the win holds at any weight and 0.02 beats plain 0.02.
- **Sum is only buying effective weight.** The gain is the row count, not the streams, and it collapses once
  the plain arm is regularised hard enough. Plain 0.02 then ties or beats sum 0.02.

**Falsifiers.**

- **Sum 0.02 beats plain 0.02 on eval `LossRecon_Comp`** → the streams matter, not the pressure. Sum becomes
  the default and the weight axis is re-read on top of it.
- **Sum 0.02 ties or loses to plain 0.02** → the 0.005 win was an effective-weight artifact. Drop the flag.
- **Sum 0.02 loses to sum 0.005** → the sum arm has its own turnover below 0.02, earlier than the plain one.

**Not doing.**

- **Sum at 0.01.** Undecided. Worth it only if 0.02 lands close enough to plain 0.02 that the bracket matters.
- **Raising `sigreg_pool_samples` to compensate the 3× row count.** It would restore `scale` to ~17 and make
  the arms pressure-matched, but pool is measured non-neutral (2026-07-31) and moving two knobs at once
  confounds the read.

## 2 Analysis

Mean of both eval sets, from the cached BSC stdout. `rank` is eval `ZPartRank` KITTI / nuScenes.

| ep 67 | plain 0.005 | **sum 0.005** | plain 0.01 |
|---|---|---|---|
| Eval `LossRecon` | 0.0589 | **0.0525** | 0.0416 |
| Eval `LossRecon_Comp` | 0.0667 | **0.0577** | 0.0449 |
| `ZPartRank` | 36.7 / 42.5 | **60.1 / 67.7** | 97.2 / 113.0 |

At its own weight the sum arm is **10.9% better on recon and 13.5% on comp**, and it lifts rank 1.6× — while
running at ~1/3 the effective pressure. It does not reach plain 0.01, which is the "half the gain of doubling
the weight" verdict already in the ledger. Sum 0.005 ran to ep 99 (recon 0.0505, comp 0.0556, rank 64.1 / 70.9).

**The plain axis is flattening at 0.02.** At the matched ep 33 the running twin sits at recon 0.0508 / comp
0.0551 / rank 75.6 · 86.1, against 0.0509 / 0.0551 / 73.9 · 85.6 for plain 0.01 — indistinguishable. If that
holds to ep 67, the plain weight optimum is at or below 0.02 and this is the right weight to test the sum at.
The 0.08 arm agrees from the other side: at ep 32 it is ~2× worse on every loss term with a *lower* rank
(38.7 · 54.3), the shape of over-regularisation (`../../TODO.md`, TODO 2 note).

## 3 Solution

One arm. Everything except `sigreg_compose_z` is held at the twin's values: tc512, compose 1.0, `ns1024`,
`pool8192`, `warmup2000`, `max_gap=9`, `bsize=2`.

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  sbatch --account=ehpc1001 --time=48:00:00 --export=ALL,COMPOSE_WEIGHT=1.0,SIGREG_WEIGHT=0.02 \
    slurm/deltatok/train_deltatok_compose_sigregsum_nozn_tc512_bsc.slurm'"
```

Submitted 2026-09-02 as **BSC:45345063**. Script unchanged (cluster md5 verified against local), account
overridden to `ehpc1001` so the file's `ehpc880` stays shared. Run
`deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.02_ns1024_pool8192_compose1.0_sigregsum` — no existing
`ckpts/`, verified before launch.

**Budget.** Sum 0.005 ran 100 epochs in 58 h = **35 min/ep**, so 48 h buys ~ep 82. The plain 0.02 twin runs at
41 min/ep under the same 48 h limit and reaches ~ep 70. Matched read at ep 67 needs no resume from either.

## 4 Results

_Pending._

### How to read it

**Primary: eval `LossRecon_Comp` against BSC:45296347 at matched ep 67**, per eval set, not pooled. Secondary:
eval `LossRecon`, the three `PredVsOrig` geometry losses, `ZPartRank`, `ZTotalVar`.

The stdout `SIGReg:` scalar is **not comparable** across the flag — one third of the rows are exact linear
combinations of the other two thirds, so the finite-sample floor sits higher. Judge on recon.

Watch `Train/ZTotalVar` drifting below 1 with recon flat: the code shrinking rather than the streams merging.
Under `z_norm=false` a shrink migrates into decoder weights.

### Tracking

| Job | Arm | State | Notes |
|---|---|---|---|
| BSC:44759943 | tc512 plain 0.005 | CANCELLED ep67 | weight control |
| BSC:45122721 | tc512 **sum** 0.005 | COMPLETED ep99 | the arm that wins at equal weight |
| BSC:45106935 | tc512 plain 0.01 | COMPLETED ep67 | Q3 winner |
| BSC:45296347 | tc512 plain 0.02 | RUNNING | **the twin for this read** |
| BSC:45297731 | tc512 plain 0.04 | RUNNING | weight axis, resubmit of the FAILED 45296348 |
| BSC:45345063 | tc512 **sum** 0.02 | SUBMITTED | this arm |

Logs: `slurm/output/train_deltatok_compose_sigregsum_nozn_tc512_bsc_45345063.{out,err}`.
TB mirror: `/mnt/d/tb_logs/deltatok_log/<run>/tb_logs/`.

## 5 Findings

_Pending. Write in the order of the falsifiers._

## → Next hypothesis

open — if sum 0.02 wins, re-read the whole tc512 weight axis with the flag on; if it ties, the flag is an
effective-weight proxy and the pool knob is the honest lever.
