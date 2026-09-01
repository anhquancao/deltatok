# BSC tokenizer arms — compose convergence & sigreg-vs-tc recon regression

Created 2026-08-26. Q1 and Q2 answered 2026-08-27; the Q2 arms were cancelled at ep 43 once answered. Q3 launched 2026-08-27,
answered 2026-09-01 — only the 0.01 half of it ran. Q4 (sigreg on the composed sum) added 2026-09-01. Q5 extends the weight axis.

## Q1 — Does the plain arm reach the compose floor if trained longer? — **ANSWERED: neither branch holds**

- **Job:** BSC:45051916 — **COMPLETED** 2026-08-27 12:40, exit 0:0, 24 h 42 m, `reached max iterations` at epoch 99 / iter 112,500.
- **Script:** `slurm/deltatok/train_deltatok_maxgap_sigreg_nozn_tc128_bsc.slurm` (account `ehpc1001`)
- **Run:** `deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192` (non-compose)
- **Compared against:** `..._compose1.0` (jobs 44635509 → 44846726), also 100 epochs.

**Result.** Compose is not a convergence speed-up on recon, and it does not lower the recon floor. The plain arm led
`LossRecon` at **every** epoch and finished ahead. What compose buys is `LossRecon_Comp`, and that gap *widens* with training
because the plain arm regresses on composition after epoch ~11.

| Metric @ ep 99 | plain | compose | Δ |
|---|---|---|---|
| Eval recon (mean of both sets) | **0.0389** | 0.0401 | compose +3.0 % |
| `LossRecon` KITTI / nuScenes | **0.0456 / 0.0322** | 0.0472 / 0.0329 | compose +3.5 % / +2.2 % |
| `LossRecon_Comp` KITTI / nuScenes | 0.1693 / 0.1256 | **0.0505 / 0.0360** | compose **3.35× / 3.49×** better |
| `LossRecon_AR` KITTI / nuScenes | **0.0470 / 0.0342** | 0.0509 / 0.0366 | compose +8.3 % / +7.0 % |
| Pointmap `PredVsOrig` KITTI / nuScenes | **0.538 / 0.478** | 0.621 / 0.493 | compose +15.5 % / +3.2 % |
| Train recon | 0.0331 | **0.0328** | tied — compose is not underfitting |
| `ZPartRank` /128 KITTI / nuScenes | 88.8 / 96.8 | 91.8 / 95.7 | matched — not a capacity story |

- **Plain composability decays.** `LossRecon_Comp` bottoms at ep 11 (KITTI 0.1238) / ep 12 (nuScenes 0.0854), then rises for
  88 straight epochs to 0.1693 / 0.1256 — **+36.8 % / +47.1 %** off its own best. Compose is monotone down.
- **Comp/Recon ratio @ ep 99:** plain 3.71× / 3.90×, compose 1.07× / 1.09× and flat since ep 11.
- **The old read was an artifact.** `docs/slide/deltatok_compose_vs_plain_slides.html` previously called 0.0401 the "compose
  floor, −3.8 % vs plain". That only held because plain had been stopped at ep 49. At equal budget the ordering reverses.

**Open:** compose is *worse* on `LossRecon_AR` despite owning `LossRecon_Comp` — composed ≠ autoregressive. And plain's
composability minimum at ep 11 suggests a short-schedule plain tokenizer may buy most of compose's composability for free; untested.

**Visual check (AR rollout).** Pulled the `eval_depth` panels straight out of both arms' TB event files at step 112,500
and cropped the AR columns plus the `GT token` reference (the teacher's own decode; identical across arms — max pixel
difference 12/255 on depth, ≤1/255 on RGB). At matched ep 99 on KITTI `08_003413` and nuScenes `scene-0017_000037`:
**both arms fall short of GT by far more than they differ from each other.** GT resolves nuScenes pedestrians as distinct
figures and holds building texture; both arms smear them and wash out RGB contrast. Compose reads marginally softer,
plain holds slightly more high-frequency detail, neither drifts. So the +8.3 % / +7.0 % `LossRecon_AR` gap is a
second-order softness penalty on top of a much larger shared shortfall — not a failure mode.

Deck: `docs/slide/deltatok_compose_vs_plain_slides.html` (8 slides, rewritten 2026-08-27; slides 05–06 are the AR visuals).

## Q2 — Is the higher-tc recon regression caused by the larger sigreg? — **ANSWERED: no**

- **Jobs:** BSC:45055387 (tc256), BSC:45055388 (tc512) — both **CANCELLED** 2026-08-27 13:48 at epoch 43 / 100, 25 h 24 m in.
  Stopped deliberately once Q2 was answered; `current.pth` is per-epoch, so a plain relaunch resumes from ep 43.
- **Scripts:** `slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc{256,512}_bsc.slurm` (account `ehpc880`)
- **Submitted:** `COMPOSE_WEIGHT=1.0 SIGREG_WEIGHT=0.002` — **not 0.005**, as an earlier version of this file claimed.
- **Runs:** `deltatok_l12_dtok64_tc{256,512}_nozn_maxgap9_vpt1to2_sigreg0.002_ns{512,1024}_pool8192_compose1.0`
- **Control:** the matching `sigreg0.005` tc256/tc512 compose arms already exist on `$SCRATCH/deltatok_log/` from the width
  sweep, so this is a clean sigreg-weight A/B at fixed tc.

**Setup:** `SIGREG_NUM_SLICES` scales with tc (2·Cz): tc256→ns512, tc512→ns1024.

**Question:** the tc-width sweep found wider tc converges *worse* on recon. Is that driven by the larger sigreg pressure at
higher tc (more slices, more channels held to N(0,I)) rather than the width itself?

**How to check:** compare `LossRecon` / `LossRecon_Comp` at matched epochs, sigreg0.002 vs sigreg0.005, within each tc.
If halving the weight recovers the recon gap → regression is sigreg-driven. If it does not → raw width.

**Result at matched epoch 42 — the hypothesis is refuted.** Halving the weight makes recon *worse* at both widths, and the
tc512-over-tc256 excess barely moves (+37.7% → +33.9%; 0.0175 → 0.0167 absolute). 95% of the width penalty survives, and the small
shrink comes from tc256 degrading *more* than tc512, not from tc512 improving.

| @ ep 42 · mean of both eval sets | tc256 0.005 | tc256 0.002 | tc512 0.005 | tc512 0.002 |
|---|---|---|---|---|
| Eval `LossRecon` | **0.0464** | 0.0493 (+6.1%) | **0.0639** | 0.0660 (+3.2%) |
| Eval `LossRecon_Comp` | **0.0502** | 0.0534 (+6.4%) | **0.0723** | 0.0752 (+4.1%) |
| Train `LossRecon` | **0.0391** | 0.0415 (+6.1%) | **0.0533** | 0.0559 (+4.9%) |
| `ZPartRank` | **86.0** | 78.0 (−9.3%) | **32.9** | 28.5 (−13.5%) |
| `ZTotalVar` | 309.0 | 340.4 (+10.2%) | 656.0 | 687.7 (+4.8%) |

- **The knob did move.** The `SIGReg:` term in the epoch line is the raw, unweighted statistic (`deltatok_trainer.py:1150`), so it reads
  how far z sits from N(0,I). At w=0.002 it runs **1.5–2.7× higher** through epoch 20 at both widths, converging back late (ep 42:
  tc512 still 1.2× above, tc256 level). The relaxation is real and lands exactly where the recon gap opens.
- **Cost in wall-clock.** Epochs to reach eval recon 0.070: tc256 13 → 19, tc512 31 → 36.
- **Mechanism: rank, not pressure.** Across the five A/B arms eval recon tracks the participation ratio at **r = −0.999**, and the
  relation holds on the independent six-width sweep at ep 67 (**r = −0.97**). Width itself predicts nothing: **Cz vs rank r = −0.10**,
  rank stays in a 37–97 band from Cz=64 to Cz=1024. Less SIGReg means fewer effective dims — it is what *builds* the rank recon lives
  on, not a tax on it.
- **tc512 is an outlier arm, not a "wide" arm.** KITTI ZPartRank at ep 67: Cz64 54.7 · Cz128 90.3 · Cz256 96.5 · **Cz512 36.7** ·
  Cz768 85.1 · Cz1024 67.9. tc512 falls behind every other width from epoch ~4 and never recovers, including versus arms with more
  channels. Its poor recon follows from its low rank; its low rank does **not** follow from its width. This matches the earlier
  width-sweep note that tc512 breaks monotonicity. Do not read the tc512 numbers here as a statement about wide bottlenecks.
- **Not a generalisation story.** Train recon is worse too, so SIGReg is not stealing capacity that recon would otherwise use.
- **Caveats.** Single seed, one weight step, epoch 43 of 100. The deficit is narrowing (tc256 peaked +15.8% at ep 15 → +6.1%; tc512
  +18.6% at ep 6 → +3.2%), so a very late crossover is not excluded, but the rank curves do not point at one. No tc128 · 0.002 twin.

**Next.** Run tc512 at sigreg **0.01**. If rank is the state variable, rank goes up and recon improves — that flips the sign of the
knob and is the strongest single check left. 45055387/45055388 were cancelled at ep 43 to free the walltime for it.

Deck: `docs/slide/deltatok_tc_sigreg_ab_slides.html` (7 slides, 2026-08-27).

## Q3 — Does *raising* sigreg at tc512 raise rank and improve recon? — **ANSWERED: yes, and it is the largest effect in the sweep**

- **Jobs:** BSC:45106935 (`SIGREG_WEIGHT=0.01`) — **COMPLETED** 2026-08-30 04:57, exit 0:0, 39 h 51 m, exited cleanly at
  epoch 67 / 100 for chain resume (`>> 9.2 min left in SLURM allocation < 36.5 min needed`). BSC:45106990
  (`SIGREG_WEIGHT=0.02`) — **never ran**: cancelled 15 min after submit while still `PENDING`, `Start=None`,
  elapsed `00:00:00`, no `slurm/output/*45106990.*` files, no run directory on `$SCRATCH`. Q3 is answered on one point, not two.
- **Script:** `slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm`, `COMPOSE_WEIGHT=1.0`, `ehpc880`, 40 h.
- **Run:** `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg0.01_ns1024_pool8192_compose1.0`
- **Controls:** tc512 `sigreg0.005` (44759943, ep 67) and tc512 `sigreg0.002` (45055388, ep 43).

**Result.** Doubling the weight improves *every* metric on *both* eval sets. The prediction from Q2 holds exactly: rank up,
recon down, and the two move together.

| @ ep 67 · KITTI / nuScenes | sigreg 0.005 | **sigreg 0.01** | Δ |
|---|---|---|---|
| Eval `LossRecon` | .0700 / .0478 | **.0493 / .0340** | **−29.6% / −28.9%** |
| Eval `LossRecon_Comp` | .0787 / .0547 | **.0528 / .0370** | −32.9% / −32.4% |
| Eval `LossRecon_AR` | .0784 / .0551 | **.0529 / .0377** | −32.5% / −31.6% |
| Pointmap `PredVsOrig` | 1.173 / 0.903 | **0.658 / 0.491** | −43.9% / −45.6% |
| Depth `PredVsOrig` | 0.700 / 0.608 | **0.416 / 0.357** | −40.5% / −41.2% |
| Raymap `PredVsOrig` | 0.696 / 0.466 | **0.404 / 0.244** | −41.9% / −47.5% |
| `ZPartRank` /512 | 36.7 / 42.5 | **97.2 / 113.0** | **2.6×** |
| `ZTotalVar` | 744 / 471 | 706 / 536 | −5% / +14% |
| Train `LossRecon` | 0.0483 | **0.0345** | −28.6% |

- **Separation is early and clean.** The 0.01 arm is ahead by epoch 2 (0.1100 vs 0.1173) and never crosses back. No warmup
  pathology — the raw `SIGReg:` term falls smoothly, no spike at the end of the 2000-iter ramp.
- **Rank is confirmed as the state variable.** Over six arms (four weights at tc512, plus tc256 and tc128 at 0.005),
  eval recon vs `ZPartRank` gives **r = −0.997 at ep 42** and **−0.993 at ep 67**; the same holds per-dataset on
  `LossRecon_Comp` (−0.98) and on all three geometry losses (−0.94 to −0.99).
- **The width penalty is gone.** tc512·0.01 ties the narrow arms at ep 67: eval recon 0.0416 vs tc256 0.0416 vs tc128 0.0414,
  ranks 97.2 / 96.5 / 90.3. On nuScenes geometry it *beats* tc128 (pointmap 0.491 vs 0.535, depth 0.357 vs 0.391, raymap
  0.244 vs 0.258). **tc512 was never a bad width — it was under-regularised for its width.** The Q2 note calling it "an
  outlier arm" and the width-sweep "wider is worse" read are both superseded: `sigreg_weight` has to scale with `Cz`.
- **AR rollout, visually.** Cropped the `Pred Depth (AR)` / `Pred RGB (AR)` columns plus the `GT token` reference out of all
  three arms' TB panels at step 76,500, forecast frames t1–t4 only (`context_views` = 1 KITTI / 2 nuScenes, so only t0 is
  context). On nuScenes `scene-0017_000037` the 0.005 arm fuses the pedestrian group into blobs and loses the far lane;
  0.01 separates the figures and recovers the lane markings. Neither reaches GT — the shared shortfall to the teacher is
  still larger than the gap between arms. GT columns are byte-identical across arms (md5 match), which checks the crop offsets.

**Still open:** the axis is monotone over 0.002 → 0.01 with **no upper turnover measured**, because 0.02 never ran. See Q5.
The winner also stopped at ep 67 of 100 while still improving (0.0435 → 0.0416 over the last 12 epochs, rank still climbing);
`current.pth` is on `$SCRATCH`, so a plain relaunch resumes it.

Deck: `docs/slide/deltatok_tc512_sigreg_weight_slides.html` (9 slides; 04–05 are the AR visuals).

## Q4 — Does SIGReg on the composed sum substitute for a bigger weight? — **ANSWERED: it helps, but less**

- **Job:** BSC:45122721 — **COMPLETED** 2026-08-31 01:08, exit 0:0, 2 d 10 h 47 m, all 100 epochs.
- **Script:** `slurm/deltatok/train_deltatok_compose_sigregsum_nozn_tc512_bsc.slurm` (`ehpc880`, 72 h), design in
  `docs/plans/sigreg_compose_z.md`, code in `1140de1` (`training.sigreg_compose_z=true`, `deltatok_trainer.py:644,960,1057`).
- **Run:** `..._sigreg0.005_ns1024_pool8192_compose1.0_sigregsum` — same 0.005 weight, but SIGReg sees `z_a`, `z_b` **and**
  `z_a+z_b` in one pooled CF test (train z rows 4.99 M vs 1.66 M, confirming the 3-way cat).

**Result at matched ep 67.** It lands between the two weights on every axis, ordered by rank exactly as Q3 predicts:

| @ ep 67 · mean of both sets | 0.005 | **sum 0.005** | 0.01 |
|---|---|---|---|
| Eval `LossRecon` | 0.0589 | 0.0525 (−10.9%) | **0.0416** (−29.4%) |
| Eval `LossRecon_Comp` | 0.0667 | 0.0577 | **0.0449** |
| `ZPartRank` KITTI | 36.7 | 60.1 | **97.2** |
| raw `SIGReg:` | 0.0071 | 0.0052 | **0.0040** |

- **No composability win beyond the rank it buys.** Comp/Recon ratio 1.132 (0.005) → 1.099 (sum) → 1.079 (0.01). The sum
  arm sits ~1% below the 0.005↔0.01 interpolation at its own rank — inside noise on one seed.
- **The plan's "hop-scale shrink" outcome is what happened**, not decorrelation: eval `ZRowMeanSquare` on the hops drops
  1.455 → 1.216 and `ZTotalVar` 744 → 622. No sign of hops going below unit, so no scale-splitting.
- **Read the eval-side z stats only.** The sum arm's *train* `ZPartRank` is a hop+sum mixture; eval rows are identical
  (53,248 / 106,496) across all arms, so the eval numbers are like-for-like.
- **Verdict:** cheaper to just raise `sigreg_weight`. Keep `sigreg_compose_z` off unless a later question needs the sum pinned.

## Q5 — Where does the weight axis turn over? — **MOVED**

Split into its own tracker: `docs/task/bsc_sigreg_weight_sweep_tc512_01-09-2026.md`, which carries the full
0.005 · 0.01 · 0.02 · 0.04 · 0.08 axis, the `weight ∝ Cz` hypothesis and the read protocol. Summary below.

Q3 leaves the knob on a slope at 2× the sweep value with no upper bracket. Three arms extend it, all at fixed tc512:

- **Weights:** `0.02`, `0.04`, `0.08` — 2×, 4× and 8× the sweep value, continuing the 0.002 → 0.005 → 0.01 axis by ~2× steps.
- **Script:** `slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm` — **no edit needed**, `SIGREG_WEIGHT` is an
  env var and feeds `RUN_NAME`. `ns1024` / `pool8192` / `warmup2000` / `COMPOSE_WEIGHT=1.0` held, as in Q3.

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  for w in 0.02 0.04 0.08; do \
    sbatch --time=48:00:00 --export=ALL,COMPOSE_WEIGHT=1.0,SIGREG_WEIGHT=\$w \
      slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm; \
  done'"
```

- **Runs:** `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg{0.02,0.04,0.08}_ns1024_pool8192_compose1.0` (fresh, no ckpts).
- **Controls:** the Q3 axis — 0.002 (45055388, ep 43), 0.005 (44759943, ep 67), 0.01 (45106935, ep 67).
- **Budget:** 48 h per arm, overriding the script's `#SBATCH --time=40:00:00` on the command line so the file stays shared
  with the Q3 arm. At the measured ~35.6 min/ep that buys ~ep 80, past the ep-67 read; chain one resume to reach ep 99.
  `training.exit_before_time_limit=true` is already set, so the split is checkpoint-safe. The cost of 48 h over a shorter
  request is queue tail only — walltime is absent from the BSC priority formula, and measured p90 wait was 8.6 h at 48 h
  against 4 min at 12 h, medians both ~3–12 min.

**Read at matched ep 67** against the three controls, same metrics as Q3: eval `LossRecon` / `LossRecon_Comp` / `LossRecon_AR`,
the three `PredVsOrig` geometry losses, train `LossRecon`, `ZPartRank`, `ZTotalVar`, and the raw `SIGReg:` term as the knob check.
Plot recon against `ZPartRank`, not against the weight — rank is the state variable, and it is what predicts the turnover.

**Falsifier.** If rank keeps rising and recon keeps falling to 0.08, the regulariser is still under-weighted at tc512 and the
question becomes where `Cz` stops paying. If rank peaks and recon turns over at 0.02 or 0.04, that point is the optimum and the
`weight ∝ Cz` rule can be calibrated from it. If a high arm diverges outright, that is the warmup pathology, not a turnover —
see below.

**Watch the high end.** On 2026-07-31, raising `sigreg_pool_samples` to 32768 grew the effective gradient scale `n_pooled/n_live`,
spiked the SIGReg statistic ~39× right at the end of the 2000-iter warmup, and training never recovered (Eval pinned at 0.12 vs
0.033 by ep 4). Weight is a different lever onto the same product `weight × ramp × scale`, so 0.08 can reproduce it. The tell is
the raw `SIGReg:` term in the epoch line around iter 2000 — a spike there, not a slow climb, means the ramp is the problem and the
arm should be relaunched with a longer `SIGREG_WARMUP` rather than read as a turnover.

## Tracking

| Job | Arm | State | Notes |
|---|---|---|---|
| BSC:45051916 | tc128 maxgap sigreg (non-compose) | **COMPLETED** ep99 | Q1 answered — compose ≠ recon technique |
| BSC:45055387 | tc256 compose sigreg**0.002** | **CANCELLED** ep43 | Q2 answered — sigreg is not the cause |
| BSC:45055388 | tc512 compose sigreg**0.002** | **CANCELLED** ep43 | Q2 answered — sigreg is not the cause |
| BSC:45106935 | tc512 compose sigreg**0.01** | **COMPLETED** ep67/100 | Q3 answered — −29.6%/−28.9% recon, rank 2.6×; resumable |
| BSC:45106990 | tc512 compose sigreg**0.02** | **NEVER RAN** | cancelled while PENDING, `00:00:00`, no logs, no run dir → Q5 |
| BSC:45122721 | tc512 compose sigreg0.005 **sigregsum** | **COMPLETED** ep99 | Q4 answered — half the gain of doubling the weight |
| — | tc512 compose sigreg**0.02** | **TO SUBMIT** | Q5 — first point above 0.01 |
| — | tc512 compose sigreg**0.04** | **TO SUBMIT** | Q5 — 4× the sweep value |
| — | tc512 compose sigreg**0.08** | **TO SUBMIT** | Q5 — 8×; watch for the warmup spike |

Logs: `slurm/output/train_deltatok_*_bsc_<jobid>.{out,err}`. TB mirror under `/mnt/d/tb_logs/.../<run>/tb_logs/`.
Decks: `docs/slide/deltatok_compose_vs_plain_slides.html` (Q1), `docs/slide/deltatok_tc_sigreg_ab_slides.html` (Q2),
`docs/slide/deltatok_tc512_sigreg_weight_slides.html` (Q3/Q4).
