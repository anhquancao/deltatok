# BSC tokenizer arms — compose convergence & sigreg-vs-tc recon regression

Created 2026-08-26. Q1 and Q2 answered 2026-08-27; the Q2 arms were cancelled at ep 43 once answered. Q3 launched 2026-08-27.

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

## Q3 — Does *raising* sigreg at tc512 raise rank and improve recon?

- **Jobs:** BSC:45106935 (`SIGREG_WEIGHT=0.01`) and BSC:45106990 (`SIGREG_WEIGHT=0.02`) — submitted 2026-08-27, `ehpc880`, 40 h each.
- **Script:** `slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm`, both with `COMPOSE_WEIGHT=1.0`.
- **Runs:** `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg{0.01,0.02}_ns1024_pool8192_compose1.0` (fresh — no prior ckpts).
- **Controls:** tc512 `sigreg0.005` (44759943, ep 67) and tc512 `sigreg0.002` (45055388, ep 43).
- **Axis:** 0.002 → 0.005 → 0.01 → 0.02, a 10× span at fixed tc512 with `ns1024` / `pool8192` / `warmup2000` held.

**Prediction from Q2.** If usable rank is the state variable and SIGReg builds it, the whole axis should be monotone: `ZPartRank`
up, `LossRecon` down. Q2 established the 0.002 → 0.005 leg (rank 28.5 → 32.9, recon 0.0660 → 0.0639). Two points above 0.005 turn
that single leg into a curve, so the shape is readable rather than just the sign.

**Falsifier, and why two points.** If 0.01 loses to 0.005, "more SIGReg is better" is wrong and 0.005 sits near an optimum. If 0.01
wins but 0.02 turns over, the optimum is bracketed between them. If both win, the knob is still on a slope at 4× the sweep value and
the ceiling is further out than tc512 can currently reach. Watch the high end for the warmup pathology measured on
2026-07-31 when `sigreg_pool_samples` was raised to 32768: the effective gradient scale `n_pooled/n_live` grew, the SIGReg statistic
spiked ~39× right at the end of the 2000-iter warmup, and training never recovered (Eval pinned at 0.12 vs 0.033 by ep 4). A 4× weight
is a different lever onto the same product `weight × ramp × scale`, so the same failure is available.

**Read at matched ep 42** against 44759943 / 45055388, same metrics as Q2: `LossRecon`, `LossRecon_Comp`, train `LossRecon`,
`ZPartRank`, `ZTotalVar`. The raw (unweighted) `SIGReg:` term from the epoch line doubles as the knob check.

## Tracking

| Job | Arm | State | Notes |
|---|---|---|---|
| BSC:45051916 | tc128 maxgap sigreg (non-compose) | **COMPLETED** ep99 | Q1 answered — compose ≠ recon technique |
| BSC:45055387 | tc256 compose sigreg**0.002** | **CANCELLED** ep43 | Q2 answered — sigreg is not the cause |
| BSC:45055388 | tc512 compose sigreg**0.002** | **CANCELLED** ep43 | Q2 answered — sigreg is not the cause |
| BSC:45106935 | tc512 compose sigreg**0.01** | PENDING | Q3 — does more sigreg help? |
| BSC:45106990 | tc512 compose sigreg**0.02** | PENDING | Q3 — brackets the optimum |

Logs: `slurm/output/train_deltatok_*_bsc_<jobid>.{out,err}`. TB mirror under `/mnt/d/tb_logs/.../<run>/tb_logs/`.
