# BSC tc512 — the sigreg-weight axis: 0.005 · 0.01 · 0.02 · 0.04 · 0.08

Created 2026-09-01 · thread `tc_width` · prior cycle: `2026-08-26_task_compose_convergence_sigreg_tc.md` (Q5) · arms: tc512 compose sigreg 0.02 / 0.04 / 0.08 · jobs: BSC:45296347 / 45296348 / 45296349 · deck: `2026-09-01_report_tc512_sigreg_weight_slides.html` (controls only, so far)

Created 2026-09-01. Splits Q5 out of `docs/research/tc_width/2026-08-26_task_compose_convergence_sigreg_tc.md`, which answered
Q1–Q4 and leaves this knob on a slope. Two of the five points are already measured; three are to submit.

## 1 Hypothesis

**Where does `sigreg_weight` turn over at tc512, and does the optimum scale with `Cz`?**

### The `weight ∝ Cz` hypothesis

tc512 · 0.01 ties tc256 · 0.005 and tc128 · 0.005 on eval recon (0.0416 / 0.0416 / 0.0414) at matched rank
(97.2 / 96.5 / 90.3). If the requirement really is `weight ∝ Cz`, tc512 needs **4× tc128's weight = 0.02**, and the
turnover lands there.

**This anchor is unverified.** 0.005 is the sweep's inherited value at tc128, never itself optimised, so the rule
predicts a ratio, not an absolute. A turnover at 0.02 supports it; one at 0.04 or later says the exponent is above 1
or the anchor was low. Either outcome calibrates the rule — that is the point of running three arms, not one.

**Falsifiers.**

- **Rank still rising and recon still falling at 0.08** → the regulariser remains under-weighted at tc512 and the
  open question moves to where `Cz` stops paying. The `∝ Cz` rule survives with an exponent above 1.
- **Rank peaks and recon turns over at 0.02** → that is the optimum and it matches `weight ∝ Cz` from the tc128
  anchor. Calibrate the rule and re-read the width sweep under it.
- **Turnover at 0.04** → the rule holds in shape but the tc128 anchor was low; the whole sweep was
  under-regularised, not just tc512.
- **Recon turns over while rank is still climbing** → rank is not sufficient. That contradicts r = −0.99 over six
  arms and would need its own arm to confirm before being believed.

**A divergent high arm is not a turnover.** On 2026-07-31, `sigreg_pool_samples=32768` grew the effective gradient
scale `n_pooled/n_live` 11.7 → 43.7, spiked the SIGReg statistic ~39× right at the end of the 2000-iter warmup, and
training never recovered (eval pinned at 0.12 vs 0.033 by ep 4). Weight is a different lever onto the same product
`weight × ramp × scale`, so 0.08 can reproduce it. The tell is the raw `SIGReg:` term in the epoch line around
iter 2000 — a **spike** there, not a slow climb. If that happens, relaunch with a longer `SIGREG_WARMUP` under a new
`RUN_NAME` and read the arm only after it is stable. See
`docs/research/sigreg/2026-07-31_findings_pool_not_weight_neutral.html`.

**Not doing.**

- **Re-running tc128 / tc256 at the higher weights.** The `∝ Cz` rule is worth calibrating on one width first; a
  second width doubles the cost before the first curve exists.
- **Touching `sigreg_num_slices` or `sigreg_pool_samples`.** Both enter the same effective-pressure product and
  would confound the weight axis. `pool8192` in particular is measured non-neutral.
- **`sigreg_compose_z`.** Q4 measured it at half the gain of simply doubling the weight. Off.
- **Resuming the 0.005 / 0.01 controls to ep 99 up front.** Only worth the walltime if the new arms land close
  enough at ep 67 that the ordering is in doubt.

## 2 Analysis

Q3 measured a 2× step, 0.005 → 0.01, and it improved *every* metric on *both* eval sets: eval `LossRecon`
−29.6% / −28.9%, `ZPartRank` 36.7 → 97.2 (2.6×). Q2 had already ruled out the opposite direction — halving to
0.002 made recon worse at tc256 *and* tc512. So the axis is monotone over 0.002 → 0.01 with **no upper bracket
measured**: the one arm that would have supplied it, BSC:45106990 at 0.02, was cancelled while still `PENDING`
(elapsed `00:00:00`, no logs, no run dir).

Across six arms eval recon tracks `ZPartRank` at **r = −0.997 (ep 42)** and **−0.993 (ep 67)**, while `Cz` vs rank
gives **r = −0.10**. Rank is the state variable; weight is the only knob that moves it. That is what this sweep
resolves.

### Controls, measured

Mean of both eval sets, from the cached BSC stdout. These are the numbers the new arms are read against.

| @ ep 67 | 0.005 | 0.01 |
|---|---|---|
| Eval `LossRecon` | 0.0589 | **0.0416** |
| Eval `LossRecon_Comp` | 0.0667 | **0.0449** |
| Train `LossRecon` | 0.0483 | **0.0345** |
| `ZPartRank` KITTI / nuScenes | 36.7 / 42.5 | **97.2 / 113.0** |
| raw `SIGReg:` | 0.0071 | 0.0040 |

At ep 42, where the 0.002 arm still exists: recon 0.0660 (0.002) · 0.0639 (0.005) · 0.0471 (0.01), rank KITTI
28.5 · 31.1 · 82.4. The axis is monotone in both, over a 5× weight range.

## 3 Solution

### Arms

Fixed at tc512, compose 1.0, `ns1024` / `pool8192` / `warmup2000` — everything except the weight is held.

| weight | job | state | epochs | source |
|---|---|---|---|---|
| 0.002 | BSC:45055388 | CANCELLED | 43 | Q2 lower anchor; off-axis on epochs, read at ep 42 only |
| **0.005** | BSC:44759943 | CANCELLED | 67 | width-sweep arm, the control |
| **0.01** | BSC:45106935 | COMPLETED | 67 | Q3 winner; `current.pth` on `$SCRATCH`, resumable |
| **0.02** | — | **TO SUBMIT** | — | the `weight ∝ Cz` prediction |
| **0.04** | — | **TO SUBMIT** | — | 8× the sweep value |
| **0.08** | — | **TO SUBMIT** | — | 16×; watch for the warmup spike |

- **Script:** `slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm` — **no edit needed**.
  `SIGREG_WEIGHT` is an env var with a `0.005` default and feeds both `training.sigreg_weight` and `RUN_NAME`.
  Account `ehpc880`, `acc` / `acc_ehpc`, `#SBATCH --time=40:00:00` in the file.
- **Runs:** `deltatok_l12_dtok64_tc512_nozn_maxgap9_vpt1to2_sigreg{0.02,0.04,0.08}_ns1024_pool8192_compose1.0` —
  fresh names, no existing `ckpts/`, so no accidental resume onto a changed objective.

### Launch

Pre-flight: the script is committed and unchanged, but the user syncs manually — confirm the cluster copy first.

```bash
ssh bsc "bash -lc 'grep -E \"account|--time|SIGREG_WEIGHT\" \
  /gpfs/projects/ehpc1001/code/deltatok/slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm'"
```

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  for w in 0.02 0.04 0.08; do \
    sbatch --account=ehpc1001 --time=48:00:00 --export=ALL,COMPOSE_WEIGHT=1.0,SIGREG_WEIGHT=\$w \
      slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc512_bsc.slurm; \
  done'"
```

Submitted 2026-09-01 as **BSC:45296347 / 45296348 / 45296349** (0.02 / 0.04 / 0.08). Account overridden to
`ehpc1001` on the command line, so the file's `--account=ehpc880` stays shared with the Q3 arm.

**Budget.** 48 h per arm, overridden on the command line so the file stays shared with the Q3 arm. The Q3 arm ran
39 h 51 m for 67 epochs = **35.7 min/ep**, so 48 h buys **~ep 80** — past the ep-67 read point with margin.
`training.exit_before_time_limit=true` is already set, so the wall split is checkpoint-safe and one chained resume
reaches ep 99. Walltime is absent from the BSC priority formula, so the only cost of asking 48 h over 12 h is queue
tail (measured p90 8.6 h vs 4 min; medians both ~3–12 min).

## 4 Results

_Pending. Read at matched ep 67, which the new arms pass about 45 h after start._

### How to read it

**Read at matched ep 67.** Both controls stopped there. The new arms pass it around 45 h, so the comparison needs no
resume; take ep 99 later from the chained halves if the ordering is close.

Same metrics as Q3: eval `LossRecon` / `LossRecon_Comp` / `LossRecon_AR`, the three `PredVsOrig` geometry losses,
train `LossRecon`, `ZPartRank`, `ZTotalVar`, and the raw `SIGReg:` term as the knob check. Report per eval set, not
pooled — KITTI and nuScenes have moved together on every arm so far, and a split would itself be the finding.

**Plot recon against `ZPartRank`, not against the weight.** Rank is the state variable and it is what predicts the
turnover; weight only sets it. A weight that raises rank and *fails* to improve recon would break the r = −0.99
relation and is the most informative single outcome available here.

The `PredVsGT` variants are teacher-ceiling-bound and will move less than `PredVsOrig`. Do not read a small
`PredVsGT` gap as the arms being close.

### Tracking

| Job | Arm | State | Notes |
|---|---|---|---|
| BSC:45055388 | tc512 compose sigreg**0.002** | **CANCELLED** ep43 | lower anchor, ep 42 read only |
| BSC:44759943 | tc512 compose sigreg**0.005** | **CANCELLED** ep67 | control |
| BSC:45106935 | tc512 compose sigreg**0.01** | **COMPLETED** ep67/100 | Q3 winner, resumable |
| BSC:45106990 | tc512 compose sigreg**0.02** | **NEVER RAN** | cancelled while PENDING, `00:00:00`, no logs — resubmit below |
| BSC:45296347 | tc512 compose sigreg**0.02** | **PENDING** | the `∝ Cz` prediction |
| BSC:45296348 | tc512 compose sigreg**0.04** | **PENDING** | 8× the sweep value |
| BSC:45296349 | tc512 compose sigreg**0.08** | **PENDING** | 16×; watch the warmup spike |

Logs: `slurm/output/train_deltatok_compose_sigreg_nozn_tc512_bsc_<jobid>.{out,err}`.
TB mirror: `/mnt/d/tb_logs/deltatok_log/<run>/tb_logs/`.
Prior decks: `docs/research/tc_width/2026-09-01_report_tc512_sigreg_weight_slides.html` (0.005 vs 0.01 vs sigregsum),
`docs/research/tc_width/2026-08-27_report_tc_sigreg_ab_slides.html` (0.002 vs 0.005).

## 5 Findings

_Pending. Write in the order of the falsifiers. If the tc128 anchor proves low, add `supersedes: 2026-08-16_report_tc_sweep_sigreg_slides.html`, since the width sweep would then have been under-regularised._

## → Next hypothesis

open — set by which falsifier lands. Turnover at 0.02: re-read the width sweep under `weight ∝ Cz`. Turnover at 0.04: re-run tc128 / tc256 at the calibrated weight. No turnover by 0.08: extend the axis.
