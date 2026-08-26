# BSC tokenizer arms — compose convergence & sigreg-vs-tc recon regression

Created 2026-08-26. All three jobs were PENDING on BSC (`acc` / `acc_ehpc`, 40 h) at creation.

## Q1 — Does the plain arm reach the compose floor if trained longer?

- **Job:** BSC:45051916
- **Script:** `slurm/deltatok/train_deltatok_maxgap_sigreg_nozn_tc128_bsc.slurm` (account `ehpc1001`)
- **Run:** `deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg<w>_ns256_pool<...>` (non-compose)

**Question:** train the non-compose tc128 arm longer — does its recon converge to the same value the compose arm reaches? I.e. is compose only a convergence speed-up, or does it move the recon floor?

**How to check:** compare `LossRecon` / `LossRecon_Comp` at matched epochs against the tc128 **compose** arm. If the plain arm keeps closing the gap and flattens at the compose value → compose only accelerates. If it plateaus above → compose buys a genuinely better floor.

## Q2 — Is the higher-tc recon regression caused by the larger sigreg?

- **Jobs:** BSC:45055387 (tc256), BSC:45055388 (tc512)
- **Scripts:** `slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc{256,512}_bsc.slurm` (account `ehpc880`)
- **Submit:** `COMPOSE_WEIGHT=1.0 SIGREG_WEIGHT=0.005`
- **Runs:** `deltatok_l12_dtok64_tc{256,512}_nozn_maxgap9_vpt1to2_sigreg0.005_ns{512,1024}_pool<...>_compose1.0`

**Setup:** `SIGREG_NUM_SLICES` scales with tc (2·Cz): tc256→ns512, tc512→ns1024, sigreg_weight fixed at 0.005.

**Question:** the [tc-width sweep](../../.claude) found wider tc converges *worse* on recon. Is that regression driven by the larger sigreg pressure at higher tc (more slices, more channels held to N(0,I)) rather than the width itself?

**How to check:** compare recon (`LossRecon`, `LossRecon_Comp`) tc256 vs tc512 at matched epochs. If recon degrades with tc in step with ns/sigreg scaling → regression is sigreg-driven, not raw width. Cross-check against a fixed-ns or lower-sigreg control if the signal is ambiguous.

## Tracking

| Job | Arm | State | Notes |
|---|---|---|---|
| BSC:45051916 | tc128 maxgap sigreg (non-compose) | PENDING | Q1 |
| BSC:45055387 | tc256 compose sigreg0.005 | PENDING | Q2 |
| BSC:45055388 | tc512 compose sigreg0.005 | PENDING | Q2 |

Logs: `slurm/output/train_deltatok_*_bsc_<jobid>.{out,err}`. TB mirror under `/mnt/d/tb_logs/.../<run>/tb_logs/`.
