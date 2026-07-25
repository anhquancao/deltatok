# SIGReg arm failed to learn — diagnosis and fix

Status as of 2026-07-25. The fix is **fully applied** in the working tree (uncommitted, branch
`revert`) — first pass plus every reviewed refinement — and re-validated. Next action is a **sync +
submit**.

## TL;DR

Job `Jeanzay:168363` (`jz_train_deltatok_multitoken_mlp_sigreg.slurm`, SIGReg weight 1.0) never
learned and was **cancelled** after 05:44:25. The cause was the SIGReg *estimator*, not the weight
and not the LayerNorm: it saw only 128–256 samples of dim 768 per call, where its own
finite-sample floor buries the collapse signal it exists to detect.

## Evidence it wasn't learning

| | baseline `162260` (`_mlp`, no sigreg) | sigreg `168363` |
|---|---|---|
| `Train/LossTot` @ ~8.5k steps | **0.078** ↓ | 0.115 |
| `Train/LossTot` @ 21.5k steps | — | 0.121 (flat since step ~700) |
| KITTI eval | **0.1267** (ep6, falling) | 0.1387 (ep18, flat) |
| nuScenes eval | **0.0928** (ep6, falling) | 0.1086 (ep18, **rising**) |

The baseline at epoch 6 beat the sigreg arm at epoch 18 on both evals. `Train/LossSIGReg` was
pinned at **0.0088 ± 0.0005 from step 716 to 21504** — dead flat for 21k steps.

## Root cause

`z` reaching SIGReg is `(M, N, K, Cz)` → `S = M·N·K` rows of dim `Cz=768`:

- `M = 2` — `bsize=2 × pairs_per_seq=1` (the subsample at `deltatok_trainer.py:886-895`
  cuts `B*(T-1)=10` rows to `B*1=2`)
- `N = 1..2` — `min_views_per_timestep=1, max_views_per_timestep=2`, varies per batch
- `K = 64` (`num_delta_tokens`)

→ **S = 128 or 256**, i.e. at most a third of the 768 dimensions. Confirmed against the log line
`effective_bsize=16 = 4 rank(s) x bsize 2 x pairs_per_seq 1 x grad_cum 2`.

Measured consequences (probes, fp32, `num_slices=256`):

| | value |
|---|---|
| floor at S=128 (perfect isotropic Gaussian) | **0.00845 ± 0.00037** |
| observed plateau | **0.0088** → **1.04× the floor**, i.e. pure estimator bias |
| rank-290 collapse at S=128 | 0.00867 — a **+0.00022** signal against **±0.00037** noise |
| gradient reproducibility at S=128 | two independent slice draws on the *same* z: **cos = 0.08** |
| `\|g_sigreg\| / \|g_recon\|` into z | **0.29×** (measured against a synthetic recon loss of 0.64; the real recon is ~0.11, so the true share is larger) |

So SIGReg was injecting ~30% of the gradient as ~92% resampling noise, through a tight
`grad_clip=0.1` that rescales the *combined* gradient — the recon component paid for it.

### Two hypotheses that were ruled out

- **LayerNorm** (`self.norm`, `deltatok_trainer.py:177`, non-affine because `norm_affine=false`;
  `encode()` returns `self.norm(z)`, which is what `return_z` hands to SIGReg). It projects out
  only **0.2%** of the SIGReg gradient and changes the statistic by <4% (0.00395 → 0.00410).
- **Weight too big.** 0.0088 is ~8% of the 0.114 total. Relative to the reference the pressure is
  ~200× too *small* (see below), not too large.

## Reference comparison (`/home/acao/code/lejepa`)

| | LeJEPA reference | run `168363` |
|---|---|---|
| samples for the statistic | `batch 512 × n_views 8` = **4096** | **128–256** |
| feature dim `D` | 512 | 768 |
| **N/D** | **8** | **0.17–0.33** (~30–48× worse) |
| `num_slices` | **1000–4096** (≈2–8× D) | **256** (0.33× D) |
| weight | `bstat_lambda` **0.01–0.1** (settled 0.05) | **1.0** |
| `× N` scaling | **kept** (`* proj.size(-2)`; `* N * world_size` in the package) | **dropped** |
| cross-rank ECF reduction | **yes** — differentiable AVG of `cos_mean`/`sin_mean` (`lejepa/univariate/epps_pulley.py:88-90`) | **no** |
| synchronized slice directions | **yes** — seeded generator + `all_reduce(MAX)` (`lejepa/multivariate/slicing.py:130-137`) | **no** |
| `clip_value` (zero below-floor slices) | available | absent |

Because our reimpl drops `× N`, the reference's effective coefficient on the *raw* statistic is
`lamb × N = 0.05 × 4096 ≈ 205` against a main-objective weight of `0.95`. Ours is `1.0` against
recon's `1.0`. **`SIGREG_WEIGHT` is a knob to raise, not lower**, once the estimator is trustworthy.

## Chosen fix: "estimator only"

Deliberately keeping `SIGREG_WEIGHT=1.0` and the dropped `× N` scaling, so the estimator is the
only variable vs `168363`. Not adopting `clip_value` or the reference's `lamb` convention yet.

### Applied in the working tree

1. `occrae/sigreg.py` — slice directions from a seeded `torch.Generator` so all ranks project onto
   the **same** directions, then `cos_mean`/`sin_mean` pooled across ranks via
   `torch.distributed.nn.all_reduce` (autograd-aware) SUM/`world_size`. Docstring updated.
2. `occrae/deltatok_trainer.py` — `_sigreg_pooled()` banks each micro-batch's detached rows in
   `_sigreg_zbuf` and runs the estimator **only on the accum window's last micro-batch**, over all
   of them. Cleared at each optimizer step and at epoch start.
3. `slurm/jz_train_deltatok_multitoken_mlp_sigreg.slurm` — `SIGREG_NUM_SLICES` 256 → **2048**
   (≥2·Cz, since directions no longer vary per rank); run-name tag gains `_pool` so it cannot
   resume `168363`'s stuck checkpoint. Header comment rewritten with the diagnosis + WATCH criteria.

Refinements from the Fable code review, all applied:

- **F1 — pooling was asymmetric.** The *first* micro-batch of each window saw an empty
  `_sigreg_zbuf`, so it got `world_size·S` (4×), not 8×. Now SIGReg runs once per window at weight
  `sigreg_weight * grad_cum`; earlier micro-batches only bank detached rows. Uniformly 8×-pooled,
  halves SIGReg compute, less code. Scaling: one call at `w·grad_cum` through
  `(loss_total/grad_cum).backward()` equals `grad_cum` calls at `w`. `Train/LossTot` is unchanged
  in expectation, so it stays directly comparable to `168363`.
- **F2 — `step` buffer and its `all_reduce(MAX)` dropped; `forward(z, seed)` now takes
  `cfg.training.iter`.** A plain int (no collective, no GPU→host sync), already checkpointed
  (`deltatok_trainer.py:766`, so directions no longer replay from seed 0 on resume), and
  rank-synchronized by construction. It also was never persisted as a buffer: only the *tokenizer*
  state dict is saved, so the old `step` reset to 0 on every resume.
  **Do not seed from the epoch**: directions frozen for ~2250 updates lets the optimizer satisfy
  2048 fixed slices while drifting non-Gaussian in unmeasured directions. That failure mode looks
  like success — `LossSIGReg` falls while the code stays collapsed.
- **F3** — `_sigreg_zbuf` now cleared at epoch start too. Dormant today (2250 micro-batches/epoch
  is divisible by `grad_cum=2`) but live on any dataset-size or `per_dataset_sampling` change.
- **F5** — documented in `sigreg.py`: the CF all-reduce is an *equal-weight rank average*, not the
  true pooled mean when `S` differs across ranks (N=1 vs 2). Unbiased, ≤~11% effective-sample loss,
  and the same weighting DDP already applies to the recon loss.
- **F6a** — the inert `torch.no_grad()` around `_directions` dropped (`randn` builds no graph).
- The **`no_sync()` trap** is now named in a comment at the all-reduce.

### Validated (4-process gloo, 4×128 samples of dim 768, stdin-piped copy of the new class)

Re-run after the refinements (`test_sigreg_pool2.py`):

- all ranks return the identical value (spread exactly 0) → directions synced and CF reduced ✅
- that value is **bit-identical** to a single-process run over the concatenated 512 rows ✅
- gradients reach every rank's own samples ✅
- **`|g_ddp| / |g_true-pooled| = 1.000000`, cos = 1.0** — the 1/W cancellation is real, and it is
  *not* the naive argument. `torch.distributed.nn.all_reduce`'s backward all-reduces the incoming
  gradient, so each rank's local grad is already W× the partial through its own rows; DDP's 1/W
  averaging cancels it. A plain per-rank mean loss measures 1.000000 in the same harness (control).
  This is exactly what the `no_sync()` comment protects. ✅
- seed advance changes the statistic (0.0023939 → 0.0022875 on fixed z) ✅
- floor drops with pooled S: `0.008166 (128) → 0.002055 (512) → 0.001029 (1024) → 0.000521 (2048)`,
  matching the earlier probe ✅
- `_sigreg_pooled` lifecycle: every estimator call gets exactly `grad_cum·S` rows (F1 symmetry),
  and a short final window leaves banked rows that the epoch-start clear removes (F3) ✅
- rank-290 discrimination (from the earlier probe, unchanged): **1.05×** the floor at S=128 →
  **2.02×** at S=2048 ✅

Both edited files syntax-check clean under the JZ python.

### Rejected from the review

**F1(b) — "the estimator bias actively rewards collapse."** Does not hold in this regime. The
`(1/S)(1−|φ_p|²)` variance term does favor concentration in isolation, but it is dominated by the
bias term's penalty on non-Gaussianity. Measured statistic vs code rank, monotonically increasing
at both sample counts:

| rank | S=128 | S=2048 |
|---|---|---|
| 768 (full) | 0.00836 | 0.00050 |
| 290 | 0.00876 (1.05×) | 0.00102 (2.02×) |
| 64 | 0.01259 (1.51×) | 0.00403 (8.0×) |
| 8 | 0.04240 (5.07×) | 0.03389 (67×) |
| 1 | 0.32714 (39×) | 0.31756 (631×) |

Collapse is penalized at every rank. What small S destroys is *discrimination*, not the sign.

## Next steps

1. ~~Sync to Jean Zay~~ — done (md5s match; the new `no_sync()` / `ready=update_grad` /
   `SIGREG_NUM_SLICES=2048` lines were confirmed present in `$TRG_WORK/code/deltatok`). That
   checkout is **not a git repo**, it's an rsync target — never sync it directly.
2. ~~Submit~~ — **`Jeanzay:247562`** queued 2026-07-25 on `gpu_p6`, 20 h, 4 tasks; logs at
   `slurm/output/jz_train_deltatok_247562.{out,err}`. Fresh
   `..._mlp_sigreg1.0_pool` dir (only the dead `_sigreg1.0` existed), so it cannot resume
   `168363`. Baseline `Jeanzay:162260` is still RUNNING and is the comparison arm.
3. **WATCH:** `Train/LossSIGReg` should now sit well above its (much lower) floor and actually
   *descend*. If it is flat again, the estimator is still under-resolved. Compare `Train/LossTot`
   against `162260` at matched steps, not against `168363`.

## Open question worth checking separately

At S=256 even a perfect `N(0,I)` measures **participation rank 191** (of 768). If the
"rank ~290 << 768" diagnostic from job `160304` that motivated adding SIGReg was computed on a
similarly small sample, that collapse may be partly a finite-sample artifact — i.e. the premise for
the whole arm. Re-measure with S ≫ 768 before investing further.

## Scratch probes

Throwaway scripts used above live in the session scratchpad (not in the repo):
`sigreg_probe.py`, `sigreg_probe2.py`, `sigreg_floor.py`, `test_sigreg_pool.py`,
`sigreg_snr_after.py`, `check_collapse_direction.py`, `tb_dump.py`, and the post-review
`test_sigreg_pool2.py`. They run via
`ssh jean-zay "bash -lc 'cd \$TRG_WORK/code/deltatok && source env_jz_h100.sh && python -'" < script.py`
(piped to stdin — the cluster checkout is never written to). The `test_sigreg_pool*.py` scripts
inline a verbatim copy of the new `SIGReg` class because the edit exists only locally; they must
use the **fork** start method and ship results as numpy (a stdin script cannot be re-imported by
`spawn`, and torch tensors over a queue use fd-passing, which dies with the child).
