# DeltaTok flow: tiny loss but mean-collapsed samples

> **⚠ CONFOUND (2026-06-26):** the overfit runs below were **not** fitting one
> fixed sample — the config left `ray_map_prob` at its `0.0` default, randomizing
> ~9 view-reorderings per batch. The "can't overfit one scene" findings are
> suspect until re-tested with `ray_map_prob=-1`. See
> [`2026-06-26_findings_overfit_data_confound.md`](2026-06-26_findings_overfit_data_confound.md).

Status: **collapse resolved on the corrected overfit** (2026-06-26); generative
(full-`t`, multi-step) test in flight. The `MSEToken≈1` plateau was the data
confound + a train/eval-`t` mismatch, not a structural wall — see
[Resolution](#resolution-the-confound--traineval-t-mismatch-2026-06-26) below.
Related: [`2026-06-23_analysis_camera_swap.md`](2026-06-23_analysis_camera_swap.md).

## Setup

Overfit debug run `deltatok_flow_overfit_deltaCtx_global_v2` (Jean Zay, single GPU):
single nuScenes scene-0625 repeated, 10 timesteps × 6 cams. Key config:
`cond_mode=delta_ctx`, `attn_mode=global`, `vit_use_camera_embed=true`,
`pred_mode=x`, `loss_mode=v`, `mu=-0.7`, `sigma=1.4`, EMA **off**, 50 Euler steps.

## Symptom

Teacher-forced flow loss is tiny (`LossFlow ≈ 0.01–0.05`) yet the flow-**sampled**
deltas decode to a **collapsed BEV trajectory** (~7 m extent vs GT ~40 m) with a
scrambled-looking camera layout — even with the per-camera embedding. "Loss is
small, why isn't the sample the GT?"

## Diagnostic added: `MSEToken`

Added a sampled-vs-GT token-space metric to `eval_one_epoch`
(`occrae/deltatok_flow_trainer.py`): `MSEToken = MSE(z_hat, z)` on predicted
(non-context) delta slots, where `z_hat` is the flow-sampled deltas (ODE from
noise) and `z` is the GT deltas. It is in `_EVAL_KEYS`, so it flows through the
per-loader reduction, the `.out` print, and TensorBoard (`Eval/<test>/MSEToken`).

Scale anchor: delta tokens exit DeltaTok through a LayerNorm → ~unit variance, so
**`MSEToken ≈ 1.0` means "predicting the mean token (≈0)"**, i.e. full collapse.

## Findings

**1. Decoder ceiling (Gap A), separate issue.** Even feeding GT deltas (GT-z
rollout) the per-frame losses are large and constant: `LossPointmap_tok ≈ 20.1`,
`LossRaymap_tok ≈ 20.1`, `LossDepth_tok ≈ 2.2`. So the frozen DeltaTok→OccRAE
decode cannot perfectly reconstruct this scene's per-frame geometry. The flow's
*ceiling* is GT-z, not GT. (These losses use per-timestep, each-in-its-own-frame
decodes, so they do **not** even measure the global ego-trajectory — only the BEV
joint-pose decode does.)

**2. Baseline `MSEToken = 1.25` while `LossFlow = 0.013`** (epoch ~101). The
sampled deltas are at collapse-to-mean level despite a tiny teacher-forced loss.

**3. Extending the overfit does not help.** Over epochs 101→118, `MSEToken` stayed
flat, bouncing 0.97–1.55, while train loss inched 0.05→0.03. → not under-training
in the usual sense; structural.

**4. ODE drift ruled out (1-step test).** With `pred_mode=x`, a 1-step sampler
(`eval_num_steps=1`) reduces `flow_euler_sample` to the **direct `x_pred` from pure
noise at t=0** (step 0 sets `z = x_pred`, step 1 is a zero-size step; no `(1−t)`
clamp involved). Result: 1-step `MSEToken ≈ 1.17` ≈ the 50-step value. So the
multi-step integration adds ~nothing — the collapse is entirely in the `x_pred`,
not in trajectory drift.

**5. `loss_mode=x` necessary but NOT sufficient.** Resume-finetune with
`loss_mode=x` (unweighted x-MSE, drops the `1/(1−t)²` weight): the teacher-forced
loss became the plain x-MSE and dropped to `≈ 0.005` (so `x_pred ≈ x` at the
GT-anchored noised points), **but 1-step `MSEToken` stayed ~1.0** (noisier:
0.59–1.23 across epochs 161–167).

## Why tiny loss coexists with huge `MSEToken`

`LossFlow` and `MSEToken` are different quantities:

- `LossFlow` (with `pred_mode=x`, `loss_mode=v`) `= mean[ (x_pred − x)² / (1−t)² ]`,
  `(1−t)` clamped at 0.05 (weight up to **400×** near t=1). It is graded only at
  **answer-anchored** points `z_t = t·x + (1−t)e` (every training input is built
  from the true `x`), and the `1/(1−t)²` weight makes it **dominated by the trivial
  near-clean (t→1) end**, where `z_t ≈ x` and the error is ~0.
- `MSEToken = mean[(z_hat − x)²]` is the plain error at the **endpoint of a
  self-generated trajectory** that starts from pure noise (t≈0).

So a model can drive `LossFlow`→0 by memorizing `x` at answer-anchored points while
the sampler — which leaves that region after step 1 and starts at the
least-supervised t≈0 — lands on the mean.

## Refined diagnosis (current best understanding)

The model learned to **denoise, not to generate**. With `loss_mode=x` it
reconstructs `x` whenever `z_t` still contains some of `x` (any t>0), but at **t=0
(pure noise, zero `x` signal — exactly where the sampler starts)** it outputs ≈0/mean
instead of the single memorized `x`. For a one-sample overfit the optimal t=0 output
*is* `x` (mean of a singleton), so `MSEToken` should be ~0 — the model is simply
**under-supervised at t≈0**: the logit-normal schedule (`mu=-0.7` → median t≈0.33)
puts little mass at the noise endpoint, and the model leans on the residual `x`-leak
at low-but-nonzero t instead of learning the constant map. EMA-off adds large
eval variance (the 0.59↔1.23 bounce).

## Next knobs (priority)

1. **EMA on** — kill the epoch-to-epoch variance for a clean read (and standard for
   sampling). Settles low → mostly EMA-off noise; settles ~1 → t≈0 starvation is the wall.
2. **Cover t≈0 in training** — shift the timestep schedule toward the noise end
   (more negative `mu`, or uniform `t`) so `x_pred(noise, t=0) → x` is supervised.
3. **Reconsider `pred_mode`** (x → v/eps) for noise-end stability — larger change.

## Side note: sanity-check eval and TensorBoard

The startup sanity pass (`eval_one_epoch(sanity_check=True)`) **skips scalar TB
logging** (guarded by `if not sanity_check:` at `deltatok_flow_trainer.py:860`) but
**still writes the eval visualization image(s) to TB** — the viz block (line ~711)
is not gated on `sanity_check` and calls `_log_viz_sample(..., log_writer=self.writer)`,
which does `add_image` (`visualization_helper.py:352`). To make sanity fully
TB-silent, pass `log_writer=(None if sanity_check else self.writer)` at the viz call.

## Resolution: the confound + train/eval-`t` mismatch (2026-06-26)

With **both** problems fixed the overfit fits — `MSEToken` drops toward 0 instead of
plateauing at ~1:

1. **Data confound fixed** — `ray_map_prob=-1` makes it a true single fixed sample
   (no per-batch view reordering), per
   [`2026-06-26_findings_overfit_data_confound.md`](2026-06-26_findings_overfit_data_confound.md).
2. **Train/eval `t` aligned** — `train_fixed_t=0` pins every training slot to the
   one timestep the 1-step eval queries (`t=0`). No schedule starvation at the
   point the sampler reads, and at `t=0` the `1/(1−t)²` weight is exactly 1 so
   `loss_mode=v` ≡ unweighted x-MSE there. Run
   `deltatok_flow_overfit_deltaCtx_global_fixedT0_wd0_raymapoff` (JZ).

So Findings 2–5 and the "denoise-not-generate / `t≈0` starvation" diagnosis above
were **artifacts of the confound and the train/eval-`t` mismatch**, not an
architectural inability to generate the sample. The model could always fit it;
it was eval-querying a `t≈0` endpoint a different objective never supervised, on a
target that was secretly moving.

## Next: remove the single timestep (full-`t` overfit)

`fixedT0` is a degenerate flow — it only learns the denoiser at one `t`, so it can't
multi-step integrate. The real generative test is whether the **full** velocity
field, trained over all `t`, still fits the single sample under a multi-step ODE.
Run `deltatok_flow_overfit_deltaCtx_global_fullT_wd0_raymapoff`
(`slurm/deltatok_flow/train_deltatok_flow_jz.slurm`), fresh (`RESUME=0`, no `fixedT0` inheritance):

- **`train_fixed_t` dropped** → sample the full logit-normal(`mu=-0.7`, `sigma=1.4`)
  `t` again.
- **`loss_mode=v` kept** (chosen) → the `1/(1−t)²` weight returns now that `t` is
  distributed. This is the knob to watch: if it re-starves `t≈0`, `loss_mode=x`
  (unweighted) is the first single-variable fallback.
- **`eval_num_steps=50`** → integrate the full ODE from noise, not the 1-step `t=0`
  probe.

**Read:** `MSEToken` → 0 under the 50-step ODE ⇒ full-distribution flow fits the
sample and the path supports multi-step sampling. Re-plateau toward ~1 ⇒ the
`1/(1−t)²` weight (→ `loss_mode=x`) or the schedule `mu` (→ shift toward the noise
end) is the remaining wall, each its own next test.
