# Why extra ODE steps degrade the tc128compose flow arm

**Date:** 2026-09-01 · **Arm:** `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit`
(ckpt iter 200000) · **Sweep:** BSC job 45296762, `eval_deltatok_flow_sampler.py --step_modes ode --num_steps 1,2,3,4,20`
· **Panels:** BSC job 45306856 · **Deck:** `../results/2026-09-01_flow_numsteps_tc128compose_slides.html`

## The observation

| steps | MSEToken | LossPointmap | LossDepth | LossRaymap |
|---|---|---|---|---|
| 1 | **0.7417** | 8.1345 | **3.6563** | 5.6041 |
| 2 | 0.7699 | **7.9687** | 3.7226 | **5.2918** |
| 3 | 0.7986 | 8.1355 | 3.7721 | 5.4819 |
| 4 | 0.8166 | 8.3646 | 3.8001 | 5.7242 |
| 20 | 0.8837 | 9.4117 | 3.8796 | 6.7993 |

Every decoded metric worsens with step count. 20 steps is the arm's own training-time eval setting and is the
worst of the five. The `_tok` teacher ceilings are byte-identical across rows, so only the sampler varied.

## The mechanism

**The sampler does not integrate. Extra steps only move the readout point in `t`.**

The live sampler is `flow_euler_sample` in `occrae/generation_helper.py`, imported at
`occrae/deltatok_flow_trainer.py:28`. (`occrae/flow_matching.py` also defines a sampler but is reached only
from `occrae/trainer.py` and `occrae/loss.py` — it is not on this path. CLAUDE.md's architecture section
points at the wrong file.)

With `pred_mode=x`, `sampler_scheduler_mode=linear` and `sampler_alpha=0`, the update at
`generation_helper.py:107-116` reduces to a convex blend:

```
z_{i+1} = (1 - 1/(N-i)) * z_i + (1/(N-i)) * x̂_i
```

At the final step `i = N-1` the mixing weight is exactly 1. So **an N-step run returns exactly `x̂` evaluated
at `t = 1 - 1/N`**, for every `N <= 20`. The `clamp(1-t, min=0.05)` guard only binds at `N >= 21`.

- `N=1` returns `x̂(noise, t=0)` — the pure regression baseline.
- `N=20` returns `x̂` at `t=0.95`, on a state that is 95% the model's own earlier outputs.

There is no error amplification in the sampler: the `1/(1-t)` v-conversion factor cancels against `Δt`.
Discretisation error shrinks with `N` and cannot produce monotone worsening. `t_shared` and the context-slot
pinning match between train and eval. All four of those are ruled out.

### Cause 1 — perception/distortion (explains the sign)

MSEToken, LossDepth, LossPointmap and LossRaymap all measure distance to one ground truth. The minimiser is
the conditional mean. A sampler that draws a plausible sample instead pays a variance penalty by
construction, up to 2x the MMSE. **"More steps = higher MSE" is the expected direction for any working
generative sampler**, not evidence of a bug. The fact that the rise is not flat also refutes strict mean
collapse: the network does read `z_t`.

### Cause 2 — t≈0 starvation (explains why even 1 step is weak)

The arm trains with `pred_mode: "x"` (`configs/deltatok_flow/train_deltatok_flow.yaml:60`) and
`model.loss_mode=v` (set only in the arm's slurm). In `flow_loss`
(`occrae/deltatok_flow_trainer.py:536-544`) that becomes an x-MSE weighted by `1/clamp(1-t, 0.05)^2`.
Under this arm's `t_dist=uniform` the gradient mass is extremely lopsided: **0.28%** of it lands below
`t=0.1`, 2.6% below `t=0.5`, and 51.3% above `t=0.95`; median `t` is 0.50 (4M Monte-Carlo draws under that
weight). Side by side with the deltatok and pointdit schedules:
`2026-09-02_pointdit_vs_deltatok.md` §1.

The 1-step "regression baseline" is the **least-trained regime in the whole model**. Its 0.7417 MSEToken is
almost certainly well above the true conditional-mean error, so the 1-vs-20 gap understates how much a
properly trained low-`t` regime would win by.

### Cause 3 — exposure bias (explains the magnitude)

Training only ever shows the net `z_t` built from ground truth plus noise. Sampling shows it `z_t` built
from its own predictions, so errors compound along the trajectory. Standard diffusion failure mode:
[Input Perturbation, arXiv 2301.11706](https://arxiv.org/abs/2301.11706),
[Epsilon Scaling, arXiv 2308.15321](https://arxiv.org/abs/2308.15321).

## Comparison with pointdit

Moved to its own thread on 2026-09-02: `2026-09-02_pointdit_vs_deltatok.md`.

The one-line version: the mechanism is **identical** — same Euler update, same `1/(1-t)^2`-weighted x-MSE,
same `t_eps` — and pointdit already ships three mitigations this arm lacks (10% of samples forced to `t=0`,
an additive unweighted x-MSE at weight 0.1, and a zeros ODE init). It never runs 20 steps; its released
default is 3 and every 256-px eval uses 1. Nothing there was measured on our data.

## Measurement bug found while checking this

`flow_noising` drew its timestep `t` from the **global** RNG (`occrae/deltatok_flow_trainer.py:474-478`),
while `_eval_noise_gen` seeded only the noise `e` (`:439-443`). The five sweep passes ran in one process
without reseeding, so **each pass evaluated `LossFlow` at different `t` values**. The LossFlow column
(1.2564 / 1.2145 / 1.2665 / 1.3280 / 1.2462) is not comparable across rows, and its non-monotonicity is
RNG, not a property of the sampler.

Fixed by drawing `t` from a dedicated `_eval_t_gen`, seeded alongside `_eval_noise_gen`. A separate
generator (rather than reusing `_eval_noise_gen`) keeps the existing ODE-init and `e` draw streams
byte-identical, so previously logged MSEToken / LossDepth / LossPointmap numbers still reproduce.

## Per-frame visual check

Panels at the last forecast frame (t4) for 4 val sequences do **not** generally reproduce the aggregate
ordering: only `s2_suburb` is monotone; `s4_lot` has 3 steps best; the night scene has 20 best. The
aggregate trend is real over 128 sequences but is not visible frame by frame. Deck slides 5-6 show one
confirming and one dissenting sequence.

Noise floor of the L1 pipeline, measured on the byte-identical GT-z column across two passes:
depth 0.004 m mean (p99 0.20 m), RGB 0.04 levels. Deck error maps use one fixed scale, vmax = pooled p99
(depth 12.35 m, RGB 117.7 levels).

## Diagnostic to-do (2026-09-02), compose arm only

Older arms (whitenpos, logitnorm, vpred) are stale and are not used. Each step gates the next. The two
pointdit-derived mitigations that used to sit here (zeros ODE init; forced `t=0` plus an additive x-MSE) are
now cycles in the `pointdit` thread: `../plan/2026-09-02_pointdit_zeroinit_ode.md` (dropped) and `../plan/2026-09-02_pointdit_lowt_recipe_finetune.md`.

1. **Seed spread at N=1 vs N=20, 8 seeds, plus a decoded depth-edge F1.** One `acc_debug` job on
   `slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm`, `NUM_STEPS=1,20`, 8 values of
   `training.seed`. Spread ≈ 0.14 MSEToken at N=20 = the rise is sample variance and the sampler works;
   much smaller = systematic drift. The edge F1 is the one realism number next to the distortion metrics;
   pointdit has the same probe (`2026-09-02_pointdit_vs_deltatok.md`). Without it cause 1
   stays unmeasured.
2. **Only if step 1 says the sampler works:** the wall is the frozen decoder and the tokenizer needs
   noise-augmented decoder training. No such option exists in `occrae/deltatok_trainer.py` today.

Most relevant paper: [PMRF, ICLR 2025 (arXiv 2410.00418)](https://arxiv.org/abs/2410.00418) — the min-MSE
estimator with perfect perceptual quality is an optimal-transport map applied on top of the MMSE
prediction, i.e. regress first, flow second.
