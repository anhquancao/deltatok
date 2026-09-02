# pointdit vs deltatok-flow — same algebra, three missing mitigations, two worth porting

Created 2026-09-02 · thread `pointdit` · prior cycle: `2026-09-01_flow_numsteps_why_more_steps_degrade.md`
· checkout: `/mnt/d/code/pointdit` (**not** `~/code/pointdit`) · no jobs, code reading only

Split out of the numsteps findings. That file answers *why more ODE steps degrade our arm*; this one answers
*what pointdit does differently and which of it is worth porting*. Nothing here is measured on our data.

## What is the same

pointdit is a single-image pointmap flow model. The generative core is the same object as ours.

- **Same Euler update.** `denoiser.py:221-228` computes `v = (x̂ - z)/clamp(1-t)` then `z += Δt·v`. On a linear
  grid that is a convex blend, and an `N`-step run returns exactly `x̂` evaluated at `t = 1 - 1/N`. Ours is
  `flow_euler_sample`, `occrae/generation_helper.py:106-116`, with the same reduction.
  (pointdit runs `N` forwards; our loop is `range(num_steps+1)` and wastes one `Δt=0` forward.)
- **Same parameterisation.** The net predicts `x`; the loss is `(v - v̂)²` with `clamp_min(1-t, 0.05)`
  (`denoiser.py:97-103`) — algebraically the `1/(1-t)²`-weighted x-MSE that our `pred_mode=x` + `loss_mode=v`
  produces at `occrae/deltatok_flow_trainer.py:536-544`.
- **Same t-eps.** pointdit `--t_eps 5e-2` (`main.py:88`); ours is the literal `min=0.05`.
- **Same practical step count.** Every `scripts/eval_*_256.sh` and every training online-eval passes
  `--num_sampling_steps 1`; the 512 evals and demos pass 3. The argparse default is 3 (`main.py:101`).
  Nothing in that repo runs 20. Our compose arm's training-time eval is `training.eval_num_steps=20`.

So "more steps degrades" is expected to hold for pointdit too, by identical algebra. **Not measured** — no
pointdit eval was run at other step counts.

## What is different — three mitigations we lack

### 1 Trains `t=0` on purpose

`--force_zero_t` defaults **on** and pins 10% of each batch to exactly `t=0` (`main.py:179-183`,
`denoiser.py:76-85`), on top of a low-centred logit-normal `t = sigmoid(-0.8 + 0.8·ε)` (`main.py:82-83`).

Gradient mass under the shared `1/clamp(1-t,0.05)²` weight, 4M Monte-Carlo draws:

| schedule | `t == 0` exactly | `t < 0.1` | `t < 0.5` | `t > 0.95` | median `t` |
|---|---|---|---|---|---|
| our compose arm — uniform | 0% | **0.28%** | 2.6% | 51.3% | 0.50 |
| our deltatok `logitnormal(-0.7, 1.4)` | 0% | 1.5% | 12.6% | 17.6% | 0.33 |
| pointdit `logitnormal(-0.8, 0.8)` + 10% forced `t=0` | 3.6% | 5.2% | 60.8% | 0.02% | 0.29 |

18× more weight below `t=0.1` than our arm, and essentially none above 0.95. **pointdit's 1-step output is a
trained regressor; ours is the least-trained point of the model.**

### 2 An additive, unweighted x-loss

`engine.py:121-140` adds `args.rel_point_loss_weight * rel_point_loss(x̂, gt)` to the v-MSE. The weight
defaults to **0.1** (`main.py:199`); the term itself is `loss.py:18`.

The v-MSE *is* the x-MSE divided by `(1-t)²`, so a sample at `t=0.95` counts 400× one at `t=0`. The extra term
adds the same x-MSE back with **no** `t` weighting, so every `t` contributes equally and at `t=0` it is the only
term that matters. The flow objective is untouched at high `t`.

pointdit's version is a relative L1 (normalised by distance from the origin) because pointmaps carry metric
scale. Ours would be a plain MSE — SIGReg already pins the latents to unit scale.

### 3 Starts the ODE from zeros, not noise — **not porting**

`--generate_noise_scale` defaults to **0.0** (`main.py:85-87`), so inference is deterministic and the
sample-variance term vanishes. Ours always draws `randn` at `occrae/generation_helper.py:251`.

De-scoped 2026-09-02, before any run. It is an eval-time sampler knob, and the two eval-time knobs already
tried on this arm (`sampler_scheduler_mode=cosine`, `sampler_alpha=0.5`) produced no resolved effect —
`../results/2026-08-20_flow_sampler_ablation.html` §11 calls them "unresolved, not refuted". The task
file `../plan/2026-09-02_pointdit_zeroinit_ode.md` is kept, marked dropped, for its patch and `sbatch` lines.

### Also there, not ported: a realism probe

`--eval_intermediate_boundary` computes a scale-invariant depth-boundary F1 at every intermediate ODE step
(`engine.py:869-893`). That is the one *realism* number sitting next to the distortion metrics. We log only
distortion (MSEToken, LossDepth, LossPointmap, LossRaymap), all of which a mean-predictor minimises, so we
cannot currently tell "worse sample" from "less mean-collapsed".

## Why this matters for us

The numsteps findings attribute the step-count degradation to three causes: perception/distortion trade-off,
`t≈0` starvation, and exposure bias. Mitigations 1 and 2 attack starvation directly; 3 attacks the variance
half of the trade-off. None of the three exists in `occrae/deltatok_flow_trainer.py` today.

Open cycle: `../plan/2026-09-02_pointdit_lowt_recipe_finetune.md` (mitigations 1 + 2). Mitigation 3 is dropped.
