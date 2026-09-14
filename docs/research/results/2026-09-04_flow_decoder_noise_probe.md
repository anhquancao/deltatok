# flow — decoder noise-tolerance probe: GT delta + isotropic noise, decoded

2026-09-04 · thread `flow` · job `BSC:45417908` (`acc_debug`, 9 min) · code `1ae449b` · asks the question left open
by [`../analysis/2026-07-19_flow_wall.html`](../analysis/2026-07-19_flow_wall.html): how much of the flow's decoded
error is the frozen decoder's intolerance to *any* latent error of that size.

**Setup.** `eval_deltatok_flow_sampler.py --noise_sigmas 0.0,0.32,0.55,0.82`, 1-step `ode`, the same 128 Waymo val
sequences and the same frozen tokenizer (tc128 · sigreg 0.005 · compose 1.0 · `epoch_100`) as every flow read this
week. On the forecast slots `z_hat := z + σ·N(0, I)` from the seeded eval generator (`deltatok_flow_trainer.py:825`);
context slots stay GT; rollout, decode and losses are the normal eval path. The flow checkpoint (control
`iter_100000`) only supplies the tokenizer reference.

| σ | MSEToken | LossDepth | LossPointmap | LossRaymap |
|---|---|---|---|---|
| 0 (sanity) | 0.0000 | 2.7923 | 4.0557 | 1.6763 |
| 0.32 | 0.1025 | 3.0104 | 4.3611 | 1.8129 |
| 0.55 | 0.3027 | 3.3617 | 9.2042 | 7.7481 |
| 0.82 | 0.6729 | 4.7519 | 18.4995 | 19.0861 |
| *GT round-trip (`*_tok`)* | 0 | 2.7923 | 4.0557 | 1.6763 |
| *flow 1-step, control `iter_100000`* | 0.6677 | 3.6325 | 8.13 (ep 100 value) | 5.0347 |

Sanity: σ=0 reproduces the round-trip to 4 decimals and MSEToken = σ² within 0.1%, so `z` is unit-scale and the
substitution is exact.

**Reading.**

- **The flow's error is already far better structured than isotropic noise.** At matched MSEToken 0.67, isotropic
  noise decodes 3.8× worse on raymap (19.09 vs 5.03), 2.3× on pointmap, 1.3× on depth. The flow's 0.67 costs what
  isotropic σ ≈ 0.50–0.61 costs. Reweighting the flow loss toward pose-carrying dims has nothing to remove.
- **Decoded loss is steeply superlinear in latent error, with the knee below MSE 0.30.** At MSE 0.10 every metric
  is within 8% of the round-trip floor; at 0.30 raymap is 4.6× the floor. A flow that stays near MSEToken 0.67 is
  invisible on the decoded metrics whatever else it does.
- **Two levers survive.** MSEToken ≲ 0.1 (far: every code so far captures 25–33% of the delta variance from
  context), or a decoder that tolerates latent error — the RAE recipe the wall analysis proposed and never ran.
  That is [`../plan/2026-09-04_flow_decoder_noise_finetune.md`](../plan/2026-09-04_flow_decoder_noise_finetune.md).

Source: `/gpfs/projects/ehpc1001/code/deltatok/slurm/output/flow_noise_probe_bsc_45417908.out` on BSC (`results/` is
rsync-excluded, no csv is written). Not re-measured: the control's pointmap 8.13 is the `iter_200000` value from
[`2026-09-01_flow_numsteps_tc128compose_slides.html`](2026-09-01_flow_numsteps_tc128compose_slides.html).
