# flow — a noise-tolerant decoder lifts the flow's decoded metrics without touching `z`

Created 2026-09-04 · thread `flow` · prior cycle:
[`../results/2026-09-04_flow_decoder_noise_probe.md`](../results/2026-09-04_flow_decoder_noise_probe.md)
· arm: `deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0_decnoise0.8_ft10`
· control: the source tokenizer `.../compose1.0/ckpts/epoch_100.pth`, read through the unchanged flow control
`deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit` at `iter_100000`
· jobs: `BSC:45421190` · deck: [`../results/2026-09-06_flow_decoder_noise_finetune_slides.html`](../results/2026-09-06_flow_decoder_noise_finetune_slides.html)

The RAE recipe the wall analysis proposed on 2026-07-19 (`../analysis/2026-07-19_flow_wall.html`, "primary:
decoder noise-robustness finetune") and never ran. The probe above made it the only cheap lever left.

## 1 Hypothesis

**Fine-tuning only the decoder half of the tc128 compose tokenizer on `z + σ·N(0, I)`, σ ~ U(0, 0.8) per sample,
encoder frozen, then dropping that decoder into the unchanged flow control, cuts the flow's 1-step `LossRaymap` from
5.03 to ≤ 3.5 (−30%) and moves `LossPointmap` from ~8.1 toward the 4.06 round-trip**, on the 128 Waymo val
sequences at `iter_100000`, N=1, while the σ=0 round-trip stays within 5% of 2.79 / 4.06 / 1.68 and `MSEToken`
stays 0.6677 to the digit (`z` and the flow are untouched).

The noise-ladder re-read on the new decoder is the independent check that the finetune did anything: σ=0.82 must
decode far below 4.75 / 18.50 / 19.09.

**Falsifiers.** Each bullet is the routing for §4.

- **Ladder flattens and the flow moves ≥ 30% on raymap.** Decoder intolerance was binding. The finetuned decoder
  becomes the tokenizer every flow read uses; re-read the 1…20-step sweep and run
  [`2026-09-04_flow_bestofk_regressor_null.md`](2026-09-04_flow_bestofk_regressor_null.md) on it — the N=20
  penalty may shrink with the decoder.
- **Ladder flattens, flow N=1 flat.** The flow's error is not what isotropic noise covers: either it sits in the
  rollout (`_rollout_from_z`, `deltatok_shared.py:374`, feeds each decode's output as the next `x_prev`, which the
  finetune never noises) or it is structured. Next: a decoder finetune on *flow samples* rather than Gaussian
  noise, or `x_prev` noise. Not this cycle.
- **Ladder does not flatten (σ=0.82 raymap stays > 10).** Ten epochs of decoder-only training cannot make the
  `z → x` map tolerant at this Cz. Then the pair needs a from-scratch noisy tokenizer, which changes `z` and is a
  `tc_width`/`sigreg` question, not a flow one.
- **σ=0 round-trip degrades > 10%.** τ = 0.8 is too wide for a decoder this size; run τ = 0.4.

**Not doing.**

- **No from-scratch noise-augmented tokenizer.** It changes `z` and invalidates every flow checkpoint; 40 h against
  ~5 h, and the from-scratch question is only reached through falsifier 3.
- **No τ sweep, no encoder unfreeze, no joint finetune.** Anything that moves the encoder moves `z`, and the whole
  point of the design is that `MSEToken` 0.6677 reproduces exactly so the decoder is the only variable.
- **No `x_prev` noise.** The AR rollout's second mismatch. `LossRecon_AR` in the tokenizer eval reads it; a
  separate cycle if falsifier 2 fires.
- **No flow retrain.** Nothing changes for the flow; the decoder swap is an eval-time `model.deltatok_ckpt` path.

## 2 Why it is worth the GPU hours

**The probe** ([`../results/2026-09-04_flow_decoder_noise_probe.md`](../results/2026-09-04_flow_decoder_noise_probe.md),
`BSC:45417908`, 128 Waymo val seqs, tokenizer `epoch_100`, 1-step):

| σ | MSEToken | LossDepth | LossPointmap | LossRaymap |
|---|---|---|---|---|
| 0 | 0 | 2.7923 | 4.0557 | 1.6763 |
| 0.32 | 0.1025 | 3.0104 | 4.3611 | 1.8129 |
| 0.55 | 0.3027 | 3.3617 | 9.2042 | 7.7481 |
| 0.82 | 0.6729 | 4.7519 | 18.4995 | 19.0861 |
| flow 1-step, `iter_100000` | 0.6677 | 3.6325 | ~8.13 | 5.0347 |

Decoded loss is superlinear in latent error with the knee below MSE 0.30, and the flow sits at 0.67. The flow's
error is already 2–4× more benign than isotropic noise of the same size, so nothing on the flow side (loss
weighting, `t` placement, sampler) can show on the decoded metrics until `MSEToken` drops below ~0.1. Every code
so far captures 25–33% of the delta variance from context (`2026-09-04_flow_bestofk_regressor_null.md` §2), so
that route is far. The decoder is the near lever.

**The recipe is RAE's, verbatim.** `third_party/RAE/src/stage1/rae.py:74-86`: during training the encoder output
gets `σ = noise_tau · U(0,1)` per sample times `randn_like`, `noise_tau = 0.8`, applied before the decoder and
nowhere else. Their decoder learns to project a ball around each code back to the image; ours has only ever seen
exact codes (`DeltaTokModule.decode`, `deltatok_trainer.py:372`). The 2026-07-19 wall analysis measured the same
intolerance on dtok64 — even the MMSE latent decoded 2.6–2.8× above the ceiling — and proposed exactly this.

**Why the encoder must stay frozen.** The flow was trained on this tokenizer's `z`. With `encoder_blocks`,
`pre_bottleneck_norm`, `z_proj_down`, `z_embed`, `xy_embed`, `norm` frozen and SIGReg off, `encode()` is
bit-identical, so the finetuned checkpoint drops into the flow eval through `model.deltatok_ckpt` and
`MSEToken` 0.6677 must reproduce exactly — the built-in check that only the decoder changed. Same design as the
stage-2 bottleneck freeze (`training.freeze_except_bottleneck`, `deltatok_trainer.py:775`), mirrored.

**Compose is covered for free.** The noise hook sits in `DeltaTokModule.forward` after `z` is fixed, so the
`z_input` branch the compose path uses (`_compose_forward`, `deltatok_trainer.py:935`) is noised too, and the
decoder learns tolerance on composed codes as well as single hops.

**Cost against the prize.** One ~5 h decoder-only job plus two 9-minute evals decide whether the flow thread has
been reading a decoder artifact for three months. Every flow checkpoint stays valid.

## 3 How it is run

### Patch 1 — the noise hook, `occrae/deltatok_trainer.py`

`DeltaTokModule.__init__` takes `decode_noise_tau: float = 0.0`. In `forward` (`:431`), after `z` is set in all
three branches (`z_input`, `return_bneck`, plain), and before `self.decode(...)`:

```python
z_dec = z
if self.training and self._decode_noise_tau > 0:
    sigma = self._decode_noise_tau * torch.rand(z.shape[0], 1, 1, 1, device=z.device, dtype=z.dtype)  # (M,1,1,1) per-sample σ ~ U(0, τ)
    z_dec = z + sigma * torch.randn_like(z)                                                          # (M, N, K, Cz) noised code, decoder input only
x_hat = self.decode(z_dec, x_prev, rope_local, rope_global)
```

`return_z` / `return_bneck` keep returning the clean `z`. `self.training` is False in the flow trainer
(`deltatok_flow_trainer.py:200`), so every flow eval is unaffected. Read the knob in `_make_deltatok_module`
(`deltatok_shared.py:475`) as `deltatok_cfg.get("decode_noise_tau", 0.0)` — the flow's yaml needs no key. Yaml key
`model.deltatok.decode_noise_tau: 0.0` in `configs/deltatok/train_deltatok.yaml`, and print the resolved value next
to the parameter breakdown in `get_network` so a silently-unset knob is not read as a null.

### Patch 2 — the freeze, `occrae/deltatok_trainer.py:775`

`training.freeze_except_decoder: false` beside `freeze_except_bottleneck` in the yaml. In `get_network`, mirror
the stage-2 block: `model.requires_grad_(False)`, then `decoder_blocks` and `z_proj_up` back to trainable. Assert
the two freezes are not both set. `_print_param_breakdown` already prints the trainable count per component; the
expected line is `encoder_blocks trainable=0.000M`.

### Patch 3 — the eval launcher takes the tokenizer from the environment

`slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm:44`:
`model.deltatok_ckpt="${DELTATOK_CKPT:-$SCRATCH/deltatok_log/.../compose1.0/ckpts/epoch_100.pth}"`, and echo it.
`EXTRA_ARGS="--cfg ..."` is not an option: `--cfg` is `nargs="*"`, a second one replaces the whole arch-flag array.

### The arm — copy the tc128 sigreg 0.005 script, then edit

The slurm script is a copy-paste of the script that trained the tokenizer being finetuned:
`slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc128_bsc.slurm` (tc128 · `SIGREG_WEIGHT` 0.005 ·
`COMPOSE_WEIGHT` 1.0, the `..._sigreg0.005_ns256_pool8192_compose1.0` run). Never rewrite from scratch — the copy
keeps the diff to the lines below.

```bash
cp slurm/deltatok/train_deltatok_compose_sigreg_nozn_tc128_bsc.slurm \
   slurm/deltatok/train_deltatok_compose_tc128_decnoise_ft_bsc.slurm
```

Change only these, and nothing else in the copy:

- `--job-name`, `--output`, `--error` together; `--time=08:00:00` (backfills in minutes at this length).
- `RUN_NAME=deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0_decnoise0.8_ft10`,
  hardcoded. A fresh name, so no `current.pth` exists and the warm start below is honoured.
- `INIT_CKPT="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0/ckpts/epoch_100.pth"`
  — weights-only load via `--ckpt` (`sh/train_deltatok.sh:35`), `iter` restarts at 0, so the LR schedule below is
  the finetune's own.
- Overrides: `model.deltatok.decode_noise_tau=0.8` · `training.freeze_except_decoder=true` ·
  `training.sigreg_weight=0` (`z` is frozen; the pool would only cost) · `training.compose_weight=1.0` (unchanged) ·
  `training.lr=1e-4` · `training.min_lr_ratio=0.1` · `training.warm_up=500` · `training.epoch=10` ·
  `training.max_iter=11250` (1125 updates/epoch: 18 000 samples at `effective_bsize` 16; `epoch_100` = iter
  112 500) · `training.save_ckpt_every_n_epochs=5`. Everything else — bsize 2, grad_clip 0.1, max_gap 9,
  `train_deltatok_nt10_bsc` — stays as the source arm trained.

Budget: the source recipe runs ~32 min/epoch on BSC with the full backward; without the encoder backward expect
≤ 30. Ten epochs ≤ 5.5 h inside an 8 h `acc_ehpc` wall.

### Pre-flight, then submit

The user syncs manually. An empty grep means the cluster copy is stale: ask, never `rsync`.

```bash
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -n decode_noise_tau occrae/deltatok_trainer.py occrae/deltatok_shared.py configs/deltatok/train_deltatok.yaml && grep -n freeze_except_decoder occrae/deltatok_trainer.py configs/deltatok/train_deltatok.yaml && grep -n DELTATOK_CKPT slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm && grep -E \"RUN_NAME=|INIT_CKPT|decode_noise_tau|freeze_except_decoder|--time\" slurm/deltatok/train_deltatok_compose_tc128_decnoise_ft_bsc.slurm'"
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/train_deltatok_compose_tc128_decnoise_ft_bsc.slurm'"
```

A job id is not a launch. Watch until the first loss line; grep the `.out` for `encoder_blocks ... trainable=
0.000M`, `decode_noise_tau=0.8`, and `Load ckpt from: .../epoch_100.pth`. `LossRecon` at iter 0 should sit near
the source arm's terminal value, not at a fresh-init value.

### The read — two eval jobs per checkpoint, `acc_debug`, ~9 min each

`epoch_10` first; `epoch_5` only if `epoch_10` moves, to see whether it was still moving.

```bash
# A: the flow through the new decoder. MSEToken must read 0.6677 (N=1) / 0.8850 (N=20) exactly.
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && \
  DELTATOK_CKPT=\$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc128_nozn_maxgap9_vpt1to2_sigreg0.005_ns256_pool8192_compose1.0_decnoise0.8_ft10/ckpts/epoch_10.pth \
  CKPT=/gpfs/scratch/ehpc1001/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc128mg9sigreg005compose_ep100tok_xxl_dit/ckpts/iter_100000.pth \
  OUTPUT_DIR=results/deltatok_flow_decnoise/flow_ep10 NUM_STEPS=1,20 STEP_MODES=ode \
  sbatch slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm'"
# B: the noise ladder on the new decoder. σ=0 is the round-trip cost of the finetune.
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && DELTATOK_CKPT=<same> CKPT=<same> \
  OUTPUT_DIR=results/deltatok_flow_decnoise/ladder_ep10 NUM_STEPS=1 STEP_MODES=ode \
  EXTRA_ARGS=\"--noise_sigmas 0.0,0.32,0.55,0.82\" sbatch slurm/eval_deltatok_flow_numsteps_tc128compose_bsc.slurm'"
```

`acc_debug` takes one job per user; run B on `--qos=acc_ehpc --time=01:00:00`.

| readout | source decoder (`epoch_100`) | finetuned `epoch_10` | finetuned `epoch_5` |
|---|---|---|---|
| σ=0 `LossDepth` / `LossPointmap` / `LossRaymap` | 2.7923 / 4.0557 / 1.6763 | | |
| σ=0.55 same | 3.3617 / 9.2042 / 7.7481 | | |
| σ=0.82 same | 4.7519 / 18.4995 / 19.0861 | | |
| flow N=1 `MSEToken` (must match) | 0.6677 | | |
| flow N=1 `LossDepth` / `LossPointmap` / `LossRaymap` | 3.6325 / ~8.13 / 5.0347 | | |
| flow N=20 `LossRaymap` | 5.8777 | | |
| tokenizer eval `LossRecon` / `LossRecon_AR` / `LossRecon_Comp` (KITTI) | source `.out`, ep 100 | | |

Into `results/2026-09-xx_flow_decoder_noise_finetune_slides.html`; verdict in `analysis/`.

## 4 Outcome

**Read 2026-09-06. `BSC:45497311` (flow N=1,20) and `BSC:45497319` (σ ladder) on `epoch_10`;
`BSC:45497830` / `BSC:45497831` on `epoch_5`; `BSC:45498382` fills in N = 2, 3, 4, 8, 12 on `epoch_10`. The source-decoder control is `BSC:45414635` — the same flow
`iter_100000` read through `epoch_100`; its N=1 `LossPointmap` is **7.5305**, not the §2 probe's approximate ~8.13.**

**Both falsifiers fire, at different step counts. Falsifier 1 at N=20; falsifier 2 at N=1.**

**The flow gains, and the gain is large.** Best-achievable against best-achievable over the measured sweep:
`LossRaymap` **4.6035 → 3.7936 (−17.6%)** and `LossPointmap` **7.2704 → 6.6117 (−9.1%)**, with the optimum moving
from N=2 to N=20; `LossDepth` is the exception at +1.2%. At matched N=20 the raymap move is **−35.5%**, which clears
falsifier 1's ≥30% bar outright — so decoder intolerance *was* binding there, and falsifier 1's prescribed action
(adopt the finetuned decoder, re-read the 1…20-step sweep) is the right routing.

**What failed is the N=1 prediction the hypothesis actually pinned**, and it failed in the wrong direction: raymap
5.0347 → 5.9274 (+17.7%) against a predicted ≤ 3.5. There, isotropic noise at the flow's own latent error decodes
3.5× *better* than the flow does, so the 1-step error is structured, not Gaussian, and a decoder taught to invert a
ball around each code does not cover it.

| readout | source decoder (`epoch_100`) | finetuned `epoch_10` | |
|---|---|---|---|
| σ=0 `LossDepth` / `LossPointmap` / `LossRaymap` | 2.7923 / 4.0557 / 1.6763 | **2.7548 / 3.9912 / 1.6484** | −1.3% / −1.6% / −1.7% |
| σ=0.32 same | 3.0104 / 4.3611 / 1.8129 | **2.8244 / 4.0606 / 1.6506** | −6.2% / −6.9% / −9.0% |
| σ=0.55 same | 3.3617 / 9.2042 / 7.7481 | **2.8892 / 4.1256 / 1.6533** | −14.1% / −55.2% / −78.7% |
| σ=0.82 same | 4.7519 / 18.4995 / 19.0861 | **3.0095 / 4.2507 / 1.6715** | −36.7% / −77.0% / **−91.2%** |
| flow N=1 `MSEToken` (must match) | 0.6677 | **0.6677** | exact |
| flow N=1 `LossDepth` / `LossPointmap` / `LossRaymap` | 3.6325 / 7.5305 / 5.0347 | **3.6765 / 8.2932 / 5.9274** | +1.2% / **+10.1%** / **+17.7%** |
| flow N=20 `MSEToken` | 0.8850 | **0.8850** | exact |
| flow N=20 `LossDepth` / `LossPointmap` / `LossRaymap` | 3.8819 / 8.6347 / 5.8777 | **3.7915 / 6.6117 / 3.7936** | −2.3% / **−23.4%** / **−35.5%** |
| tokenizer eval `LossRecon` / `_AR` / `_Comp` (KITTI) | 0.0472 / 0.0509 / 0.0505 | 0.0484 / 0.0530 / 0.0515 | +2.5% / +4.1% / +2.0% |
| tokenizer eval `PredVsOrig` depth / pmap / raymap (KITTI) | 0.4248 / 0.6214 / 0.3933 | 0.4238 / 0.6165 / 0.3908 | −0.2% / −0.8% / −0.6% |

**Every design check held.** `MSEToken` reproduces to the digit at both N, so `encode()` is bit-identical and the
decoder is the only thing that changed. The σ=0 round-trip did not degrade — it *improved* 1.3–1.7%, well inside
the 5% band, so falsifier 4 (τ too wide) does not fire. The tokenizer's own eval pays 2.0–4.1% in token space and
nothing in decoded geometry.

**The ladder is the headline.** At σ=0.82 raymap goes 19.09 → 1.67, i.e. the decoder is now essentially flat in
latent error across the whole range the flow lives in — the superlinear knee below MSE 0.30 is gone. The 2026-07-19
wall analysis was right that the decoder was intolerant, and ten epochs of RAE's recipe fixed it outright.

**The N=1 cell is the one loss.** The prediction was raymap 5.03 → ≤ 3.5. It went the other way,
to 5.93. Set the two σ=0.82 rows against the flow's N=1 row on the *same* decoder: MSE 0.6729 of isotropic noise
decodes to 3.01 / 4.25 / 1.67, while MSE 0.6677 of flow error decodes to 3.68 / 8.29 / 5.93. **Same latent-error
magnitude, 3.5× the raymap.** The probe's framing inverts: before the finetune the flow's error looked 2–4× *more*
benign than isotropic noise; after it, isotropic noise is 3.5× more benign than the flow's.

**The full sweep, `BSC:45498382`, N = 1, 2, 3, 4, 8, 12, 20 on `epoch_10`:**

| N | 1 | 2 | 3 | 4 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|
| `MSEToken` | 0.6677 | 0.7109 | 0.7554 | 0.7826 | 0.8398 | 0.8638 | 0.8850 |
| `LossDepth` | **3.6765** | 3.7324 | 3.7423 | 3.7581 | 3.7802 | 3.7894 | 3.7915 |
| `LossPointmap` | 8.2932 | 7.6756 | 7.3669 | 7.1248 | 6.7778 | 6.6715 | **6.6117** |
| `LossRaymap` | 5.9274 | 5.1114 | 4.7003 | 4.4063 | 3.9965 | 3.8630 | **3.7936** |

The finetuned curve is **monotone in N** for pointmap and raymap and has not plateaued at 20; the source decoder
instead has an interior optimum at **N=2** (raymap 4.6035, pointmap 7.2704) and degrades after it. Depth is
monotonically worse with N for both, so depth wants N=1 and the other two want N=20.

**Best achievable against best achievable** — the comparison that matters when N is free:
raymap **3.7936** (ft, N=20) vs 4.6035 (source, N=2) = **−17.6%**; pointmap **6.6117** vs 7.2704 = **−9.1%**;
depth 3.6765 vs 3.6325 = **+1.2%**, the source. Two of three to the finetuned decoder — and it clears the source's
best on both at **N=4** (raymap 4.4063, pointmap 7.1248), so the gain does not require twenty steps.

**N=20 is the largest single gain: raymap 5.8777 → 3.7936 (−35.5%).** The step-count ordering flips with it — the source
decoder read N=1 better than N=20 on raymap (5.03 vs 5.88), the finetuned decoder reads N=20 better (5.93 vs 3.79).
The N=20 sample carries *more* token error (`MSEToken` 0.8850 vs 0.6677) and still decodes better, which is the
same statement as above from the other side: what matters is the shape of the error, not its size. The
`2026-09-04_flow_bestofk_regressor_null.md` N=20 penalty is a decoder artifact at least in part, and should be
re-read on this checkpoint.

**Routing.** Falsifier 2's own text: *"either it sits in the rollout (`_rollout_from_z`, `deltatok_shared.py:374`)
or it is structured. Next: a decoder finetune on flow samples rather than Gaussian noise, or `x_prev` noise."*
That is the next cycle. The finetuned decoder is strictly better than the source at every σ and at N=20, so it
**replaces `epoch_100` as the tokenizer every flow read uses**, and the 1…20-step sweep is re-read on it first.
