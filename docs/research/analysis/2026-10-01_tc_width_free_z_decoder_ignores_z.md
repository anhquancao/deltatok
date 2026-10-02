# Why free-z tc1536 runs get stuck: the decoder stops depending on z

**Date:** 2026-10-01 · **Stuck:** `BSC:46860758` (compose 1.0, seed 2), `46838176` (compose 1.0, sw2000), `46641176` /
`46860759` (compose off, triplet), `46250879` (compose off, pair) · **Escaped:** `BSC:45949269` (control, seed 1),
`46013197` (no SIGReg), `45725881` (decnoise, pool 8192) · **Diagnostics:** layer-scale probes `BSC:46877154` /
`46877752`, z-swap probe `BSC:46880769` (scripts in `/gpfs/scratch/ehpc1001/acao_probe/`) · **prior cycle:**
`plan/2026-09-16_tc_width_zrowclip_seed_tc1536.md`, `analysis/2026-09-14_tc_width_forced_bneck_stall.md` · **Code:** none

## Answer

- **The main issue: the stuck decoder does not depend on z.** Swapping a window's z for another window's changes KITTI
  `LossRecon` by at most 0.2% in all 5 stuck checkpoints. In escaped runs the same swap costs +370%, which is worse
  than copying `x_prev`.
- **It found a predictor that never reads z.** Stuck decoders still beat copying by 65–73%. That is the 0.13 plateau.
- **It is not a z-scale artifact.** The swap test does not depend on z's scale. At matched eval z scale (ep 7–9) the
  absolute noise ladder also moves stuck loss 7–15× less than in the control.
- **Escaping is a draw at this recipe.** Counting distinct data streams, compose on escaped 4 of 7 draws and compose
  off 0 of 4. The earlier "reliable" escapes were one draw replayed three times.
- **Encoder signature when stuck.** One learned channel is amplified in 8–11 of 12 encoder MLP layer scales. Decoder
  attention layer scales are 2× smaller. Cause or consequence is open.

## 1 The decoder ignores z: z-swap probe

Each `epoch_10.pth` decodes every consecutive pair of 64 KITTI windows from `x_prev` with a perturbed z. Loss is the
change against the unperturbed decode. Decoding the stored z reproduces the forward loss exactly in every run, and
every checkpoint loads with 0 missing and 0 unexpected keys. nuScenes (64 windows, 2 cameras) gives the same picture.

| run | base | swap z | z → batch mean | noise 1σ per channel | copy `x_prev` | top-channel share of z energy |
|---|---|---|---|---|---|---|
| E control `45949269` | 0.0483 | +370% | +244% | +111% | +361% | 0.4% |
| E no SIGReg `46013197` | 0.0485 | +370% | +284% | +115% | +359% | 0.7% |
| E decnoise p8192 `45725881` | 0.0366 | +536% | +475% | +455% | +507% | 79% |
| S compose s2 `46860758` | 0.1294 | +0.1% | +0.7% | +1.3% | +72% | 18% |
| S compose sw2000 `46838176` | 0.1310 | +0.0% | +4.1% | +4.3% | +70% | 22% |
| S triplet s1 `46641176` | 0.1298 | +0.0% | +0.0% | −0.0% | +71% | 26% |
| S triplet s2 `46860759` | 0.1285 | −0.0% | +0.1% | −0.1% | +73% | 60% |
| S pair s1 `46250879` | 0.1345 | +0.0% | −0.0% | −0.0% | +65% | 16% |

The two compose-on stuck runs react 1–5% to z's mean or zero but about 0% to the swap. So they read at most a
property shared by every window, not the window's own content.

## 2 The noise ladder was not fooled by z's scale

The ladder adds absolute noise (σ = 0.82), so a large z makes it relatively small. At ep 7–9 eval z had the same
scale in both runs, and the stuck run had more relative noise. KITTI `[Eval/206]`:

| epoch | control z mean square | control loss rise | stuck `46860758` z mean square | stuck loss rise |
|---|---|---|---|---|
| 7 | 1.59 | +0.0130 | 1.69 | +0.0017 |
| 8 | 1.43 | +0.0230 | 0.88 | +0.0031 |
| 9 | 1.31 | +0.0242 | 1.59 | +0.0016 |

Epochs where z was exploded (stuck ep 0–6 and 10, control ep 1–3) are uninformative.

## 3 Escaping is a draw, not the recipe

- **Step-0 loss fingerprints the data.** At init every block is near identity (`layer_scale_init` 1e-5), so the
  decoder copies `x_prev`. `45949269`, `45949271` and `46013197` all start at 0.173695, so identical data and init.
- **The 2026-09-30 eval change shifted the stream.** 3 sanity loaders instead of 2 draw from the global RNG before the
  train iterator, so seed 1 got different cameras and triplet gaps. See `deltatok_trainer.py:563` and `:1769`.
- **Scenes are identical in every run.** The batch sampler seeds with `epoch + 777` (`batched_sampler.py:36`), so only
  camera picks, triplet gaps and model init differ.

| recipe | escaped draws | stuck draws |
|---|---|---|
| compose on | 4: seed-1 stream, `45727710`, `45725881`, `45610897` | 3: `46860758`, `46838176` (sw2000), `45861170` (reg4) |
| compose off | 0 | 4 draws, 6 jobs |

## 4 Timeline

- **Same until iter ~4k.** Train `LossRecon` is about 0.105 in all 15 runs. They split at iter 4–5k, where the lr
  warmup ends (`warm_up` 5000).
- **The z blow-up is not the trap.** It hits escaped and stuck runs alike at iter ~1k. `46860812` (compose off, row
  clip 4) never spiked and stuck. `46013197` (no SIGReg) never spiked and escaped.
- **Lead, found after looking:** among compose-on runs, a SIGReg statistic above 1 at iter 1.5–2k marks all 3 stuck
  runs and none of 5 escaped.

## 5 Layer scales at epoch 10

20 checkpoints at iter 11,250, all initialised at 1e-5.

| | escaped (9) | stuck (11) |
|---|---|---|
| encoder MLP max \|λ\|, avg over blocks | 0.105–0.176 | 0.345–0.432 |
| same channel is the max in most encoder MLP blocks | no | yes, 8–11 of 12 |
| encoder attention mean \|λ\| | large on local blocks, small on global | flat at ~0.025 |
| decoder attention mean \|λ\|, avg / block 11 | 0.031–0.040 / 0.063–0.081 | 0.015–0.018 / 0.013–0.021 |

- **The channel is learned and differs per run:** 1075, 1335/1195, 145/510, 528, 134/1112, or 1105, the DA3 input's
  own large channel.
- **A dominant channel in z does not block reading by itself.** Escaped `45725881` holds 79% of z energy in one channel
  and its decoder reads it: replacing that channel with its mean costs +449%.

## 6 What does not explain it

- **Row clip, registers, half SIGReg:** `46860812`, `45861170` and `46717607` are all stuck with the channel.
- **SIGReg warmup 2000:** the warmup twins ran a different data stream from the control. Confounded, see §3.
- **`max_gap`:** it does not touch the triplet draw, which picks 3 of 10 slots uniformly (mean hop 2.75, 30% gap 1).
  The August pair sweep (max_gap ≥ 4 escaped 4/4, ≤ 3 escaped 1/3) is one seed per arm, 14% likely by chance.

## 7 Open and running

- **Why some runs never start reading z.** Not answered.
- **Replay `BSC:46868776`:** seed 1 on the old eval sets; pass check is step-0 `Train/LossRecon` 0.173695.
- **Layer scale `BSC:46869812` (1e-2) / `46869813` (0.1):** the stuck `46860758` draw with only `layer_scale_init`
  changed. Read: train `LossRecon` < 0.09 at iter ~6k.
- **Not applied:** the train-loader generator fix (own `torch.Generator` for the train loader) and a triplet gap
  warm start.
