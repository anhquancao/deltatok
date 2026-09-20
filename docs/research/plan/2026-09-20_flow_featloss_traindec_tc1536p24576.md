# Plan: feature loss with a trainable DeltaTok decoder on the tc1536 pool-24576 flow (QUESTION 24)

**Date:** 2026-09-20 · **Thread:** flow · **Cluster:** BSC
· **Control:** `BSC:46012704`, `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc1536mg9sigreg002pool24576compose_dndetach_ep30tok_featloss1_xxl_dit` (frozen decoder)
· **Baseline:** `BSC:45861488`, `..._ep30tok_xxl_dit` (no feature loss)
· **Arm:** `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc1536mg9sigreg002pool24576compose_dndetach_ep30tok_featloss1_traindec_xxl_dit`
· jobs: `BSC:46185767` (smoke), `BSC:46186126` (prod) · deck: _pending_ · prior cycle: `plan/2026-09-17_flow_featloss_tc1536p24576.md`

Two knobs, `model.feat_loss_train_decoder` (false = the control) and `training.decoder_lr`. When on, the tokenizer's
`decoder_blocks` (and `z_proj_up`, absent at tc1536) take gradient from `LossFeat` in their own optimizer group; the
encoder, `z_embed` and `xy_embed` stay frozen, so z, the flow target and `MSEToken` are unchanged. The decoder state is
saved in every flow ckpt and restored on resume and in the sampler script. Eval gains `LossFeat_tok`, the decoder's own
recon of GT z. The decoder stays in eval mode: `DINOv3ViTConfig` defaults `attention_dropout` and `drop_path_rate` to 0,
so train and eval forwards are identical.

All line numbers below are today's files, before any edit.

## 1. Trainer — `occrae/deltatok_flow_trainer.py`

**1a.** After line 37 (`"LossFeat", ...`) in `_EVAL_KEYS`:

```python
    "LossFeat_tok",                                          # decode of GT z vs GT layer-12 feats: the (finetuned) decoder's own recon
```

**1b.** After line 106 (`print(f"feat_loss_weight=...")`):

```python
        # Trainable decoder: LossFeat also updates deltatok.decoder_blocks (encoder frozen, z fixed).
        self.train_decoder = bool(self.cfg.model.get("feat_loss_train_decoder", False))
        assert not self.train_decoder or self.feat_loss_weight > 0, "feat_loss_train_decoder needs feat_loss_weight > 0"
        print(f"feat_loss_train_decoder={self.train_decoder} decoder_lr={self.cfg.training.get('decoder_lr', 0.0)}")
```

**1c.** Optimizer. Replace lines 145–148:

```python
        # Decoder group gets its own lr (pretrained module); a ready dict keeps get_optim's
        # resume load seeing the same group count the ckpt was saved with.
        groups = [{"params": list(self.vit.parameters())}]
        if self._decoder_params:
            groups.append({"params": self._decoder_params, "lr": float(self.cfg.training.decoder_lr)})
        self.optim = self.get_optim(
            groups, self.cfg.training.lr, betas=(0.9, 0.999),
            weight_decay=self.cfg.training.weight_decay, mode=self.cfg.training.optimizer
        )
```

**1d.** Checkpoint provenance. Replace lines 171–174 (`_ckpt_extra`):

```python
    def _ckpt_extra(self):
        """Provenance for every flow ckpt: whitened weights are meaningless without
        the same stats. Kept in one place so all three save sites agree."""
        extra = {"whiten_stats": getattr(self, "_whiten_stats_path", None)}
        if self.train_decoder:
            # Finetuned decoder travels with the flow weights (the tokenizer ckpt no longer matches).
            sd = self.deltatok.state_dict()
            extra["deltatok_decoder_state"] = {k: v.detach() for k, v in sd.items() if k.startswith(("decoder_blocks.", "z_proj_up."))}
        return extra
```

**1e.** `_build_deltatok`. After line 211 (`model.requires_grad_(False)`):

```python
        self._decoder_params = []                                     # trainable tokenizer params; empty when frozen
        if self.train_decoder:
            mods = [model.decoder_blocks] + ([model.z_proj_up] if model.z_proj_up is not None else [])
            for m in mods:
                m.requires_grad_(True)
            self._decoder_params = [p for m in mods for p in m.parameters()]
            print(f"[INFO] trainable DeltaTok decoder: {sum(p.numel() for p in self._decoder_params) / 1e6:.1f}M params")
            resume_ckpt = self.get_resume_checkpoint_path()
            if resume_ckpt is not None:                               # resume: overlay the finetuned decoder
                self.load_decoder_state(model, torch.load(resume_ckpt, map_location="cpu", weights_only=False, mmap=True))
```

New method after line 217 (`return model`, before `def get_network`):

```python
    def load_decoder_state(self, model, ckpt):
        """Overlay a finetuned decoder saved by _ckpt_extra; no-op for frozen-decoder ckpts."""
        state = ckpt.get("deltatok_decoder_state")
        if state is None:
            return False
        keys = set(model.state_dict())
        assert set(state) <= keys, sorted(set(state) - keys)[:5]
        model.load_state_dict(state, strict=False)                    # encoder keys stay as loaded from deltatok_ckpt
        print(f"[INFO] loaded finetuned DeltaTok decoder ({len(state)} tensors)")
        return True
```

**1f.** Train loop. After line 611 (`window_grad = deque(...)`):

```python
        window_grad_dec = deque(maxlen=print_freq)                # () pre-clip decoder grad norms; empty when frozen
```

Replace lines 668–670 (`grad_norm = ...` through `self.optim.step()`):

```python
                grad_norm = nn.utils.clip_grad_norm_(self.vit.parameters(), self.cfg.training.grad_clip)
                window_grad.append(grad_norm.detach())
                if self._decoder_params:
                    if self.distributed:                              # not DDP-wrapped: average grads by hand
                        for p in self._decoder_params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    grad_norm_dec = nn.utils.clip_grad_norm_(self._decoder_params, self.cfg.training.grad_clip)
                    window_grad_dec.append(grad_norm_dec.detach())
                self.optim.step()
```

Replace line 699 (`peaks = torch.stack(...)`):

```python
                    grad_dec_max = torch.stack(tuple(window_grad_dec)).float().max() if window_grad_dec else torch.zeros_like(loss_mean)
                    # (3,) [window-max loss, window-max pre-clip grad norm, same for the decoder (0 when frozen)]:
                    # spike detectors, so reduced with MAX (worst rank wins).
                    peaks = torch.stack((window.max(), torch.stack(tuple(window_grad)).float().max(), grad_dec_max))
```

Replace line 712:

```python
                        loss_max, grad_max, grad_dec_max = peaks.tolist()
```

After line 720 (`self.log_add_scalar('Train/GradNorm', ...)`):

```python
                        if window_grad_dec:
                            self.log_add_scalar('Train/GradNormDec', grad_dec_max, self.cfg.training.iter)
                            self.log_add_scalar('Train/LearningRateDec', self.optim.param_groups[-1]['lr'], self.cfg.training.iter)
```

Replace line 730 (`f"grad: {grad_max:.2f}  "`):

```python
                            f"grad: {grad_max:.2f}  " + (f"gradDec: {grad_dec_max:.2f}  " if window_grad_dec else "") +
```

**1g.** Eval. After line 960 (`loss_feat = self.feat_loss(x_pred, ...)`), same indent:

```python
                            loss_feat_tok = self.feat_loss(x_spatial, tokens, H, W, num_cameras)  # decode(GT z): the decoder's own recon
```

After line 962 (`batch_losses["LossFeat"] = ...`):

```python
                        batch_losses["LossFeat_tok"] = loss_feat_tok.item()
```

## 2. Base trainer — `occrae/abstract_trainer.py`

`get_optim` flattens a list of modules into one group. Let a list element be a ready param-group dict. Replace lines 237–242:

```python
        if isinstance(net, list):
            params = []
            for n in net:
                params += [n] if isinstance(n, dict) else list(n.parameters())   # dict = a ready param group
        else:
            params = list(net.parameters())
```

The adamw/adam/sgd paths take the mixed list as is; `initial_lr` is set per group at line 268, and the resume load at
line 280 now sees two groups, as saved. The muon path is unchanged and does not accept dict groups.

## 3. Sampler — `eval_deltatok_flow_sampler.py`

After line 247 (the `[INFO] Loaded flow ViT` print):

```python
    trainer.load_decoder_state(trainer.deltatok, ckpt)   # finetuned decoder if the run trained one; no-op otherwise
```

No flag needed: the helper reads the ckpt. The `.out` line `loaded finetuned DeltaTok decoder` is the proof the
re-eval used it.

## 4. Config — `configs/deltatok_flow/train_deltatok_flow.yaml`

After line 62 (`feat_loss_weight: 0.0`):

```yaml
  feat_loss_train_decoder: false  # LossFeat also trains deltatok.decoder_blocks (encoder frozen); needs feat_loss_weight > 0
```

After line 106 (`lr: 1e-4`):

```yaml
  decoder_lr: 1e-5       # lr of the trainable DeltaTok decoder group (feat_loss_train_decoder); warmup/schedule as lr
```

## 5. Slurm — `slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc.slurm`

```bash
cp slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_bsc.slurm slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc.slurm
```

Change only:

```bash
#SBATCH --job-name=deltatok_flow_tc1536_p24576_featloss_traindec
#SBATCH --output=slurm/output/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc_%j.out
#SBATCH --error=slurm/output/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc_%j.err

# Featloss arm BSC:46012704 + trainable DeltaTok decoder (model.feat_loss_train_decoder, QUESTION 24).
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc.slurm'"

export RUN_NAME="${RUN_NAME:-deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc1536mg9sigreg002pool24576compose_dndetach_ep30tok_featloss1_traindec_xxl_dit}"
```

and, after the `model.feat_loss_weight=...` line in `EXTRA_CFG_ARGS`:

```bash
    model.feat_loss_train_decoder=${TRAIN_DECODER:-true}   # decoder_blocks take LossFeat grad; the control ran false
    training.decoder_lr=${DECODER_LR:-1e-4}                # = the flow lr; the 09-04 decoder finetune moved at 1e-4
```

Everything else (2 nodes × 4 GPU, `bsize` 8, `effective_bsize` 64, 48 h, `acc_ehpc`, `ehpc880`) stays as the control's.

## Pre-flight

1. **Cluster copy.** The user syncs. Before any `sbatch`:
   ```bash
   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && grep -c feat_loss_train_decoder occrae/deltatok_flow_trainer.py configs/deltatok_flow/train_deltatok_flow.yaml && grep -c \"isinstance(n, dict)\" occrae/abstract_trainer.py && grep -c load_decoder_state eval_deltatok_flow_sampler.py && grep -n \"RUN_NAME=\\|train_decoder\\|decoder_lr\" slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc.slurm'"
   ```
2. **Smoke**, 30 min on `acc_debug`, from a login shell at the repo root:
   ```bash
   RUN_NAME=smoke_featloss1_traindec sbatch --qos=acc_debug --time=00:30:00 slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc.slurm
   ```
   Pass when the `.out` shows `feat_loss_train_decoder=True decoder_lr=0.0001` (the smoke `BSC:46185767` ran the earlier 1e-05 default; plumbing only), a `trainable DeltaTok decoder: ...M params`
   line (expect ~340 M; record the exact count), a sanity eval line with `LossFeat_tok` below `LossFeat` (0.1144 on the
   control's sanity pass) and on the `LossRecon` scale (0.02–0.05), and a `Training (Epoch 0)` line with a non-zero
   `gradDec:`. Record `time:` (control 0.82 s/update) and `max gpu mem` (control 44389 of 64805 MB; expect +~5.4 GB of
   fp32 grads + Adam state). **Measured on `BSC:46185767`:** 340.5M decoder params, `gradDec` 0.20-0.21, sanity `LossFeat_tok`
   0.0194 against `LossFeat` 0.1144, 52037 of 64805 MB (+7.6 GB), 0.93 s/update (+13%). Every other sanity-eval key
   reproduced the control's exactly, as it must at iter 0. Wait for the first `current.pth` (iter 1000), then `scancel`.
   - OOM: drop to `training.bsize=4` (`grad_cum` 2, `effective_bsize` unchanged) and re-smoke.
   - `gradDec: 0.00`: the decoder is not in the graph. Check `requires_grad_(True)` ran before the first `feat_loss`.
3. **Resume check** — SKIPPED at the user's call 2026-09-20; the two-group optimizer restore and the decoder overlay are untested, and a relaunch after the 48 h wall is the first thing that will exercise them. Relaunch the smoke with the same `RUN_NAME`. Pass when the `.out` shows
   `loaded finetuned DeltaTok decoder (N tensors)`, `Number of iteration(s): 1000`, and no optimizer state error. This is
   the 48 h wall's path; do not skip it. Then delete the smoke's `ckpts/`.
4. **Prod**, one 48 h job, no chain:
   ```bash
   sbatch slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_featloss_traindec_bsc.slurm
   ```
   Watch until `RUNNING` and the first loss line. The epoch-0 `_tok` rows must still equal the control's
   (`LossPointmap_tok` 3.8634, `LossDepth_tok` 2.6313, `LossRaymap_tok` 1.6268): the decoder has not moved yet. From
   epoch 1 on they drift; that is the read, not a bug.
5. **Read.** Training-log evals are rank-matched to `BSC:46012704` (8 ranks) but not to `BSC:45861488` (4). The read is
   the 1-GPU common-seed sweep on `iter_100000.pth` (matched ep 50):
   ```bash
   export NUM_STEPS="1,5,10,15,20"
   CKPT=$SCRATCH/quan/deltatok_flow_log/<ARM RUN_NAME>/ckpts/iter_100000.pth sbatch --export=ALL slurm/eval_deltatok_flow_numsteps_fd_tc1536p24576_bsc.slurm
   ```
   The `.out` must print `loaded finetuned DeltaTok decoder`. Keys: `LossPointmap` / `LossDepth` / `LossRaymap` /
   `MSEToken` / `FVD` / `LossFeat` / `LossFeat_tok` and the `_tok` rows, against `BSC:46182963` (control, same iter)
   and `BSC:46182962` (baseline). `Train/GradNormDec`, `Train/LearningRateDec` from TB or the `gradDec:` stdout field.

## Notes

- `feat_loss` is unchanged: with `requires_grad` on the decoder blocks, autograd already reaches their parameters
  through the same call. Activation memory is the same as the control's; the additions are grads + Adam state.
- The all-reduce runs once per optimizer update over ~150 decoder tensors with `ReduceOp.AVG` (NCCL), matching DDP's
  mean. Under `grad_cum` > 1 it sits inside `if update_grad`, so accumulated micro-batches reduce once.
- `current.pth` and `iter_*.pth` grow by the decoder state (~1.4 GB fp32). Saves are every 1000 iters; fine on `$SCRATCH`.
- The decoder group inherits `weight_decay` 0.05 and the betas from the optimizer defaults, and `adapt_learning_rate`
  scales it from its own `initial_lr`, so warmup and schedule match the ViT's. Prod runs 1e-4 = the flow lr (yaml default 1e-5 is the conservative base): the tokenizer's own cosine sat near 8e-4 at ep 30, and the 2026-09-04 decoder finetune moved at 1e-4.
- `use_ema` is false in this recipe, so eval runs the same live ViT the decoder co-adapted to. If EMA is ever turned on
  here, the decoder needs its own EMA or eval sees a mismatched pair.
- `LossFeat_tok` at eval decodes GT z teacher-forced, one step per forecast slot, like `LossFeat`. It is logged on every
  run, so the frozen-decoder controls carry the number too once re-evaluated.
- `load_decoder_state` asserts the saved keys exist in the module, so a decoder saved from a different tokenizer arch
  fails loudly rather than loading partially.
