# Plan: eval-only Fréchet distance on OccAny tokens, and a step sweep on BSC:45861488 (QUESTION 10)

**Date:** 2026-09-16, updated 2026-09-17 · **Thread:** flow · **Cluster:** BSC
· **Target:** `BSC:45861488`, `deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc1536mg9sigreg002pool24576compose_dndetach_ep30tok_xxl_dit`

FD is eval-only. The trainer collects pooled tokens only when `eval_deltatok_flow_sampler.py --fvd` switches it on,
so training evals are unchanged. The script computes FD with `torchaudio.functional.frechet_distance` and prints one
line per pass. Patch tokens of each forecast view are mean-pooled to one vector per image. One new slurm script
sweeps steps 1, 5, 10, 15, 20; each pass already prints `LossPointmap`, `LossDepth`, `LossRaymap`.

All line numbers below are today's files, before any edit.

## 1. Trainer — `occrae/deltatok_flow_trainer.py` (collect only, off by default)

**1a.** After line 76 (`self._err_spectrum = {}`):

```python
        self._fvd_feats = None        # {test_name: {"flow"|"tok"|"gt": [(B*F, C)]}}; None = off, set by the sampler script's --fvd
```

**1b.** After line 762 (`err_rows = 0`):

```python
                if self._fvd_feats is not None:
                    self._fvd_feats[test_name] = {"flow": [], "tok": [], "gt": []}  # fresh per loader, per pass
```

**1c.** After line 868 (`pred_slice = ...`):

```python
                        if self._fvd_feats is not None:
                            pfx = self._num_prefix_tokens
                            for key, tk in (("flow", full_tokens), ("tok", full_tokens_tok), ("gt", tokens)):  # each (B, V, N_tok, C)
                                # forecast views, patch tokens only, mean over patches -> one vector per image
                                self._fvd_feats[test_name][key].append(
                                    tk[:, pred_slice, pfx:].float().mean(2).reshape(-1, tk.shape[-1]).cpu())  # (B*F, C)
```

## 2. Eval script — `eval_deltatok_flow_sampler.py`

**2a.** After line 34 (`import torch`):

```python
from torchaudio.functional import frechet_distance
```

**2b.** After line 120 (end of the `--viz_rgb` argument):

```python
    parser.add_argument(
        "--fvd", action="store_true",
        help="Fréchet distance on OccAny patch tokens (mean-pooled per forecast image) vs GT: "
             "sampled rollout (FVD) and GT-delta rollout (FVD_tok). One line per pass.",
    )
```

**2c.** After `_sanitize` (lines 126–127):

```python
def _frechet_distance(a, b):
    """FD between two (N, C) feature sets, each fitted as a Gaussian."""
    a, b = a.double(), b.double()                                           # (N1, C), (N2, C)
    return frechet_distance(a.mean(0), torch.cov(a.T), b.mean(0), torch.cov(b.T)).item()  # scalar
```

**2d.** After line 244 (`_bank_z_basis(trainer, cfg)`):

```python
    if args.fvd:
        trainer._fvd_feats = {}  # switches on the trainer's pooled-token collection
```

**2e.** After line 272 (the `_err_spectrum` `torch.save`), at the `if trainer._err_spectrum:` indent:

```python
                for test_name, f in (trainer._fvd_feats or {}).items():
                    f = {k: torch.cat(v, 0) for k, v in f.items()}                    # {"flow"|"tok"|"gt": (N, C)}
                    print(f"[FVD/{test_name}] steps={n_steps} mode={mode} sigma={sigma} n={f['gt'].shape[0]}  "
                          f"FVD: {_frechet_distance(f['flow'], f['gt']):.4f}  "
                          f"FVD_tok: {_frechet_distance(f['tok'], f['gt']):.4f}", flush=True)
```

## 3. Slurm — `slurm/eval_deltatok_flow_numsteps_fd_tc1536p24576_bsc.slurm`

```bash
cp slurm/eval_deltatok_flow_numsteps_tc768_bsc.slurm slurm/eval_deltatok_flow_numsteps_fd_tc1536p24576_bsc.slurm
```

Change only:

```
#SBATCH --job-name=deltatok_flow_numsteps_fd_tc1536
#SBATCH --qos=acc_debug
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/eval_deltatok_flow_numsteps_fd_tc1536p24576_bsc_%j.out
#SBATCH --error=slurm/output/eval_deltatok_flow_numsteps_fd_tc1536p24576_bsc_%j.err
```

Header, lines 14–20:

```
# Step sweep + FD on the tc1536 pool-24576 flow arm (BSC 45861488): one eval_one_epoch
# per NUM_STEPS entry, ode only; steps=20 reproduces the run's logged eval.
# Arch flags mirror slurm/deltatok_flow/train_deltatok_flow_waymo_xxl_tc1536mg9_pool24576_bsc.slurm.
#   ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/eval_deltatok_flow_numsteps_fd_tc1536p24576_bsc.slurm'"
```

Lines 35–37:

```
: "${CKPT:=$SCRATCH/deltatok_flow_log/deltatok_flow_waymo_consec5cam0_ctx3fwd2_tc1536mg9sigreg002pool24576compose_dndetach_ep30tok_xxl_dit/ckpts/current.pth}"
: "${OUTPUT_DIR:=results/deltatok_flow_numsteps_fd_tc1536p24576}"
: "${NUM_STEPS:=1,5,10,15,20}"
```

Lines 42–44:

```
    # epoch_30: the exact tokenizer the flow arm trained against.
    model.deltatok_ckpt="$SCRATCH/deltatok_log/deltatok_l12_dtok64_tc1536_nozn_maxgap9_vpt1to2_sigreg0.02_ns3072_pool24576_compose1.0_decnoise0.8_detach_sw0/ckpts/epoch_30.pth"
    model.deltatok.target_channels=1536        # == hidden size, no z bottleneck
```

Line 66: `--config-name train_deltatok_flow_waymo_ep128k_bsc \` (the run's config; test set inherited).
After line 70 (`--step_modes`): `--fvd \`.

The other `CFG_ARGS` already match the run's `EXTRA_CFG`.

## Pre-flight

- **Cluster copy:** grep `_fvd_feats`, `frechet` and the new slurm on BSC before `sbatch`. The user syncs.
- **torchaudio:** `2.5.1+cu121` in the `env_bsc.sh` venv (checked 2026-09-17).
- **Walltime:** the tc768 sweep ran 246 steps over 5 passes in 10 min (`BSC:44663476`); this one is 51 steps.
- **Watch** until the first `[Eval/` line.
- **Checks:** the steps=20 row matches the run's `[Eval/128 @ WaymoSeqMultiView]` line at the `global_epoch` the
  script prints. `FVD_tok` is identical on every pass. `FVD_tok < FVD`.
- **Read:** `grep -E "=====|\[Eval/|\[FVD/"` on the `.out`; each `[Eval/` line follows its `eval_num_steps=N` header.

## Notes

- `n` = 128 seqs × 2 forecast frames × cam 0 = 256 vectors in C = 1536, so the covariance has rank ≤ 255. Compare FD
  within this sweep only.
- The eval script runs one process, so there is no cross-rank gather.
- Key named `FVD` to match QUESTION 10. It is a per-image Fréchet distance, not a video one.
