# DeltaTok Geom-Supervision: Gradient-Checkpointed `decode_grad` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the A100-40GB OOM in `train_deltatok_geom.sh` by gradient-checkpointing the DA3 decoder blocks walked by `OccRAE.decode_grad`, without modifying any vendored DA3 code.

**Architecture:** A small context manager in `occany/` temporarily swaps each `block.forward` of `self.model.backbone.pretrained.blocks[start_layer:]` with a `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` shim for the duration of one `decode_grad` call, then restores the originals. Gradient flow into `x_hat` is preserved; per-block activations are recomputed in backward instead of stored.

**Tech Stack:** PyTorch (`torch.utils.checkpoint`), nn.Module attribute override, contextlib.contextmanager. No third_party changes.

**Verified access path:** `OccRAE.model` → `DA3Wrapper`. `DA3Wrapper._get_pretrained_backbone()` (`occany/model/model_da3.py:63-69`) already unwraps PEFT and returns the `DinoVisionTransformer` whose `.blocks` is the per-layer `ModuleList` iterated by `_get_intermediate_layers_from_layer` (`third_party/.../vision_transformer.py:470-485`).

---

### Task 1: Block-checkpointing context manager

**Files:**
- Create: `occany/model/checkpoint_utils.py`

- [ ] **Step 1.1: Implement the context manager**

```python
# occany/model/checkpoint_utils.py
"""Scope-bounded gradient checkpointing for frozen backbone blocks.

Used by OccRAE.decode_grad to let gradients flow back to the input tokens
(x_hat from the DeltaTok tokenizer) without storing per-block activations
across the DA3 decoder stack. Frozen weights mean no parameter gradients
are computed — only the inputs need gradient — which is exactly the
gradient-checkpointing sweet spot.
"""
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.utils.checkpoint as _ckpt


@contextmanager
def checkpointed_blocks(blocks: Iterable[torch.nn.Module]):
    """Temporarily wrap each block.forward with torch.utils.checkpoint.

    No-op when autograd is disabled (e.g. under torch.no_grad), so the
    same context manager is safe to wrap around mixed grad/no_grad call
    sites. Always restores the original bound methods, even on exception.
    """
    if not torch.is_grad_enabled():
        yield
        return

    blocks = list(blocks)
    originals = []
    for blk in blocks:
        originals.append(blk.forward)
        orig = blk.forward

        def make_shim(_orig):
            def _ckpt_forward(*args, **kwargs):
                return _ckpt.checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _ckpt_forward

        blk.forward = make_shim(orig)
    try:
        yield
    finally:
        for blk, orig in zip(blocks, originals):
            blk.forward = orig
```

- [ ] **Step 1.2: Commit**

```bash
git add occany/model/checkpoint_utils.py
git commit -m "feat(occany): add checkpointed_blocks context manager for backbone blocks"
```

---

### Task 2: Wire into `OccRAE.decode_grad`

**Files:**
- Modify: `occany/model/occ_rae.py` (top of file + `decode_grad`, lines 165-193)

- [ ] **Step 2.1: Import the helper**

At the top of `occany/model/occ_rae.py`, after `from occany.utils.io_da3 import ...`:

```python
from occany.model.checkpoint_utils import checkpointed_blocks
```

- [ ] **Step 2.2: Wrap the decoder call in `decode_grad`**

Replace the body of `decode_grad` (`occany/model/occ_rae.py:179-193`):

```python
        x = latents["tokens"]
        h = latents["H"]
        w = latents["W"]
        start_layer = self.encode_layer + 1

        # Frozen DA3 weights → activations are pure recompute candidates.
        # Checkpoint every walked block so memory is O(1) in layer count
        # instead of O(L). Costs ~30% extra step time; cuts ~5-7x peak.
        backbone = self.model._get_pretrained_backbone()
        ckpt_blocks = backbone.blocks[start_layer:]

        with checkpointed_blocks(ckpt_blocks):
            return self.model.inference_batch_from_layer(
                x=x,
                start_layer=start_layer,
                h=h,
                w=w,
                local_x=None,  # layer 18 is local → x == local_x
                pose_from_depth_ray=pose_from_depth_ray,
                pose_from_cam_dec=pose_from_cam_dec,
                point_from_depth_and_pose=point_from_depth_and_pose,
            )
```

- [ ] **Step 2.3: Commit**

```bash
git add occany/model/occ_rae.py
git commit -m "feat(occrae): gradient-checkpoint DA3 decoder blocks in decode_grad"
```

---

### Task 3: Correctness test (CPU, fast)

**Files:**
- Create: `tests/test_checkpointed_blocks.py`

- [ ] **Step 3.1: Write the tests**

```python
# tests/test_checkpointed_blocks.py
"""Pure correctness tests for occany.model.checkpoint_utils. No GPU needed."""
import torch
import torch.nn as nn

from occany.model.checkpoint_utils import checkpointed_blocks


def _make_blocks(n=4, dim=8):
    torch.manual_seed(0)
    return nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])


def test_outputs_identical_with_and_without_checkpoint():
    blocks = _make_blocks()
    x = torch.randn(2, 8, requires_grad=True)

    y_ref = x
    for blk in blocks:
        y_ref = blk(y_ref)

    x2 = x.detach().clone().requires_grad_(True)
    y_ckpt = x2
    with checkpointed_blocks(blocks):
        for blk in blocks:
            y_ckpt = blk(y_ckpt)

    torch.testing.assert_close(y_ref, y_ckpt)

    y_ref.sum().backward()
    y_ckpt.sum().backward()
    torch.testing.assert_close(x.grad, x2.grad)


def test_forward_restored_after_context_exit():
    blocks = _make_blocks(n=2)
    originals = [blk.forward for blk in blocks]
    with checkpointed_blocks(blocks):
        assert all(blk.forward is not orig for blk, orig in zip(blocks, originals))
    assert all(blk.forward is orig for blk, orig in zip(blocks, originals))


def test_forward_restored_on_exception():
    blocks = _make_blocks(n=2)
    originals = [blk.forward for blk in blocks]
    try:
        with checkpointed_blocks(blocks):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert all(blk.forward is orig for blk, orig in zip(blocks, originals))


def test_no_grad_is_noop():
    blocks = _make_blocks(n=2)
    originals = [blk.forward for blk in blocks]
    with torch.no_grad(), checkpointed_blocks(blocks):
        assert all(blk.forward is orig for blk, orig in zip(blocks, originals))
```

- [ ] **Step 3.2: Run tests**

```bash
conda activate occany && pytest tests/test_checkpointed_blocks.py -v
```

Expected: 4 passed.

- [ ] **Step 3.3: Commit**

```bash
git add tests/test_checkpointed_blocks.py
git commit -m "test(checkpoint_utils): cover identity, restoration, no_grad no-op"
```

---

### Task 4: GPU smoke + memory verification on Karolina

**Files:**
- Create: `scripts/smoke_decode_grad_memory.py` (one-off; can delete after verification)

- [ ] **Step 4.1: Write the smoke script**

```python
# scripts/smoke_decode_grad_memory.py
"""Compare peak GPU memory of OccRAE.decode_grad with and without the
checkpointed_blocks context manager. Run on Karolina; expects an A100/H100.

Usage:
    conda activate occany && python scripts/smoke_decode_grad_memory.py \
        --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth
"""
import argparse
import contextlib
import torch

from occany.model.occ_rae import OccRAE
from occany.model.checkpoint_utils import checkpointed_blocks


def run_once(occ_rae, tokens, H, W, use_ckpt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokens = tokens.detach().clone().requires_grad_(True)

    backbone = occ_rae.model._get_pretrained_backbone()
    blocks = backbone.blocks[occ_rae.encode_layer + 1:]
    cm = checkpointed_blocks(blocks) if use_ckpt else contextlib.nullcontext()

    with cm:
        out = occ_rae.decode_grad({"tokens": tokens, "H": H, "W": W})
    loss = out["pointmap"].float().pow(2).mean() + out["depth"].float().pow(2).mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"use_checkpoint={use_ckpt}: peak={peak:.2f} GiB  loss={loss.item():.4f}  "
          f"grad_norm={tokens.grad.float().norm().item():.4f}")
    return peak, tokens.grad.detach().clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--occany_recon_ckpt", required=True)
    p.add_argument("--num_views", type=int, default=5)
    p.add_argument("--H", type=int, default=294)
    p.add_argument("--W", type=int, default=518)
    args = p.parse_args()

    torch.manual_seed(0)
    occ_rae = OccRAE(weights_path=args.occany_recon_ckpt, device="cuda").cuda().eval()
    occ_rae.requires_grad_(False)

    backbone = occ_rae.model._get_pretrained_backbone()
    C = backbone.embed_dim
    n_patches = (args.H // 14) * (args.W // 14) + 1
    tokens = torch.randn(1, args.num_views, n_patches, C, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        peak_off, g_off = run_once(occ_rae, tokens, args.H, args.W, use_ckpt=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        peak_on, g_on = run_once(occ_rae, tokens, args.H, args.W, use_ckpt=True)

    diff = (g_off.float() - g_on.float()).norm() / g_off.float().norm().clamp_min(1e-8)
    print(f"\nrelative input-grad diff (bf16 noise expected): {diff.item():.2e}")
    print(f"peak reduction: {peak_off / peak_on:.2f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run on Karolina**

```bash
ssh karolina "cd OccAny && conda activate occany && \
  python scripts/smoke_decode_grad_memory.py \
  --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth"
```

Expected: peak with checkpointing ≥3× smaller than without; input-grad diff < 1e-2 (bf16 noise).

---

### Task 5: End-to-end retry of the failing training command

- [ ] **Step 5.1: Re-run the originally failing command**

```bash
ssh karolina "cd OccAny && conda activate occany && bash sh/train_deltatok_geom.sh" \
  2>&1 | tee /tmp/deltatok_geom_smoke.log
```

Expected: training advances past the first geom step without `torch.OutOfMemoryError`.

- [ ] **Step 5.2: If OOM persists, escalate**

Next levers (do NOT add preemptively):
1. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `sh/train_deltatok_geom.sh`.
2. Reduce `pairs_per_batch` further or chunk over the camera dim (V) inside `decode_grad`.

Surface the new error and decide together.
