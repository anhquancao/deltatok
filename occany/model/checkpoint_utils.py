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

    # Why we poke at blk.__dict__ directly:
    #
    # Python attribute lookup for `blk.forward` checks the instance dict
    # (blk.__dict__) first, then falls back to the class dict
    # (type(blk).__dict__). For a normal nn.Module, `forward` lives on the
    # class — so `'forward' in blk.__dict__` is False, and `blk.forward`
    # resolves via the class descriptor.
    #
    # Assigning `blk.forward = shim` does NOT modify the class; it adds a
    # 'forward' entry to blk.__dict__ that *shadows* the class descriptor.
    # To restore the block to its original shape on exit we must therefore
    # `del blk.__dict__['forward']` — overwriting with the bound method we
    # captured would leave a permanent instance attribute behind, a subtle
    # state leak (the block would no longer be in its pristine form).
    #
    # The `had` flag handles one edge case: if 'forward' was *already* an
    # instance attribute on entry (rare — e.g. monkey-patched upstream),
    # we save that value and put it back verbatim instead of deleting.
    blocks = list(blocks)
    prev_attr = []  # (had_entry, saved_value) per block
    for blk in blocks:
        had = "forward" in blk.__dict__  # was 'forward' an instance attr already?
        prev_attr.append((had, blk.__dict__.get("forward")))

        orig = blk.forward

        def make_shim(_orig):
            def _ckpt_forward(*args, **kwargs):
                return _ckpt.checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _ckpt_forward

        blk.forward = make_shim(orig)
    try:
        yield
    finally:
        for blk, (had, val) in zip(blocks, prev_attr):
            if had:
                blk.forward = val
            elif "forward" in blk.__dict__:
                del blk.__dict__["forward"]
