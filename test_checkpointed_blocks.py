"""Correctness tests for occany.model.checkpoint_utils.

Requires CUDA: torch.utils.checkpoint exercises CUDA RNG preservation and
device-side autograd graph manipulation, so we test on the same hardware
the production callers (OccRAE.decode_grad) actually use.
"""
import sys

import torch
import torch.nn as nn

from occany.model.checkpoint_utils import checkpointed_blocks


if not torch.cuda.is_available():
    print("SKIP: CUDA unavailable; checkpointed_blocks tests require GPU.")
    sys.exit(0)


DEVICE = torch.device("cuda")


def _make_blocks(n=4, dim=8):
    torch.manual_seed(0)
    return nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)]).to(DEVICE)


def test_outputs_identical_with_and_without_checkpoint():
    blocks = _make_blocks()
    x = torch.randn(2, 8, device=DEVICE, requires_grad=True)

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
    # nn.Module.forward resolves via the class descriptor; instance __dict__
    # has no 'forward' entry by default. The CM must restore that state.
    blocks = _make_blocks(n=2)
    assert all("forward" not in blk.__dict__ for blk in blocks)
    with checkpointed_blocks(blocks):
        assert all("forward" in blk.__dict__ for blk in blocks)
    assert all("forward" not in blk.__dict__ for blk in blocks)


def test_forward_restored_on_exception():
    blocks = _make_blocks(n=2)
    try:
        with checkpointed_blocks(blocks):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert all("forward" not in blk.__dict__ for blk in blocks)


def test_no_grad_is_noop():
    # Under no_grad the CM must not touch instance state — torch.utils.checkpoint
    # warns/errors when grad is disabled, so we don't want shims installed.
    blocks = _make_blocks(n=2)
    with torch.no_grad(), checkpointed_blocks(blocks):
        assert all("forward" not in blk.__dict__ for blk in blocks)


def test_nested_call_restores_correctly():
    # Two nested CMs on the same blocks should still restore to the original
    # state on outer exit (regression guard for stacked use, e.g. if decode_grad
    # is ever called from within an outer checkpointed scope).
    blocks = _make_blocks(n=2)
    with checkpointed_blocks(blocks):
        with checkpointed_blocks(blocks):
            assert all("forward" in blk.__dict__ for blk in blocks)
        # Outer CM still active → 'forward' still shadowed.
        assert all("forward" in blk.__dict__ for blk in blocks)
    assert all("forward" not in blk.__dict__ for blk in blocks)


if __name__ == "__main__":
    test_outputs_identical_with_and_without_checkpoint()
    print("test_outputs_identical_with_and_without_checkpoint: PASS")
    test_forward_restored_after_context_exit()
    print("test_forward_restored_after_context_exit: PASS")
    test_forward_restored_on_exception()
    print("test_forward_restored_on_exception: PASS")
    test_no_grad_is_noop()
    print("test_no_grad_is_noop: PASS")
    test_nested_call_restores_correctly()
    print("test_nested_call_restores_correctly: PASS")
