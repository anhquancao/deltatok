#!/usr/bin/env python3
"""Check that encoding only a pair's 2 timesteps matches encoding all T.

DeltaTok training pushes just the timesteps each kept transition needs through the
frozen backbone (``pair_t`` in ``_extract_pair_feats``). That is only valid because
blocks 0..encode_layer are per-view local attention with a view-index-free rope and
cls/register tokens, so a view's features cannot depend on which other views share
the batch. This asserts it on the real backbone, and separately checks the
vectorized view-select arithmetic against an obvious Python twin.

Usage
-----
python test_pair_slice_exact.py --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth
"""
import argparse

import torch

from occany.utils.runtime_paths import prepend_vendored_import_paths

prepend_vendored_import_paths()

from occany.model.occ_rae import OccRAE


def kept_views_reference(pair_t, num_cameras):
    """Plain-Python twin of the vectorized view select: one row of view indices per
    (sequence, kept pair). View v = t*num_cameras + cam, and transition t spans t, t+1."""
    rows = []
    for b in range(pair_t.shape[0]):
        for t in pair_t[b].tolist():
            rows.append([(t + dt) * num_cameras + cam for dt in (0, 1) for cam in range(num_cameras)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occany_recon_ckpt", default="checkpoints/occany_plus_recon_1B.pth")
    ap.add_argument("--encode_layer", type=int, default=12)
    ap.add_argument("--timesteps", type=int, default=6)       # T, as in the train configs
    ap.add_argument("--num_cameras", type=int, default=2)     # len(fixed_cams)
    ap.add_argument("--bsize", type=int, default=2)
    ap.add_argument("--pairs_per_seq", type=int, default=1)
    ap.add_argument("--height", type=int, default=294)        # 518x294 == the nuScenes preset
    ap.add_argument("--width", type=int, default=518)
    args = ap.parse_args()

    torch.manual_seed(0)
    occ_rae = OccRAE(
        weights_path=args.occany_recon_ckpt, device="cuda", encode_layer=args.encode_layer,
    )
    occ_rae.eval()
    occ_rae.requires_grad_(False)

    B, T, N, n_pairs = args.bsize, args.timesteps, args.num_cameras, args.pairs_per_seq
    M = B * n_pairs                                                    # rows after folding pairs into batch
    imgs = torch.randn(B, T * N, 3, args.height, args.width, device="cuda")  # (B, T*N, 3, H, W)
    pair_t = torch.argsort(torch.rand(B, T - 1, device="cuda"), dim=1)[:, :n_pairs]  # (B, n_pairs)

    # The vectorized select from _extract_pair_feats, re-derived here.
    steps = torch.stack([pair_t, pair_t + 1], dim=-1)                  # (B, n_pairs, 2) timesteps per pair
    cams = torch.arange(N, device=imgs.device)                         # (num_cameras,)
    views = (steps[..., None] * N + cams).reshape(B, -1)               # (B, n_pairs*2*num_cameras)
    seqs = torch.arange(B, device=imgs.device)[:, None]                # (B, 1)

    ref = kept_views_reference(pair_t.cpu(), N)                        # M rows of 2*num_cameras
    got = views.reshape(M, 2 * N).tolist()
    assert got == ref, f"view select mismatch:\n got {got}\n ref {ref}"
    print(f"view select OK (rows of kept view idx): {got}")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = occ_rae.encode(imgs)["tokens"]                          # (B, T*N, N_tok, C)
        sliced_imgs = imgs[seqs, views].reshape(M, 2 * N, 3, args.height, args.width)
        sliced = occ_rae.encode(sliced_imgs)["tokens"]                 # (M, 2*N, N_tok, C)

    # Line the full encode's kept views up with the sliced encode's rows.
    idx = torch.tensor(ref, device=full.device)                        # (M, 2*N) view idx per row
    row_seq = torch.arange(B, device=full.device).repeat_interleave(n_pairs)  # (M,) sequence of each row
    expected = full[row_seq[:, None], idx]                             # (M, 2*N, N_tok, C)

    scale = expected.float().abs().max().item()
    max_diff = (expected.float() - sliced.float()).abs().max().item()
    print(f"encoded {tuple(sliced.shape)} vs full {tuple(full.shape)}: "
          f"max|diff| {max_diff:.3e} on scale {scale:.3e} (rel {max_diff / scale:.2e}), "
          f"exactly equal: {torch.equal(expected, sliced)}")
    # Mathematically identical, so the only residual is kernel nondeterminism from the
    # changed batch shape: a few ulp of bf16 (eps = 2^-8 = 3.9e-3), accumulated over 13
    # blocks. A mispairing instead gives relative error ~1, so 5e-2 separates them 20x
    # either way -- tightening this below one ulp only buys spurious failures.
    assert max_diff / scale < 5e-2, "sliced encode does not match the full encode"
    print(f"PASS: encoding 2 of {T} timesteps reproduces the full encode on the kept views.")


if __name__ == "__main__":
    main()
