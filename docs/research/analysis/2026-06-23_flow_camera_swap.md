# DeltaTok flow: camera swap in generated views

Status: **open / partially addressed** (2026-06-23). Fix implemented (learned
per-camera embedding); not yet confirmed on a run. Reference image:
`2026-06-23_flow_camera_swap.png`.

## Symptom

In the DeltaTok flow eval viz (overfit run on nuScenes scene-0625), the generated
("Flow") views have their cameras **swapped** — content/pose that belongs to one
surround camera shows up in another camera's slot.

## Key evidence: the swap is generator-side, not pipeline-side

The viz has two rollout columns built by the **same** decode path
(`_rollout_from_z`), same t0 seed, differing only in the source of the delta token `z`:

- **GT-z** (rollout fed the *true* tokenizer deltas): **not swapped**.
- **Flow** (rollout fed the *flow-sampled* deltas `z_hat`): **swapped**.

So the rollout, the `view(B, T, num_cameras, ...)` reshapes, the tokenizer decode,
and the viz ordering are all fine (GT-z shares them). The swap is entirely in how
the flow model produces `z_hat`. Confirmed also that:
- `flow_euler_sample` does pass `cross_cond` every step (generation_helper.py:74).
- `z_hat`'s layout matches GT `z` (the `permute(0,2,3,1)` inverts the encode-side
  `permute(0,3,1,2)`), so no reshape/permute bug.

## Diagnosis: generator binding failure

The flow ViT emits one delta per camera slot but has no *stable, strong* per-slot
identity:
- spatial pos-embed is shared across slots (`ref_spatial_size=(1,1)` broadcasts one
  vector to every camera),
- the camera rope was either shuffled per-forward (no stable identity) or removed,
- so the only per-slot signal is `cross_cond` content, and its residual is
  AdaLN-zero-gated (`alpha_cross` starts at 0).

Slots are therefore near-exchangeable. Teacher-forced flow loss can stay low without
firmly binding each delta to its camera (the 6 surround deltas have similar
statistics), but free sampling from per-slot independent noise lets slots swap.

## "But decode is anchored on the t0 frame — how can it swap?"

The decode is **slot-faithful**: output camera `i` is written from input slot `i` and
residual-anchored to `x_prev_i`. Local blocks are per-camera (independent); global
blocks concat all cameras, self-attend, then split back, but with a residual
(`x = x + attn`) so output slot `i` = input slot `i` + a perturbation. So content does
**not** freely bleed/swap *between* camera slots in the decode — GT-z proves the
correspondence holds (same global `pos_nodiff` layers, no swap). (An earlier version of
this note claimed cross-camera "bleed"; that was wrong — the input↔output slot
correspondence is structural.)

That *sharpens* the diagnosis: since the decode is slot-faithful, the swap must be
carried **inside `z_hat`** — slot `i`'s generated delta is effectively another camera's
delta, and the decode faithfully applies it to `frame0_i`. The anchor constrains the
*start*, not the *destination*:
- `z` is threaded through **every** decoder block and rewrites the spatials at each
  layer (deltatok_trainer.py:346-361); a trained `z` carries enough to turn frame `t`
  into `t+1` across the whole grid, so a mis-bound `z` can push `frame0_i` a long way.
- Only the **first** transition uses the true t0; the rollout feeds its own output back
  (`x_hat_prev = x_hat_t`, deltatok_shared.py:242) → compounding.

How visibly it swaps depends on how much `z` dominates the t0 anchor: if deltas were
purely incremental the anchor would win and a wrong `z` would only smear, so a clean
swap implies `z` carries real absolute content (heavy 1-token-per-camera bottleneck)
and/or the swap is clearest in pose (BEV) / later compounded frames. The permute-GT-z
probe below settles which.

## Why not RoPE

RoPE is **relative**: in attention it only enters `q·k` as a function of the offset
`(i−j)`, and values are never rotated — so it never writes an absolute "I am camera
`i`" tag into a token's content/residual stream. The tokenizer gets away with camera
rope because its decode is seeded with `x_prev` (camera `i`'s real patches), so
absolute identity is already in the content. The flow generator starts from noise, so
it needs an **absolute** identity signal, which RoPE is not.

## Fix implemented

Absolute, additive, ungated **learned per-camera embedding** over the flow-ViT camera
(spatial) axis — one distinct vector per slot, like `temporal_pos` but over cameras —
so each generated delta binds to a fixed camera.

- `occrae/network/efficient_transformer.py`: `Transformer(use_camera_embed=...)`
  builds `nn.Embedding(max_cameras, hidden_dim)`, added into the position sum in
  `forward` as `repeat(camera_embed(arange(S)), 's d -> (t s) d', t=t)` (S = N_cam).
- `occrae/deltatok_flow_trainer.py`: passes `vit_use_camera_embed`.
- `configs/train_deltatok_flow.yaml`: `vit_use_camera_embed: false` default.
- `slurm/deltatok_flow/train_deltatok_flow_jz.slurm`: overfit run `deltatok_flow_overfit_camEmbed`
  — `model.vit_use_camera_embed=true`, tokenizer reverted to **const-global-rope**
  (config-default ckpt `deltatok_surround_constGlobalRope_layer12`, no camera rope).

The shared `(1,1)` `spatial_pos` is left in place (a harmless constant offset); the
`camera_embed` carries the identity.

## Ruled out

- **cross_cond dropped during sampling** — `flow_euler_sample` passes `cross_cond`
  every step (generation_helper.py:74).
- **`z_hat` layout vs GT `z`** — `permute(0,2,3,1)` inverts the encode-side
  `permute(0,3,1,2)`; same `(B, T-1, N, C)`.
- **cross-camera bleed in decode** — decode is slot-faithful (residual-anchored,
  input slot `i` → output slot `i`); GT-z proves it.
- **encode/decode camera-order mismatch in the flow path** — the camera (N) axis is
  the `feats` order everywhere (target `z`, `cross_cond`, model in/out, rollout seed),
  with no data shuffle (the old rope `randperm` only permuted rope *positions*, not
  the data; `camera_embed` uses `arange`). Decisive: GT-z shares the full
  encode→rollout path and is not swapped, so any shared-order mismatch would show
  there too.

## TODO / diagnostic to run later

Prove the mechanism: in eval, take the **GT** `z`, permute it across the camera axis
(`z[:, :, perm]`), and decode with the *unpermuted* t0 frames. If
`decode(permuted GT-z, frame0)` reproduces the Flow-column swap, that confirms
"mis-bound `z` → visible swap despite the t0 anchor" and shows how much the delta
overrides the anchor.
