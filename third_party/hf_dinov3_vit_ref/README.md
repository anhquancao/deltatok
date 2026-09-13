# HF DINOv3 ViT — reference copy

Verbatim copy of `transformers/models/dinov3_vit/{modeling,configuration}_dinov3_vit.py` from
transformers 5.0.0 (the BSC venv, 2026-09-13), for reading only. `occrae/deltatok_trainer.py`
still imports `DINOv3ViTLayer` and `DINOv3ViTRopePositionEmbedding` from the installed package;
nothing imports this folder, and the relative imports inside make it non-importable on its own.

Read `DINOv3ViTLayer.forward` (line ~387): pre-norm, `h = h + ls1·Attn(LN1(h))`, `h = h + ls2·MLP(LN2(h))`.
The residual `h` is never normalised — that is the un-normed `z` at `target_channels == hidden_size`.
