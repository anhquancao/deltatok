# OccAny+ DINOv3 Backbone (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. All GPU work (forward smokes, training launches) must use the `karolina-job` skill — there is no local GPU.

**Spec:** `docs/superpowers/specs/2026-05-21-occany-dinov3-backbone-design.md`

**Goal:** Train an OccAny+ recon model with the same loss stack as `sh/train_occany_plus_recon_1B_infinite_depth.sh`, but with DINOv3 ViT-H+/16 (LVD-1689M) as a vanilla per-view backbone replacing DA3-GIANT-1.1. No cross-view attention, no SAM3, no gen mode — Stage 1 validates the simplest possible swap before Stage 2 adds cross-view machinery.

**Architecture:** A thin `DinoV3PerViewBackbone` module wraps an unmodified `dinov3.models.vision_transformer.DinoVisionTransformer`, taps four intermediate layers via `get_intermediate_layers(...)`, and produces a list of `(patch_feat, cls_token)` tuples shaped to drop into DA3's existing `DualDPT` head. A new recon-only `Dinov3Wrapper` (does **not** subclass `DepthAnything3`) owns the backbone and head directly — exposing `model.backbone` and `model.head` (one level shallower than `DA3Wrapper`'s `model.model.backbone`). The existing `training_da3.py` is forked into `training_dinov3.py` with all gen / SAM3 / aux paths deleted and every `model.model.X` access rewired to `model.X`. The DA3-side `'da3'` branch in datasets is widened to also accept `'dinov3'` (same ImageNet normalization — verified bit-for-bit).

**Tech stack:** PyTorch + bf16 autocast + DDP, `dinov3` vendored package (added to `runtime_paths` and `train_common.sh`), DA3's `DualDPT` head imported in isolation, `karolina-job` skill for cluster execution, SLURM `qgpu` partition / account `eu-25-92` / 4×8 A100 40Gb.

---

## File map

**Create:**

| Path | Responsibility |
|---|---|
| `occany/configs/dinov3/vith16plus.yaml` | DINOv3 ViT-H+/16 arch kwargs + DPT tap layers |
| `occany/model/dinov3_backbone.py` | `DinoV3PerViewBackbone` + `build_dinov3_per_view` factory |
| `occany/model/model_dinov3.py` | `Dinov3Wrapper` (recon-only, per-view) |
| `occany/training_dinov3.py` | Recon-only training loop (fork of `training_da3.py`) |
| `launch_dinov3.py` | torchrun/DDP entrypoint (clone of `launch_da3.py`) |
| `sh/train_occany_plus_recon_dinov3_vith16plus.sh` | Shell wrapper |
| `slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm` | SLURM wrapper for Karolina |
| `test_dinov3_pca.py` | Standalone PCA-feature visualization CLI |

**Modify (small, additive):**

| Path | What changes |
|---|---|
| `occany/utils/runtime_paths.py` | Add `third_party/dinov3` to `_VENDORED_PATHS` |
| `sh/train_common.sh` | Add `third_party/dinov3` to `occany_prepend_pythonpath` |
| `occany/datasets/base_seq_dataset.py:179` | Widen `== 'da3'` → `in ('da3', 'dinov3')` |
| `occany/datasets/base_seq_dataset.py:499` | Widen `== 'da3'` → `in ('da3', 'dinov3')` |
| `occany/datasets/kitti.py:493` | Widen `== 'da3'` → `in ('da3', 'dinov3')` |
| `occany/datasets/nuscenes.py:835` | Widen `== 'da3'` → `in ('da3', 'dinov3')` |

**Out of scope (left untouched):** `occany/datasets/eval_helper.py` (only used by `extract_output_occany.py` eval pipeline, which is follow-up); `occany/model/model_da3.py`, `occany/training_da3.py`, `launch_da3.py`; the `dinov3` vendored tree itself.

---

## Task 1: Preflight verification

**Files:** none modified — Karolina ssh sanity checks only.

Verifications from spec §6, run before writing any code. Karolina sync runs continuously, so the repo state on Karolina mirrors local.

- [ ] **Step 1.1: Confirm DINOv3 checkpoint exists on Karolina**

```bash
ssh karolina 'cd ~/OccAny && ls -lh checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth 2>&1 || echo MISSING'
```

Expected: file present (~3.2 GB) at `~/OccAny/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`. The `-7c1da9a5` hash suffix matches the hash in `dinov3.hub.backbones.dinov3_vith16plus` (`kwargs["hash"] = "7c1da9a5"`) — this is the native Meta release format that loads with `strict=True` into `DinoVisionTransformer` (no key remapping needed; verified in Step 4.3). If MISSING: ask the user to download from the Meta DINOv3 release page (the license requires accepting the access form — only the user can do this). Do not proceed until the file exists.

- [ ] **Step 1.2: Confirm `DualDPT` imports in isolation**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  PYTHONPATH=third_party/Depth-Anything-3/src python -c \
  "from depth_anything_3.model.dualdpt import DualDPT; \
   import inspect; sig = inspect.signature(DualDPT.__init__); print(\"DualDPT OK\", sig)"'
```

Expected output: `DualDPT OK (self, dim_in: int, *, patch_size: int = 14, output_dim: int = 2, activation: str = 'exp', conf_activation: str = 'expp1', features: int = 256, out_channels: Sequence[int] = (256, 512, 1024, 1024), pos_embed: bool = True, down_ratio: int = 1, aux_pyramid_levels: int = 4, aux_out1_conv_num: int = 5, head_names: Tuple[str, str] = ('depth', 'ray'))`

If import fails (e.g., pulls in unavailable deps), fall back to a hand-rolled DPT head — Task 5 has Plan B written out.

- [ ] **Step 1.3: Confirm DINOv3 normalization matches DA3 (bit-for-bit)**

```bash
ssh karolina 'cd ~/OccAny && grep -n "NORMALIZE\s*=\|IMAGENET_DEFAULT" \
  third_party/Depth-Anything-3/src/depth_anything_3/utils/io/input_processor.py \
  third_party/dinov3/dinov3/data/transforms.py'
```

Expected: both files use mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. If they diverge, stop and re-read the spec — the dataset-edit shortcut assumes parity.

- [ ] **Step 1.4: Confirm `DinoVisionTransformer` is buildable from the YAML kwargs and accepts strict checkpoint load**

Defer to Task 4 step 4.5 (it requires the YAML + module to exist). For now, only verify the file layout: `ls third_party/dinov3/dinov3/models/vision_transformer.py` returns the file.

- [ ] **Step 1.5: Confirm there are no out-of-scope `base_model == 'da3'` branches that would diverge for DINOv3**

```bash
cd /home/acao/code/OccAny && grep -rn "base_model" occany/ | grep -v "^Binary" | grep -v __pycache__
```

Expected sites:
- `occany/datasets/base_seq_dataset.py:67,72,179,485,499` — 67/72 are param/assignment, 485 is a comment, 179/499 are edit targets.
- `occany/datasets/kitti.py:99,102,493` — 99/102 are param/assignment, 493 is edit target.
- `occany/datasets/nuscenes.py:85,128,835` — 85/128 are param/assignment, 835 is edit target.
- `occany/datasets/eval_helper.py:77,84,89,130,171` — 77 is param default, 130/171 are pass-through. **84 and 89 branch on `'da3'` to set `output_resolution = (518, 168)` / `(518, 294)`** — this is only consumed by `extract_output_occany.py` (eval pipeline), which Stage 1 does **not** run. Leave unchanged. Document in commit message of Task 7.
- `occany/model/model_da3.py:59-60` — unrelated to this branch logic.

No other sites. If grep returns additional surprises, stop and reconcile before Task 7.

---

## Task 2: Add `third_party/dinov3` to vendored import paths

**Files:**
- Modify: `occany/utils/runtime_paths.py`
- Modify: `sh/train_common.sh`

The `dinov3` package lives at `third_party/dinov3/dinov3/`. The existing vendored path list includes `third_party/` (which makes `third_party/dust3r` etc. importable as packages) but **not** `third_party/dinov3` — meaning `import dinov3.models.vision_transformer` would resolve to `third_party/dinov3/models/...` (doesn't exist) instead of `third_party/dinov3/dinov3/models/...`. Both helpers need a one-line addition.

- [ ] **Step 2.1: Add `third_party/dinov3` to `runtime_paths.py`**

Edit `occany/utils/runtime_paths.py` — the `_VENDORED_PATHS` tuple at line ~8. Append one entry:

```python
_VENDORED_PATHS = (
    Path("third_party"),
    Path("third_party/dust3r"),
    Path("third_party/croco/models/curope"),
    Path("third_party/Grounded-SAM-2"),
    Path("third_party/Grounded-SAM-2/grounding_dino"),
    Path("third_party/sam3"),
    Path("third_party/Depth-Anything-3/src"),
    Path("third_party/InfiniDepth"),
    Path("third_party/dinov3"),
)
```

- [ ] **Step 2.2: Add `third_party/dinov3` to `train_common.sh`**

Edit `sh/train_common.sh`, the `occany_prepend_pythonpath` array. Append one entry to the `vendored_paths` array:

```bash
occany_prepend_pythonpath() {
    local repo_root="$1"
    local -a vendored_paths=(
        "$repo_root/third_party"
        "$repo_root/third_party/dust3r"
        "$repo_root/third_party/croco/models/curope"
        "$repo_root/third_party/Grounded-SAM-2"
        "$repo_root/third_party/Grounded-SAM-2/grounding_dino"
        "$repo_root/third_party/sam3"
        "$repo_root/third_party/Depth-Anything-3/src"
        "$repo_root/third_party/dinov3"
    )
    ...
}
```

- [ ] **Step 2.3: Smoke-test the import on Karolina**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "from occany.utils.runtime_paths import prepend_vendored_import_paths; \
             prepend_vendored_import_paths(); \
             from dinov3.models.vision_transformer import DinoVisionTransformer; \
             print(\"OK\", DinoVisionTransformer.__module__)"'
```

Expected output: `OK dinov3.models.vision_transformer`. If it fails with `ModuleNotFoundError: No module named 'dinov3'`, the edits didn't land — re-check both files.

- [ ] **Step 2.4: Commit**

```bash
git add occany/utils/runtime_paths.py sh/train_common.sh
git commit -m "feat(occany): add third_party/dinov3 to vendored import paths"
```

---

## Task 3: DINOv3 YAML config

**Files:**
- Create: `occany/configs/dinov3/vith16plus.yaml`

The `arch:` block mirrors `dinov3.hub.backbones._make_dinov3_vit` kwargs passed by `dinov3_vith16plus` (verified against `third_party/dinov3/dinov3/hub/backbones.py:413-449`). `img_size: 224` is the standard default and only affects `PatchEmbed.__init__` bookkeeping — per-block RoPE is computed from the *actual* patch grid at forward time, so the same weights load and run unchanged at 528×176.

- [ ] **Step 3.1: Create the directory and YAML**

```bash
mkdir -p occany/configs/dinov3
```

Then create `occany/configs/dinov3/vith16plus.yaml` with:

```yaml
arch:
  img_size: 224
  patch_size: 16
  in_chans: 3
  embed_dim: 1280
  depth: 32
  num_heads: 20
  ffn_ratio: 6.0
  ffn_layer: swiglu
  layerscale_init: 1.0e-05
  norm_layer: layernormbf16
  n_storage_tokens: 4
  mask_k_bias: true
  qkv_bias: true
  proj_bias: true
  ffn_bias: true
  drop_path_rate: 0.0
  pos_embed_rope_base: 100.0
  pos_embed_rope_normalize_coords: separate
  pos_embed_rope_rescale_coords: 2
  pos_embed_rope_dtype: fp32

# DPT tap layers — quartered across the 32-block depth, by analogy to DA3-GIANT.
# Tunable after the first training run.
out_layers: [17, 21, 26, 31]
```

- [ ] **Step 3.2: Commit**

```bash
git add occany/configs/dinov3/vith16plus.yaml
git commit -m "feat(occany): add DINOv3 ViT-H+/16 backbone YAML config"
```

---

## Task 4: `DinoV3PerViewBackbone` + factory

**Files:**
- Create: `occany/model/dinov3_backbone.py`

A small `nn.Module` that owns the published DINOv3 ViT and exposes a per-view `forward_multiview` returning `(patch_feat, cls_token)` tuples shaped to drop into `DualDPT`. The DINOv3 ViT's `get_intermediate_layers(...)` already strips storage tokens internally (verified at `third_party/dinov3/dinov3/models/vision_transformer.py:285-323`); we just call it.

- [ ] **Step 4.1: Write the file**

Create `occany/model/dinov3_backbone.py`:

```python
"""Per-view DINOv3 ViT-H+/16 backbone wrapper for OccAny+ Stage 1.

No cross-view machinery. Each view is processed independently through an
unmodified `dinov3.models.vision_transformer.DinoVisionTransformer`, with
multiple intermediate layers tapped for the DPT head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import yaml

from dinov3.models.vision_transformer import DinoVisionTransformer


class DinoV3PerViewBackbone(nn.Module):
    def __init__(self, vit: DinoVisionTransformer, out_layers: Iterable[int]):
        super().__init__()
        self.vit = vit
        self.out_layers = tuple(int(idx) for idx in out_layers)
        if len(self.out_layers) == 0:
            raise ValueError("DinoV3PerViewBackbone requires at least one tap layer")
        if max(self.out_layers) >= self.vit.n_blocks:
            raise ValueError(
                f"out_layers={self.out_layers} exceeds n_blocks={self.vit.n_blocks}"
            )

    def forward_multiview(
        self, images: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run DINOv3 ViT per-view and return tapped features.

        Args:
            images: (B, T, 3, H, W). H, W must be multiples of patch_size (16).

        Returns:
            List of length len(out_layers). Each entry is a tuple
            `(patch_feat, cls_token)` where:
              - patch_feat: (B, T, N_patch, embed_dim) with N_patch = (H//16)*(W//16)
              - cls_token: (B, T, embed_dim) — included for signature parity with DA3,
                not consumed by the recon path in Stage 1.
        """
        if images.ndim != 5:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(images.shape)}")
        B, T, C, H, W = images.shape
        flat = images.reshape(B * T, C, H, W)

        # DINOv3's get_intermediate_layers strips the storage tokens internally,
        # returns the patch tokens (post-norm) and the cls token per tap.
        taps = self.vit.get_intermediate_layers(
            flat,
            n=self.out_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for patch_tokens, cls_token in taps:
            # patch_tokens: (B*T, N_patch, embed_dim)
            # cls_token:    (B*T, embed_dim)
            N_patch = patch_tokens.shape[-2]
            E = patch_tokens.shape[-1]
            results.append(
                (
                    patch_tokens.reshape(B, T, N_patch, E),
                    cls_token.reshape(B, T, E),
                )
            )
        return results

    def get_backbone_metadata(self) -> dict:
        """Return shape-compatible metadata dict matching DA3Wrapper.get_backbone_metadata."""
        embed_dim = int(self.vit.embed_dim)
        return {
            "name": "dinov3_vith16plus",
            "token_dim": embed_dim,
            "feature_dim": embed_dim,            # cat_token disabled in Stage 1
            "out_layers": tuple(self.out_layers),
            "alt_start": -1,                      # sentinel: no cross-view in Stage 1
            "total_layers": int(self.vit.n_blocks),
            "num_heads": int(self.vit.num_heads),
            "cat_token": False,
        }


def build_dinov3_per_view(
    yaml_path: str | Path,
    weights_path: str | Path | None = None,
) -> DinoV3PerViewBackbone:
    """Build a `DinoV3PerViewBackbone` from a YAML config + optional checkpoint."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    arch_kwargs = dict(cfg["arch"])
    out_layers = tuple(int(idx) for idx in cfg["out_layers"])

    vit = DinoVisionTransformer(**arch_kwargs)

    if weights_path is not None:
        # weights_only=True mirrors `torch.hub.load_state_dict_from_url` behavior
        # in dinov3.hub.backbones._make_dinov3_vit.
        state_dict = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )
        missing, unexpected = vit.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"DINOv3 strict load failed: "
                f"missing={missing[:5]}... unexpected={unexpected[:5]}..."
            )

    return DinoV3PerViewBackbone(vit, out_layers=out_layers)
```

- [ ] **Step 4.2: CPU build smoke (no weights)**

Run on the local box — random init, fast, validates that arch kwargs are accepted:

```bash
cd /home/acao/code/OccAny && \
  PYTHONPATH=third_party/dinov3 python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.model.dinov3_backbone import build_dinov3_per_view
bb = build_dinov3_per_view('occany/configs/dinov3/vith16plus.yaml', weights_path=None)
md = bb.get_backbone_metadata()
print('metadata:', md)
assert md['token_dim'] == 1280
assert md['total_layers'] == 32
assert md['num_heads'] == 20
assert md['out_layers'] == (7, 15, 23, 31)
print('CPU build OK')
"
```

Expected last line: `CPU build OK`. If `DinoVisionTransformer(**arch_kwargs)` raises a `TypeError` for an unknown kwarg, an entry in the YAML doesn't match the constructor signature at `third_party/dinov3/dinov3/models/vision_transformer.py:60-91` — reconcile before continuing.

- [ ] **Step 4.3: GPU forward + strict checkpoint load smoke on Karolina**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
import os, torch
from occany.model.dinov3_backbone import build_dinov3_per_view

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
bb = build_dinov3_per_view(\"occany/configs/dinov3/vith16plus.yaml\", weights_path=ckpt)
bb = bb.eval().cuda()
print(\"strict load OK\")

x = torch.randn(2, 4, 3, 528, 176, device=\"cuda\")
with torch.no_grad(), torch.autocast(\"cuda\", dtype=torch.bfloat16):
    feats = bb.forward_multiview(x)
print(\"num taps:\", len(feats))
for i, (f, cls) in enumerate(feats):
    print(f\"tap{i}: feat={tuple(f.shape)} cls={tuple(cls.shape)}\")
# (528//16)*(176//16) = 33*11 = 363
assert len(feats) == 4
assert feats[-1][0].shape == (2, 4, 363, 1280), feats[-1][0].shape
assert feats[-1][1].shape == (2, 4, 1280), feats[-1][1].shape
print(\"GPU forward OK\")
"'
```

Expected output: `strict load OK`, `num taps: 4`, `tap{0..3}: feat=(2, 4, 363, 1280) cls=(2, 4, 1280)`, `GPU forward OK`.

If strict load fails: re-check arch kwargs against the checkpoint. If shapes don't match: fix the YAML. Do not proceed past this step.

- [ ] **Step 4.4: Commit**

```bash
git add occany/model/dinov3_backbone.py
git commit -m "feat(occany): add DinoV3PerViewBackbone with multi-tap forward"
```

---

## Task 5: `Dinov3Wrapper` (recon-only)

**Files:**
- Create: `occany/model/model_dinov3.py`

Recon-only wrapper. Does **not** inherit from `DepthAnything3`. Exposes `self.backbone` and `self.head` directly (one level shallower than `DA3Wrapper`'s `self.model.backbone` / `self.model.head` nesting).

Try **Plan A first** (import `DualDPT` from `depth_anything_3.model.dualdpt`, sized by backbone metadata). Plan B (hand-rolled DPT) is the fallback documented below if Plan A's import fails.

The DA3 `DualDPT` head accepts `feats` as a list of `(patch_tensor, cam_token)` tuples (per `occany/model/model_da3.py:717-721` which iterates `f[0]` / `f[1:]`), with `patch_tensor` of shape `(B, T, N, E)` and `H, W` and `patch_start_idx=0`. It returns a dict with `depth`, `depth_conf`, `ray`, `ray_conf` keys. The shape returned by `DinoV3PerViewBackbone.forward_multiview` matches this contract exactly.

- [ ] **Step 5.1: Write `occany/model/model_dinov3.py` (Plan A)**

```python
"""Recon-only DINOv3 wrapper for OccAny+ Stage 1.

No cross-view machinery, no SAM3 head, no gen mode, no aux teacher path.
Each view is processed independently through DINOv3 ViT-H+/16, and the same
DA3 DualDPT head produces per-view depth / depth_conf / ray / ray_conf.

Public surface deliberately mirrors a subset of `occany.model.model_da3.DA3Wrapper`:
- `forward(images, **kwargs)` -> `inference_batch(images, **kwargs)`
- `inference_batch` returns the same dict keys as DA3Wrapper's recon path
- `get_backbone_metadata()` returns a dict with the same shape

Differences from DA3Wrapper:
- `self.backbone` / `self.head` live directly on the wrapper (no `self.model.` nesting)
- `c2w`, `intrinsics`, `sam_feats` are always None in Stage 1
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from occany.model.dinov3_backbone import build_dinov3_per_view

# Plan A: reuse DA3's DualDPT. Falls back to Plan B (hand-rolled head) only if
# the import drags in unavailable deps — see the matching comment block below.
from depth_anything_3.model.dualdpt import DualDPT


class Dinov3Wrapper(nn.Module):
    def __init__(
        self,
        config_path: str | Path,
        weights_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.backbone = build_dinov3_per_view(config_path, weights_path=weights_path)
        meta = self.backbone.get_backbone_metadata()
        # DualDPT defaults: output_dim=2, features=256, out_channels=(256, 512, 1024, 1024),
        # head_names=("depth", "ray"). Patch size is fixed at 16 for DINOv3 ViT-H+/16.
        self.head = DualDPT(
            dim_in=meta["feature_dim"],
            patch_size=16,
            output_dim=2,
            features=256,
            out_channels=(256, 512, 1024, 1024),
            head_names=("depth", "ray"),
        )

    def get_backbone_metadata(self) -> dict:
        return self.backbone.get_backbone_metadata()

    def forward(self, images: torch.Tensor, **kwargs):
        return self.inference_batch(images, **kwargs)

    def inference_batch(self, images: torch.Tensor, **_unused):
        """Per-view recon. Matches DA3Wrapper.inference_batch's output schema."""
        b, t, c, h, w = images.shape
        feats = self.backbone.forward_multiview(images)
        return self._process_depth_output(feats=feats, h=h, w=w)

    def _process_depth_output(self, feats, h, w):
        """Direct port of DA3Wrapper._process_depth_output with:
        - head_sam branch removed (sam_feats always None)
        - pose_from_cam_dec branch removed (no camera decoder)
        - pose_from_depth_ray branch removed (c2w/intrinsics always None)
        - save_outputs debug block removed
        """
        output = self.head(feats, h, w, patch_start_idx=0)

        default_scale = 20

        depth = output["depth"] * default_scale          # (B, T, H, W)
        depth_conf = output["depth_conf"]                # (B, T, H, W)

        ray = output["ray"]                               # (B, T, H, W, 6)
        ray_conf = output["ray_conf"]                     # (B, T, H, W)
        # Scale ray origins (last 3 of the 6-channel ray field).
        ray[..., 3:] = ray[..., 3:] * default_scale

        # DA3 paper: P = t + D * d (unnormalized ray direction; depth preserves scale).
        pointmap = depth.unsqueeze(-1) * ray[..., :3] + ray[..., 3:]

        return {
            "pointmap": pointmap,
            "depth": depth,
            "depth_conf": depth_conf,
            "ray": ray,
            "ray_conf": ray_conf,
            "c2w": None,
            "intrinsics": None,
            "sam_feats": None,
        }
```

- [ ] **Step 5.2: GPU smoke — forward then loss + backward**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
import os, torch
from occany.model.model_dinov3 import Dinov3Wrapper

ckpt = \"checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth\"
m = Dinov3Wrapper(\"occany/configs/dinov3/vith16plus.yaml\", weights_path=ckpt).cuda()
m.train()

x = torch.randn(1, 2, 3, 528, 176, device=\"cuda\")
with torch.autocast(\"cuda\", dtype=torch.bfloat16):
    out = m(x)
print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in out.items()})

# Tiny scalar loss + backward to exercise every learnable param path.
loss = sum(v.float().mean() for k, v in out.items() if torch.is_tensor(v))
loss.backward()
print(\"backward OK; loss=\", float(loss))
"'
```

Expected: every recon key present, `c2w=None`, `intrinsics=None`, `sam_feats=None`, and `backward OK`. If `DualDPT.__init__` rejects `dim_in=1280` (DA3-GIANT uses `dim_in=3072` due to `cat_token=True`), that's still valid — `dim_in` is the per-tap input channel count, which is exactly the patch embedding dim for Stage 1.

- [ ] **Step 5.3: (Plan B fallback only) If Plan A's `DualDPT` import or instantiation fails irrecoverably**

If `from depth_anything_3.model.dualdpt import DualDPT` raises at import time, or `DualDPT(**kwargs)` fails because Plan A's kwargs are misaligned with the DA3 head's actual signature, write a minimal DPT head and replace the `self.head = DualDPT(...)` line. The head must:

- Accept `feats: list[tuple[Tensor(B,T,N,E), Tensor(B,T,E)]]`, `H: int`, `W: int`, `patch_start_idx: int`.
- Return `{"depth": (B, T, H, W), "depth_conf": (B, T, H, W), "ray": (B, T, H, W, 6), "ray_conf": (B, T, H, W)}` — with non-negative depth/conf (e.g., `softplus(x) + 1e-6` activation on depth, `sigmoid` on confs).

A ~120-line scratch+fusion DPT head following `third_party/Depth-Anything-3/src/depth_anything_3/model/dpt.py` patterns suffices. **Do not write this unless Plan A actually fails the smoke in Step 5.2.**

- [ ] **Step 5.4: Commit**

```bash
git add occany/model/model_dinov3.py
git commit -m "feat(occany): add Dinov3Wrapper (recon-only, per-view)"
```

---

## Task 6: Standalone PCA visualization script

**Files:**
- Create: `test_dinov3_pca.py`

CLI sanity-check that the backbone is loading and producing reasonable dense features. The spec referenced `third_party/dinov3/notebooks/pca.ipynb` but **that path does not exist on disk** (verified — only `dinov3/notebooks/` is absent). Implement PCA with torch SVD; foreground-mask trick optional.

Pattern matches existing standalone test scripts (`test_occ_rae.py`, `test_view_sampling.py` etc.) — argparse + `main()` + `if __name__ == "__main__"`.

- [ ] **Step 6.1: Write `test_dinov3_pca.py`**

```python
"""Standalone PCA-feature visualization for the DINOv3 backbone.

Confirms the backbone is loading and producing reasonable dense features
before launching any training. Output is a per-image PNG showing the top-3
PCA components of patch features, RGB-mapped.

Usage:
    python test_dinov3_pca.py \
        --weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
        --input_dir demo_data/input \
        --output_dir results/dinov3_pca

Success criterion: output PNGs show spatially coherent object-vs-background
structure (matching DINOv3 paper's example visualizations). Noise = bad
normalization or wrong checkpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from occany.utils.runtime_paths import prepend_vendored_import_paths

prepend_vendored_import_paths()

from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: E402
from occany.model.dinov3_backbone import build_dinov3_per_view  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="occany/configs/dinov3/vith16plus.yaml")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True,
                   help="Directory of images, or a single image path.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="Tap index to PCA; default = last value of out_layers.")
    p.add_argument("--resolution", type=int, nargs=2, default=(528, 528),
                   metavar=("H", "W"), help="Resize input to (H, W); both multiples of 16.")
    p.add_argument("--foreground_mask", action="store_true",
                   help="Apply the PCA-1 foreground mask trick before fitting PCAs 2..4.")
    p.add_argument("--device", type=str, default="cuda")
    return p


def torch_pca(x: torch.Tensor, n_components: int) -> torch.Tensor:
    """Centered SVD-based PCA. x: (N, D) -> projections (N, n_components)."""
    x_centered = x - x.mean(dim=0, keepdim=True)
    # torch.linalg.svd: x = U @ diag(S) @ Vh. Top components = first n_components rows of Vh.
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    components = vh[:n_components]              # (n_components, D)
    return x_centered @ components.T            # (N, n_components)


def minmax_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each channel to [0, 255]."""
    mn = x.amin(dim=(0, 1), keepdim=True)
    mx = x.amax(dim=(0, 1), keepdim=True)
    return ((x - mn) / (mx - mn).clamp(min=1e-8) * 255).round().clamp(0, 255).byte()


def list_images(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def main():
    args = get_args_parser().parse_args()
    H, W = args.resolution
    assert H % 16 == 0 and W % 16 == 0, f"resolution must be multiples of 16, got {(H, W)}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bb = build_dinov3_per_view(args.config, weights_path=args.weights)
    bb = bb.eval().to(args.device)

    out_layers = bb.out_layers
    if args.layer is None:
        tap_idx = len(out_layers) - 1
    else:
        if args.layer not in out_layers:
            raise SystemExit(f"--layer {args.layer} not in out_layers {out_layers}")
        tap_idx = out_layers.index(args.layer)

    H_patch = H // 16
    W_patch = W // 16

    for img_path in list_images(Path(args.input_dir)):
        pil = Image.open(img_path).convert("RGB").resize((W, H), Image.BICUBIC)
        x_norm = InputProcessor.NORMALIZE(to_tensor(pil))         # (3, H, W)
        x = x_norm[None, None].to(args.device)                     # (1, 1, 3, H, W)

        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
            feats = bb.forward_multiview(x)

        patch_feats, _ = feats[tap_idx]                            # (1, 1, N, E)
        patches = patch_feats.reshape(-1, patch_feats.shape[-1]).float()  # (N, E)

        feat_norm = patches.norm(dim=-1).mean().item()

        if args.foreground_mask:
            # PCA-1 foreground mask: split patches by sign of first component.
            pca1 = torch_pca(patches, n_components=1).squeeze(-1)  # (N,)
            # Pick the half with larger mean abs as foreground (sign is arbitrary).
            mask = pca1 > pca1.median()
            if mask.float().mean().item() < 0.5:
                mask = ~mask
            fg = patches[mask]
            comps_fg = torch_pca(fg, n_components=3)               # (N_fg, 3)
            rgb = torch.zeros((patches.shape[0], 3), device=patches.device)
            rgb[mask] = comps_fg
        else:
            rgb = torch_pca(patches, n_components=3)                # (N, 3)

        rgb_img = rgb.reshape(H_patch, W_patch, 3)
        rgb_u8 = minmax_to_uint8(rgb_img).cpu().numpy()
        # Bilinear upsample to (H, W) via PIL.
        out_pil = Image.fromarray(rgb_u8, mode="RGB").resize((W, H), Image.BILINEAR)
        out_path = output_dir / f"pca_{img_path.stem}.png"
        out_pil.save(out_path)

        print(
            f"{img_path} -> {out_path}  tap={out_layers[tap_idx]}  "
            f"patch_grid=({H_patch}, {W_patch})  feat_norm_mean={feat_norm:.3f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run on Karolina against a handful of `demo_data/input` images**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python test_dinov3_pca.py \
    --weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
    --input_dir demo_data/input \
    --output_dir results/dinov3_pca_smoke \
    --resolution 528 528'
```

Expected: one-line summary per image (input → output path, tap layer, patch grid, feature L2 norm mean ~10–40 for healthy DINOv3 features). Inspect a few output PNGs (`scp` or `rsync` back); they should show coherent object/background separation, not noise.

If outputs are noise: the normalization is wrong (check `InputProcessor.NORMALIZE` is being applied) or the checkpoint didn't load correctly (re-run Step 4.3). Do not proceed past this step.

- [ ] **Step 6.3: Commit**

```bash
git add test_dinov3_pca.py
git commit -m "feat(occany): add DINOv3 PCA visualization sanity script"
```

---

## Task 7: Widen `'da3'` dataset branch to also accept `'dinov3'`

**Files:**
- Modify: `occany/datasets/base_seq_dataset.py:179`
- Modify: `occany/datasets/base_seq_dataset.py:499`
- Modify: `occany/datasets/kitti.py:493`
- Modify: `occany/datasets/nuscenes.py:835`

DINOv3 uses the same ImageNet mean/std as DA3 (verified in Step 1.3), so routing it through the same `InputProcessor.NORMALIZE(to_tensor(...))` path is correct and minimal.

- [ ] **Step 7.1: Edit `occany/datasets/base_seq_dataset.py:179`**

```python
# Before:
            if self.base_model == 'da3':
                
                view['img'] = InputProcessor.NORMALIZE(to_tensor(view['img']))
            else:
                view['img'] = self.transform(view['img'])

# After:
            if self.base_model in ('da3', 'dinov3'):
                view['img'] = InputProcessor.NORMALIZE(to_tensor(view['img']))
            else:
                view['img'] = self.transform(view['img'])
```

(Also drops the dangling blank line on the original line 180.)

- [ ] **Step 7.2: Edit `occany/datasets/base_seq_dataset.py:499`**

```python
# Before:
            if self.base_model == 'da3':
                view['img'] = InputProcessor.NORMALIZE(to_tensor(img_pil))
            elif pil_jitter is not None:
                view['img'] = ImgNorm(img_pil)
            else:
                view['img'] = self.transform(img_pil)

# After:
            if self.base_model in ('da3', 'dinov3'):
                view['img'] = InputProcessor.NORMALIZE(to_tensor(img_pil))
            elif pil_jitter is not None:
                view['img'] = ImgNorm(img_pil)
            else:
                view['img'] = self.transform(img_pil)
```

- [ ] **Step 7.3: Edit `occany/datasets/kitti.py:493`**

```python
# Before:
            if self.base_model == 'da3':
                imgs.append(InputProcessor.NORMALIZE(to_tensor(downscaled_img)))
            else:
                imgs.append(ImgNorm(np.array(downscaled_img)))

# After:
            if self.base_model in ('da3', 'dinov3'):
                imgs.append(InputProcessor.NORMALIZE(to_tensor(downscaled_img)))
            else:
                imgs.append(ImgNorm(np.array(downscaled_img)))
```

- [ ] **Step 7.4: Edit `occany/datasets/nuscenes.py:835`**

```python
# Before:
        if self.base_model == 'da3':
            img_tensor = InputProcessor.NORMALIZE(to_tensor(downscaled_img))
        else:
            img_tensor = ImgNorm(np.array(downscaled_img))

# After:
        if self.base_model in ('da3', 'dinov3'):
            img_tensor = InputProcessor.NORMALIZE(to_tensor(downscaled_img))
        else:
            img_tensor = ImgNorm(np.array(downscaled_img))
```

- [ ] **Step 7.5: Re-grep to confirm no other diverging sites slipped in**

```bash
cd /home/acao/code/OccAny && grep -rn "base_model" occany/ | grep -v __pycache__ | grep -v "Binary"
```

The list should match Step 1.5's expected list. `eval_helper.py:84,89` still use `'da3'` — leave them; eval pipeline is out of scope.

- [ ] **Step 7.6: Smoke-test dataset construction with the new value**

```bash
ssh karolina 'cd ~/OccAny && . ~/.bashrc && conda activate occany && \
  python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.datasets.kitti import KittiSeqMultiView
import os
ds = KittiSeqMultiView(
    KITTI_PREPROCESSED_ROOT=os.path.join(os.environ[\"SCRATCH\"], \"data/kitti_processed\"),
    seq_pkl_name=\"seq_exact_len_sub5_stride9.pkl\",
    frame_interval=1, min_memory_num_views=2, max_memory_num_views=2,
    reverse_seq=False, min_num_timesteps=1, no_partial_views=False,
    z_far=50, split=\"val\", seed=42, recon_view_idx=[0], ray_map_idx=[1],
    resolution=[(528, 176)], base_model=\"dinov3\")
sample = ds[0]
imgs = sample[0][\"img\"] if isinstance(sample, list) else sample[\"img\"]
print(\"sample type:\", type(sample), \"first img shape:\", imgs.shape, \"dtype:\", imgs.dtype)
print(\"img min/max/mean:\", float(imgs.min()), float(imgs.max()), float(imgs.mean()))
"'
```

Expected: `img` is a normalized float tensor (range roughly `[-2.5, 2.5]` post-ImageNet-normalization). No exceptions.

- [ ] **Step 7.7: Commit**

```bash
git add occany/datasets/base_seq_dataset.py occany/datasets/kitti.py occany/datasets/nuscenes.py
git commit -m "feat(occany): accept base_model='dinov3' via existing DA3 normalization path"
```

---

## Task 8: Launcher `launch_dinov3.py`

**Files:**
- Create: `launch_dinov3.py`

Clone of the 16-line `launch_da3.py` with one module-import change.

- [ ] **Step 8.1: Create the file**

```python
from pathlib import Path
import torch.multiprocessing as mp

from occany.utils.runtime_paths import prepend_vendored_import_paths

mp.set_sharing_strategy("file_descriptor")
prepend_vendored_import_paths(Path(__file__).resolve().parent)

from occany.training_dinov3 import get_args_parser, train

if __name__ == "__main__":
    train(get_args_parser().parse_args())
```

- [ ] **Step 8.2: Commit**

```bash
git add launch_dinov3.py
git commit -m "feat(occany): add launch_dinov3.py DDP entrypoint"
```

(Note: `from occany.training_dinov3 import ...` will fail until Task 9 lands — that's fine, the launcher is a thin import shim; we'll smoke-test it in Task 11.)

---

## Task 9: `training_dinov3.py` — fork `training_da3.py`

**Files:**
- Create: `occany/training_dinov3.py` (forked from `occany/training_da3.py`)

This is the largest task. `training_da3.py` is 1301 lines. The fork is mostly **deletions** of paths Stage 1 doesn't use, plus a small number of **rewires** for the attribute-nesting difference (`model.model.X` → `model.X`) and the model-construction call. Don't reshape the recon loop body — it stays identical so the existing loss-name logging is preserved.

- [ ] **Step 9.1: Copy `training_da3.py` → `training_dinov3.py`**

```bash
cp occany/training_da3.py occany/training_dinov3.py
```

- [ ] **Step 9.2: Rename argparse flags — drop `--da3_model_name`, add `--dinov3_config` / `--dinov3_weights`**

Find the argparse block where `--da3_model_name` is defined (search `grep -n "da3_model_name" occany/training_dinov3.py`). Remove that argument entirely. Add the two new flags:

```python
    parser.add_argument(
        "--dinov3_config", type=str, required=True,
        help="Path to DINOv3 backbone YAML (e.g. occany/configs/dinov3/vith16plus.yaml).",
    )
    parser.add_argument(
        "--dinov3_weights", type=str, required=True,
        help="Path to DINOv3 pretrained .pth checkpoint.",
    )
```

- [ ] **Step 9.3: Remove SAM3 distillation argparse flags**

Search for and **delete** the definitions of these args (they're SAM3-specific and the wrapper has no `head_sam`):

- `--distill_model`
- `--distill_criterion`
- `--sam3_proj_lr_mult`
- `--sam3_use_dpt_proj`
- `--distill_k_schedule` (if present)

- [ ] **Step 9.4: Remove other unused argparse flags**

- `--loss_enc_feat`
- `--aux_branch_layers`

- [ ] **Step 9.5: Rewire model construction**

Find the block (≈line 325 in the original) where `DA3Wrapper.from_pretrained(args.da3_model_name)` is called. Replace it with:

```python
    from occany.model.model_dinov3 import Dinov3Wrapper

    model = Dinov3Wrapper(args.dinov3_config, weights_path=args.dinov3_weights)
    model = model.to(device)
    backbone_metadata = model.get_backbone_metadata()
```

Remove the import of `DA3Wrapper` at the top of the file (search `from occany.model.model_da3 import DA3Wrapper`).

- [ ] **Step 9.6: Rewire `model.model.X` → `model.X` attribute accesses**

Search `grep -n "model\.model\." occany/training_dinov3.py`. Replace each site as follows (the line numbers correspond to the `training_da3.py` source — your forked file may shift slightly):

| What | From | To |
|---|---|---|
| Backbone params (full freeze)         | `model.model.backbone.parameters()`                       | `model.backbone.vit.parameters()` |
| Per-layer fine-tune                   | `model.model.backbone.pretrained.blocks[layer_idx]`       | `model.backbone.vit.blocks[layer_idx]` |
| Head params                           | `model.model.head.parameters()`                           | `model.head.parameters()` |
| `model.model.cam_enc` references      | (any branch checking this)                                | **delete the whole branch** |
| `model.model.cam_dec` references      | (any branch checking this)                                | **delete the whole branch** |
| `model.model.gs_head` references      | (any branch checking this)                                | **delete the whole branch** |
| `model.model.gs_adapter` references   | (any branch checking this)                                | **delete the whole branch** |

After this step, `grep -n "model\.model\." occany/training_dinov3.py` should return zero matches. **Run it to verify.**

- [ ] **Step 9.7: Delete generation-mode paths**

In `training_da3.py` lines ≈456–593, gen mode constructs `model_recon` (frozen) + `model_gen` (trainable). Delete:

- The entire `if args.gen:` setup block that creates `model_recon` and `model_gen`.
- Any call to `init_gen_encoders()` / `forward_gen()` / `set_slice_layer()` in the train and eval loops.
- The `loss_of_one_batch_occany_da3_gen` branch in the train/eval loop bodies (search `args.gen`); replace `if args.gen: ... else: ...` with the recon-only branch.
- The `--gen_*` argparse flags (search `gen_input_encoder`, `model_gen`, `gen_*`).

Concretely, `grep -n "args\.gen\b\|model_gen\|forward_gen\|init_gen_encoders\|set_slice_layer\|loss_of_one_batch_occany_da3_gen" occany/training_dinov3.py` should return zero matches after this step.

- [ ] **Step 9.8: Delete SAM3 distillation init paths**

In `training_da3.py` lines ≈426–451:

- Delete the `init_sam3_head(...)` call and the SAM3 neck-weight loading block.
- Delete the SAM3 distill loss computation and logging keys in the train loop.
- Delete the `--distill_model` branch entirely.

After: `grep -n "head_sam\|sam3\|distill_model\|SAM3\|init_sam3_head\|forward_sam_features" occany/training_dinov3.py` should return zero matches.

- [ ] **Step 9.9: Delete `--loss_enc_feat` path**

Search `grep -n "loss_enc_feat\|enc_feat" occany/training_dinov3.py` and delete the conditional loss computation + logging key.

- [ ] **Step 9.10: Delete `--aux_branch_layers` / `init_aux_branch` path**

Search `grep -n "aux_branch_layers\|init_aux_branch\|aux_head\|aux_blocks" occany/training_dinov3.py` and delete the conditional aux-teacher init + the aux loss computation. The recon loop uses `--infinidepth_pseudo_supervision` (precomputed pseudo files), which is independent of the aux path — keep that intact.

- [ ] **Step 9.11: Verify DDP `find_unused_parameters=True` is still set**

```bash
grep -n "find_unused_parameters" occany/training_dinov3.py
```

Expected: at least one match showing `find_unused_parameters=True`. Recon-only training touches every learnable param every step, but keeping `True` is safer and matches the inherited DA3 setting.

- [ ] **Step 9.12: Verify the recon loss assembly + log-key block is intact**

```bash
grep -n "loss_pointmap_lidar\|loss_pointmap_pseudo\|depth_loss_depth_lidar\|depth_loss_depth_pseudo\|raymap" occany/training_dinov3.py
```

Expected: log-key strings still present in the loss aggregation block — these are what the existing eval / tensorboard infrastructure consumes. Do not rename.

- [ ] **Step 9.13: Argparse-only smoke (CPU)**

```bash
cd /home/acao/code/OccAny && PYTHONPATH=third_party python -c "
from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()
from occany.training_dinov3 import get_args_parser
ap = get_args_parser()
# Must contain new flags, must NOT contain dropped flags.
flag_set = {a.option_strings[0] for a in ap._actions if a.option_strings}
assert '--dinov3_config' in flag_set
assert '--dinov3_weights' in flag_set
assert '--da3_model_name' not in flag_set
assert '--distill_model' not in flag_set
assert '--loss_enc_feat' not in flag_set
assert '--aux_branch_layers' not in flag_set
assert '--sam3_proj_lr_mult' not in flag_set
print('argparse OK')
"
```

Expected: `argparse OK`.

- [ ] **Step 9.14: Commit**

```bash
git add occany/training_dinov3.py
git commit -m "feat(occany): add training_dinov3.py — recon-only fork of training_da3"
```

---

## Task 10: Training shell wrapper

**Files:**
- Create: `sh/train_occany_plus_recon_dinov3_vith16plus.sh`

Clone of `sh/train_occany_plus_recon_1B_infinite_depth.sh` (the **non-SAM3** variant) with the changes listed in spec §5. Use chmod +x to mark executable.

- [ ] **Step 10.1: Create the file**

```bash
#!/bin/bash
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

export EXP_NAME="occany_plus_recon_dinov3_vith16plus"


: ${BATCH_SIZE:=1}
: ${EFFECTIVE_BATCH_SIZE:=64}
: ${N_WORKERS:=12}

export EPOCHS=100

# Default values for multi-node setup
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

export ACCUM_ITER
ACCUM_ITER="$(occany_compute_accum_iter "$EFFECTIVE_BATCH_SIZE" "$NUM_GPU_PER_NODE" "$BATCH_SIZE" "$NUM_NODE")"
occany_log_train_config "$EXP_NAME"
CMD="$(occany_select_train_cmd 'launch_dinov3.py')"
occany_log_start_cmd "$CMD"

WIDTH=528
HEIGHT=176



RAY_MAP_PROB=-1

$CMD \
    --train_dataset="5000 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(528, 288), (528, 272), (528, 256), (528, 208), (528, 176)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        2000 @ VKittiSeqMultiView(VKITTI_PROCESSED_ROOT='$SCRATCH/data/vkitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=10, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(528, 288), (528, 272), (528, 256), (528, 208), (528, 176)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        5000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$SCRATCH/data/ddad_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(528, 288), (528, 272), (528, 256), (528, 208), (528, 176)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        5000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$SCRATCH/data/pandaset_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=6, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(528, 288), (528, 272), (528, 256), (528, 208), (528, 176)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True) + \
        5000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$SCRATCH/data/once_processed', \
        seq_pkl_name='seq_surround_temporal_sub5_stride9_all.pkl', \
        min_memory_num_views=2, frame_interval=1, max_memory_num_views=30, \
        num_views_per_timestep=5, ray_map_prob=-1, \
        min_num_timesteps=1, no_partial_views=False, \
        aug_crop=128, z_far=50, split='train', \
        resolution=[(528, 288), (528, 272), (528, 256), (528, 208), (528, 176)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, base_model='dinov3', load_infinidepth_pseudo=True)"  \
    --test_dataset="206 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$SCRATCH/data/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, reverse_seq=False, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, recon_view_idx=[0, 2, 4, 6, 8], ray_map_idx=[1, 3, 5, 7], \
        resolution=[(528, 176)], base_model='dinov3') + \
        206 @ Occ3dNuscenesSeqMultiView(NUSCENES_PREPROCESSED_ROOT='$SCRATCH/data/occ3d_nuscenes_processed', \
        seq_pkl_name='seq_surround_temporal_sub1_stride9_all.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, num_views_per_timestep=6, \
        min_num_timesteps=1, no_partial_views=False, \
        z_far=50, split='val', seed=42, fixed_cams=[0,1], \
        resolution=[(528, 256)], base_model='dinov3')" \
    --lr=5e-5 --min_lr=1e-6 --warmup_epochs=3 --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE --accum_iter=$ACCUM_ITER \
    --save_freq=3 --keep_freq=5 --eval_freq=1  --num_workers=$N_WORKERS --multiview \
    --amp bf16 --fixed_eval_set \
    --output_dir="$PROJECT/tb_log_occany/$EXP_NAME" \
    --training_objective pointmap_depth_ray --fine_tune_layers 26,27,28,29,30,31 \
    --dinov3_config occany/configs/dinov3/vith16plus.yaml \
    --dinov3_weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
    --loss_type L1 --pointmap_lambda_c 1.0 --depth_lambda_c 0.0 --raymap_lambda_c 1.0 \
    --infinidepth_pseudo_supervision \
     --lambda_depth 1.0 --lambda_pointmap 1.0 \
     --lambda_pointmap_lidar 1.0 --lambda_pointmap_pseudo 1.0 \
     --lambda_depth_lidar 1.0 --lambda_depth_pseudo 1.0
```

Key deltas vs `sh/train_occany_plus_recon_1B_infinite_depth.sh`:

- `EXP_NAME` changed
- `WIDTH=528`, `HEIGHT=176` (was 518/168)
- All five training-dataset `resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)]` → `resolution=[(528, 288), (528, 272), (528, 256), (528, 208), (528, 176)]`
- KITTI test `(518, 168)` → `(528, 176)`; Occ3dNuscenes test `(518, 266)` → `(528, 256)`
- `distill_model_name='SAM3'` dropped from every dataset constructor
- `base_model='da3'` → `base_model='dinov3'` on every dataset constructor
- `--da3_model_name depth-anything/DA3-GIANT-1.1` → `--dinov3_config ... --dinov3_weights ...`
- `--sam3_proj_lr_mult 10.0`, `--sam3_use_dpt_proj`, `--loss_enc_feat` removed
- `--fine_tune_layers 34,35,36,37,38,39` → `--fine_tune_layers 26,27,28,29,30,31` (last 6 of 32 DINOv3 blocks)
- Launcher: `launch_da3.py` → `launch_dinov3.py`
- SAM3 / aux comment block dropped

- [ ] **Step 10.2: Make executable**

```bash
chmod +x sh/train_occany_plus_recon_dinov3_vith16plus.sh
```

- [ ] **Step 10.3: Syntax-check on Karolina (does not start training)**

```bash
ssh karolina 'cd ~/OccAny && bash -n sh/train_occany_plus_recon_dinov3_vith16plus.sh && echo SYNTAX_OK'
```

Expected: `SYNTAX_OK`.

- [ ] **Step 10.4: Commit**

```bash
git add sh/train_occany_plus_recon_dinov3_vith16plus.sh
git commit -m "feat(sh): add DINOv3 ViT-H+/16 training wrapper"
```

---

## Task 11: SLURM file

**Files:**
- Create: `slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm`

Clone of `slurm/karolina_train_occany_plus_recon_1B_infinite_depth_sam3_distill.slurm` with the SAM3-specific memory-budget comment table dropped and the script invocation rewired.

- [ ] **Step 11.1: Create the file**

```bash
#!/bin/bash
#SBATCH --job-name=train_occany_plus_recon_dinov3_vith16plus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --partition=qgpu
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/train_occany_plus_recon_dinov3_vith16plus_%j.out
#SBATCH --error=slurm/output/train_occany_plus_recon_dinov3_vith16plus_%j.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-4}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"

NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" \
NUM_NODE="$NUM_NODE" \
N_WORKERS="${N_WORKERS:-10}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
bash sh/train_occany_plus_recon_dinov3_vith16plus.sh
```

- [ ] **Step 11.2: Syntax-check**

```bash
ssh karolina 'cd ~/OccAny && bash -n slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm && echo SYNTAX_OK'
```

Expected: `SYNTAX_OK`.

- [ ] **Step 11.3: Commit**

```bash
git add slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm
git commit -m "feat(slurm): add Karolina SLURM wrapper for DINOv3 recon training"
```

---

## Task 12: End-to-end single-GPU smoke on Karolina

**Files:** none.

Submit a tiny 1-GPU job (or run interactively on `qgpu_exp`) for 1–2 iterations to confirm:

1. Forward + backward run end-to-end with the real DINOv3 weights.
2. DDP wraps the model without `RuntimeError: marked ready twice` or `parameter not used in producing loss` failures.
3. Loss values are finite for the first step.
4. The per-step log lines include all expected keys: `loss_pointmap_lidar`, `loss_pointmap_pseudo`, `depth_loss_depth_lidar`, `depth_loss_depth_pseudo`, `loss_raymap`.

Use the `karolina-job` skill for the submission.

- [ ] **Step 12.1: Submit a 1-node 1-GPU smoke (qgpu_exp, ~10 min)**

```bash
# via karolina-job skill; equivalent direct sbatch:
ssh karolina 'cd ~/OccAny && NUM_NODE=1 NUM_GPU_PER_NODE=1 \
  BATCH_SIZE=1 EFFECTIVE_BATCH_SIZE=1 N_WORKERS=2 EPOCHS=1 \
  sbatch --partition=qgpu_exp --time=00:30:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
    --job-name=dinov3_smoke --output=slurm/output/dinov3_smoke_%j.out \
    --error=slurm/output/dinov3_smoke_%j.err \
    --wrap="eval \"\$(conda shell.bash hook)\" && conda activate occany && \
            bash sh/train_occany_plus_recon_dinov3_vith16plus.sh"'
```

- [ ] **Step 12.2: Wait + tail the smoke output**

Use the `karolina-job` skill to tail. Look for, in order:

1. Successful argparse + dataset construction.
2. `Dinov3Wrapper` load message (`strict load OK` equivalent — should be silent unless the assert fires).
3. First training-step log line with finite `loss_*` values and the expected key names from Step 9.12.
4. No `marked ready twice`, `parameter not used in producing loss`, or `Found dtype Float but expected ...` errors.

- [ ] **Step 12.3: If `find_unused_parameters` warnings appear, investigate**

If the log includes warnings about unused parameters from DDP, run:

```bash
# 1-step grad-presence audit
ssh karolina '... single-step run with hooks dumping param names whose .grad is None ...'
```

Decision matrix:
- If the unused params are all in `head_sam` / `aux_head` / `gen_*` — Step 9 deletions were incomplete; fix and re-run.
- If they're real params from `model.head` or `model.backbone.vit.blocks[*]` (within the `--fine_tune_layers` range), that's a wiring bug in `_process_depth_output` — investigate before Task 13.
- If they're outside the fine-tune range and inside the frozen backbone, that's expected — keep `find_unused_parameters=True`.

- [ ] **Step 12.4: Aspect-ratio metrics sanity (spec §6 step 8)**

After the smoke job completes, glance at the `--fixed_eval_set` eval-time depth/IoU metrics in the log. The patch-16 preset change shifts the KITTI longest aspect ratio from 3.08 → 3.00 (~3% loss); the absolute eval numbers will be worse than DA3 because the head is randomly initialized, but they should not be NaN, inf, or wildly degenerate (e.g., `depth_loss_depth_lidar > 1e3`).

- [ ] **Step 12.5: Commit any fixes from 12.3, then proceed**

If the smoke required code fixes:

```bash
git add <fixed files>
git commit -m "fix(occany): <specific issue from smoke>"
```

---

## Task 13: Full multi-node Karolina launch

**Files:** none.

Once the single-GPU smoke is clean, submit the production 4-node × 8-GPU job.

- [ ] **Step 13.1: Submit**

```bash
ssh karolina 'cd ~/OccAny && sbatch slurm/karolina_train_occany_plus_recon_dinov3_vith16plus.slurm'
```

Record the job ID.

- [ ] **Step 13.2: Tail the first ~30 minutes via `karolina-job` skill**

Watch for:
- All 32 ranks initialize without `NCCL` / `Address already in use` errors.
- First epoch's per-step loss is in a sane range (compare to DA3-GIANT baseline as orientation only — Stage 1 will be worse since the head is freshly initialized).
- Memory utilization is well under 80 GiB per H100 (DINOv3 ViT-H+ is ~1.1 B params vs DA3-GIANT's 1.1 B; expect similar VRAM footprint).

If anything fails at scale (NCCL hang, OOM at multi-node), pause and investigate before iterating.

- [ ] **Step 13.3: Once the run is stable, hand off to the user**

Report: job ID, output directory (`$PROJECT/tb_log_occany/occany_plus_recon_dinov3_vith16plus`), expected first-eval timestamp. Do **not** auto-iterate beyond Task 13 — the spec's open questions (out_layers retuning, multi-aspect-ratio balance) are post-Stage-1 follow-ups.

---

## Self-review notes

**Spec coverage:** All sections of `2026-05-21-occany-dinov3-backbone-design.md` are mapped to a task:
- §1 goal & two-stage approach → plan header
- §2 file map → "File map" section + Tasks 3-11
- §3 backbone module → Task 4
- §4 wrapper module → Task 5
- §5 training entry + shell + SLURM → Tasks 9, 10, 11
- §5b PCA script → Task 6
- §6 verification steps 1-8 → Task 1 (steps 1-3, 5), Task 4 (steps 4, 7), Task 12 (steps 7-8); step 4 (DPT import) is Task 1 step 1.2
- §7 open questions / risks → not implementation steps; called out in Task 12.4 (aspect ratio) and Task 13.3 (hand-off note)
- §8 Stage 2 deferred → explicitly not in this plan

**Discrepancies between spec and code reality, surfaced during planning:**

1. The spec said the PCA notebook exists at `third_party/dinov3/notebooks/pca.ipynb`. **It does not** — `third_party/dinov3/notebooks/` does not exist on disk. Task 6 implements PCA from scratch via torch SVD (with optional foreground mask), matching the spirit of the official viz.

2. The spec listed `runtime_paths` and `train_common.sh` under "Reused unchanged", but `third_party/dinov3` is **not** in either path list — and `dinov3` lives at `third_party/dinov3/dinov3/`, not directly under `third_party/`. Task 2 adds it to both. (Two-line additive edits, not a structural change.)

3. The spec mentioned widening 3 dataset files (4 sites). `occany/datasets/eval_helper.py:84,89` also branch on `'da3'`, but only for the eval pipeline that Stage 1 does not run — Task 1 step 1.5 documents this; Task 7 step 7.5 reconfirms after the edits.

4. The spec's `forward_multiview` returns `list[(feat, cam_token)]` tuples. Verified at `occany/model/model_da3.py:717-721` that DA3's `DualDPT` indeed expects tuples (its code path indexes `f[0]` and `f[1:]`). Task 5's wrapper passes tuples through directly.

**No placeholders / TBDs / `# implement later` strings.** Every step contains the actual command or code to run.

**Type consistency:** `Dinov3Wrapper` consistently exposes `self.backbone` (with `.vit`, `.out_layers`, `.get_backbone_metadata()`) and `self.head` across Tasks 5, 8, 9. The `get_backbone_metadata()` dict shape (`name`, `token_dim`, `feature_dim`, `out_layers`, `alt_start`, `total_layers`, `num_heads`, `cat_token`) is defined once in Task 4 and consumed by Task 5 only.
