# OccRAE

`OccRAE` is a lightweight wrapper around the OccAny DA3 reconstruction model that exposes two operations:

- `encode(images)`: run the backbone up to an intermediate layer and return cached tokens
- `decode(latents)`: resume the backbone from those cached tokens and produce the normal OccAny reconstruction output

This is useful when you want to:

- save reusable intermediate features instead of full outputs
- measure how much information is preserved by an intermediate representation
- run encode/decode roundtrip tests against the pretrained OccAny 1B reconstruction model

The current public implementation is intentionally fixed to `encode_layer=18`.

## Why layer 18

The current implementation is fixed to `encode_layer=18`.

For the DA3-Giant backbone used by OccAny 1B:

- `alt_start = 13`
- layers `>= 13` alternate between local and global attention
- even layers in that region are local-attention layers

Layer 18 is therefore a local-attention layer. After a local-attention block, `x == local_x`, so only one tensor needs to be saved. If extraction happened at layer 19 instead, both `x_19` and `local_x_19` would be needed.

This cuts the cached feature size roughly in half while still allowing decode to resume from layer 19 and reproduce the normal model output path.

## API

Implementation: [occany/model/occ_rae.py](occany/model/occ_rae.py)

### Constructor

```python
from occany.model.occ_rae import OccRAE

occ_rae = OccRAE(
    weights_path="checkpoints/occany_plus_recon_1B.pth",
    output_resolution=(518, 518),
    device="cuda",
    encode_layer=18,
)
```

Arguments:

- `weights_path`: path to the OccAny reconstruction checkpoint
- `output_resolution`: square input size used when instantiating the DA3 model
- `device`: target torch device
- `encode_layer`: must be `18`; other layers are not currently supported by the public API

### `encode(images)`

Input:

- `images`: tensor with shape `(B, S, C, H, W)`
- images must already be normalized the same way as DA3 inputs in the reconstruction pipeline

Return value:

```python
{
    "tokens": x_18,  # (B, S, N, D), includes cls token
    "H": H,
    "W": W,
}
```

Notes:

- `tokens` includes the cls token at position `0`
- the token tensor is taken from the raw backbone state exported at `encode_layer`
- for layer 18, `local_x` is not stored separately because it is identical to `x`
- non-18 layers are rejected, because decode would otherwise need both `x` and `local_x`

### `decode(latents)`

Input:

- the dict returned by `encode()`

Return value:

- same output schema as `DA3Wrapper.inference_batch()`
- includes `pointmap`, `depth`, `depth_conf`, `ray`, `ray_conf`, `c2w`, `intrinsics`

Internally, decode:

1. resumes the ViT backbone from `encode_layer + 1`
2. collects DPT features from the configured `out_layers`
3. runs the normal DA3 depth / pointmap head

## Roundtrip test script

Test script: [test_occ_rae.py](test_occ_rae.py)

This script mirrors the demo input pipeline from [inference.py](inference.py), but instead of calling the model directly it runs:

1. demo RGB loading
2. `OccRAE.encode()`
3. `OccRAE.decode()`
4. save output in the same `pts3d_*.npy` format used by [vis_viser.py](vis_viser.py)

### Basic usage

```bash
python test_occ_rae.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --input_dir ./demo_data/input \
    --output_dir ./demo_data/output_occ_rae
```

This writes scene folders under `./demo_data/output_occ_rae` containing:

- `pts3d_render.npy`: OccRAE encode/decode roundtrip output

`test_occ_rae.py` always enables `pose_from_depth_ray=True` so the saved output contains distinct camera poses for viser visualization.

### Compare against direct inference

```bash
python test_occ_rae.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --input_dir ./demo_data/input \
    --output_dir ./demo_data/output_occ_rae \
    --compare
```

With `--compare`, each scene directory also contains:

- `pts3d_direct.npy`: direct `inference_batch()` output from the same model

The script also prints numeric differences for:

- pointmap mean / max absolute error
- depth mean / max absolute error

## Visualizing with viser

Viewer: [vis_viser.py](vis_viser.py)

After running the test script:

```bash
python vis_viser.py --input_folder ./demo_data/output_occ_rae
```

The viewer scans each scene folder and loads files matching `pts3d_*.npy`. In the dropdown you can switch between:

- `render`: OccRAE decode output
- `direct`: direct inference output, if `--compare` was enabled

The saved file format is compatible with the viewer because each `.npy` contains:

- `pts3d`
- `pts3d_local`
- `colors`
- `conf`
- `focal`
- `c2w`

## Feature extraction script

Extraction script: [extract_occany_features.py](extract_occany_features.py)

This script is for dataset-wide token dumping rather than demo visualization. It:

- runs on training-style datasets
- extracts cached tokens at layer 18
- saves `.pth` files under `<processed_root>/<scene>/<frame>.pth`

Example:

```bash
python extract_occany_features.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --train_dataset "6000 @ WaymoSeqMultiView(...)" \
    --output_dir ./outputs/occany_features \
    --resolution 518 294
```

Saved fields:

- `tokens`
- `1st_frame_relpath`
- `timesteps`
- `output_resolution`

## Saving validation latents during OccRAE training

Training launcher: [sh/train_occrae.sh](sh/train_occrae.sh)

The training wrapper now enables validation latent dumps by default. On each validation run it writes sampled `.pth` latents under the run directory:

- `validation_latents/step_<train_step>/<processed_root>/<scene>/<frame>.pth`

Each file uses the same payload contract as [extract_occany_features.py](extract_occany_features.py), so it can be passed directly into [test_occ_rae.py](test_occ_rae.py) with `--latent_path`.

The wrapper exposes one switch before launch:

- `SAVE_VALIDATION_LATENTS=0` to disable validation latent dumps

Example decode of a saved validation latent:

```bash
python test_occ_rae.py \
    --occany_recon_ckpt checkpoints/occany_plus_recon_1B.pth \
    --latent_path /path/to/run/validation_latents/step_0001000/ddad_processed/scene_name/000046_5.pth \
    --output_dir ./demo_data/output_occ_rae
```

That command writes `pts3d_render.npy`, which can then be visualized with [vis_viser.py](vis_viser.py).

## Current limitations

- the current design assumes the saved state comes from a local-attention layer when `local_x` is omitted
- only `encode_layer=18` is currently exposed by the public API and scripts
- the implementation is currently targeted at the DA3-based OccAny reconstruction model
- the roundtrip test script is focused on the demo RGB folder structure used by [inference.py](inference.py)

## Minimal example

```python
import torch

from occany.model.occ_rae import OccRAE

occ_rae = OccRAE(
    weights_path="checkpoints/occany_plus_recon_1B.pth",
    output_resolution=(518, 518),
    device="cuda",
    encode_layer=18,
)

images = torch.randn(1, 5, 3, 294, 518, device="cuda")
latents = occ_rae.encode(images)
output = occ_rae.decode(latents)

print(latents["tokens"].shape)
print(output["pointmap"].shape)
```

Expected shapes:

- `latents["tokens"]`: `(B, S, N, 1536)`
- `output["pointmap"]`: `(B, S, H, W, 3)`
