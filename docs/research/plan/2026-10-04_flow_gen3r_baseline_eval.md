# Plan: pretrained Gen3R on the all-data ctx2fwd8 flow benchmark

**Date:** 2026-10-04 · **Thread:** flow · **Cluster:** BSC
· **Reference:** DeltaTok-flow `BSC:46861907`–`18`; VGGT-World `cityscapes.pt` stride1 `BSC:46962868`–`70`
· **Model:** Gen3R (arXiv 2601.04090), vendored at `third_party/Gen3R`, HF checkpoint `JaceyH919/Gen3R` (no retrain)
· jobs: _pending_ · deck: _pending_ · prior cycle: `plan/2026-10-01_flow_vggt_world_baseline_eval.md`

## 0. What is scored

- **Windows:** the same per-set configs and loaders as VGGT-World (`eval_deltatok_flow_alldata_ctx2fwd8_{kitti,nuscenes,waymotest}_bsc`).
  Frames 0–1 are context; frames 2–9 are the forecast (`slice(2, 10)`).
- **Gen3R input** (user choice 2026-10-04):
  - **Context:** both frames, no camera. Ctx 0–1 go in slots 0–1 of a 13-frame clip; Gen3R needs 4k+1 frames, so 10 is not allowed.
    Forecast frame k is read from slot k (slots 2–9). Slots 10–12 are dropped. One call per window, with no unroll.
  - **Cameras:** zero Plücker, their camera-free mode.
  - **Text:** empty prompt. This matches VGGT-World's Table 3 row for Gen3R with no text and no pose (arXiv 2603.12655 §4.2).
  - **Resolution:** native aspect. KITTI 336×1008; nuScenes and Waymo 448×896. Depth is resized back to the loader size.
- **Rows:**
  - **Main:** DA3METRIC scale on the decoded context slots, the same `_da3_scale` as VGGT-World.
  - **`_oracle`:** median GT scale on the forecast frames.
  - **No `_tok` row.** Gen3R has no 10-frame true-token decode.
- **Metrics:** `_compute_frame_losses` plus `compute_chamfer_metrics` (Acc / Comp / Chamfer), unchanged from `eval_vggt_world.py`.

## 1. Pre-flight

1. **BSC reachable.** On 2026-10-04 SSH to `alogin1`, `glogin1` and `transfer1` timed out; Jean Zay answered.
2. **Checkpoint**, 40.5 GB: `transformer` 6.5, `text_encoder` (umT5-xxl) 22.7, `clip_image_encoder` 4.8, `vggt` 4.8,
   `geo_adapter` 1.3, `wan_vae` 0.5 GB.
   - Locally run `ssh -N -R 15432 bsc` in the background.
   - Then on the BSC login node:
     `HTTPS_PROXY=socks5h://localhost:15432 huggingface-cli download JaceyH919/Gen3R --local-dir /gpfs/scratch/ehpc1001/quan/gen3r/checkpoints`.
3. **Env:**
   - On `alogin1`, `source env_bsc.sh`, then import `diffusers transformers accelerate open3d cv2 matplotlib imageio easydict einops`.
   - Then try `from gen3r.pipeline import Gen3RPipeline` with `third_party/Gen3R` on `sys.path`.
   - If anything is missing: `python -m venv --system-site-packages /gpfs/projects/ehpc1001/venvs/gen3r`.
     Pip-install only the missing packages, at the pins in `third_party/Gen3R/requirements.txt`, through the same proxy.
     Never reinstall `torch`.
   - Flash-attn, xformers and sageattention are optional: `wan_transformer3d.py:33-58` wraps them in try/except.
4. **Code on BSC:** sync with the `monitor-sync` skill, then md5 the 4 edited/new files against local.

## 2. `third_party/Gen3R` patches (2 files)

**`gen3r/models/geometry_adapter/geometry_adapter.py`** — the decoder hard-codes a 560×560 grid.

- `Interpolate` (line 27, currently unused) gains an exact integer ratio:
  ```python
  class Interpolate(nn.Module):
      def __init__(self, size=None, ratio=None):
          super().__init__()
          self.size = size
          self.ratio = ratio  # (num, den): size = dim * num // den

      def forward(self, x):
          size = self.size
          if self.ratio is not None:
              size = (x.shape[-2] * self.ratio[0] // self.ratio[1], x.shape[-1] * self.ratio[0] // self.ratio[1])
          return F.interpolate(x.float(), size=size, mode='nearest-exact').type_as(x)
  ```
- Line 124: `Upsample(size=(80, 80), mode='nearest-exact'),  # 70, 70 -> 80, 80`
  → `Interpolate(ratio=(8, 7)),  # 70, 70 -> 80, 80 at 560`.
  - The module has no parameters, so the state-dict keys are unchanged.
  - At 560 it is the same `F.interpolate` call to size (80, 80).
  - At other sizes: KITTI latent 42×126 → 48×144 → pad → 24×72 = 336/14 × 1008/14.
    nuScenes and Waymo: 56×112 → 64×128 → 32×64.

**`gen3r/pipeline/pipeline_gen3r.py`** — `__call__` puts a second frame at the last slot (`[0, -1]`, line 699), not slot 1.

- Add the kwarg `control_index: Optional[List[int]] = None` after `min_max_depth_mask` (line 584).
- Line 695: `if control_images.shape[1] == 1:` becomes
  ```python
  if control_index is not None:  # frames already at their slots in an F-long control_images
      pass
  elif control_images.shape[1] == 1:
  ```
- The default `None` leaves their 1view, 2view and allview paths unchanged.

## 3. `eval_gen3r.py` (new)

`cp eval_vggt_world.py eval_gen3r.py`, then:

- **Docstring and usage:** Gen3R, the slot mapping from §0, plan path.
- **Imports:**
  - `extra=["third_party/pyTorchChamferDistance", "third_party/Gen3R"]`.
  - Drop the `tensorboardX` stub and the VGGT-World imports.
  - Add `from gen3r.pipeline import Gen3RPipeline`, `from gen3r.utils.common_utils import convert_to_token_list`,
    `from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri`, and `from einops import rearrange`.
- **Constants:**
  - `GEN3R_HW = {(168, 518): (336, 1008), (266, 518): (448, 896)}  # loader (H, W) -> multiples of 112, ~560² px`
  - `NEG_PROMPT = "bad detailed"  # infer.py`
- **Args:**
  - Drop `--fm_steps`, `--rollout` and `--resolution`.
  - `--ckpt` defaults to `/gpfs/scratch/ehpc1001/quan/gen3r/checkpoints`.
  - Add `--steps 50`, `--guidance 5.0` (infer.py) and `--slot_step 1`. `5` is the 49-frame, 10 Hz layout, kept as an option.
  - Add `--ctx_mode {both,first}`, default `both`. `first` is ctx 0 only, their trained 1view mask; it is used in the smoke only.
  - Add `--prompt ""`, which is text-free (§0). Gen3R trained with 20% prompt dropout (`train_dit.py:1444`).
  - Drop `--bsize`; windows are always one per batch.
  - `--output_dir` defaults to `results/gen3r_alldata_ctx2fwd8`. The subdir is `<ctx_mode>_step<slot_step>_native`.
- **`_load_gen3r(path, device)`:** `Gen3RPipeline.from_pretrained(path).to(device).to(torch.bfloat16)` as `infer.py`. Print the
  component classes and the scheduler class.
- **`_generate(pipe, x, ctx_mode, slot_step, n_slots, hw, args, gen)`**, one window per call (the pipeline's camera reshape is batch-1, `pipeline_gen3r.py:684`):
  ```python
  x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)  # (2, 3, h, w) ctx in [0, 1]
  ctrl = torch.zeros(1, n_slots, 3, *hw, device=x.device)                             # (1, S, 3, h, w); S = 13
  idx = [0, slot_step] if ctx_mode == "both" else [0]
  ctrl[0, idx] = x[: len(idx)]
  cams = torch.zeros(1, n_slots, 6, *hw, device=x.device)                              # (1, S, 6, h, w) camera-free
  return pipe(prompt=args.prompt, negative_prompt=NEG_PROMPT, control_cameras=cams, control_images=ctrl.to(torch.bfloat16),
              control_index=idx, num_frames=n_slots, height=hw[0], width=hw[1], num_inference_steps=args.steps,
              guidance_scale=args.guidance, generator=gen, output_type="latent", return_dict=False)[0]  # (1, 16, (S-1)/4+1, h/8, 2w/8)
  ```
- **`_decode(pipe, latents, slots, out_hw)`** returns the same `(depth, conf, K, c2w)` as VGGT-World `_decode`:
  ```python
  geo = latents.chunk(2, dim=-1)[1]                                                    # (1, 16, (S-1)/4+1, h/8, w/8)
  tok = pipe.geo_adapter.decode(geo).sample                                            # (1, 5C, S, h/14, w/14)
  agg, frames = convert_to_token_list(rearrange(tok, "b c f h w -> b f h w c"), 14)    # 4 x (1, S, 5+P, C)
  pose_enc = pipe.vggt.camera_head(agg)[-1]                                            # (1, S, 9) trunk attends over all S slots
  w2c, K = pose_encoding_to_extri_intri(pose_enc[:, slots], frames.shape[-2:])        # (1, T, 3, 4), (1, T, 3, 3)
  depth, conf = pipe.vggt.depth_head([a[:, slots] for a in agg], frames[:, slots], 5) # (1, T, h, w, 1), (1, T, h, w)
  ```
  - The rest is VGGT-World `_decode` from `depth = depth[..., 0]` on: the resize to `out_hw`, the K rescale and `c2w`, in float32.
  - The depth head is per frame (chunked), so slicing the slots before it gives the same output.
- **Main loop changes:**
  - Remove the VGGT-World model and `_forecast_tokens`. Keep `_da3_scale`.
    Scoring is in `eval_forecast_metrics.py` (`plan/2026-10-05_flow_forecast_dump_then_score.md`).
  - `slots = [k * args.slot_step for k in range(T)]` and `n_slots = 4 * ((slots[-1] + 3) // 4) + 1`. That gives 13 at step 1 and 49 at step 5.
  - Each window gets `gen = torch.Generator("cuda").manual_seed(args.seed + it)`.
  - **Debug prints:** the scorer prints per-frame depth L1 for all T frames. Context frames 0–1 check that the slot-0/1 conditioning landed.
    It also prints the predicted step length after the oracle scale, next to GT.
- **Output:** `<run>/<set>/<n:05d>.pt` (depth, K, c2w), scored and drawn by `eval_forecast_metrics.py`.

## 4. `slurm/eval_gen3r_alldata_ctx2fwd8_bsc.slurm` (new)

`cp slurm/eval_vggt_world_alldata_ctx2fwd8_bsc.slurm slurm/eval_gen3r_alldata_ctx2fwd8_bsc.slurm`, then:

- **SBATCH lines:** `--job-name=eval_gen3r_alldata_ctx2fwd8` and `--output` / `--error` `slurm/output/eval_gen3r_alldata_ctx2fwd8_bsc_%j.{out,err}`.
- **Header comment:** Gen3R, 1 GPU.
- **Venv:** after `source env_bsc.sh`, add `[ -d /gpfs/projects/ehpc1001/venvs/gen3r ] && source /gpfs/projects/ehpc1001/venvs/gen3r/bin/activate`, only if §1.3 creates it.
- **Defaults:** replace `CKPT` / `ROLLOUT` / `RESOLUTION` with `: "${CKPT:=/gpfs/scratch/ehpc1001/quan/gen3r/checkpoints}"`
  `: "${CTX_MODE:=both}"` and `: "${SLOT_STEP:=1}"`. Set `OUTPUT_DIR:=/gpfs/scratch/ehpc1001/quan/forecast_preds/gen3r_alldata_ctx2fwd8`.
- **Command:** `srun python eval_gen3r.py ... --ctx_mode "$CTX_MODE" --slot_step "$SLOT_STEP"`.
- **Walltime:** stays at `03:00:00` until the smoke sizes it (§6).

## 5. Smoke

- **Jobs:** 6, for `SET` ∈ {kitti, nuscenes, waymotest} × `CTX_MODE` ∈ {both, first}, with `NUM_ITEMS=4` and 1 GPU.
  - Override at submit: `--time=00:30:00`.
  - One job goes to `acc_debug` (MaxSubmitPU=1); the other 5 go to `acc_ehpc`.
  - One scorer job per dump: `slurm/eval_forecast_metrics_alldata_ctx2fwd8_bsc.slurm`, `NUM_ITEMS=4`, `--dependency=afterok`.
- **Checks:**
  - **Load:** the geo-adapter load prints no missing or unexpected keys.
  - **Inputs:** `x01` is in [0, 1].
  - **Context slots:** the frame-0/1 depth L1 is close to VGGT-World `_tok` on the same sets, so the conditioning landed.
  - **`both` vs `first`:** compare forecast depth L1 on the same windows (same seeds).
  - **Cost:** s/window and peak memory (`torch.cuda.max_memory_allocated`).

## 6. Full runs

- **Jobs:** one job per set, `CTX_MODE=both`, 1 GPU, as VGGT-World.
- **Walltime:** smoke s/window × windows × 1.3. Windows are KITTI 2,013, nuScenes 2,382 and Waymo 2,018.
  - If that exceeds 72 h, run fewer windows with `--num_items`.
- **Scoring:** one scorer job per set, `afterok`.
- **Output:** `/gpfs/scratch/ehpc1001/quan/forecast_preds/gen3r_alldata_ctx2fwd8/both_step1_native/<set>.json`.
