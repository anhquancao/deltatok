# CLAUDE.md

## Interaction rules

- Never add `Co-Authored-By` or `Claude-Session` trailers to commit messages.
- Ask clarifying questions or offer choices via `AskUserQuestion`, never inline in plain text.
- On multi-step tasks, post a one-line progress update as each step finishes, not in a batch.

## Response style

Write for a reader who is scanning, not studying.

- **Answer first.** One or two sentences with the conclusion, before any detail.
- **One idea per sentence.** Split anything longer than ~20 words.
- **No stacked clauses.** Chains of em-dashes, parentheses, and "which/so that" tails become separate sentences.
- **Plain words.** "makes it faster", not "constitutes a speedup". Avoid leverage, surface, orthogonal, modulo, non-trivial.
- **Bullets when there are 3+ items**, with a bold label starting each one.
- **Name things concretely.** `file.py:42`, a job ID, a config key — never "the relevant script".
- **Numbers need units and a baseline.** "loss 0.41 (was 0.55)", not "loss improved".
- **End with what I did not do.** Skipped steps, failures, assumptions — one line each.
- **Stop when the answer is delivered.** No closing summary of the summary.

## Critical: never run code locally

This machine has no GPUs and no env. SSH to a cluster for anything that runs.

| | Jean Zay | BSC (MareNostrum) |
|---|---|---|
| SSH alias | `jean-zay` (reverse tunnel — see `jeanzay-karolina-tunnel` skill if refused) | `bsc` |
| Account | `trg@h100` | `ehpc1001` or `ehpc880` |
| Partition / QoS | `gpu_p6` via `-C h100`; `qos_gpu_h100-t3` (20 h) / `-t4` (100 h) / `-dev` (2 h) | `acc`; `acc_ehpc` (prod, 72 h) / `acc_debug` (2 h) |
| Node shape | 4 GPU, `--cpus-per-task=24` | 4 H100, `--cpus-per-task=20` |
| Repo path | `$TRG_WORK/code/deltatok` | `/gpfs/projects/ehpc1001/code/deltatok` |
| Env | `source env_jz_h100.sh` | `source env_bsc.sh` |

```bash
ssh jean-zay "cd \$TRG_WORK/code/deltatok && source env_jz_h100.sh && <command>"
ssh bsc "cd /gpfs/projects/ehpc1001/code/deltatok && source env_bsc.sh && <command>"
```

Many checked-in scripts carry **stale accounts** (`zuw@h100`, `cya@h100`, `lwy@h100`, `ehpc551`, `ehpc793`). New and edited files use `trg@h100` / `ehpc1001` / `ehpc880`.

**Karolina is deprecated.** Anything tagged `_karolina`, any un-suffixed `slurm/*.slurm`, and any `conda activate occany` is dead. The bare `sh/train_*.sh` wrappers still default `CONFIG_NAME` to a `_karolina` config; the slurm scripts override it.

## Environment bootstrap

There is no conda on either site. Source the cluster's own env script before any `python`.

Jean Zay has `env_jz_{h100,a100,v100}.sh`. Default to H100. Never use one cluster's script on the other — `env_bsc.sh` runs `module purge` and activates a venv that does not exist on Jean Zay.

Submit from a **login shell**: `ssh <host> "bash -lc '... sbatch ...'"`. Only the login profile defines `module`, and `--export=ALL` will not carry it into the job. Symptom when you forget: the job dies after ~1 s, and `*.err` shows `module: command not found` then `ModuleNotFoundError: No module named 'torch'`.

`PROJECT` and `SCRATCH` are part of the repo contract. Eval helpers default datasets to `$PROJECT/data/...` and processed artifacts to `$SCRATCH/...`.

## Job monitor / knowledge base

`../monitor_jobs` tracks this repo's SLURM jobs. **Read its `data/*.json` directly. Never start its server or call its HTTP API.** Schemas are in the "RAG knowledge base" section of `../monitor_jobs/CLAUDE.md`.

- `monitor_jobs.json`, `archived_jobs.json` — job records keyed `"<Cluster>:<JobID>"`. Search both for full history.
- `logs/<Cluster>/<name>_<jobid>.{out,err}` — cached stdout/stderr. Grep here for "why did job N fail".
- `research.json` — journal entries per research direction. `results.json` — benchmark tables.
- `projects.json` — the `Deltatok` entry holds deploy paths and the rsync `excludes` that `syncer.py` reads. A new local output dir must be excluded here **and** in `.gitignore`.

**Grep metrics from the `*.out`/`*.err` logs first.** Trainers print scalars to stdout via `metric_logger`. Live logs are in `slurm/output/`.

**TensorBoard is the fallback, and it has scalars the logs lack** — a metric added to `log_add_scalar` before its stdout echo exists only in TB. Event files are mirrored locally under `/mnt/d/tb_logs/<log_root>/<run>/tb_logs/`, so read them here, not on the cluster.

**Always fast-scan an event file. Never use `EventAccumulator` or `EventFileLoader`** — these runs store images, so a file is ~4.7 GB and a full parse crawls (~400 events/s, killed by the BSC login watchdog). Walk the TFRecord frames yourself and *seek past* any record over 8 KB; only small records are scalars. 4.7 GB then takes ~100 s instead of hours.

```python
# frame = uint64 len | uint32 crc(len) | payload | uint32 crc(payload)
hdr = f.read(12)
ln = struct.unpack('<Q', hdr[:8])[0]
if ln > 8192: f.seek(ln + 4, 1); continue        # image/histogram — skip, do not parse
e = Event.FromString(f.read(ln)); f.read(4)      # tensorboard.compat.proto.event_pb2
```

## Running jobs

```bash
bash sh/train_deltatok.sh            # DeltaTok tokenizer   → train_deltatok.py
bash sh/train_deltatok_flow.sh       # flow matching        → train_deltatok_flow.py
bash sh/train_occrae_img_decoder.sh  # OccRAE image decoder → train_occrae_img_decoder.py
ssh bsc "bash -lc 'cd /gpfs/projects/ehpc1001/code/deltatok && sbatch slurm/deltatok/<name>_bsc.slurm'"
```

Scripts are named `<name>_<cluster>.slurm`, and the cluster tag (`_jz`, `_bsc`) is **always the last suffix**. Tokenizer arms live in `slurm/deltatok/`, flow arms in `slurm/deltatok_flow/`, eval and diagnostics at `slurm/` root. Each training folder has an `archive/` for retired arms. Paths inside the scripts are relative to the repo root, so always `sbatch slurm/deltatok[_flow]/<name>.slurm` from there.

Training **always** resumes from `<RESULTS_DIR>/<RUN_NAME>/ckpts/current.pth` when that file exists. There is no `RESUME` env var and no `--resume` flag. A plain relaunch continues the run. To start an arm over, change `RUN_NAME` or delete its `ckpts/`.

Use the `chain-slurm-jobs` skill for chained resume jobs. On BSC, chain at 40 h rather than the 72 h cap. The scheduler is `sched/backfill` and walltime does **not** enter the priority formula, so a shorter request only ever fits more gaps. Measured p90 queue wait was 4 min at 12 h against 8.6 h at 48 h. `training.exit_before_time_limit=true` keeps the split checkpoint-safe.

### Critical: verify the cluster copy before `sbatch`

The user syncs manually. Never assume a local edit reached the cluster, and never sync it yourself — no `scp`, no `rsync`. Grep the remote file first, and ask the user to sync if it is stale.

```bash
ssh jean-zay "bash -lc 'grep -E \"account|partition|time\" \$TRG_WORK/code/deltatok/slurm/<script>.slurm'"
```

### Critical: watch the job until training starts

A returned job ID is **not** a successful launch. Watch until the job is `RUNNING` *and* its log shows a first iteration or loss line. Report only then. Most breakage lands in the first ~60 s: missing `module`, Hydra config errors, checkpoint shape mismatches, OOM, DDP port collisions.

Run the watch as an `until` loop with `run_in_background: true` (foreground `sleep` is blocked) that exits on the first loss line **or** when the job leaves `squeue`, so silence is never mistaken for running. `Monitor` is for open-ended per-event tails, not this one-shot check. Loop on `squeue`, then tail both streams:

```bash
ssh jean-zay "bash -lc 'squeue -j <jobid> -h -o \"%T %M %R\"'"
ssh jean-zay "bash -lc 'tail -40 \$TRG_WORK/code/deltatok/slurm/output/<name>_jz_<jobid>.err'"
```

If it dies, report the actual error, not the submission. Test log conditions with `grep -q`. Do not use `grep -c ... || echo 0` — it emits two values and breaks the integer test. A long `PENDING` wait is the one case not worth watching; hand back the job ID.

## Architecture

The repo trains **DeltaTok** (tokenizer), **DeltaTok-flow** (flow matching), and **OccRAE** over cached Depth Anything 3 (DA3) features. All three are Hydra-configured and SLURM-distributed. Nothing else is trained here.

`occany/` is a **library, not a training target**. The trainers import `DA3Wrapper` (`occany/model/model_da3.py`), `occany/training_da3.py`, `occany/datasets`, and `occany/loss` from it. `occany/model/must3r_blocks/` is historically named shared code that this path still imports, so it is not dead. Everything else under `occany/` and the root occupancy entrypoints are legacy and out of scope.

**OccRAE** caches DA3 tokens at **layer 12**, a pre-fusion local layer. `OccRAE.encode()` early-exits through `DA3Wrapper.encode_to_layer` and runs only blocks 0–12 of the giant's 40. That fast path holds *only* while `encode_layer < alt_start=13`. `encode_layer: 18` is post-fusion and falls back to a full 40-block forward, roughly 3× the backbone cost, so treat it as a deliberate expense. See `docs/occrae/occrae.md`. Components: `extract_occany_features.py` (dump tokens dataset-wide), `occrae/{deltatok,deltatok_flow,img_decoder}_trainer.py`, `occrae/dataset/occrae_tokens.py`, `occrae/network/` and `deltatok_shared.py` (nets), `occrae/sigreg.py` and `z_spread.py` (latent regulariser and diagnostics), `occrae/flow_matching.py` (ODE/sampler), `occrae/metric_logger.py` (scalar logging to stdout + TB).

**Configs** mirror `slurm/`: `configs/deltatok/`, `configs/deltatok_flow/`, `configs/rae/`. Each wrapper's `CONFIG_DIR` points at its own folder, so `CONFIG_NAME` stays a bare name. Configs are per-cluster, with the tag (`_bsc`, `_jeanzay`) always the last suffix. Edit the one for the cluster you submit to. Hydra resolves `defaults:` relative to the config's own folder, so a config and its parent must share a folder.

**`third_party/`** must be importable from the checkout. Entrypoints prepend the paths via `occany.utils.runtime_paths.prepend_vendored_import_paths()`, and shell wrappers use `occany_prepend_pythonpath`.

DeltaTok is the active research direction. Read `docs/README.md` first. Research is filed by stage: `docs/research/{plan,results,analysis}/<date>_<thread>_<slug>.<ext>`, and `docs/research/README.md` ledgers every thread. Runbooks and incidents are in `docs/infra/`, OccRAE notes in `docs/occrae/`.

## Conventions

- **Edits:** surgical. Change only what the task requires — no drive-by refactors, renames, or reformatting. Explain each edit so a human can verify it.
- **Comments:** terse. The fewest words that convey the *why*, not the *what*. Default to one short line, in shell and slurm and config files too.
- **Never lengthen a comment you edit.** The replacement must be no longer than what it replaces. Cut a stale line to make room. Measurements, dated findings, and sweep history belong in `docs/research/`, not a script header.
- **Tensor code:** comment each line and annotate the resulting shape inline, e.g. `x = rearrange(x, 'b (t s) d -> (b s) t d', t=t, s=s)  # (B*S, T, D)`. Define each shape symbol where it first appears.
- **New variant scripts:** copy the closest existing file (`cp old.py new.py`), then edit. Never rewrite from scratch — copying keeps the diff reviewable.
- **Code search:** prefer `grep` over `rg`.
- **Slide decks** go in `docs/research/results/<date>_<thread>_<slug>_slides.html`: see the section below.
- **Dataset strings** in `configs/**/*.yaml` are `eval()`-ed by `occany.datasets.get_data_loader()`, so they must stay valid Python. Every entry states `num_timesteps` (the consecutive-window length) and names its cameras via `fixed_cams`. Cameras and view budgets are never sampled.
- **Sharding** uses `--world` and `--pid` in `extract_occany_features.py` and `dataset_setup/waymo/preprocess_waymo.py`.

## Slide decks

`docs/research/results/<date>_<thread>_<slug>_slides.html`, self-contained, light theme only, white slide background. No dark-mode variant,
no `prefers-color-scheme` block. Copy the closest existing deck and edit — the chrome (1280x720
stage, nav, print CSS, the `lineChart` helper) is shared and already works.

- **Plots over prose.** Small multiples split by eval set, one metric per panel. Prose only where a
  number needs a caveat.
- **Verify by rendering, never by reading.** Headless Chrome, then assert
  `body.scrollHeight - clientHeight == 0` on every slide, and screenshot each one.
  `--window-size` height is not the viewport height — add ~90 px or the shots come out cropped.
- **Two layout traps:** SVG is inline by default, so without `svg{display:block}` every panel
  overflows ~7 px; and `height:100%` on a grid whose parent also holds a caption overflows by the
  caption's height.
- **End-of-line series tags collide** when two curves converge. Suppress them on the faint
  reference arms and let the legend carry those.

### Arm-comparison frames

From the `eval_depth/<n> @ <dataset>/<scene>_t0-4` panels in each arm's TB event file. Scan with the
seek-skip TFRecord walker (sniff tag+step from the first ~1.6 KB, seek past the payload); a 6-10 GB
file indexes in ~7 s. Run it on the login node when the local mirror lags, and return PNGs only.

- **Columns are `arm | arm | ... | GT token`, always.** Without the GT column the comparison reads as
  a null — the shared gap to the teacher is larger than the gap between arms. GT is byte-identical
  across arms (frozen teacher), so an md5 match is a free check on the crop offsets.
- **AR only, last forecast frame only.** `context_views = num_cameras if num_cameras > 1 else 1`
  (`deltatok_trainer.py:1389`), so t0 is context and its AR cells are blanked. Crop to the final
  forecast timestep and drop the TF columns. Show the rollout only when the rollout is the point.
- **Panel geometry:** 7 cols x 518 px, 36 px header, one row per timestep — KITTI 168 px, nuScenes
  532 px (2 stacked cams, take the top one). Cols `0 RGB | 1 PredDepth(TF) | 2 PredDepth(AR) |
  3 GTTokenDepth | 4 PredRGB(TF) | 5 PredRGB(AR) | 6 GTTokenRGB`; a deck uses `0, 2, 3, 5, 6`.
- nuScenes `scene-1062_005538` is the one night scene — never pick it for an RGB visual.

### L1 error maps

Decodes alone hide small differences. Pair every decode row with its own error row directly below,
same columns.

- `|arm - GT token|` per pixel. **Depth first inverts the JET LUT:** `depth2rgb` runs at a fixed
  `min_depth=0 / max_depth=50` (`_depths_to_rgb_panel`), so colour maps to metres exactly — assert
  the inversion residual is 0. Do the LUT distance in float32; int16 overflows and silently returns
  garbage argmins that look like a peaky error distribution.
- **One fixed colour scale across every panel in the deck.** Never per-panel autoscale — arms stop
  being comparable. Take vmax from the p99 pooled over all arms, frames and datasets, print the range
  and the units on the slide, and say the top 1% clips. `inferno`; the GT column stays as a black
  `error = 0` tile, which also anchors the scale.
- Quote the mean L1 next to the maps. The eye reads the map, the number settles it.
