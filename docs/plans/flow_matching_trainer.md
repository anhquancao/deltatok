# Plan: Simplify OccRAETrainer for Token Flow Matching

Make `train_occrae.py` work with `OccRAETrainer` from `occrae/occrae_trainer.py` by fixing broken
imports, removing video/VAE/CLIP dependencies, and simplifying the trainer into a clean
flow-matching network that predicts OccAny DA3 layer-18 features from pre-extracted token dumps.

## Missing files (from external codebase, not in this repo)

**Direct dependencies of `occrae_trainer.py`:**
1. `Trainer.generation_helper` → `GenerationHelper` — MISSING
2. `Dataset.train_dataloader` → `get_data()` — MISSING
3. `Utils.utils` → `resize_positional_embeddings()` — MISSING

**Transitive deps via `occrae/evaluation_helper.py` (10+ missing modules):**
- `Dataset.dataloader`, `Dataset.train_dataset`, `Metrics.inception_metrics`,
  `Metrics.pixel_metrics`, `Metrics.scenario_ade_metric`, `analyze_traj.trajectory_diversity`,
  `Trainer.occany_model_loader`, `Trainer.da3_pseudo_labeler`, `Network.da3_inference`,
  `Trainer.trajectory_preview_helper`

**Wrong import paths (files exist locally but capitalized path doesn't resolve):**
- `Trainer.abstract_trainer` → `occrae.abstract_trainer`
- `Trainer.evaluation_helper` → `occrae.evaluation_helper`
- `Network.efficient_transformer` → `occrae.network.efficient_transformer`
- `Network.ema` → `occrae.network.ema`
- `Utils.training_summary` → `occrae.util.training_summary`
- In `efficient_transformer.py`: `Network.transformer_block` → `occrae.network.transformer_block`

## Approach

Strip `OccRAETrainer` down to a token-only flow-matching trainer:
- Inherits from `abstract_trainer.Trainer` (logging, optimizer, lr scheduling)
- Uses `efficient_transformer.Transformer` as the denoising network
- Loads data from `OccRAETokenDataset` (pre-extracted DA3 layer-18 `.pth` dumps)
- Flow matching from `occrae/flow_matching.py` (logit-normal timestep, v/x/e modes)
- Drops all VAE/CLIP/video/GenerationHelper/EvaluationHelper complexity

**Key design — token→spatial reshape:**
DA3 tokens `(V, seq_len, 1536)` are treated as `(B, C=1536, T=num_views, H=seq_len, W=1)`.
`efficient_transformer.Transformer` with `proj=1` then gives:
- Spatial attention = within-view token-to-token attention
- Temporal attention = cross-view attention per spatial token position
This preserves meaningful structure: tokens attend within a view and propagate across views.

## Steps

### Phase 1 — Fix import paths (no logic changes)

1. `occrae/network/efficient_transformer.py`: fix `from Network.transformer_block import ...`
   → `from occrae.network.transformer_block import ...`

2. `occrae/occrae_trainer.py`: fix all capitalized imports to `occrae.*` paths; remove imports
   for missing files (`GenerationHelper`, `get_data`, `resize_positional_embeddings`).

### Phase 2 — Rewrite `OccRAETrainer` for token flow matching

3. `occrae/occrae_trainer.py` — rewrite the class:
   - Keep `Trainer` base class (logging, `get_optim`, `save_network`, `adapt_learning_rate`)
   - `__init__`: accept Hydra cfg + (device, rank, world_size, distributed) — same interface as
     `OccRAETrainerOld`. Build `Transformer(input_size=(1536, num_views, seq_len, 1), proj=1)`.
     Set up EMA via `occrae.network.ema.EMA`. Set up optimizer via `get_optim()`.
   - `fit()`: load `OccRAETokenDataset` + `ProcessedRootBatchSampler`; run epoch loop.
   - `train_one_epoch()`: reshape batch tokens `(B,V,C,S,1)` → `(B,C,V,S,1)`, apply flow
     noising (logit-normal t), forward transformer, compute flow loss, gradient accumulation,
     EMA update, TensorBoard logging.
   - `eval_one_epoch()`: same under `torch.no_grad()` + EMA scope; return flow loss.
   - Add `_tokens_to_spatial()` / `_spatial_to_tokens()` shape helpers.
   - Remove: `self.ae`, `self.clip`, `GenerationHelper`, `EvaluationHelper`, `repa_loss`,
     video logging, `training_summary.json`.
   - Keep: flow noising, flow loss, EMA, gradient accumulation, periodic checkpointing.

### Phase 3 — Adapt entrypoint

4. `train_occrae.py`: change import to `occrae.occrae_trainer.OccRAETrainer`. Adapt `main()`
   to call `trainer.run()` (same as `OccRAETrainerOld`).

### Phase 4 — Config

5. Add `configs/train_occrae_efficient.yaml` with `efficient_transformer.Transformer` arch params
   (`hidden_dim`, `depth`, `heads`, `input_size=(1536, num_views, seq_len, 1)`), flow matching
   hyperparams (`mu`, `sigma`, `pred_mode`, `loss_mode`), and dataset paths.

## Relevant files

| File | Action |
|------|--------|
| `occrae/occrae_trainer.py` | Major rewrite — strip to token flow matching |
| `occrae/abstract_trainer.py` | No changes (base class) |
| `occrae/network/efficient_transformer.py` | Fix one import line |
| `occrae/network/transformer_block.py` | No changes |
| `occrae/network/ema.py` | No changes |
| `occrae/dataset/occrae_tokens.py` | No changes |
| `train_occrae.py` | Fix import + adapt init args |
| `configs/train_occrae_efficient.yaml` | New config file |
| `occrae/trainer.py` | Keep as-is (`OccRAETrainerOld` backward compat) |

## Decisions

- **Model**: `efficient_transformer.Transformer` with factored spatial+temporal attention.
- **Flow matching only**: no transport/GLD dependency.
- **No evaluation metrics**: validation = flow loss only (no FID/FVD/ADE).
- **Trajectory conditioning**: wiring stays in `efficient_transformer` but unused here.
- **Keep `OccRAETrainerOld`**: preserve existing GLD-based training path.

## Verification

```bash
python -c "from occrae.occrae_trainer import OccRAETrainer"
python -c "from occrae.network.efficient_transformer import Transformer"
source env_bsc.sh && python train_occrae.py \
  --config-name train_occrae_efficient \
  --cfg dataset.train_feature_path=./demo_data/occrae_emb_overfit