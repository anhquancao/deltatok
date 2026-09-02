# Plan: Online Token Extraction Integration

## Objective
Replace the offline `OccRAETokenDataset` with the `PreprocessedSequenceDataset` in `train_occrae.py` to perform online token extraction during flow-matching training. This deprecates the pre-extracted token workflow by directly utilizing the original images to extract features on the fly.

## Scope & Impact
* **Dataset Loading:** Shifts from reading `(V, seq_len, dim)` tokens off `.pth` dumps to reading raw `(V, C, H, W)` `.npz` sequence images.
* **Trainer Workflow:** `OccRAETrainer` will manage a frozen `OccRAE` encoder and extract tokens online before handing them to the main `vit` transformer training logic.
* **Configurations:** Hydra config parameters for dataset paths will be overhauled to point to `preprocess_data_root` instead of feature paths, while encoder checkpoint settings move into the model section.
* **Impact:** Eliminates the intermediate token-saving step but adds encoding time to the training loop. Requires careful memory management so the frozen encoder and flow matcher can coexist on the same GPU, including mixed-precision handling for encoded tokens.

## Key Files
1. `occrae/dataset/preprocessed_sequence.py` (New file or added to `occrae_tokens.py`)
2. `occrae/occrae_trainer.py`
3. `configs/train_occrae*.yaml`
4. `train_occrae.py`
5. `occany/model/occ_rae.py`

## Proposed Solution (Implementation Steps)

### Step 1: Migrate `PreprocessedSequenceDataset`
* Move `PreprocessedSequenceDataset`, `build_sequence_records`, `SequenceRecord`, `ProcessedDatasetConfig`, `PROCESSED_DATASET_CONFIGS`, and helper functions from `extract_occany_features.py` into a new file `occrae/dataset/preprocessed_sequence.py` (or add them to `occrae/dataset/occrae_tokens.py`).
* Add a `get_processed_root_index_groups(self)` method to `PreprocessedSequenceDataset` that mirrors the existing `OccRAETokenDataset` contract expected by `ProcessedRootBatchSampler`. It should build a lookup from `processed_root` to dataset indices.
* Update `ProcessedDatasetConfig` to accept a list/tuple of `(W, H)` resolution tuples. Update `SequenceRecord` to carry either a single resolution or a list of allowed resolutions for that root.
* Keep `PreprocessedSequenceDataset.__getitem__` responsible for loading raw sequence images and metadata only. Do not randomly resize per sample in `__getitem__`; instead, return the original image tensors plus the allowed resolution list so batch collation can choose one resolution for the entire batch.

### Step 2: Update Configuration Files
* Update `configs/train_occrae.yaml`, `configs/train_occrae_fm.yaml` and `configs/train_occrae_fm_overfit.yaml`.
* **Remove:** `train_feature_path`, `val_feature_path`.
* **Add to `dataset`:** `preprocess_data_root` (e.g., `/gpfs/scratch/ehpc558/quan/occrae_data`), `subsampling_rate`, `max_stride`, and optional multi-resolution settings.
* **Add to `model`:** `occany_recon_ckpt` (e.g., `checkpoints/occany_plus_recon_1B.pth`) and `encode_layer` (default 18).
* **Resolution Support:** Modify `ProcessedDatasetConfig` and dataset config sections to accept a list of multiple target resolutions in `(W, H)` order, for example `[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)]`. The dataloader collate function, not `__getitem__`, should randomly sample one resolution per batch to match the dynamic scaling approach used in `sh/train_occany_recon.sh` while preserving homogeneous batch shapes.

### Step 3: Update `OccRAETrainer` Dataset Loading
* In `occrae/occrae_trainer.py` `fit()` method:
  * Remove `OccRAETokenDataset` instantiation.
  * Use `build_sequence_records` and instantiate `PreprocessedSequenceDataset` for both training and validation splits.
  * Pass the new dataset to `ProcessedRootBatchSampler`, preserving the existing `set_epoch()` behavior so DDP shuffling remains epoch-aware.
  * Replace the default dataloader collation with a custom `collate_fn`. Because `ProcessedRootBatchSampler` already groups examples by `processed_root`, each batch should share the same allowed resolution set. The `collate_fn` should:
    * pick one target resolution for the batch,
    * apply `crop_resize_if_necessary` to every image in every sequence in that batch,
    * stack images into `(B, V, C, H, W)`, and
    * return the chosen `output_resolution` alongside `imgs` and other metadata.
  * Avoid per-sample random resizing and avoid padding-based batching. Uniform per-batch resizing is simpler, matches the existing multi-resolution training pattern, and keeps the encoder input dense and efficient.

### Step 4: Online Encoding in Training Loops
* Update `train_one_epoch()` and `eval_one_epoch()` in `OccRAETrainer`.
* Since different roots can have different resolutions, implement a dynamic cache for the frozen `OccRAE` encoder inside `OccRAETrainer`, keyed by `output_resolution`, following the same idea as the `occ_rae_cache` used in `extract_occany_features.py`.
* Initialize `OccRAE` from `model.occany_recon_ckpt` and `model.encode_layer`, keep it frozen, and run `encode()` under `torch.no_grad()`.
* Retrieve `batch["imgs"]`, move them to `self.device`, and select the cached encoder instance for `batch["output_resolution"]`.
* Call `occ_rae.encode(imgs)` to compute `tokens` online. The encoder output is `(B, V, S, C)` in trainer naming, where `V` is the number of views, `S` is the per-view token sequence length, and `C` is the embedding dimension.
* Convert encoder output into the existing training contract with:

  ```python
  x_tokens = latents["tokens"].permute(0, 1, 3, 2).unsqueeze(-1).contiguous()
  ```

  This produces `(B, V, C, S, 1)`, which is what `_tokens_to_spatial()` and the flow-matching path already expect.
* Cast encoded tokens to the trainer autocast dtype if needed so the online path matches the existing mixed-precision training behavior.
* Proceed with the existing `flow_noising` and `vit` training steps in both training and evaluation, so `eval_one_epoch()` uses the same online encoding logic instead of reading `batch["tokens"]`.

### Step 5: Update `train_occrae.py` Arguments
* Add parser overrides for the new config keys that need runtime control, at minimum `--preprocess_data_root`, `--occany_recon_ckpt`, and optionally `--encode_layer`.
* Keep the existing result-directory and run-name override flow unchanged.

## Verification & Testing
1. **Unit Validation:** Ensure `get_processed_root_index_groups()` properly maps sequence records to indices so `ProcessedRootBatchSampler` produces root-homogeneous batches.
2. **Batch Collation Validation:** Verify the custom `collate_fn` always returns dense `(B, V, C, H, W)` tensors plus a single `(W, H)` `output_resolution` for the batch.
3. **Shape Validation:** Log online encoder outputs and confirm the transformed tensor matches `(B, V, C, S, 1)` before entering `_tokens_to_spatial()`.
4. **Local Run:** Run `train_occrae.py` locally or with an overfit config (`train_occrae_fm_overfit.yaml`) to verify that both `train_one_epoch()` and `eval_one_epoch()` execute with online encoding.
5. **OOM & Performance:** Observe memory usage to ensure running the frozen `OccRAE` encoder during the main training loop does not cause out-of-memory issues, and verify token casting does not introduce dtype mismatches.
6. **DDP Validation:** Confirm `train_sampler.set_epoch(e)` still drives deterministic reshuffling under distributed training with the new dataset and custom collation path.
