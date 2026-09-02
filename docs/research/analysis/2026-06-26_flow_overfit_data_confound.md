# DeltaTok flow overfit: the "single scene" was secretly randomized data

Status: **resolved (config fix), re-test in flight** (2026-06-26). The overfit runs
behind [`2026-06-25_flow_mse_collapse.md`](2026-06-25_flow_mse_collapse.md) were **never
fitting one fixed sample** — the overfit config silently randomized the data per
batch. Found while adding a frozen-encode cache. Fix: add `ray_map_prob=-1` to the
overfit dataset string. Clean re-test = JZ job 937913 (`..._fixedT0_wd0_raymapoff`).

## Part 1 — Token-input caching (`cache_frozen_encode`)

**Motivation.** Each training step ran the frozen ~1B OccRAE/DA3 backbone
(`_extract_pair_feats`) + frozen DeltaTok (`_encode_deltas`) on the same scene.
For an overfit the encode output is identical every step, so it should run once.

**What it caches.** Only what's consumed downstream — never the full `feats`:
- `z` — GT delta tokens `(B, T-1, N, C)`, the flow target.
- `feat0` — frame-0 patch feats `(B, N, P, C)` (cross-cond + rollout seed).
- `tokens` — full OccRAE tokens `(B, V, N_tok, C)`, eval-only (decode path).

**Key.** Per data item: `(scene_name, frame_ids, timesteps)` — fully pins the
OccRAE input, so a key match guarantees identical feats. Self-correcting: if the
realized frames ever differ, the key differs and it recomputes (never serves stale
feats). The speedup just requires deterministic per-sample preprocessing.

**Where.** `occrae/deltatok_flow_trainer.py`: `_encode_inputs()` +
`_item_cache_keys()`; wired into `train_one_epoch` (`want_tokens=False`) and
`eval_one_epoch` (`want_tokens=True`). Flag: `training.cache_frozen_encode`
(default `false`; **overfit-scoped** — do not enable on the full multi-dataset
config, memory grows per unique sample). Flag-off path is behaviorally identical.

It prints one line per *new* unique sample:
`[INFO] cache_frozen_encode: stored encode, N unique sample(s)`.

## Part 2 — The bug the cache exposed

On a `max_seqs=1` "single scene" overfit, the cache printed a **growing** count:

```
stored encode, 4 unique sample(s)
stored encode, 5 unique sample(s)
...
stored encode, 8 unique sample(s)   # still climbing toward ~9
```

A true single-sample overfit must print exactly `1`. So the dataset was emitting
distinct samples per batch.

### Root cause: `ray_map_prob` defaulted to `0.0`

The overfit config (`configs/train_deltatok_flow_overfit_jeanzay.yaml`) **omitted
`ray_map_prob=-1`**, which every real dataset in the base config sets. The dataset
default is `0.0` (`BaseSeqDatasetMultiView.__init__`), and in the batch sampler
(`third_party/dust3r/dust3r/datasets/base/batched_sampler.py:117-128`):

```python
if self.ray_map_prob < 0:           # 0.0 is NOT < 0 -> skipped
    ray_map_idx = []
elif len(self.ray_map_idx) > 0:     # default [] -> skipped
    ray_map_idx = self.ray_map_idx
else:
    ray_map_idx = []
    for j in range(memory_num_views[i]):
        if j != 0 and rng.random() < self.ray_map_prob:   # prob 0.0 -> never
            ray_map_idx.append(j)
    if len(ray_map_idx) == 0:
        ray_map_idx.append(rng.integers(1, memory_num_views[i]))  # <-- FALLBACK
```

So with `ray_map_prob=0.0` the fallback fires every batch: **one random gen-view
index in `{1..9}`** (here `memory_num_views=10`). That's ~9 possibilities → the
~8 observed unique keys.

### Why it changes the sample (not just a flag)

That gen-view is pulled out and **reordered to the end** of the returned views
(`occany/datasets/base_seq_dataset.py:530-565`: `recon_view_idx` first, then
`gen_view_idx` appended, with `is_raymap=True` + an added `ray_map` field). The
flow encode (`_extract_pair_feats`) reshapes the flat view axis `V → (T, N)`
assuming **timestep-major, camera-major** contiguous order; moving one view to the
end scrambles that mapping for the affected timestep. So each batch is a different
reordering — the model fit ~9 variants of scene-0625, not one sample.

### Everything else was genuinely pinned

`max_seqs=1` → `ResizedDataset` maps every index to seq 0 → per-item reseed
`seed+idx` is constant (42); `min=max_memory_num_views=10`; single resolution →
`resolution_idx=0`; `fixed_cams=[0,1]` → no camera randomness; `reverse_seq`
defaults False. The **only** per-batch variation was the ray-map pick.

## Fix

Add `ray_map_prob=-1` to both `train_dataset` and `test_dataset` in
`configs/train_deltatok_flow_overfit_jeanzay.yaml`. Then `ray_map_idx=[]` → no gen
views → no reordering → every batch identical → `1 unique sample(s)`, true overfit,
and the cache hits after epoch 0. (Test already used the int-index path with
`ray_map_idx=[]`, so the test edit is for symmetry.)

## Impact on the MSE-collapse investigation

The conclusions in [`2026-06-25_flow_mse_collapse.md`](2026-06-25_flow_mse_collapse.md)
— "can't overfit one scene", "noise-end mean-collapse", "weak per-slot
conditioning SNR" — were drawn on this confounded data and are **suspect**. The
plateau may have been a moving target (~9 reorderings), not a structural collapse.
**Re-test first:** the now-true single-sample overfit (`ray_map_prob=-1`,
`cache_frozen_encode=true`). If `MSEToken → 0`, the SNR/architecture story was moot;
only if it still plateaus do the embed-std / AdaLN-routing knobs become relevant.

Clean re-test: JZ job **937913**, `RUN_NAME=..._fixedT0_wd0_raymapoff`, fresh
(`RESUME=0`) so it does not inherit the confounded checkpoint.
