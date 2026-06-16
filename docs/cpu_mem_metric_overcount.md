# BUG: `cpu mem` log overcounts with many dataloader workers

**Status:** open — to fix 2026-06-18
**Severity:** low (cosmetic / misleading metric; not a real OOM)
**Area:** `occrae/metric_logger.py`

## Symptom

Training log shows the reported CPU memory *exceeding* the reported budget, e.g.
observed on the Jean Zay overfit run (job 514410, `num_workers: 16`):

```
... cpu mem: 232529 (120000)
```

i.e. "current" (232 GB) > "budget" (117 GB), yet the job keeps running and is
never OOM-killed. The number also jumped hugely when `num_workers` went 0 -> 16
(`16209` -> `232529`).

## Root cause

The two numbers measure different things:

- **First number** `rank_cpu_mem_mb()` (`occrae/metric_logger.py:176`) sums the
  **RSS of the main process plus every dataloader-worker child**:
  ```python
  rss = proc.memory_info().rss
  for child in proc.children(recursive=True):   # the N=num_workers workers
      rss += child.memory_info().rss
  ```
  Workers are **forked**, so they share most pages **copy-on-write** with the
  parent (DA3/DeltaTok model weights, the loaded dataset, torch/Python libs).
  `RSS` counts each shared page **once per process**, so summing across
  `1 + num_workers` processes multiply-counts all shared memory. The existing
  docstring already flags this ("...counted once per process, so this slightly
  overestimates") — with 16 workers and a large shared model, the overcount is
  large, not slight.

- **Second number** `rank_cpu_budget_mb()` (`occrae/metric_logger.py:200`) is the
  **cgroup / OOM budget** (SLURM `--mem`, ~117 GB = 24/96 cores * node RAM) divided
  by ranks-on-node.

The OOM killer counts **unique physical pages** (shared COW pages once total),
while the metric counts them **once per process**. So the metric is inflated and
can exceed the budget without any real breach — true unique RSS stays under the
limit, hence no kill.

## Fix options

1. **Use USS instead of summed RSS (recommended).** `psutil.memory_full_info().uss`
   is the memory unique to each process; summing USS across the parent + workers
   counts shared COW pages correctly (once total). Replace the RSS sum in
   `rank_cpu_mem_mb()` with USS:
   ```python
   info = proc.memory_full_info()          # has .uss
   total = info.uss
   for child in proc.children(recursive=True):
       try:
           total += child.memory_full_info().uss
       except (psutil.NoSuchProcess, psutil.AccessDenied):
           pass
   return total / (1024.0 * 1024.0)
   ```
   Caveats: `memory_full_info()` is slower than `memory_info()` (reads
   `/proc/<pid>/smaps`) and may need permissions; it is only called once per
   `log_iter`, so the cost is negligible. Keep the `ImportError`/AccessDenied
   fallbacks. Consider PSS as an alternative (proportional set size) if USS is
   too conservative.

2. **Report the cgroup's own current usage** (`memory.current` / 
   `memory.usage_in_bytes`) from the same cgroup files `total_cpu_mem_mb()`
   already walks (`occrae/metric_logger.py:150`). This matches exactly what the
   OOM killer sees, so "current (budget)" become directly comparable. Cleanest
   semantically, but ties the metric to cgroup v1/v2 file layout.

## Recommendation

Go with option 1 (USS sum) for a minimal, portable change to `rank_cpu_mem_mb()`,
and update the docstring to note USS is summed (no longer overcounts COW). Verify
on a `num_workers: 16` run that the reported value drops below the budget and
tracks `num_workers` sanely.

## References

- `occrae/metric_logger.py:176` — `rank_cpu_mem_mb()` (the overcounting sum)
- `occrae/metric_logger.py:200` — `rank_cpu_budget_mb()` (the budget)
- `occrae/metric_logger.py:150` — `total_cpu_mem_mb()` (cgroup limit reader)
- `occrae/deltatok_flow_trainer.py:517` — the log line that prints both
- Context: surfaced while overfitting the DeltaTok flow run on Jean Zay
  (`train_deltatok_flow_overfit_jeanzay`, `num_workers: 16`, job 514410).
