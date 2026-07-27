# fsn1 purge incident — dataset bleed and repair (2026-07-26)

Training job `Jeanzay:276534` died on a missing data file. The investigation found the
Jean Zay working tier (`fsn1`, i.e. `$SCRATCH`) was actively deleting dataset files at up
to ~2,000/min. Renaming the tree stopped it. This is the record of what was observed,
what was done, and what is still unknown.

## Symptom

`276534` (`jz_train_deltatok_multitoken_mlp_sigreg.slurm`) ran fine into epoch 2, then:

```
[rank3]: FileNotFoundError: [Errno 2] No such file or directory:
         '.../occany_data/once_processed/000424/010135_cam09.npz'
[rank3]: RuntimeError: Failed to load dataset sample: ...
srun: error: jzxh323: task 3: Exited with exit code 1
```

Raised by `occany/datasets/base_seq_dataset.py:399`. Rank 3 exited, SLURM SIGTERM'd ranks
0-2 — the `pycolmap`/`cuStreamSynchronize` stack traces at the end of the `.err` are that
teardown, not the cause. A single missing file out of millions kills the whole job.

Initial scope against the pickle the run uses
(`seq_surround_temporal_sub5_stride9_fs1_once_5cam_all.pkl`): 52,874 unique `.npz` files
missing, affecting 17,083 of 845,311 sequences (2.0%). All present on Karolina — so the
data was lost on Jean Zay, not missing at source.

## Root cause

`fsn1` purges files unread for 30 days (documented; confirmed by the user). The dataset
lives there because it is the only tier with the inodes for it — 15.2M files total, versus
WORK's 500k inode quota (90% used) and STORE's 100k.

**The deletion did not respect timestamps.** Two independent observations:

- Files with `atime` set to *today* were deleted anyway. Directly measured: kitti held
  41,610 files at 21:52 and 41,606 at 21:57, with `atime_older_1d=0` — every survivor
  touched that day, and four still vanished.
- Files with `mtime` of 2026-02-03 had survived for ~6 months. So mtime is not the
  criterion either.

Best remaining hypothesis: the purge deletes from a candidate list built by an earlier
scan and does not re-check timestamps at deletion time. Unconfirmed.

## Deletion rates (measured 22:25-22:27, 77s apart)

| folder | delta | rate |
|---|---|---|
| `once_processed` | -2,645 | ~2,060/min |
| `occ3d_nuscenes_processed` | -226 | ~176/min |
| `kitti_processed` | -6 | ~5/min |
| `ddad`, `pandaset`, `vkitti`, `waymo` | 0 | stable |

## What stopped it: renaming the tree

At **22:30:35**, `mv $SCRATCH/occany_data $SCRATCH/occany_dataset`. Rationale: if the
deleter resolves a precomputed list of paths, renaming the parent invalidates every entry
atomically. A rename is a metadata operation — instant, no data moved.

It worked. Formal 5-minute delta (22:36:33 -> 22:41:44), where the prior rate predicted
~10,000 losses from `once` alone:

```
ddad_processed             197412 ->    197412   +0
kitti_processed             40653 ->     40654   +1
occ3d_nuscenes_processed    40729 ->     40849   +120
once_processed           13229313 -> 13229314   +1
pandaset_processed          96966 ->     96972   +6
vkitti_processed            42528 ->     42528   +0
waymo_processed           1577712 ->   1577726   +14
```

Every delta zero or positive; the positives are the restore sync landing files. No
deletions since.

**This is a workaround, not a fix.** When the next scan indexes `occany_dataset`, it may
resume.

## Timestamp refresh

`slurm/jz_touch_occany_data.slurm` — array `0-9` on `archive`, scene dirs sharded
round-robin, `xargs -P 8 touch` (both atime and mtime). Job `284624`: all 10 shards
COMPLETED, 13-16 min each.

An earlier single-process attempt over `once_processed` (job `281191`,
`find | xargs touch -a` across 13.2M files) was **OOM-killed** after 1h23m — it printed its
"done" line but a child died partway, so most of `once` was never touched. That is why
`once` bled fastest. Shard the walk; do not stream 13M entries through one process.

## Repair

Karolina -> Jean Zay, seven datasets in parallel, each its own SSH connection through the
`localhost:2222` tunnel:

```
rsync -a --no-times --size-only -e 'ssh -p 2222' \
  /scratch/project/eu-25-92/data/$d/ uyl37fq@localhost:$SCRATCH/occany_dataset/$d/
```

- `--no-times` — transferred files get mtime = now instead of inheriting Karolina's old
  mtimes, so they are not born purge-eligible.
- `--size-only` — without `-t` rsync cannot compare mtimes; also makes it a pure
  gap-filler. Blind spot: cannot detect a right-sized but corrupt file.

Files rsync *skips* get no timestamp refresh — that is what the touch array covers.

| dataset | JZ final | Karolina src | files restored |
|---|---|---|---|
| `kitti_processed` | 50,473 | 50,473 | 9,820 |
| `occ3d_nuscenes_processed` | 48,315 | 48,314 | 7,586 |
| `pandaset_processed` | 96,972 | 96,972 | 6 |
| `ddad_processed` | 197,412 | 197,412 | 0 (intact) |
| `vkitti_processed` | 42,528 | 42,528 | 0 (intact) |
| `waymo_processed` | 1,898,163 | 1,898,162 | ~320,000 |
| `once_processed` | in progress | 13,855,391 | ~626,000 total gap |

As of 00:03 on 2026-07-27, `once_processed` had 406,398 files left, moving ~50/s
(~48 MB/s, tunnel-limited) — ETA ~2.25h.

Parallel beat serial decisively here: these are metadata-bound walks, so a dataset with
197k files and 7 missing spends nearly all its time stat'ing with the tunnel idle. Three
datasets finished within a minute of each other once run concurrently.

## Backup tier (`fsstor`, not purged)

`$TRG_STORE/datasets_preprocess_backup` holds per-scene uncompressed tars:

| dataset | archives | usable for restore? |
|---|---|---|
| `once_processed` | 571 | yes |
| `argoverse2_processed` | 520 | yes, but not in `backup_folders` |
| `ddad_processed` | 200 | yes |
| `pandaset_processed` | 103 | yes |
| `vkitti_processed` | 50 | yes |
| `nuscenes_processed` | 4 | yes |
| `kitti_processed` | 1 | no — `sequences.tar.gz`, 671 MB of a 58.8 GB tree |
| `waymo_processed` | 0 | no — tfrecords only |
| `occ3d_nuscenes_processed` | 0 | was never in `backup_folders` (now added) |

kitti and waymo have no usable archive, so Karolina is their only restore path.

### Re-backup 2026-07-27 (closes all three gaps)

Job `304546` (array 0-9, `archive`, `slurm/jz_backup.slurm`) tarred all seven fsn1 datasets
into a **new** dir, `$TRG_STORE/datasets_preprocess_backup_2026-07` — 9.9 TB, all 10 shards
COMPLETED 0:0 in under 2h20m. `backup.py` gained `--dst_name` (default unchanged) so a
re-backup never overwrites the old set.

The 2025 dir is deliberately kept: `argoverse2_processed` (521 archives, 621 GB) exists
**only** there — not on fsn1, not on Karolina, and not in `backup_folders`, so no re-backup
regenerates it. `kitti_pair_processed` and `nuscenes_processed` are likewise absent from
fsn1 (both still on Karolina).

Verified: archive count == scene-dir count for all seven (waymo 798 vs 799 dirs is the
`tmp` dir `backup.py` skips by design); logs clean; and `tar -tf` member counts match the
source exactly on spot checks including the largest archives of the three formerly-broken
sets — kitti `02` (9,323 files, 9.9 GB), occ3d `scene-0017` (1,600, 626 MB), waymo
`segment-268278198029493143_1…` (1,990, 2.0 GB).

Two empty scene dirs surfaced, neither a backup fault: `kitti_processed/resized_512` (empty
on JZ, absent on Karolina — a stray) and `occ3d_nuscenes_processed/scene-1109` (empty on
both). Empty dirs are invisible to a file-count diff, so they never showed in the parity
check.

## Changes committed

- deltatok `bc30735` — all JZ paths -> `occany_dataset`; added `dataset_setup/backup.py`,
  `slurm/jz_backup.slurm`, `slurm/jz_extract_backup.slurm`,
  `slurm/jz_touch_occany_data.slurm`.
- OccAny `89e8458` — 31 files repointed; merge of `origin/revert-cvpr` resolved by
  accepting upstream's deletion of four `jz_*.slurm` scripts we had only path-edited.

BSC's `/gpfs/scratch/ehpc793/occany_data` was deliberately left unchanged.

`backup.py` extraction stamps atime+mtime to now on every extracted member *and* walks the
whole scene dir, so companions written after the backup (`*.infinidepth.png`) are also
refreshed. Extraction is therefore self-sufficient against the purge.

## Verification (2026-07-27)

The restore finished at 02:42; all seven rsyncs exited clean. Verified three independent
ways.

**1. Counts.** Every dataset on Jean Zay is now >= Karolina, and >= its own 07-26 22:41
value — so nothing bled after the rename:

| dataset | JZ 07-26 22:41 | JZ 07-27 | Karolina | delta |
|---|---|---|---|---|
| `once_processed` | 13,229,314 | 13,855,393 | 13,855,391 | +2 |
| `waymo_processed` | 1,577,726 | 1,898,163 | 1,898,162 | +1 |
| `ddad_processed` | 197,412 | 197,412 | 197,412 | 0 |
| `pandaset_processed` | 96,972 | 96,972 | 96,972 | 0 |
| `kitti_processed` | 40,654 | 50,473 | 50,473 | 0 |
| `occ3d_nuscenes_processed` | 40,849 | 48,315 | 48,314 | +1 |
| `vkitti_processed` | 42,528 | 42,528 | 42,528 | 0 |

`once_processed` recounted at 10:50 and 13:09 — identical. ~15h since the rename with zero
deletions.

**2. Per-scene folders.** Counts alone can't detect a short scene offset by a surplus
elsewhere. Walked both trees at scene granularity: **2,153 folders on each side, all
matched by name, 0 folders short on Jean Zay.** The four surplus files are:

| folder | JZ-only file | what |
|---|---|---|
| `once_processed/000076` | `.010025_cam09.infinidepth.png.F2aQV8` | rsync temp, interrupted transfer |
| `waymo_processed/segment-1833171384….tfrecord` | `.00118_1.npz.GFxbr2` | same |
| `once_processed/000047` | `003710_cam05.infinidepth.p` | truncated name, partial write |
| `occ3d_nuscenes_processed/scene-0018` | `000143_0.npz` | genuine extra, absent on Karolina |

The three partials are leftovers, not losses — each folder is Karolina's count *+1*, so the
real file sits alongside the stray. All three were 0 bytes with mtimes inside the restore
window (22:50-23:09), and all three counterparts had real content (337 KB / 137 KB /
952 KB) written seconds apart. Deleted 2026-07-27 14:23; counterparts verified intact. JZ
now differs from Karolina only by `occ3d_nuscenes_processed/scene-0018/000143_0.npz`.

**3. File-level parity.** Dry-run rsync (`--dry-run --size-only`, same flags as the
restore) — a true set-difference by name+size. **0 files would transfer for all seven
datasets.** Combined with JZ counts >= Karolina, the sets are identical.

Caveat: `--size-only` compares name and size, never content. A right-sized but corrupt file
is invisible to all three checks — same blind spot the restore had.

### Operational lesson: shard the dry-run too

A single rsync over `once_processed` (13.8M entries) **died twice**. First attempt:
`connection unexpectedly closed (660 bytes received)`, code 12. Second: the JZ receiver
process vanished while the Karolina sender sat in `core_sys_select`, having moved zero
bytes for 18 minutes — a hang that looks exactly like slow progress unless you sample
`/proc/<pid>/io` twice and compare.

Sharding per scene dir (571 rsyncs, 4 concurrent) completed with 0 missing and 0 errors.
Same lesson as the OOM-killed touch job: do not stream 13M entries through one process on a
login node. A sharded run also gives a real progress signal — a completed-shard counter
distinguishes working from stalled, which a monolithic run cannot.

## Open items

1. **Mechanism unexplained.** Deletion ignored fresh atime; neither atime nor mtime
   explains the selection. Worth an IDRIS support ticket with the evidence above.
2. **Rename may only buy time** — the next scan could rebuild its list against the new path.
   Held for ~15h as of 2026-07-27 13:09, but that is not yet a full purge cycle.
3. **Training is fragile by design.** One missing file out of 13M aborts a 20h job.
   Making `_get_views` resample instead of raising would decouple training from this
   entirely.
4. `../monitor_jobs/data/projects.json` may still carry the old path for rsync deploys —
   not touched.
