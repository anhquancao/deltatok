#!/usr/bin/env python3
"""Cache the flow's frozen-encode targets: GT delta tokens per frame pair.

Flow training re-encodes all 10 frames of every window at every step (frozen DA3, then frozen
DeltaTok on each consecutive pair). The delta token z of a pair depends only on its two frames,
so this script computes each pair once and saves it.

Output, per scene and height:
  <out_root>/<W>x<H>/<DatasetClass>/<scene>/  z.npy (n_pairs, K, C) float32, pairs.json
  Row i of z.npy is the pair pairs.json[i].

Steps (see main):
  1. Build the frozen encoder from the flow training config.
  2. Build the same train datasets as training; list every scene with its pairs.
  3. Keep this array task's share of the scenes: scenes[pid::world].
  4. Split the scenes into units of 512 pairs. One DataLoader loads the frames of every unit in order,
     so workers load ahead while the GPU runs DA3 per frame and DeltaTok per pair, at each height.
  5. After a scene's last unit, save z + pairs per height. Each z.npy lands atomically; a rerun
     skips scenes with all of them.
  6. --verify N: compare the cache with the normal training encode on N random windows.

Usage: sbatch slurm/extract_deltatok_flow_zcache_bsc.slurm (smoke: WORLD=200 VERIFY=32 on acc_debug)
"""

import argparse
import json
import os
import os.path as osp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from occany.utils.runtime_paths import prepend_vendored_import_paths

prepend_vendored_import_paths(
    Path(__file__).resolve().parent,
    extra=["third_party/pyTorchChamferDistance", "third_party/GLD/src", "third_party/deltatok"],
)

# Same as train_deltatok_flow.py
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.transforms.functional import to_tensor  # noqa: E402
from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: E402
from occrae.deltatok_flow_trainer import DeltaTokFlowMatchingTrainer  # noqa: E402
import occany.datasets as occany_datasets  # noqa: E402
from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView  # noqa: E402

# dust3r's dataset import sets file_system; its /dev/shm unlink aborted 14 tasks of 47037211.
torch.multiprocessing.set_sharing_strategy("file_descriptor")


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeltaTok flow: cache GT delta tokens per frame pair.")
    p.add_argument("--config-dir", default="configs/deltatok_flow")
    p.add_argument("--config-name", default="train_deltatok_flow_alldata_ctx2fwd8_bsc")
    p.add_argument("--cfg", nargs="*", default=[], help="Hydra overrides; must carry the frozen tokenizer flags.")
    p.add_argument("--out_root", required=True)
    p.add_argument("--heights", default="280,266,210,168", help="Heights of the 518-wide resolutions to cache.")
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--pid", type=int, default=0)
    p.add_argument("--chunk", type=int, default=512, help="Pairs encoded together (one unit).")
    p.add_argument("--frame_bsize", type=int, default=32)
    p.add_argument("--pair_bsize", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--verify", type=int, default=0, help="Train windows checked against the online encode.")
    return p


def _base_datasets(ds):
    """The train string `32000 @ A(...) + 20000 @ B(...)` builds wrapper datasets.
    Unwrap `N @` (.dataset) and `+` (.datasets) down to the 5 real sources (BaseSeqDatasetMultiView)."""
    if isinstance(ds, BaseSeqDatasetMultiView):
        return [ds]
    if hasattr(ds, "datasets"):
        return [b for d in ds.datasets for b in _base_datasets(d)]
    return _base_datasets(ds.dataset)


def _unique(xs):
    """Items of xs, each once, in first-seen order."""
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _scene_pairs(ds):
    """List what to cache for one train source.

    Returns {scene_idx: pairs}: every (frame_t, frame_t+1) of some training window, each once,
    in first-seen order. Windows overlap, so most pairs appear in many windows.
    """
    # The cache holds one camera; all 5 train sources use fixed_cams=[0] and loose npz files.
    assert len(ds.cams) == 1 and ds.max_views_per_timestep is None, "z-cache is single-camera"
    assert not ds.use_tar, "z-cache reads loose npz files"
    # vpt: cameras stored per timestep in a record; T: frames per window (10); cam: the camera kept.
    vpt, T, cam = ds.num_views_per_timestep, ds.num_timesteps, ds.cams[0]
    # Per scene: a list in first-seen order, plus a set to check "already added?" fast.
    pairs, seen = defaultdict(list), defaultdict(set)
    # A pkl record: (scene index, flat list of frame indices, unused); timestep-major, camera-minor.
    for scene_idx, seq, _ in ds.seqs:
        # Every window start _get_views can draw. Our records hold exactly T timesteps, so start = 0.
        for start in range(len(seq) // vpt - T + 1):
            # Frame ids of the window's T timesteps, camera `cam`.
            ids = [str(ds.frames[seq[(start + t) * vpt + cam]]) for t in range(T)]
            for pr in zip(ids[:-1], ids[1:]):       # the window's T-1 consecutive pairs
                if pr not in seen[scene_idx]:       # first time this pair appears
                    seen[scene_idx].add(pr)
                    pairs[scene_idx].append(pr)
    return pairs


def _scene_dirs(out_root, resolutions, cls, scene):
    """One cache dir per height: <out_root>/<W>x<H>/<cls>/<scene>."""
    return [osp.join(out_root, f"{w}x{h}", cls, scene) for w, h in resolutions]


class _Frames(torch.utils.data.Dataset):
    """One frame per item, for a DataLoader over the whole shard: the npz is read once, then
    resized and normalized at every height exactly as the training loader does."""

    def __init__(self, items, resolutions):
        self.items, self.resolutions = items, resolutions              # items: [(ds, scene, frame id)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ds, scene, fid = self.items[i]
        # Raw npz: full-size image, depth, intrinsics (the loader's own reader).
        image, depthmap, intrinsics, _ = ds._load_raw_frame(scene, fid, osp.join(ds.ROOT, scene), None)
        imgs = []
        for res in self.resolutions:
            # The loader's crop + resize to (518, H). rng=None: a square image (random transpose) fails loudly.
            img, _, _, _ = ds._resize_image_and_sparse_depthmap(image, depthmap, intrinsics.copy(), res, None, info=(scene, fid))
            # DA3 input normalization, as in BaseSeqDatasetMultiView.__getitem__.
            imgs.append(InputProcessor.NORMALIZE(to_tensor(img)))     # (3, H, W)
        return imgs


@torch.no_grad()
def encode_chunk(trainer, imgs, fids, chunk, args):
    """Encode the M pairs of one unit.
    imgs: per height (F, 3, H, W), row f is frame fids[f]. -> per height (M, K, C) float32, row m is chunk[m]."""
    row = {f: i for i, f in enumerate(fids)}
    a = torch.tensor([row[f] for f, _ in chunk])                       # (M,) row of each pair's frame t
    b = torch.tensor([row[f] for _, f in chunk])                       # (M,) row of its frame t+1
    zs = []
    for x_all in imgs:                                                 # one height at a time
        # DA3 blocks 0-12 on each frame alone. Those blocks are per-view, so this equals the 10-frame call.
        feats = []
        for s in range(0, len(fids), args.frame_bsize):
            x = x_all[s:s + args.frame_bsize].to(trainer.device)[:, None]  # (n, 1, 3, H, W) one view each
            _, f, _, _, H, W = trainer._extract_pair_feats(x, num_cameras=1, return_pairs=False)  # (n, 1, 1, P, C)
            feats.append(f[:, 0])                                      # (n, 1, P, C)
        feats = torch.cat(feats)                                       # (F, 1, P, C)
        # Frozen DeltaTok encoder on each pair (frame t, frame t+1) -> its delta token z.
        z = []
        for s in range(0, len(chunk), args.pair_bsize):
            i = slice(s, s + args.pair_bsize)
            x = torch.stack([feats[a[i]], feats[b[i]]], dim=1)         # (m, 2, 1, P, C) a 2-frame clip per pair
            z.append(trainer._encode_deltas(x, H, W)[:, 0, 0])         # (m, K, C) float32
        zs.append(torch.cat(z).cpu().numpy())                          # (M, K, C)
    return zs


@torch.no_grad()
def verify(trainer, items, resolutions, args):
    """Random train windows: encode them the normal training way (all T frames at once)
    and check the cache returns the same z."""
    picks = np.random.default_rng(0).choice(len(items), size=min(args.verify, len(items)), replace=False)
    worst = 0.0
    for p in picks:
        ds, seq_idx = items[p]
        ds._resolutions = list(resolutions)                            # __getitem__ indexes this list
        cls = type(ds).__name__
        for r, res in enumerate(resolutions):
            # The window exactly as the training loader returns it.
            views = ds[(seq_idx, r)]
            ids, scene = [v["frame_id"] for v in views], views[0]["scene_name"]
            # Online encode, as in training: all T frames through DA3, then DeltaTok.
            imgs = torch.stack([v["img"] for v in views])[None].to(trainer.device)  # (1, T, 3, H, W)
            _, f, _, _, H, W = trainer._extract_pair_feats(imgs, num_cameras=1, return_pairs=False)  # (1, T, 1, P, C)
            z_on = trainer._encode_deltas(f, H, W)[0, :, 0].float().cpu()            # (T-1, K, C)
            # Cache: look up the window's T-1 pairs in pairs.json, read those rows of z.npy.
            d = _scene_dirs(args.out_root, [res], cls, scene)[0]
            row = {tuple(pr): i for i, pr in enumerate(json.load(open(osp.join(d, "pairs.json"))))}
            zc = np.load(osp.join(d, "z.npy"), mmap_mode="r")[[row[pr] for pr in zip(ids[:-1], ids[1:])]]  # (T-1, K, C)
            rel = float((z_on - torch.from_numpy(zc)).norm() / z_on.norm())
            worst = max(worst, rel)
            print(f"[VERIFY] {cls}/{scene} {ids[0]} {res[0]}x{res[1]}: rel L2 {rel:.2e}", flush=True)
    verdict = "FAIL"
    if worst <= 1e-2:
        verdict = "PASS"
    print(f"[VERIFY] {verdict}: worst rel L2 {worst:.2e} (<= 1e-2)", flush=True)


def main() -> None:
    args = get_args_parser().parse_args()
    resolutions = [(518, int(h)) for h in args.heights.split(",")]

    # Step 1: the frozen encoder. Built through the flow trainer so DA3 + DeltaTok load exactly
    # as in training; the flow ViT is built too but never used.
    with initialize_config_dir(version_base=None, config_dir=str(Path(args.config_dir).resolve())):
        cfg = compose(config_name=args.config_name, overrides=args.cfg)
    with open_dict(cfg):
        if cfg.model.get("img_decoder", None) is not None:
            cfg.model.img_decoder.ckpt_path = None                     # nothing is decoded
        cfg.training.writer_log = ""                                   # no TensorBoard
        cfg.training.vit_folder = osp.join(os.environ.get("TMPDIR", "/tmp"), "extract_deltatok_flow_zcache") + "/"
    trainer = DeltaTokFlowMatchingTrainer(
        args=argparse.Namespace(resume=False, ckpt=None, test_only=True, eval_only=True, debug=False, is_multi_gpus=False),
        cfg=cfg, device=torch.device("cuda"), rank=0, world_size=1, distributed=False,
    )
    print(f"[INFO] tokenizer {cfg.model.deltatok_ckpt} encode_layer {cfg.model.encode_layer}\n"
          f"[INFO] {resolutions} -> {args.out_root}", flush=True)

    # Step 2: the train datasets, built from the same config string training uses.
    # cfg.dataset.train_dataset is Python source text:
    #   "32000 @ WaymoSeqMultiView(ROOT=..., num_timesteps=10, ...) + 20000 @ VKittiSeqMultiView(...) + ..."
    # Training's get_data_loader runs eval() on it inside occany/datasets/__init__.py. Passing that
    # module's names (vars(occany_datasets)) lets WaymoSeqMultiView etc. resolve the same way here.
    train_set = eval(str(cfg.dataset.train_dataset), vars(occany_datasets))
    # `N @ ds` wraps ds to draw N windows per epoch, `a + b` concatenates; unwrap to the 5 sources.
    bases = _base_datasets(train_set)
    # One entry per scene to cache: (class, scene, ds, pairs).
    #   class:  dataset class name, e.g. "WaymoSeqMultiView"; cache subfolder, = the loader's view['dataset'].
    #   scene:  scene folder name under the dataset ROOT; cache subfolder.
    #   ds:     the source dataset object, used to load and resize this scene's frames.
    #   pairs:  from _scene_pairs.
    # Example, a real Waymo scene (153 windows -> 193 pairs):
    #   ("WaymoSeqMultiView",
    #    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord",
    #    <the WaymoSeqMultiView object>,
    #    [("00000_1", "00005_1"), ("00005_1", "00010_1"), ..., ("00040_1", "00045_1"), ("00001_1", "00006_1"), ...])
    #   Frame id "00005_1" = frame 5, camera suffix _1 (the fixed_cams=[0] camera). Window k holds frames
    #   k, k+5, ..., k+45. Windows 0-4 each add 9 new pairs; from window 5 on, only pair (k+40, k+45)
    #   is new: 5*9 + 148 = 193 pairs.
    scenes = []
    for ds in bases:
        for scene_idx, pairs in _scene_pairs(ds).items():
            scenes.append((type(ds).__name__, ds.scenes[scene_idx], ds, pairs))
    # Sorted by (class, scene), so every array task builds the same list.
    scenes.sort(key=lambda x: x[:2])
    assert len({x[:2] for x in scenes}) == len(scenes), "one (class, scene) in two train sources"

    # Step 3: this task takes every world-th scene, starting at pid; finished scenes are skipped
    # (each z.npy lands atomically, so all present = scene done).
    mine = scenes[args.pid::args.world]
    todo = []
    for cls, scene, ds, pairs in mine:
        if not all(osp.exists(osp.join(d, "z.npy")) for d in _scene_dirs(args.out_root, resolutions, cls, scene)):
            todo.append((cls, scene, ds, pairs))
    n_pairs = sum(len(pairs) for _, _, _, pairs in todo)
    print(f"[INFO] shard {args.pid}/{args.world}: {len(mine)} of {len(scenes)} scenes, "
          f"{len(mine) - len(todo)} already done, {n_pairs} pairs to encode", flush=True)

    # Step 4: units of --chunk pairs, each with the frames it needs, so a 14k-frame ONCE scene fits in memory.
    units = []                                                         # (scene index in todo, chunk of pairs, chunk's frame ids)
    frame_items = []                                                   # every frame to load, unit after unit: (ds, scene, frame id)
    for s, (cls, scene, ds, pairs) in enumerate(todo):
        for c in range(0, len(pairs), args.chunk):
            chunk = pairs[c:c + args.chunk]
            fids = []
            for a, b in chunk:                                         # frame t, frame t+1 of each pair
                fids += [a, b]
            fids = _unique(fids)                                       # each frame once, first-seen order
            units.append((s, chunk, fids))
            for f in fids:
                frame_items.append((ds, scene, f))
    # One DataLoader over frame_items, in order. About one unit of frames is in flight,
    # so loading the next unit overlaps the GPU work on this one.
    frames = _Frames(frame_items, resolutions)
    it = iter(torch.utils.data.DataLoader(frames, batch_size=None, num_workers=args.num_workers,
                                          prefetch_factor=args.chunk // args.num_workers + 2))

    t_all, done_pairs, zs = time.time(), 0, []                         # zs: per unit, per height (M, K, C)
    for k, (s, chunk, fids) in enumerate(units):
        loaded = [next(it) for _ in fids]                              # per frame: per height (3, H, W)
        imgs = [torch.stack([fr[r] for fr in loaded]) for r in range(len(resolutions))]  # per height (F, 3, H, W)
        zs.append(encode_chunk(trainer, imgs, fids, chunk, args))
        done_pairs += len(chunk)
        if k + 1 < len(units) and units[k + 1][0] == s:                # more units of this scene to come
            continue

        # Step 5: the scene's last unit is done; save. z rows follow `pairs`. Write to a tmp name,
        # then rename: a job killed mid-write leaves no z.npy, so the scene reruns.
        cls, scene, _, pairs = todo[s]
        for r, d in enumerate(_scene_dirs(args.out_root, resolutions, cls, scene)):
            os.makedirs(d, exist_ok=True)
            json.dump([list(pr) for pr in pairs], open(osp.join(d, "pairs.json"), "w"))
            np.save(osp.join(d, "z.tmp.npy"), np.concatenate([u[r] for u in zs]))
            os.replace(osp.join(d, "z.tmp.npy"), osp.join(d, "z.npy"))
        zs = []
        dt = time.time() - t_all
        print(f"[{s + 1}/{len(todo)}] {cls}/{scene}: {len(pairs)} pairs; total {done_pairs}/{n_pairs} pairs "
              f"in {dt:.0f}s ({done_pairs / dt:.1f} pairs/s)", flush=True)

    # Step 6: check random training windows from this task's scenes against the cache.
    if args.verify > 0:
        keys = {(cls, scene) for cls, scene, _, _ in mine}
        items = []                                                     # (source dataset, window index)
        for ds in bases:
            for i, (scene_idx, _, _) in enumerate(ds.seqs):
                if (type(ds).__name__, ds.scenes[scene_idx]) in keys:
                    items.append((ds, i))
        verify(trainer, items, resolutions, args)


if __name__ == "__main__":
    main()
