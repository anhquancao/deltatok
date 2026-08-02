"""Test the view sampling in BaseSeqDatasetMultiView._get_views.

Creates a temp dir with dummy npz files (labeled with timestep/camera), builds a
mock instance of the real class, and calls the real _get_views. Asserts the
contract: one ORDERED, CONSECUTIVE run of `num_timesteps` at a random offset,
named cameras (never sampled), and dense 0..k-1 timestep labels.

Needs no GPU and no real data:
  source env_jz_h100.sh && python test_view_sampling.py
"""
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()

from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView

IMG_H, IMG_W = 32, 32
TRIALS = 60


def make_dummy_npz_dir(tmpdir, scene_name, avail, max_vpt):
    scene_dir = os.path.join(tmpdir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    K = np.array([[500, 0, IMG_W / 2], [0, 500, IMG_H / 2], [0, 0, 1]], dtype=np.float32)
    for i in range(avail * max_vpt):
        np.savez(
            os.path.join(scene_dir, f"frame_{i:04d}.npz"),
            image=np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
            depthmap=np.ones((IMG_H, IMG_W), dtype=np.float32),
            intrinsics=K,
            cam2world=np.eye(4, dtype=np.float32),
        )


def make_mock_instance(tmpdir, max_vpt, avail, num_timesteps, fixed_cams=None):
    """Real class, minimal state: _get_views only reads these attributes."""
    scene_name = "scene_0"
    make_dummy_npz_dir(tmpdir, scene_name, avail, max_vpt)

    obj = object.__new__(BaseSeqDatasetMultiView)
    obj.ROOT = tmpdir
    obj.num_views_per_timestep = max_vpt
    obj.num_timesteps = num_timesteps
    obj.cams = list(fixed_cams) if fixed_cams is not None else list(range(max_vpt))
    obj.scenes = [scene_name]
    obj.frames = [f"frame_{i:04d}" for i in range(avail * max_vpt)]
    # ts is no longer read by _get_views; keep the pkl's raw stride offsets to prove it.
    obj.seqs = [(0, list(range(avail * max_vpt)),
                 [t * 5 for t in range(avail) for _ in range(max_vpt)])]

    def mock_resize(image, depthmap, intrinsics, resolution, rng=None, info=None,
                    pseudo_depthmap=None):
        return Image.fromarray(image), depthmap, intrinsics.copy(), pseudo_depthmap

    obj._resize_image_and_sparse_depthmap = mock_resize
    return obj


def decode(views, max_vpt):
    """-> [(source_timestep, cam), ...] recovered from the loaded frame ids."""
    out = []
    for view in views:
        idx = int(view["frame_id"].split("_")[1])
        out.append((idx // max_vpt, idx % max_vpt))
    return out


def run_case(name, max_vpt, avail, num_timesteps, fixed_cams=None):
    tmpdir = tempfile.mkdtemp(prefix="viewsampling_")
    failures = []
    starts = set()
    try:
        obj = make_mock_instance(tmpdir, max_vpt, avail, num_timesteps, fixed_cams)
        cams = obj.cams
        vpt = len(cams)

        for trial in range(TRIALS):
            rng = np.random.default_rng(seed=trial)
            views = obj._get_views(0, (IMG_W, IMG_H), rng)
            dec = decode(views, max_vpt)

            if len(views) != num_timesteps * vpt:
                failures.append(f"trial {trial}: {len(views)} views, want {num_timesteps * vpt}")
                continue

            # dense, ascending labels: [0]*vpt, [1]*vpt, ...
            want_labels = [t for t in range(num_timesteps) for _ in range(vpt)]
            got_labels = [v["timestep"] for v in views]
            if got_labels != want_labels:
                failures.append(f"trial {trial}: labels {got_labels} != {want_labels}")

            # source timesteps form a consecutive run
            src = [t for t, _ in dec]
            uniq = sorted(set(src))
            if uniq != list(range(uniq[0], uniq[0] + num_timesteps)):
                failures.append(f"trial {trial}: source timesteps {uniq} not consecutive")
            starts.add(uniq[0])
            if uniq[-1] >= avail:
                failures.append(f"trial {trial}: window runs past the record ({uniq[-1]} >= {avail})")

            # cameras are exactly the named set, same order, at every timestep
            for t in uniq:
                got = [c for tt, c in dec if tt == t]
                if got != cams:
                    failures.append(f"trial {trial}: t{t} cams {got} != {cams}")
                    break

        want_starts = set(range(avail - num_timesteps + 1))
        if starts != want_starts:
            failures.append(f"origins seen {sorted(starts)} != all of {sorted(want_starts)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    status = "PASS" if not failures else "FAIL"
    print(f"[{status}] {name}")
    for f in failures[:5]:
        print(f"         {f}")
    return not failures


def run_too_short():
    """num_timesteps > available timesteps must fail loudly, not silently truncate."""
    tmpdir = tempfile.mkdtemp(prefix="viewsampling_")
    try:
        obj = make_mock_instance(tmpdir, max_vpt=5, avail=3, num_timesteps=5, fixed_cams=[0])
        try:
            obj._get_views(0, (IMG_W, IMG_H), np.random.default_rng(seed=0))
        except AssertionError:
            print("[PASS] short record raises AssertionError")
            return True
        print("[FAIL] short record did not raise")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ok = []
    # ONCE/Waymo flow+tokenizer arms: monocular front cam over a 10-timestep record.
    ok.append(run_case("monocular cam0, k=5 of 10", max_vpt=5, avail=10, num_timesteps=5,
                       fixed_cams=[0]))
    # nuScenes eval: two named cameras of a 6-cam rig.
    ok.append(run_case("two cams [0,1], k=5 of 10", max_vpt=6, avail=10, num_timesteps=5,
                       fixed_cams=[0, 1]))
    # A non-zero-based camera set must be honoured verbatim, in order.
    ok.append(run_case("cams [2,0] order preserved", max_vpt=5, avail=8, num_timesteps=4,
                       fixed_cams=[2, 0]))
    # fixed_cams=None -> the whole rig.
    ok.append(run_case("full rig (cams None)", max_vpt=3, avail=6, num_timesteps=3))
    # Window exactly fills the record: only one possible origin.
    ok.append(run_case("k == avail (single window)", max_vpt=5, avail=5, num_timesteps=5,
                       fixed_cams=[0]))
    ok.append(run_too_short())

    print(f"\n{sum(ok)}/{len(ok)} cases passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
