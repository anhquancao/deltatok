"""Generate sequence-of-surround-view pkls (timestep-major: all cameras at each timestep).

Self-contained port of OccAny-main's surround_temporal sequence generation
(occrae/util/seq_helper.py + dataset_setup/base_make_seq.py) so it lives entirely
in this file and does NOT touch dataset_setup/base_make_seq.py or sh/make_seqs.sh.

Output layout (per scene, flat & timestep-major):
    seq = [t0_cam0, t0_cam1, ..., t0_camN-1, t1_cam0, ..., tS_camN-1]
    ts  = [s0,      s0,      ..., s0,        s1,      ..., sS]        # stride per cam
so camera ``c`` at timestep ``t`` lives at flat index ``t * N + c`` where
``N = len(cameras)``. A sequence is emitted only when EVERY stride is present in
ALL cameras, giving exactly ``(max_stride + 1) * N`` entries.

Windows are a fully overlapping sliding window by default (``frame_stride=1``):
the base frame advances by one each time, e.g. with strides [0,5,10,15] you get
[0,5,10,15], [1,6,11,16], [2,7,12,17], ... — matching the existing temporal
``_generate_seq`` in base_make_seq.py.

Filename: ``seq_surround_temporal_sub{subsampling_rate}_stride{max_stride}{_fsK}_{dataset}_all.pkl``
The default overlapping window (frame_stride=1) is tagged ``_fs1`` (e.g.
seq_surround_temporal_sub5_stride9_fs1_once_all.pkl); non-overlapping (frame_stride<=0)
carries no tag. ``_{dataset}`` is the dataset config name so configs that share a
preprocessed dir but differ in rig (once vs once_5cam) don't collide.
"""

import argparse
import glob
import os
import pickle
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


# Per-dataset rig + filename-format config. Cameras must match the suffixes the
# preprocessing wrote (e.g. waymo "*_1.npz"; once "*_cam01.npz"). frame_id_format
# is the zero-pad width used for the numeric frame id (":05d" -> 5 digits).
DATASET_CONFIGS: Dict[str, dict] = {
    "waymo": dict(
        preprocessed_dir="waymo_processed",
        cameras=list(range(1, 6)),
        frame_id_format=":05d",
        file_ext=".npz",
    ),
    # Waymo v1.4.2 validation split (202 segments), eval only.
    "waymo_test": dict(
        preprocessed_dir="waymo_test_processed",
        cameras=list(range(1, 6)),
        frame_id_format=":05d",
        file_ext=".npz",
    ),
    # vKITTI2 is single-camera: a degenerate "surround" of 1 view per timestep.
    # Native 10Hz (frame_ids spaced 1), so it supports both HZ tiers like waymo.
    "vkitti": dict(
        preprocessed_dir="vkitti_processed",
        cameras=[0],
        frame_id_format=":05d",
        file_ext=".npz",
    ),
    "ddad": dict(
        preprocessed_dir="ddad_processed",
        cameras=list(range(6)),
        frame_id_format=":06d",
        file_ext=".npz",
    ),
    "pandaset": dict(
        preprocessed_dir="pandaset_processed",
        cameras=list(range(6)),
        frame_id_format=":06d",
        file_ext=".npz",
    ),
    "once": dict(
        preprocessed_dir="once_processed",
        # Old TEMPORAL pkl (seq_exact_len_*_all) camera set {cam06,07,08,09} — the set the
        # converging base recipe used. NB the old SURROUND preset was {cam01,05,06,07,08}; the
        # unified pkl drives BOTH the temporal and surround blocks from ONE camera list, so this
        # set now also drives once's surround (a 4-cam surround, vs the old 5-cam ring). Reverting
        # to the old temporal cameras here targets the trajectory-ADE regression localized to
        # once's temporal block (see docs/surround_temporal_trajectory_regression.md).
        cameras=["cam06", "cam07", "cam08", "cam09"],
        frame_id_format=":06d",
        file_ext=".npz",
    ),
    # 5-cam once ring: the 4-cam temporal set above plus cam05. Shares once_processed
    # with "once"; the _{dataset} filename tag keeps the once_5cam pkl from clobbering
    # the once one.
    "once_5cam": dict(
        preprocessed_dir="once_processed",
        cameras=["cam05", "cam06", "cam07", "cam08", "cam09"],
        frame_id_format=":06d",
        file_ext=".npz",
    ),
    # 6-cam once rig: the 5-cam ring plus the narrow-FOV front cam03 (fx/W~0.86 -> ~447px
    # focal at the 518-wide model input). cam03 is the only real train camera near the
    # nuScenes eval focal band (~409-440px), which falls in the gap between the wide
    # (~250-300px) and tele (~540-610px) training clusters — see
    # docs/research/results/2026-07-05_traj_nuscenes_ade_metric_scale.html. Front camera listed first,
    # mirroring the waymo/ddad/pandaset convention. All 571 scenes have cam03 with the
    # same frame ids as the ring cams, so sequence coverage matches once_5cam.
    "once_6cam": dict(
        preprocessed_dir="once_processed",
        cameras=["cam03", "cam05", "cam06", "cam07", "cam08", "cam09"],
        frame_id_format=":06d",
        file_ext=".npz",
    ),
    # 10Hz nuScenes tier (sub1). Dir name mirrors base_make_seq.py's
    # occ3d_nuscenes_all branch — CONFIRM against $SCRATCH/data on Karolina.
    "occ3d_nuscenes_all": dict(
        preprocessed_dir="occ3d_nuscenes_processed_all",
        cameras=list(range(6)),
        frame_id_format=":06d",
        file_ext=".npz",
    ),
    "occ3d_nuscenes": dict(
        preprocessed_dir="occ3d_nuscenes_processed",
        cameras=list(range(6)),
        frame_id_format=":06d",
        file_ext=".npz",
    ),
    # OpenScene: 8-cam per-scene tar store (see occany/datasets/tar_store.py), cam ids
    # 0..7, native 2Hz. use_tar switches scene/frame discovery to the .idx.npz sidecars.
    # mini, trainval, and test are separate tar dirs (scene names collide across splits),
    # so distinct entries -> distinct pkl tags.
    "openscene_mini": dict(
        preprocessed_dir="openscene_mini",
        cameras=list(range(8)),
        frame_id_format=":06d",
        file_ext=".npz",
        use_tar=True,
    ),
    "openscene_trainval": dict(
        preprocessed_dir="openscene_trainval",
        cameras=list(range(8)),
        frame_id_format=":06d",
        file_ext=".npz",
        use_tar=True,
    ),
    "openscene_test": dict(
        preprocessed_dir="openscene_test",
        cameras=list(range(8)),
        frame_id_format=":06d",
        file_ext=".npz",
        use_tar=True,
    ),
}


# --- helpers ported verbatim from OccAny-main/occrae/util/seq_helper.py ---

def parse_frame_stem(frame_stem: str) -> Tuple[int, str]:
    prefix, camera_id = frame_stem.split("_", 1)
    return int(prefix), camera_id


def format_frame_stem(frame_id: int, camera_id: str, frame_digits: int) -> str:
    return f"{frame_id:0{frame_digits}d}_{camera_id}"


def generate_surround_temporal_sequences(
    frame_stems: Sequence[str],
    cameras: Sequence[str],
    frame_digits: int,
    subsampling_rate: int,
    max_stride: int,
    frame_stride: Optional[int] = None,
) -> List[Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Temporal sequences of surround views: all cameras at each timestep.

    A single camera (e.g. vkitti) is allowed: it degenerates to a one-view-per-
    timestep temporal sequence with (max_stride + 1) entries.
    """
    if not cameras:
        return []

    strides = [stride * subsampling_rate for stride in range(max_stride + 1)]
    frame_ids_by_camera: Dict[str, set] = defaultdict(set)

    for frame_stem in frame_stems:
        frame_id, camera_id = parse_frame_stem(frame_stem)
        if camera_id in cameras:
            frame_ids_by_camera[camera_id].add(frame_id)

    if len(frame_ids_by_camera) < len(cameras):
        return []

    complete_frame_ids = sorted(
        set.intersection(*(frame_ids_by_camera[cam] for cam in cameras))
    )
    if not complete_frame_ids:
        return []

    complete_set = set(complete_frame_ids)
    num_cameras = len(cameras)

    sequences: List[Tuple[Tuple[str, ...], Tuple[int, ...]]] = []
    idx = 0
    while idx < len(complete_frame_ids):
        base_frame_id = complete_frame_ids[idx]
        flat_stems: List[str] = []
        flat_timesteps: List[int] = []
        valid = True
        for stride in strides:
            target_frame_id = base_frame_id + stride
            if target_frame_id not in complete_set:
                valid = False
                break
            for cam_id in cameras:
                flat_stems.append(
                    format_frame_stem(target_frame_id, cam_id, frame_digits)
                )
            flat_timesteps.extend([stride] * num_cameras)

        if valid and len(flat_stems) == len(strides) * num_cameras:
            sequences.append((tuple(flat_stems), tuple(flat_timesteps)))
            if frame_stride is not None:
                next_frame_id = base_frame_id + frame_stride
                idx = bisect_right(complete_frame_ids, next_frame_id - 1)
            else:
                sequence_end_frame_id = base_frame_id + strides[-1]
                idx = bisect_right(complete_frame_ids, sequence_end_frame_id)
        else:
            idx += 1

    return sequences


class SurroundTemporalSeqMaker:
    """Glob a preprocessed dataset and write a surround-temporal sequence pkl.

    Modeled on dataset_setup/base_make_seq.py::SeqMaker but specialized to the
    surround_temporal mode (always all cameras, suffix "_all").
    """

    def __init__(self, preprocessed_dir, cameras,
                 frame_id_format=":06d", file_ext=".npz",
                 img_track_pattern="*_{camera_id}",
                 subsampling_rate=5, max_stride=9, frame_stride=1,
                 prefix="", suffix="_all", dataset_tag="", use_tar=False):
        self.processed_root = self._get_processed_root(preprocessed_dir)
        self.cameras = cameras
        self.dataset_tag = dataset_tag
        # use_tar: discover scenes/frames from per-scene tar sidecars, not loose dirs.
        self.use_tar = use_tar
        self.frame_id_format = frame_id_format
        self.file_ext = file_ext
        self.img_track_pattern = img_track_pattern
        self.subsampling_rate = subsampling_rate
        self.max_stride = max_stride
        self.frame_stride = frame_stride
        self.prefix = prefix
        self.suffix = suffix
        self.scenes = []
        self.frames = []
        self.seqs = []

    def _get_processed_root(self, preprocessed_dir):
        # DATA_ROOT lets non-Karolina clusters point at their processed-data tier
        # directly. Falls back to Karolina's $SCRATCH/data when unset.
        data_root = os.environ.get("DATA_ROOT")
        if not data_root:
            assert "SCRATCH" in os.environ, "Set DATA_ROOT or SCRATCH"
            data_root = os.path.join(os.environ["SCRATCH"], "data")
        return os.path.join(data_root, preprocessed_dir)

    def _scene_frame_stems(self, scene_name):
        """Frame stems ("<frame_id>_<cam>") for one scene, rig cameras only."""
        stems = []
        if self.use_tar:
            # Read the sidecar index with bare numpy (occany.datasets.tar_store's
            # package __init__ imports torch). The suffix filter mirrors the dir-mode
            # glob "*_{cam}<ext>" and drops .infinidepth.png members.
            cam_suffixes = tuple(f"_{c}{self.file_ext}" for c in self.cameras)
            idx_path = os.path.join(self.processed_root, scene_name + ".tar.gz.idx.npz")
            with np.load(idx_path) as idx:
                names = [str(n) for n in idx["names"]]  # "<scene>/<frame>_<cam>.npz"
            for name in names:
                base = name.rsplit("/", 1)[-1]
                if base.endswith(cam_suffixes):
                    stems.append(base[:-len(self.file_ext)])
        else:
            path = os.path.join(self.processed_root, scene_name)
            for camera_id in self.cameras:
                pattern = self.img_track_pattern.format(camera_id=camera_id) + self.file_ext
                for fp in glob.glob(os.path.join(path, pattern)):
                    stems.append(fp.split('/')[-1].replace(self.file_ext, ''))
        return sorted(stems)

    def _load_scenes_and_frames(self):
        print("Loading scenes and frames...")
        if self.use_tar:
            # One <scene>.tar.gz per scene (all repo writers emit .tar.gz).
            tars = glob.glob(os.path.join(self.processed_root, "*.tar.gz"))
            self.scenes = sorted(os.path.basename(t)[:-len(".tar.gz")] for t in tars)
        else:
            self.scenes = [
                d for d in os.listdir(self.processed_root)
                if os.path.isdir(os.path.join(self.processed_root, d)) and "tmp" not in d
            ]
            self.scenes.sort()
        print(f"Loaded {len(self.scenes)} scenes")
        frames_set = set()
        pbar = tqdm(self.scenes, desc="Loading frames")
        for scene in pbar:
            frames_set.update(self._scene_frame_stems(scene))
            pbar.set_postfix_str(f"Scene {scene}: unique frames {len(frames_set)}")

        self.frames = sorted(list(frames_set))
        print(f"Loaded {len(self.frames)} frames")

    def _generate_surround_temporal_seq(self):
        print(f"Generating surround temporal sequences for {len(self.cameras)} cameras")
        frame_to_index = {frame: idx for idx, frame in enumerate(self.frames)}
        scene_to_index = {scene: idx for idx, scene in enumerate(self.scenes)}
        cameras = [str(c) for c in self.cameras]
        frame_digits = int(self.frame_id_format[1:-1])

        pbar = tqdm(self.scenes, desc="Generating surround temporal seq")
        for scene_name in pbar:
            scene_idx = scene_to_index[scene_name]
            scene_stems = self._scene_frame_stems(scene_name)

            for frame_seq, timesteps in generate_surround_temporal_sequences(
                frame_stems=scene_stems,
                cameras=cameras,
                frame_digits=frame_digits,
                subsampling_rate=self.subsampling_rate,
                max_stride=self.max_stride,
                frame_stride=self.frame_stride,
            ):
                frame_indices = [frame_to_index[stem] for stem in frame_seq]
                self.seqs.append([scene_idx, frame_indices, list(timesteps)])
            pbar.set_postfix({"seqs": len(self.seqs)})

    def save_seq(self):
        print(len(self.seqs), "seq generated.")
        # frame_stride=1 (the overlapping default) -> "_fs1"; non-overlapping
        # (frame_stride=None) -> no tag. Matches OccAny-main's naming and the
        # already-generated *_fs1_all.pkl files.
        fs_tag = f"_fs{self.frame_stride}" if self.frame_stride is not None else ""
        # Tag with the dataset config name so configs that share a preprocessed dir
        # but differ in rig (e.g. once's 4-cam vs once_5cam) write distinct pkls
        # instead of overwriting.
        ds_tag = f"_{self.dataset_tag}" if self.dataset_tag else ""
        save_filename = (
            f"{self.prefix}seq_surround_temporal_sub{self.subsampling_rate}"
            f"_stride{self.max_stride}{fs_tag}{ds_tag}{self.suffix}.pkl"
        )
        save_path = os.path.join(self.processed_root, save_filename)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'scenes': self.scenes,
                'frames': self.frames,
                'seqs': self.seqs,
            }, f)
        print(f"Saved to {save_path}")

    def run(self):
        self._load_scenes_and_frames()
        print("Generating seq in surround_temporal mode...")
        self._generate_surround_temporal_seq()
        self.save_seq()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create surround-temporal sequence pkls.")
    parser.add_argument('--dataset', type=str, required=True, choices=sorted(DATASET_CONFIGS.keys()),
                        help='Dataset name (selects preprocessed dir + camera rig)')
    parser.add_argument('--prefix', type=str, default="", help='Optional prefix for the output PKL filename')
    parser.add_argument('--subsampling_rate', type=int, default=5, help='Frame stride per timestep unit')
    parser.add_argument('--max_stride', type=int, default=9,
                        help='Max stride -> strides [0..max_stride], i.e. (max_stride + 1) timesteps')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Stride between sequence start frames. 1 = fully overlapping '
                             'sliding window (default; base advances by 1). Larger K = sparser '
                             'starts. Pass 0 (or negative) for non-overlapping windows.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = DATASET_CONFIGS[args.dataset]
    # frame_stride <= 0 means non-overlapping (the helper's None path).
    frame_stride = args.frame_stride if args.frame_stride > 0 else None
    seq_maker = SurroundTemporalSeqMaker(
        preprocessed_dir=config["preprocessed_dir"],
        cameras=config["cameras"],
        frame_id_format=config["frame_id_format"],
        file_ext=config["file_ext"],
        subsampling_rate=args.subsampling_rate,
        max_stride=args.max_stride,
        frame_stride=frame_stride,
        prefix=args.prefix,
        dataset_tag=args.dataset,
        use_tar=config.get("use_tar", False),
    )
    seq_maker.run()
