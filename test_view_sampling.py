"""Test the unified view sampling in BaseSeqDatasetMultiView._get_views.

Creates a temp dir with dummy npz files (labeled with timestep/camera),
builds a mock instance of the real class, calls the real _get_views,
and saves sampled image grids to ./results/test_sampling/.
"""
import os
import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from occany.utils.runtime_paths import prepend_vendored_import_paths
prepend_vendored_import_paths()

from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView

RESULTS_DIR = "./results/test_sampling"
IMG_H, IMG_W = 128, 128
GRID_PAD = 4

COLORS = [
    (230, 100, 100),  # cam 0 — red
    (100, 200, 100),  # cam 1 — green
    (100, 130, 230),  # cam 2 — blue
    (230, 200, 80),   # cam 3 — yellow
    (200, 120, 230),  # cam 4 — purple
    (100, 210, 210),  # cam 5 — cyan
]

try:
    FONT_LARGE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except OSError:
    FONT_LARGE = ImageFont.load_default()
    FONT_SMALL = FONT_LARGE


def make_labeled_image(timestep, cam_idx):
    color = COLORS[cam_idx % len(COLORS)]
    img = np.full((IMG_H, IMG_W, 3), color, dtype=np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text((IMG_W / 2, IMG_H / 2), f"t{timestep} c{cam_idx}",
              fill=(0, 0, 0), anchor="mm", font=FONT_LARGE)
    return np.array(pil)


def make_dummy_npz_dir(tmpdir, scene_name, num_timesteps, max_vpt):
    scene_dir = os.path.join(tmpdir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    K = np.array([[500, 0, IMG_W / 2], [0, 500, IMG_H / 2], [0, 0, 1]], dtype=np.float32)
    for i in range(num_timesteps * max_vpt):
        np.savez(
            os.path.join(scene_dir, f"frame_{i:04d}.npz"),
            image=make_labeled_image(i // max_vpt, i % max_vpt),
            depthmap=np.ones((IMG_H, IMG_W), dtype=np.float32),
            intrinsics=K,
            cam2world=np.eye(4, dtype=np.float32),
        )


def make_mock_instance(tmpdir, max_vpt, min_vpt, num_timesteps, reverse_seq, anchor_cam=0, memory_num_views=10):
    num_frames = num_timesteps * max_vpt
    scene_name = "scene_0"

    make_dummy_npz_dir(tmpdir, scene_name, num_timesteps, max_vpt)

    obj = object.__new__(BaseSeqDatasetMultiView)
    obj.ROOT = tmpdir
    obj.num_views_per_timestep = max_vpt
    obj.min_views_per_timestep = min_vpt
    obj.reverse_seq = reverse_seq
    obj.anchor_cam = anchor_cam
    obj.min_memory_num_views = memory_num_views
    obj.max_memory_num_views = memory_num_views

    obj.scenes = [scene_name]
    obj.frames = [f"frame_{i:04d}" for i in range(num_frames)]
    obj.load_infinidepth_pseudo = False
    seq_indices = list(range(num_frames))
    timesteps = []
    for t in range(num_timesteps):
        timesteps.extend([t] * max_vpt)
    obj.seqs = [(0, seq_indices, timesteps)]

    def mock_resize(image, depthmap, intrinsics, resolution, rng=None, info=None, pseudo_depthmap=None):
        return Image.fromarray(image), depthmap, intrinsics.copy(), pseudo_depthmap

    obj._resize_image_and_sparse_depthmap = mock_resize
    return obj


def decode_views(views, max_vpt):
    result = []
    for view in views:
        idx = int(view["frame_id"].split("_")[1])
        result.append((idx // max_vpt, idx % max_vpt))
    return result


def group_by_timestep(decoded):
    by_timestep = {}
    for i, (t, c) in enumerate(decoded):
        by_timestep.setdefault(t, []).append((i, c))
    return by_timestep


def get_timestep_visit_order(decoded):
    """Return unique timesteps in first-appearance order."""
    seen = set()
    order = []
    for t, _ in decoded:
        if t not in seen:
            seen.add(t)
            order.append(t)
    return order


def save_grid(views, by_timestep, decoded, path):
    """Save sampled views as a grid: rows = timesteps, columns = camera slots."""
    all_cams = list(dict.fromkeys(c for _, c in decoded))
    cam_to_col = {c: i for i, c in enumerate(all_cams)}

    timesteps = sorted(by_timestep.keys())
    n_cols = len(all_cams)
    n_rows = len(timesteps)

    grid_w = n_cols * (IMG_W + GRID_PAD) + GRID_PAD
    grid_h = n_rows * (IMG_H + GRID_PAD) + GRID_PAD
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))

    cell_counts = {}
    for row, t in enumerate(timesteps):
        for view_idx, cam in by_timestep[t]:
            col = cam_to_col[cam]
            x = GRID_PAD + col * (IMG_W + GRID_PAD)
            y = GRID_PAD + row * (IMG_H + GRID_PAD)
            img = views[view_idx]["img"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            grid.paste(img, (x, y))
            cell_counts[(x, y)] = cell_counts.get((x, y), 0) + 1

    draw = ImageDraw.Draw(grid)
    for (x, y), count in cell_counts.items():
        draw.text((x + 2, y + 2), f"x{count}", fill=(255, 255, 255), font=FONT_SMALL)

    grid.save(path)


def run_test(name, description, expected, max_vpt, min_vpt, num_timesteps,
             memory_num_views, reverse_seq, expect_fallback=False, n_trials=10,
             anchor_cam=0, check_ordering=False):
    tmpdir = tempfile.mkdtemp()
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "")
    test_dir = os.path.join(RESULTS_DIR, safe_name)
    os.makedirs(test_dir, exist_ok=True)
    try:
        obj = make_mock_instance(tmpdir, max_vpt, min_vpt, num_timesteps, reverse_seq,
                                 anchor_cam=anchor_cam, memory_num_views=memory_num_views)
        resolution = (IMG_W, IMG_H)

        lines = []
        header = [
            f"{'=' * 60}",
            f"TEST: {name}",
            f"  What: {description}",
            f"  Expected: {expected}",
            f"  Config: num_timesteps={num_timesteps}, max_vpt={max_vpt}, "
            f"min_vpt={min_vpt}, memory_num_views={memory_num_views}, "
            f"reverse_seq={reverse_seq}, anchor_cam={anchor_cam}",
            f"{'=' * 60}",
            "",
        ]
        for h in header:
            print(h)
        lines.extend(header)

        saw_forward = False
        saw_reverse = False

        for trial in range(n_trials):
            rng = np.random.default_rng(seed=trial)
            views = obj._get_views(0, resolution, memory_num_views, rng)

            assert len(views) == memory_num_views, \
                f"Trial {trial}: expected {memory_num_views} views, got {len(views)}"

            decoded = decode_views(views, max_vpt)
            by_timestep_indexed = group_by_timestep(decoded)
            by_timestep_cams = {t: [c for _, c in entries] for t, entries in by_timestep_indexed.items()}

            assert decoded[0][1] == anchor_cam, \
                f"Trial {trial}: views[0] has cam {decoded[0][1]}, expected anchor_cam={anchor_cam}"

            cam_sets = [tuple(cams) for cams in by_timestep_cams.values()]

            if cam_sets:
                max_cam_count = max(len(cs) for cs in cam_sets)
                for t, cams in by_timestep_cams.items():
                    if len(cams) == max_cam_count:
                        assert anchor_cam in cams, \
                            f"Trial {trial}: anchor_cam={anchor_cam} missing from full timestep {t}, cams={cams}"

                if not expect_fallback and len(cam_sets) > 1:
                    full_sets = [cs for cs in cam_sets if len(cs) == max_cam_count]
                    partial_sets = [cs for cs in cam_sets if len(cs) < max_cam_count]
                    assert len(set(full_sets)) == 1, \
                        f"Trial {trial}: cameras differ across full timesteps: {by_timestep_cams}"
                    for ps in partial_sets:
                        assert set(ps).issubset(set(full_sets[0])), \
                            f"Trial {trial}: partial cams {ps} not subset of {full_sets[0]}"

            visit_order = get_timestep_visit_order(decoded)
            is_ascending = visit_order == sorted(visit_order)
            is_descending = visit_order == sorted(visit_order, reverse=True)

            if check_ordering and not expect_fallback:
                assert is_ascending or is_descending, \
                    f"Trial {trial}: timesteps not monotonic: {visit_order}"
                if is_descending and len(visit_order) > 1:
                    assert visit_order[0] == max(visit_order), \
                        f"Trial {trial}: reverse order should start from latest timestep, got {visit_order}"

            if is_ascending:
                saw_forward = True
            if is_descending and len(visit_order) > 1:
                saw_reverse = True

            actual_vpt = len(cam_sets[0]) if cam_sets else 0

            grid_path = os.path.join(test_dir, f"trial_{trial:02d}_vpt{actual_vpt}.png")
            save_grid(views, by_timestep_indexed, decoded, grid_path)

            order_label = "REV" if (is_descending and len(visit_order) > 1) else "FWD"
            trial_lines = [
                f"  trial {trial}: actual_vpt={actual_vpt}, order={order_label}, "
                f"timesteps={visit_order}, "
                f"selected_cams={list(cam_sets[0]) if cam_sets else []}",
            ]
            for t in visit_order:
                trial_lines.append(f"    t{t}: cams {by_timestep_cams[t]}")
            trial_lines.append(f"    -> {grid_path}")
            trial_lines.append("")
            for tl in trial_lines:
                print(tl)
            lines.extend(trial_lines)

        if check_ordering and reverse_seq and not expect_fallback:
            assert saw_forward and saw_reverse, \
                f"Expected both forward and reverse orderings across {n_trials} trials, " \
                f"got forward={saw_forward}, reverse={saw_reverse}"
            lines.append(f"  Ordering check: saw_forward={saw_forward}, saw_reverse={saw_reverse}")

        if check_ordering and not reverse_seq:
            assert saw_forward, \
                f"Expected forward ordering when reverse_seq=False, but saw_forward={saw_forward}"
            assert not saw_reverse, \
                f"Expected no reverse ordering when reverse_seq=False, but saw_reverse={saw_reverse}"
            lines.append(f"  Ordering check: always forward (reverse_seq=False)")

        passed = f"  PASSED ({n_trials} trials)"
        print(passed)
        lines.append(passed)
        lines.append("")
        return lines
    finally:
        shutil.rmtree(tmpdir)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_lines = ["View Sampling Test Results", "=" * 60, ""]

    all_lines += run_test(
        name="Sequence backward compat",
        description="max_vpt=1, min_vpt=1: each timestep has 1 camera, so actual_vpt is always 1. "
                    "This is the old sequence behavior.",
        expected="Each row has exactly 1 cell. All rows show different timesteps with 1 camera each. "
                 "Grid is a single column of different timesteps.",
        max_vpt=1, min_vpt=1, num_timesteps=20,
        memory_num_views=8, reverse_seq=True,
    )

    all_lines += run_test(
        name="Temporal fixed backward compat",
        description="max_vpt=5, min_vpt=5: always select all 5 cameras per timestep. "
                    "This is the old temporal surround behavior.",
        expected="Every row has exactly 5 cells (all cameras). Same 5 columns in every row. "
                 "Colors should be identical across rows (same cameras everywhere).",
        max_vpt=5, min_vpt=5, num_timesteps=10,
        memory_num_views=15, reverse_seq=False,
    )

    all_lines += run_test(
        name="Unified (max_vpt=5, min_vpt=1)",
        description="Randomly sample 1-5 cameras per timestep. This is the new unified behavior "
                    "that replaces separate sequence + temporal entries.",
        expected="actual_vpt varies across trials (1 to 5). Within each trial, all full-timestep "
                 "rows must have the same camera set (same colors in same columns). "
                 "Last row may have fewer cells (partial) but only cameras from the selected set.",
        max_vpt=5, min_vpt=1, num_timesteps=10,
        memory_num_views=10, reverse_seq=True,
    )

    all_lines += run_test(
        name="Unified 6-cam (max_vpt=6, min_vpt=1)",
        description="6-camera setup (DDAD/Pandaset-like). Randomly sample 1-6 cameras per timestep.",
        expected="Same as unified 5-cam but with up to 6 columns. Camera consistency across rows.",
        max_vpt=6, min_vpt=1, num_timesteps=10,
        memory_num_views=12, reverse_seq=True,
    )

    all_lines += run_test(
        name="Partial timestep (max_vpt=5, min_vpt=3, views=7)",
        description="min_vpt=3, so actual_vpt is 3-5. With 7 views, the last timestep gets "
                    "a partial camera subset.",
        expected="Full rows have 3-5 cameras. Last row has fewer cameras, all from the same "
                 "selected set. No cameras outside the selected set appear.",
        max_vpt=5, min_vpt=3, num_timesteps=10,
        memory_num_views=7, reverse_seq=False,
    )

    all_lines += run_test(
        name="Fallback (short seq, max_vpt=5, min_vpt=1, views=20)",
        description="Only 3 timesteps available but 20 views requested. Triggers the fallback "
                    "path which cycles through timesteps with consistent cameras.",
        expected="Same camera set at every timestep. Timesteps repeat (cycle) to fill 20 views. "
                 "Grid shows repeated timestep rows but with consistent camera columns.",
        max_vpt=5, min_vpt=1, num_timesteps=3,
        memory_num_views=20, reverse_seq=False,
        expect_fallback=True,
    )

    all_lines += run_test(
        name="Anchor cam non-zero (anchor_cam=2, max_vpt=5)",
        description="anchor_cam=2: cam 2 must always be views[0] and present in every timestep. "
                    "Verifies the anchor_cam parameter works for non-default values.",
        expected="First view is always cam 2 (blue). Cam 2 appears in every row. "
                 "Remaining cameras are random from {0,1,3,4}.",
        max_vpt=5, min_vpt=2, num_timesteps=10,
        memory_num_views=8, reverse_seq=False,
        anchor_cam=2,
    )

    all_lines += run_test(
        name="Anchor cam single view per timestep (max_vpt=1)",
        description="max_vpt=1, min_vpt=1: only 1 camera per timestep, forced to be the anchor. "
                    "Matches temporal-only datasets (KITTI, VKitti).",
        expected="Every row has exactly 1 cell showing cam 0 (the anchor). Single column grid.",
        max_vpt=1, min_vpt=1, num_timesteps=10,
        memory_num_views=5, reverse_seq=False,
        anchor_cam=0,
    )

    all_lines += run_test(
        name="Anchor cam fallback path (anchor_cam=3, short seq)",
        description="Fallback path with anchor_cam=3. Cam 3 must be views[0] even when "
                    "timesteps are cycled.",
        expected="First view is always cam 3. Cam 3 present in every timestep. "
                 "Timesteps repeat to fill views.",
        max_vpt=6, min_vpt=2, num_timesteps=2,
        memory_num_views=10, reverse_seq=False,
        expect_fallback=True,
        anchor_cam=3,
    )

    # ---- Reverse ordering tests ----

    all_lines += run_test(
        name="Reverse ordering (max_vpt=1)",
        description="reverse_seq=True with single-cam timesteps. ~50% of trials should have "
                    "descending timestep order, ~50% ascending. Verifies the stochastic "
                    "do_reverse=rng.random()<0.5 branch.",
        expected="Across 20 trials, both FWD and REV orderings observed. Each trial is "
                 "strictly monotonic (ascending or descending). Anchor timestep is 0 for "
                 "FWD, num_timesteps-1 for REV.",
        max_vpt=1, min_vpt=1, num_timesteps=15,
        memory_num_views=8, reverse_seq=True,
        check_ordering=True, n_trials=20,
    )

    all_lines += run_test(
        name="Reverse ordering multi-cam (max_vpt=5, min_vpt=2)",
        description="reverse_seq=True with multi-cam timesteps. Checks that timestep visit "
                    "order is monotonic and both directions are observed.",
        expected="Across 20 trials, both FWD and REV orderings observed. Camera selection "
                 "is consistent within each trial regardless of direction.",
        max_vpt=5, min_vpt=2, num_timesteps=10,
        memory_num_views=10, reverse_seq=True,
        check_ordering=True, n_trials=20,
    )

    all_lines += run_test(
        name="Forward only (reverse_seq=False, check_ordering)",
        description="reverse_seq=False: all trials must be forward. Verifies that disabling "
                    "reverse_seq guarantees ascending timestep order.",
        expected="All 20 trials have ascending timestep order. No reverse ordering observed.",
        max_vpt=5, min_vpt=2, num_timesteps=10,
        memory_num_views=10, reverse_seq=False,
        check_ordering=True, n_trials=20,
    )

    all_lines += run_test(
        name="Fallback reverse (reverse_seq=True, short seq)",
        description="Fallback path (short seq) with reverse_seq=True. Timesteps cycle "
                    "in reverse order (2,1,0,2,1,0...) ~50% of the time.",
        expected="Across 20 trials, both forward (0,1,2,0,...) and reverse (2,1,0,2,...) "
                 "cycling observed. Cameras remain consistent.",
        max_vpt=5, min_vpt=1, num_timesteps=3,
        memory_num_views=20, reverse_seq=True,
        expect_fallback=True, n_trials=20,
    )

    output = "\n".join(all_lines)
    print(output)

    out_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nSaved summary to {out_path}")
    print(f"Image grids in {RESULTS_DIR}/*/")


if __name__ == "__main__":
    main()
