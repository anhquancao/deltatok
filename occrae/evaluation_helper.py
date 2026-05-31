import hashlib
import json
import os
from contextlib import contextmanager, nullcontext

import numpy as np
import imageio.v2 as imageio
from einops import rearrange
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from Dataset.dataloader import get_data
from Dataset.train_dataset import NatixEvalWithTrajectoryDataset
from Metrics.inception_metrics import MultiInceptionMetrics
from Metrics.pixel_metrics import PixelMetrics
from Metrics.scenario_ade_metric import ScenarioADEMetric
from analyze_traj.trajectory_diversity import ACTION_ORDER

from Trainer.occany_model_loader import setup_occany_da3_models
from types import SimpleNamespace
from Trainer.da3_pseudo_labeler import DA3Pseudolabeler
from Network.da3_inference import inference_occany_da3
from Trainer.trajectory_preview_helper import compose_video_with_trajectory_panel
from Trainer.trajectory_preview_helper import compose_video_pair_with_trajectory_panel

class EvaluationHelper:
    def __init__(self, trainer, generation_helper):
        self.fm = trainer
        self.generation_helper = generation_helper
        self.occany_model = None
        self._occany_img_size = None
        self._occany_normalize = None
        # OccAny is a frozen helper model. We distinguish the device where it runs
        # from the device where it stays parked between uses so training does not
        # keep paying its VRAM cost.
        self._occany_runtime_device = None
        self._occany_storage_device = None
        self._occany_keep_on_runtime_device = False
        self._init_occany_eval()

    def _resolve_runtime_device(self):
        device = getattr(self.fm.args, "device", "cpu")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
        return torch.device(device)

    def _should_keep_occany_on_runtime_device(self):
        return bool(getattr(self.fm.args, "test_only", False))

    @contextmanager
    def occany_model_scope(self):
        if self.occany_model is None:
            yield False
            return

        if self._occany_keep_on_runtime_device:
            yield True
            return

        # Move to GPU only for one coarse-grained use, then park back on CPU.
        # This avoids paying a permanent VRAM tax during normal training steps.
        self.occany_model = self.occany_model.to(self._occany_runtime_device)
        try:
            yield True
        finally:
            self.occany_model = self.occany_model.to(self._occany_storage_device)
            if self._occany_runtime_device.type == "cuda":
                torch.cuda.empty_cache()

    def _should_enable_occany_eval(self):
        if getattr(self.fm.args, "test_only", False):
            return True
        return bool(getattr(self.fm.args, "enable_occany_ade_during_training", False))

    def _init_occany_eval(self):
        """Load OccAny recon model for trajectory ADE evaluation if args are set."""
        is_master = bool(getattr(self.fm.args, "is_master", True))
        occany_ckpt = str(getattr(self.fm.args, "eval_occany_ckpt", "")).strip()
        if not occany_ckpt:
            return

        if not self._should_enable_occany_eval():
            if is_master:
                print("[EvalHelper] OccAny ADE evaluation disabled during training to save memory")
            return

        occany_base = str(getattr(self.fm.args, "eval_occany_base_da3_ckpt", "")).strip()
        if not occany_base:
            if is_master:
                print("[EvalHelper] --eval-occany-base-da3-ckpt not set; ADE evaluation disabled")
            return

        occany_img_size = list(getattr(self.fm.args, "eval_occany_img_size", [378, 504]))
        if len(occany_img_size) == 1:
            occany_img_size = occany_img_size * 2

        self._occany_runtime_device = self._resolve_runtime_device()
        self._occany_keep_on_runtime_device = self._should_keep_occany_on_runtime_device()
        self._occany_storage_device = (
            self._occany_runtime_device if self._occany_keep_on_runtime_device else torch.device("cpu")
        )

        # Load onto the storage device first. For training runs this is CPU, so
        # OccAny stays out of GPU memory until occany_model_scope() is entered.
        # gen=False means we only need the reconstruction-side OccAny model.
        loader_args = SimpleNamespace(
            device=self._occany_storage_device,
            img_size=occany_img_size,
            gen=False,
        )

        if is_master:
            print(f"[EvalHelper] Loading OccAny recon model from: {occany_ckpt}")
        _, model_recon, _ = setup_occany_da3_models(
            weights_path=occany_ckpt,
            da3_ckpt_path=occany_base,
            args=loader_args,
            output_resolution=occany_img_size,
        )
        self.occany_model = model_recon
        self._occany_img_size = tuple(occany_img_size)
        self._occany_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        if is_master:
            device_label = str(self._occany_storage_device)
            print(f"[EvalHelper] OccAny ADE evaluation enabled (input {self._occany_img_size}, parked on {device_label})")

    @staticmethod
    def _poses_to_xy_trajectory(c2w):
        """Convert c2w poses to (B, T, 2) xy trajectory — same as DA3Pseudolabeler."""
        c2w_np = c2w.detach().float().cpu().numpy()

        if c2w_np.shape[-2:] == (3, 4):
            c2w_h = np.zeros(c2w_np.shape[:-2] + (4, 4), dtype=np.float32)
            c2w_h[..., :3, :4] = c2w_np
            c2w_h[..., 3, 3] = 1.0
        elif c2w_np.shape[-2:] == (4, 4):
            c2w_h = c2w_np.astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"Unsupported c2w shape: {tuple(c2w_np.shape)}")

        bsz, num_frames = c2w_h.shape[:2]
        trajectories = np.zeros((bsz, num_frames, 2), dtype=np.float32)

        for b in range(bsz):
            pose0_inv = np.linalg.inv(c2w_h[b, 0])
            local_pose = np.matmul(pose0_inv, c2w_h[b])
            trajectories[b, :, 0] = local_pose[:, 0, 3]
            trajectories[b, :, 1] = local_pose[:, 2, 3]

        return trajectories

    def _get_front_camera_index(self):
        cameras = getattr(self.fm.args, "cameras", None)
        if isinstance(cameras, (list, tuple)) and "FRONT" in cameras:
            return cameras.index("FRONT")
        return 0

    def _select_front_camera_from_combined_video(self, video):
        if video.dim() != 5:
            raise ValueError(f"Expected generated video shape (B, C, T, H, W), got {tuple(video.shape)}")

        num_cameras = max(1, int(getattr(self.fm.args, "nb_cam", 1)))
        width_total = video.shape[-1]
        if width_total % num_cameras != 0:
            raise ValueError(
                f"Combined video width {width_total} is not divisible by num cameras {num_cameras}"
            )

        frame_width = width_total // num_cameras
        front_cam_idx = min(self._get_front_camera_index(), num_cameras - 1)
        front_video = video[:, :, :, :, front_cam_idx * frame_width:(front_cam_idx + 1) * frame_width]
        return rearrange(front_video, "b c t h w -> b t c h w").contiguous()

    def _select_front_camera_from_camera_batch(self, video):
        if video.dim() != 5:
            raise ValueError(f"Expected camera batch shape (B, T, C, H, W), got {tuple(video.shape)}")

        num_cameras = max(1, int(getattr(self.fm.args, "nb_cam", 1)))
        front_cam_idx = min(self._get_front_camera_index(), num_cameras - 1)
        if num_cameras == 1:
            return video
        if video.shape[0] <= front_cam_idx:
            raise ValueError(
                f"Cannot select front camera index {front_cam_idx} from batch with size {video.shape[0]}"
            )
        return video[front_cam_idx::num_cameras].contiguous()

    def _extract_trajectory_from_video(self, gen_frames):
        """Run OccAny on generated frames and return (B, T, 2) trajectory.

        Args:
            gen_frames: (B, T, C, H, W) tensor in [-1, 1] range for a single camera.
        Returns:
            np.ndarray of shape (B, T, 2) or None on failure.
        """
        # Convert [-1, 1] → [0, 1]
        frames_01 = (gen_frames.float() + 1.0) * 0.5
        frames_01 = torch.clamp(frames_01, 0.0, 1.0)

        # Resize to OccAny expected resolution if needed
        _, t, c, h, w = frames_01.shape
        target_h, target_w = self._occany_img_size
        if h != target_h or w != target_w:
            frames_flat = frames_01.reshape(-1, c, h, w)
            frames_flat = F.interpolate(frames_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            frames_01 = frames_flat.reshape(-1, t, c, target_h, target_w)

        # Apply ImageNet normalization
        frames_norm = self._occany_normalize(frames_01.reshape(-1, c, target_h, target_w))
        frames_norm = frames_norm.reshape(-1, t, c, target_h, target_w)

        return self._extract_trajectory_from_preprocessed_video(frames_norm)

    def _extract_trajectory_from_preprocessed_video(self, frames_norm):
        """Run OccAny on frames already preprocessed like DA3Pseudolabeler."""
        _, _, _, height, width = frames_norm.shape
        assert (height, width) == self._occany_img_size, (
            f"OccAny input size mismatch: expected={self._occany_img_size}, got={(height, width)}"
        )

        # occany_model_scope() is responsible for making sure the model is on the
        # runtime device before we call inference_occany_da3().
        recon_views = DA3Pseudolabeler._build_occany_recon_views(frames_norm)
        output = inference_occany_da3(
            recon_views,
            self.occany_model,
            device=self._occany_runtime_device,
            dtype=torch.float32,
            pose_from_depth_ray=True,
            point_from_depth_and_pose=False,
        )

        c2w = DA3Pseudolabeler._get_field(output, "c2w")
        if c2w is None:
            return None

        return self._poses_to_xy_trajectory(c2w)

    @staticmethod
    def _video_to_png_strip(video):
        """Convert a normalized video tensor (T, C, H, W) to a horizontal PNG strip."""
        video = video.detach().float().cpu()
        if video.numel() == 0:
            raise ValueError("Cannot save an empty video preview")
        video = torch.clamp(video, -1.0, 1.0)
        frames = ((video + 1.0) * 127.5).round().to(torch.uint8)
        frames = frames.permute(0, 2, 3, 1).numpy()
        return np.concatenate(list(frames), axis=1)

    @staticmethod
    def _dist_is_active():
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _log_eval_stage(self, message):
        if self._dist_is_active():
            print(f"[EvalHelper][rank{dist.get_rank()}] {message}", flush=True)
        elif self.fm.args.is_master:
            print(f"[EvalHelper] {message}", flush=True)

    @staticmethod
    def _scenario_key(scenario_name):
        return str(scenario_name).strip().replace(" ", "_").upper()

    @staticmethod
    def _sanitize_path_component(value):
        sanitized = str(value).strip().replace(os.sep, "__")
        sanitized = sanitized.replace(" ", "_")
        sanitized = sanitized.replace(":", "_")
        return sanitized or "sample"

    def _extract_eval_batch_metadata(self, batch, batch_size, batch_idx):
        scenario_values = batch.get("scenario")
        if isinstance(scenario_values, (list, tuple)):
            scenario_names = [str(v) for v in scenario_values]
        elif scenario_values is None:
            scenario_names = ["unknown"] * batch_size
        else:
            scenario_names = [str(scenario_values)] * batch_size

        sample_id_values = batch.get("sample_id")
        if isinstance(sample_id_values, (list, tuple)):
            rel_paths = [str(v) for v in sample_id_values]
        elif sample_id_values is not None:
            rel_paths = [str(sample_id_values)] * batch_size
        else:
            rel_path_values = batch.get("rel_path")
            if isinstance(rel_path_values, (list, tuple)):
                rel_paths = [str(v) for v in rel_path_values]
            elif rel_path_values is None:
                rel_paths = [f"sample_{batch_idx}_{sample_idx}" for sample_idx in range(batch_size)]
            else:
                rel_paths = [str(rel_path_values)] * batch_size

        return scenario_names, rel_paths

    def _should_save_eval_trajectory_visualizations(self):
        return int(getattr(self.fm.args, "eval_trajectory_viz_per_scenario", 0)) > 0

    def _resolve_eval_trajectory_visualization_dir(self):
        configured_dir = str(getattr(self.fm.args, "eval_trajectory_viz_dir", "")).strip()
        if configured_dir:
            return os.path.abspath(configured_dir)

        output_root = str(getattr(self.fm.args, "vit_folder", "")).strip()
        if not output_root:
            output_root = "./outputs"
        return os.path.abspath(os.path.join(output_root, "eval_trajectory_viz"))

    def _save_eval_traj_vis_videos(
        self,
        real_sample,
        gen_sample,
        trajectory_cond,
        scenario_names,
        rel_paths,
        gen_traj,
        real_traj,
        traj_vis_output_dir,
        traj_vis_limit_per_scenario,
        traj_vis_counts_by_scenario,
        traj_vis_fps,
        traj_vis_saved_paths,
    ):
        if trajectory_cond is None:
            return

        front_real = self._select_front_camera_from_camera_batch(real_sample)
        front_gen = self._select_front_camera_from_camera_batch(gen_sample)
        for b_idx in range(front_gen.shape[0]):
            scenario_name = scenario_names[b_idx] if b_idx < len(scenario_names) else "unknown"
            saved_count = traj_vis_counts_by_scenario.get(scenario_name, 0)
            if saved_count >= traj_vis_limit_per_scenario:
                continue

            traj_vis_input_trajectory = trajectory_cond[b_idx:b_idx + 1].detach().cpu()
            estimated_gen_trajectory = None
            estimated_real_trajectory = None
            if gen_traj is not None:
                estimated_gen_trajectory = torch.from_numpy(np.asarray(gen_traj[b_idx:b_idx + 1], dtype=np.float32))
            if real_traj is not None:
                estimated_real_trajectory = torch.from_numpy(np.asarray(real_traj[b_idx:b_idx + 1], dtype=np.float32))

            traj_vis_video = compose_video_pair_with_trajectory_panel(
                rearrange(front_gen[b_idx:b_idx + 1], "b t c h w -> b c t h w").contiguous(),
                rearrange(front_real[b_idx:b_idx + 1], "b t c h w -> b c t h w").contiguous(),
                traj_vis_input_trajectory,
                comparison_trajectories=[estimated_gen_trajectory, estimated_real_trajectory],
                trajectory_label="Input trajectory",
                comparison_labels=["Estimated gen", "Estimated gt"],
                left_video_label="Generated",
                right_video_label="Ground truth",
            )
            traj_vis_video_np = traj_vis_video[0].permute(1, 2, 3, 0).cpu().numpy()
            traj_vis_video_np = ((traj_vis_video_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            scenario_dir = os.path.join(traj_vis_output_dir, self._sanitize_path_component(self._scenario_key(scenario_name)))
            os.makedirs(scenario_dir, exist_ok=True)
            rel_path_value = rel_paths[b_idx] if b_idx < len(rel_paths) else f"sample_{b_idx}"
            rel_path_stem = os.path.splitext(self._sanitize_path_component(rel_path_value))[0]
            output_path = os.path.join(
                scenario_dir,
                f"{saved_count + 1:02d}_{rel_path_stem}.mp4",
            )
            imageio.mimsave(output_path, traj_vis_video_np, fps=traj_vis_fps)
            traj_vis_counts_by_scenario[scenario_name] = saved_count + 1
            traj_vis_saved_paths.append(output_path)
            print(f"[EvalHelper] Saved evaluation trajectory visualization: {output_path}")

    def _broadcast_object(self, obj):
        if not self._dist_is_active():
            return obj

        # Rank 0 builds the Python object once, then broadcast_object_list copies
        # it to the other ranks so every process evaluates the exact same samples.
        object_list = [obj if self.fm.args.is_master else None]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]

    def _read_non_empty_lines(self, path):
        with open(path, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]

    def _build_uneven_distributed_subset(self, dataset):
        if not self.fm.args.is_multi_gpus or not self._dist_is_active():
            return dataset

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Strided slicing keeps the full evaluation set without sampler padding,
        # so some ranks may receive one extra item when len(dataset) is uneven.
        indices = list(range(rank, len(dataset), world_size))
        return Subset(dataset, indices)

    def _select_eval_video_paths(self, video_rel_paths, num_video):
        if num_video is None or int(num_video) <= 0:
            return video_rel_paths

        target_videos = max(1, num_video // max(1, self.fm.args.nb_cam))
        if len(video_rel_paths) <= target_videos:
            return video_rel_paths

        print("[EvalutionHelper] Deterministic sampling of validation items")
        seed = self.fm.args.seed if self.fm.args.seed >= 0 else 0
        keyed_paths = []
        for rel_path in video_rel_paths:
            stable_key = hashlib.sha256(f"{seed}:{rel_path}".encode("utf-8")).digest()
            keyed_paths.append((stable_key, rel_path))

        keyed_paths.sort(key=lambda item: (item[0], item[1]))
        selected_paths = [rel_path for _, rel_path in keyed_paths[:target_videos]]
        selected_paths.sort()
        return selected_paths

    @staticmethod
    def _normalize_real_sample(real_sample):
        real_sample = real_sample.float()
        if real_sample.numel() == 0:
            return real_sample
        if real_sample.max() > 1.5:
            real_sample = real_sample / 255.0
        return torch.clamp((real_sample * 2.0) - 1.0, -1.0, 1.0).contiguous()

    def _resolve_natix_eval_list_path(self):
        val_file_list = getattr(self.fm.args, "val_file_list", "")
        if val_file_list:
            return val_file_list

        val_video_list = getattr(self.fm.args, "val_video_list", "")
        if val_video_list:
            if self.fm.args.data == "natix_feat_wds":
                metadata_path = os.path.join(os.path.dirname(val_video_list), "metadata.json")
                if not os.path.isfile(metadata_path):
                    raise FileNotFoundError(
                        f"Expected WebDataset metadata next to shard manifest: {metadata_path}"
                    )
                with open(metadata_path, "r", encoding="utf-8") as file_obj:
                    metadata = json.load(file_obj)
                source_manifest = metadata.get("source_manifest")
                if not source_manifest:
                    raise KeyError(f"Missing 'source_manifest' in {metadata_path}")
                return source_manifest
            return val_video_list

        eval_folder = self.fm.args.eval_folder
        if eval_folder.endswith(".txt"):
            return eval_folder

        default_list_path = os.path.join(eval_folder, "natix_video_small.txt")
        if os.path.isfile(default_list_path):
            return default_list_path

        raise FileNotFoundError(
            "Unable to resolve a Natix evaluation video list. "
            "Provide --val-file-list, --val-video-list, or set --eval-folder to a manifest file."
        )

    def _build_eval_loader(self, tot_frames, num_video):
        if not self.fm.args.use_trajectory_cond:
            return get_data(
                data=self.fm.args.data.split("_")[0],
                img_size=self.fm.args.img_size,
                n_frames=tot_frames,
                seed=self.fm.args.seed,
                data_folder=self.fm.args.eval_folder,
                cameras=self.fm.args.cameras,
                bsize=2,
                num_workers=2,
                is_multi_gpus=self.fm.args.is_multi_gpus,
            )[0]

        eval_list_path = self._resolve_natix_eval_list_path()
        video_rel_paths = self._read_non_empty_lines(eval_list_path)
        video_rel_paths = self._select_eval_video_paths(video_rel_paths, num_video)
        visualize_traj_only = bool(getattr(self.fm.args, "visualize_traj_only", False))
        max_samples_per_scenario = 0
        if visualize_traj_only:
            max_samples_per_scenario = max(0, int(getattr(self.fm.args, "eval_trajectory_viz_per_scenario", 0)))
        samples = None
        if not self._dist_is_active() or self.fm.args.is_master:
            # Build the Natix sample list only on rank 0 when distributed, then
            # share it with the other ranks instead of re-scanning pseudo-labels.
            samples = NatixEvalWithTrajectoryDataset.build_samples(
                video_rel_paths=video_rel_paths,
                pseudo_label_folder=self.fm.args.pseudo_label_folder,
                show_progress=not self._dist_is_active() or self.fm.args.is_master,
                max_samples_per_scenario=max_samples_per_scenario,
            )
        samples = self._broadcast_object(samples)
        dataset = NatixEvalWithTrajectoryDataset(
            samples=samples,
            eval_folder=self.fm.args.eval_folder,
            img_size=self.fm.args.img_size,
            cameras=self.fm.args.cameras,
        )
        if len(dataset) == 0:
            raise RuntimeError(
                "Natix trajectory-aware evaluation dataset is empty. "
                "Check --eval-folder, --pseudo-label-folder, and the validation manifest."
            )
        total_eval_items = len(dataset)
        eval_dataset = self._build_uneven_distributed_subset(dataset)
        loader = DataLoader(
            eval_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            # Keep the final partial batch so every manifest entry is evaluated once.
            drop_last=False,
        )
        loader.total_eval_items = total_eval_items
        loader.scenario_names = list(ACTION_ORDER)
        return loader

    @torch.no_grad()
    def compute_score(self, num_video=None, num_step=50, tot_frames=16, save_exemple=False, compute_pr=False, use_precomputed_feat="", save_sample=False):
        if num_video is None:
            num_video = int(getattr(self.fm.args, "metrics_num_video", 500))
        visualize_traj_only = bool(getattr(self.fm.args, "visualize_traj_only", False))
        if visualize_traj_only:
            # In visualization-only mode we want to traverse the full validation list.
            num_video = -1
        metrics_hf_root = getattr(self.fm.args, "metrics_hf_root", "pretrained_ckpts")

        video_metrics = MultiInceptionMetrics(
            device=self.fm.args.device,
            compute_manifold=compute_pr,
            num_inception_chunks=10,
            manifold_k=3,
            model="i3d",
            use_precomputed_feat=use_precomputed_feat,
            huggingface_root=metrics_hf_root,
        )

        images_metrics = MultiInceptionMetrics(
            device=self.fm.args.device,
            compute_manifold=compute_pr,
            num_inception_chunks=10,
            manifold_k=3,
            model="inception",
            use_precomputed_feat=use_precomputed_feat,
            huggingface_root=metrics_hf_root,
        )

        pixel_metrics = PixelMetrics(device=self.fm.args.device)

        ade_per_scenario = bool(getattr(self.fm.args, "ade_per_scenario", False))
        if ade_per_scenario:
            print("[EvalHelper] ADE per scenario enabled: compute ADE per scenario")
        eval_ade = self.occany_model is not None
        if visualize_traj_only and bool(getattr(self.fm.args, "is_multi_gpus", False)):
            raise RuntimeError("--visualize-traj-only is only supported in single-GPU mode")
        traj_vis_enabled = self.fm.args.is_master and visualize_traj_only
        configured_traj_vis_limit = int(getattr(self.fm.args, "eval_trajectory_viz_per_scenario", 0))
        traj_vis_limit_per_scenario = max(0, configured_traj_vis_limit)
        traj_vis_counts_by_scenario = {}
        traj_vis_output_dir = None
        traj_vis_fps = float(getattr(self.fm.args, "eval_trajectory_viz_fps", 4.0))
        traj_vis_saved_paths = []
        if traj_vis_enabled:
            traj_vis_output_dir = self._resolve_eval_trajectory_visualization_dir()
            os.makedirs(traj_vis_output_dir, exist_ok=True)
            mode_label = " (visualize-traj-only)" if visualize_traj_only else ""
            print(
                f"[EvalHelper] Saving up to {traj_vis_limit_per_scenario} evaluation traj_vis_video files per scenario to {traj_vis_output_dir}{mode_label}"
            )

        eval_data = self._build_eval_loader(tot_frames, num_video)
        ade_scenario_names = getattr(eval_data, "scenario_names", []) if ade_per_scenario else []
        ade_real_vs_gen_metric = ScenarioADEMetric(
            device=self._resolve_runtime_device(),
            metric_key="ADE_REAL_VS_GEN",
            scenario_names=ade_scenario_names,
        )
        ade_real_vs_pseudolabel_metric = ScenarioADEMetric(
            device=self._resolve_runtime_device(),
            metric_key="ADE_REAL_VS_PSEUDOLABEL",
            scenario_names=ade_scenario_names,
        )

        try:
            total_videos = int(getattr(eval_data, "total_eval_items", 0))
            if total_videos <= 0:
                total_videos = len(eval_data) * int(getattr(eval_data, "batch_size", 1) or 1)
                if self.fm.args.is_multi_gpus:
                    total_videos *= self.fm.args.nb_gpus
        except TypeError:
            total_videos = num_video
        if self.fm.args.is_multi_gpus:
            total_videos = len(eval_data) * int(getattr(eval_data, "batch_size", 1) or 1)
        bar = tqdm(total=total_videos, leave=False, desc="Metrics Evaluation") if self.fm.args.is_master else None
        rollout_steps = (tot_frames - 1) // self.fm.args.n_frames + 1
        # Keep OccAny on GPU for the whole metric pass instead of moving it for
        # every batch. When this scope exits, the model is parked back on CPU.
        occany_scope = self.occany_model_scope() if eval_ade else nullcontext(False)
        with self.fm.ema_scope(), occany_scope:
            for i, batch in enumerate(eval_data):
                real_sample = batch["images"].to(self.fm.args.device)
                trajectory_cond = batch.get("trajectory")
                if trajectory_cond is not None:
                    trajectory_cond = trajectory_cond.to(self.fm.args.device)
                    if rollout_steps > 1 and trajectory_cond.shape[1] == self.fm.args.trajectory_length:
                        trajectory_cond = trajectory_cond.repeat(1, rollout_steps, 1)

                b, c, t, h, w_total = real_sample.size()
                w = w_total // self.fm.args.nb_cam

                gen_sample = self.generation_helper.generate_samples(
                    x_ctx=real_sample,
                    nb_video=b,
                    latent_context=1,
                    num_steps=num_step,
                    text=None,
                    cfg_w=0,
                    alpha=0.0,
                    scheduler_mode=self.fm.args.scheduler_mode,
                    rollout_steps=rollout_steps,
                    use_clip=False,
                    trajectory=trajectory_cond,
                )

                gen_sample = gen_sample[:, :, :tot_frames, :, :]
                gen_sample = rearrange(gen_sample, "b c t h (w cam) -> (b cam) t c h w", b=b, c=c, t=tot_frames, h=h, w=w, cam=self.fm.args.nb_cam)
                gen_sample = torch.clamp(gen_sample, -1, 1).contiguous().float()

                real_sample = rearrange(real_sample, "b c t h (w cam) -> (b cam) t c h w", b=b, c=c, t=tot_frames, h=h, w=w, cam=self.fm.args.nb_cam)
                real_sample = self._normalize_real_sample(real_sample)

                video_metrics.update(gen_sample, image_type="fake")
                video_metrics.update(real_sample, image_type="real")

                images_metrics.update(gen_sample, image_type="fake")
                images_metrics.update(real_sample, image_type="real")

                pixel_metrics.update(gen_sample, real_sample)

                front_cam_idx = self._get_front_camera_index()
                scenario_names, rel_paths = self._extract_eval_batch_metadata(batch, b, i)

                # --- ADE: compare OccAny trajectories from real and generated front-camera frames ---
                real_traj = None
                gen_traj = None
                if eval_ade:
                    try:
                        real_front = real_sample[front_cam_idx::self.fm.args.nb_cam]
                        real_traj = self._extract_trajectory_from_video(real_front)

                        gen_front = gen_sample[front_cam_idx::self.fm.args.nb_cam]
                        gen_traj = self._extract_trajectory_from_video(gen_front)

                        # os.makedirs("outputs", exist_ok=True)
                        # imageio.imwrite("outputs/real.png", self._video_to_png_strip(real_sample[front_cam_idx]))
                        # imageio.imwrite("outputs/gen.png", self._video_to_png_strip(gen_sample[front_cam_idx]))

                        if real_traj is not None and gen_traj is not None:
                            assert gen_traj.shape == real_traj.shape, (
                                f"Trajectory shape mismatch: generated={gen_traj.shape}, "
                                f"real={real_traj.shape}"
                            )
                            batch_ade_real_vs_gen = []
                            batch_ade_real_vs_gen_scenarios = []
                            for b_idx in range(gen_traj.shape[0]):
                                ade = float(np.mean(np.linalg.norm(
                                    real_traj[b_idx] - gen_traj[b_idx], axis=1,
                                )))
                                batch_ade_real_vs_gen.append(ade)
                                batch_ade_real_vs_gen_scenarios.append(scenario_names[b_idx])
                            ade_real_vs_gen_metric.update(
                                batch_ade_real_vs_gen,
                                batch_ade_real_vs_gen_scenarios if ade_per_scenario else None,
                            )

                        if real_traj is not None and trajectory_cond is not None:
                            loaded_traj = trajectory_cond.detach().cpu().numpy()
                            assert real_traj.shape == loaded_traj.shape, (
                                f"Trajectory shape mismatch: real={real_traj.shape}, "
                                f"loaded={loaded_traj.shape}"
                            )
                            batch_ade_real_vs_pseudolabel = []
                            batch_ade_real_vs_pseudolabel_scenarios = []
                            for b_idx in range(real_traj.shape[0]):
                                ade_real_pseudolabel = float(np.mean(np.linalg.norm(
                                    real_traj[b_idx] - loaded_traj[b_idx], axis=1,
                                )))
                                batch_ade_real_vs_pseudolabel.append(ade_real_pseudolabel)
                                batch_ade_real_vs_pseudolabel_scenarios.append(scenario_names[b_idx])
                            ade_real_vs_pseudolabel_metric.update(
                                batch_ade_real_vs_pseudolabel,
                                batch_ade_real_vs_pseudolabel_scenarios if ade_per_scenario else None,
                            )
                    except Exception as exc:
                        if self.fm.args.is_master:
                            print(f"[EvalHelper] ADE extraction failed for batch {i}: {exc}")

                if traj_vis_enabled:
                    self._save_eval_traj_vis_videos(
                        real_sample=real_sample,
                        gen_sample=gen_sample,
                        trajectory_cond=trajectory_cond,
                        scenario_names=scenario_names,
                        rel_paths=rel_paths,
                        gen_traj=gen_traj,
                        real_traj=real_traj,
                        traj_vis_output_dir=traj_vis_output_dir,
                        traj_vis_limit_per_scenario=traj_vis_limit_per_scenario,
                        traj_vis_counts_by_scenario=traj_vis_counts_by_scenario,
                        traj_vis_fps=traj_vis_fps,
                        traj_vis_saved_paths=traj_vis_saved_paths,
                    )

                if save_sample and i == 0 and self.fm.args.is_master:
                    os.makedirs("./outputs", exist_ok=True)
                    video = torch.cat([gen_sample[0], real_sample[0]], dim=-1)
                    video = video.permute(0, 2, 3, 1).cpu().numpy()
                    video = ((video + 1) * 127.5).astype(np.uint8)
                    imageio.mimsave("./outputs/generated_video.mp4", video, fps=4)
                    print("[EvalHelper] Saved video: ./outputs/generated_video.mp4")

                    if trajectory_cond is not None:
                        front_real = self._select_front_camera_from_camera_batch(real_sample)[:1]
                        front_gen = self._select_front_camera_from_camera_batch(gen_sample)[:1]
                        preview_video = rearrange(front_gen, "b t c h w -> b c t h w").contiguous()
                        preview_trajectory = trajectory_cond[:1].detach().cpu()
                        estimated_gen_trajectory = None
                        estimated_real_trajectory = None
                        if self.occany_model is not None:
                            try:
                                estimated_gen_trajectory_np = self._extract_trajectory_from_video(front_gen)
                                if estimated_gen_trajectory_np is not None:
                                    estimated_gen_trajectory = torch.from_numpy(estimated_gen_trajectory_np)
                                estimated_real_trajectory_np = self._extract_trajectory_from_video(front_real)
                                if estimated_real_trajectory_np is not None:
                                    estimated_real_trajectory = torch.from_numpy(estimated_real_trajectory_np)
                            except Exception as exc:
                                print(f"[EvalHelper] OccAny trajectory extraction failed for evaluation sample preview: {exc}")
                        preview_video = compose_video_with_trajectory_panel(
                            preview_video,
                            preview_trajectory,
                            comparison_trajectories=[estimated_gen_trajectory, estimated_real_trajectory],
                            comparison_labels=["Gen (OccAny)", "Real (OccAny)"],
                        )
                        preview_video = preview_video[0].permute(1, 2, 3, 0).cpu().numpy()
                        preview_video = ((preview_video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        imageio.mimsave("./outputs/generated_video_with_trajectory.mp4", preview_video, fps=4)
                        print("[EvalHelper] Saved video: ./outputs/generated_video_with_trajectory.mp4")
                    
                if self.fm.args.is_master:
                    remaining = max(0, total_videos - bar.n)
                    bar.update(min(b, remaining))

        self._log_eval_stage(
            f"completed generation loop over {len(eval_data)} local batches; starting metric reductions"
        )
        self._log_eval_stage("starting video metric reduction")
        video_m = video_metrics.compute()
        self._log_eval_stage("finished video metric reduction")
        self._log_eval_stage("starting image metric reduction")
        images_m = images_metrics.compute()
        self._log_eval_stage("finished image metric reduction")
        self._log_eval_stage("starting pixel metric reduction")
        pixel_m = pixel_metrics.compute()
        self._log_eval_stage("finished pixel metric reduction")

        metrics = {**video_m, **images_m, **pixel_m}
        self._log_eval_stage("starting ADE metric reduction")
        metrics.update(ade_real_vs_gen_metric.compute())
        metrics.update(ade_real_vs_pseudolabel_metric.compute())
        self._log_eval_stage("finished ADE metric reduction")

        metrics = {f"{k}": round(v, 4) for k, v in metrics.items()}

        if self.fm.args.is_master:
            bar.close()
            if not getattr(self.fm.args, "test_only", False):
                print(metrics)
            if traj_vis_saved_paths:
                print(
                    f"[EvalHelper] Saved {len(traj_vis_saved_paths)} evaluation traj_vis_video files to {traj_vis_output_dir}"
                )
            if save_exemple:
                name = str(self.fm.sampler).replace(" ", "_").replace(",", "").replace(":", "")
                with open(f"./results/" + name, "w") as file:
                    file.write(str(metrics))

        if "batch" in locals():
            del batch
        if "real_sample" in locals():
            del real_sample
        if "gen_sample" in locals():
            del gen_sample
        if "video" in locals():
            del video
        if "video_m" in locals():
            del video_m
        if "images_m" in locals():
            del images_m
        if "pixel_m" in locals():
            del pixel_m
        if "eval_data" in locals():
            del eval_data
        if "video_metrics" in locals():
            del video_metrics
        if "images_metrics" in locals():
            del images_metrics
        if "pixel_metrics" in locals():
            del pixel_metrics
        if "ade_real_vs_gen_metric" in locals():
            del ade_real_vs_gen_metric
        if "ade_real_vs_pseudolabel_metric" in locals():
            del ade_real_vs_pseudolabel_metric

        if self.fm.args.device != "cpu":
            torch.cuda.empty_cache()

        return metrics

