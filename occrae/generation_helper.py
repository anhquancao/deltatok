import math
from contextlib import nullcontext

import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

import torch


@torch.no_grad()
def flow_euler_sample(
    model,
    z,
    *,
    pred_mode="x",
    context=0,
    num_steps=50,
    alpha=0.5,
    scheduler_mode="cosine",
    cross_cond=None,
    cfg_w=0,
    cross_cond_zeros=None,
    autocast_ctx=None,
    desc="Sampling",
):
    """Euler flow-matching sampler for (B, C, T, H, W) latents.

    The first ``context`` frames along T are kept fixed as conditioning.

    Args:
        model: Transformer model (in eval mode).
        z: ``(B, C, T, H, W)`` initial state with context frames already set.
        pred_mode: Model prediction target (``"x"``, ``"v"``, or ``"e"``).
        context: Number of context frames to keep fixed.
        num_steps: Number of Euler integration steps.
        alpha: Per-frame timestep offset strength.
        scheduler_mode: ``"cosine"``, ``"square"``, or ``"linear"``.
        cross_cond: Optional conditioning for cross-attention.
        cfg_w: Classifier-free guidance weight (0 disables CFG).
        cross_cond_zeros: Zero embedding for CFG (required when ``cfg_w > 1``).
        autocast_ctx: Context manager for mixed-precision. Defaults to no-op.
        desc: Progress bar description.

    Returns:
        ``(B, C, T, H, W)`` denoised output.
    """
    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    B = z.shape[0]
    t_dim = z.shape[2]

    for i in tqdm(range(num_steps + 1), leave=False, desc=desc):
        if scheduler_mode == "cosine":
            progress = 1 - math.cos(i / num_steps * math.pi / 2)
            progress_next = 1 - math.cos((i + 1) / num_steps * math.pi / 2)
        elif scheduler_mode == "square":
            progress = (i / num_steps) ** 0.5
            progress_next = ((i + 1) / num_steps) ** 0.5
        else:
            progress = i / num_steps
            progress_next = (i + 1) / num_steps

        offsets = 1 + (torch.linspace(1, 0, t_dim, device=z.device)[None, :] * alpha)
        t_curr = torch.clamp(progress * offsets, 0, 1).expand(B, t_dim).clone()
        t_next = torch.clamp(progress_next * offsets, 0, 1)
        if context > 0:
            t_curr[:, :context] = 1
            t_next[:, :context] = 1

        with autocast_ctx:
            pred = model(
                x=z, ada_cond=t_curr, cross_cond=cross_cond,
            )
            if cfg_w > 1:
                pred_uncond = model(
                    x=z, ada_cond=t_curr, cross_cond=cross_cond_zeros,
                )
                pred = pred_uncond + cfg_w * (pred - pred_uncond)

        if pred_mode == "x":
            v = (pred - z) / torch.clamp(1 - t_curr.view(B, 1, t_dim, 1, 1), min=0.05)
        elif pred_mode == "e":
            v = (z - pred) / torch.clamp(1 - t_curr.view(B, 1, t_dim, 1, 1), min=0.05)
        else:
            v = pred

        if context > 0:
            v[:, :, :context] = 0.0

        z = z + (t_next - t_curr).view(B, 1, t_dim, 1, 1) * v

    return z


class GenerationHelper:
    def __init__(self, trainer):
        self.fm = trainer

    @torch.no_grad()
    def extract_clip_emb(self, x_ctx=None, text=None):
        """
        x_ctx can be:
        - uint8 [0,255]
        - float [-1,1]
        - float [0,1]

        Always converted to [0,1] before CLIP.
        """

        def normalize_to_01(x):
            if x.dtype == torch.uint8:
                x = x.float()

            xmin = x.min()
            xmax = x.max()

            if xmax > 1.5:
                x = x / 255.0
            elif xmin < 0:
                x = (x + 1.0) / 2.0

            return torch.clamp(x, 0.0, 1.0)

        if text is not None and text != "":
            text_inputs = self.fm.clip_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self.fm.args.device)

            clip_emb = self.fm.clip.get_text_features(**text_inputs)
            if hasattr(clip_emb, "pooler_output"):
                clip_emb = clip_emb.pooler_output

        elif x_ctx is not None:
            if x_ctx.dim() == 5:
                imgs = x_ctx[:, :, 0]
            else:
                imgs = x_ctx

            imgs = normalize_to_01(imgs)
            imgs = imgs.permute(0, 2, 3, 1)

            imgs = self.fm.clip_processor(images=imgs, return_tensors="pt")
            imgs = imgs["pixel_values"].to(self.fm.args.device)

            clip_emb = self.fm.clip.get_image_features(imgs)
            if hasattr(clip_emb, "pooler_output"):
                clip_emb = clip_emb.pooler_output

        else:
            print("no input for clip")
            clip_emb = None

        return clip_emb

    @torch.no_grad()
    def generate_samples(
        self,
        x_ctx=None,
        trajectory=None,
        nb_video=1,
        latent_context=1,
        num_steps=50,
        text=None,
        cfg_w=0,
        alpha=0.5,
        scheduler_mode="cosine",
        use_clip=True,
        rollout_steps=1,
    ):
        model = self.fm.vit if not self.fm.args.is_multi_gpus else self.fm.vit.module
        model.eval()
        c, t_dim, h, w = self.fm.args.input_size

        all_chunks = []
        if x_ctx is not None:
            nb_video = x_ctx.shape[0]

        if self.fm.args.use_trajectory_cond:
            if trajectory is None:
                raise ValueError("Trajectory conditioning is enabled, but no trajectory was provided for sampling.")
            trajectory = trajectory.to(self.fm.args.device)
            if trajectory.dim() != 3 or trajectory.shape[2] != 2:
                raise ValueError(
                    f"Expected trajectory sampling shape (B, T, 2), got {tuple(trajectory.shape)}"
                )
            if trajectory.shape[0] != nb_video:
                raise ValueError(
                    f"Trajectory batch size {trajectory.shape[0]} does not match sampling batch size {nb_video}"
                )

            expected_rollout_traj_len = self.fm.args.trajectory_length
            if trajectory.shape[1] == expected_rollout_traj_len:
                trajectory_chunks = [trajectory] * rollout_steps
            elif trajectory.shape[1] == expected_rollout_traj_len * rollout_steps:
                trajectory_chunks = list(trajectory.split(expected_rollout_traj_len, dim=1))
            else:
                raise ValueError(
                    "Trajectory sampling length must match either one rollout window "
                    f"({expected_rollout_traj_len}) or the full rollout length "
                    f"({expected_rollout_traj_len * rollout_steps}), got {trajectory.shape[1]}"
                )
        else:
            trajectory = None
            trajectory_chunks = None

        if use_clip:
            if text is not None:
                clip_emb = self.extract_clip_emb(text=text)
            else:
                clip_emb = self.extract_clip_emb(x_ctx=x_ctx[:, :, 0, :, :w // self.fm.args.nb_cam])
        else:
            clip_emb = None

        base_bs = nb_video if clip_emb is None else clip_emb.shape[0]
        clip_emb_zeros = torch.zeros(base_bs, 768, device=self.fm.args.device)

        if x_ctx is not None and latent_context > 0:
            with self.fm.autocast:
                c_ctx = self.fm.ae.encode(x_ctx.clone())[:, :, :latent_context]
        else:
            c_ctx = None
        nb_frames_in_context = max(1 + (4 * (latent_context - 1)), 0)
        for rollout in range(rollout_steps):
            rollout_trajectory = trajectory_chunks[rollout] if trajectory_chunks is not None else None
            z = torch.randn(nb_video, c, t_dim, h, w, device=self.fm.args.device)

            if c_ctx is not None and latent_context > 0:
                z[:, :, :latent_context] = c_ctx

            for i in tqdm(range(num_steps + 1), leave=False, desc=f"Sampling rollout {rollout + 1}/{rollout_steps}"):
                if scheduler_mode == "cosine":
                    progress = 1 - torch.cos(torch.tensor(i / num_steps) * torch.pi / 2)
                    progress_next = 1 - torch.cos(torch.tensor((i + 1) / num_steps) * torch.pi / 2)
                elif scheduler_mode == "square":
                    progress = (i / num_steps) ** 0.5
                    progress_next = ((i + 1) / num_steps) ** 0.5
                else:
                    progress = (i / num_steps)
                    progress_next = ((i + 1) / num_steps)

                offsets = 1 + (torch.linspace(1, 0, t_dim, device=z.device)[None, :] * alpha)
                t_curr = torch.clamp(progress * offsets, 0, 1).expand(nb_video, t_dim)
                t_next = torch.clamp(progress_next * offsets, 0, 1)
                if latent_context > 0:
                    t_curr[:, :latent_context] = 1
                    t_next[:, :latent_context] = 1

                with self.fm.autocast:
                    pred = model(x=z, ada_cond=t_curr, cross_cond=clip_emb, trajectory_cond=rollout_trajectory)
                    if cfg_w > 1:
                        pred_uncond = model(x=z, ada_cond=t_curr, cross_cond=clip_emb_zeros, trajectory_cond=rollout_trajectory)
                        pred = pred_uncond + cfg_w * (pred - pred_uncond)

                if self.fm.args.pred_mode == "x":
                    v = (pred - z) / torch.clamp(1 - t_curr.view(nb_video, 1, t_dim, 1, 1), min=0.05)
                elif self.fm.args.pred_mode == "e":
                    v = (z - pred) / torch.clamp(1 - t_curr.view(nb_video, 1, t_dim, 1, 1), min=0.05)
                else:
                    v = pred

                if latent_context > 0:
                    v[:, :, :latent_context] = 0.0

                z = z + (t_next - t_curr).view(nb_video, 1, t_dim, 1, 1) * v

            with self.fm.autocast:
                video_chunk = self.fm.ae.decode(z)

            if latent_context > 0 and rollout == 0:
                _video_chunk = video_chunk.clone()
            else:
                _video_chunk = video_chunk[:, :, nb_frames_in_context:].clone()

            all_chunks.append(_video_chunk)
            latent_context = max(latent_context, 1)

            if latent_context > 0 and rollout < rollout_steps - 1:
                _x_ctx = video_chunk[:, :, -nb_frames_in_context:]
                _x_ctx = (_x_ctx + 1) / 2.0
                _x_ctx = torch.clamp(_x_ctx, 0.0, 1.0)
                x_ctx_uint8 = (_x_ctx * 255)

                with self.fm.autocast:
                    c_ctx = self.fm.ae.encode(x_ctx_uint8.clone())
                if use_clip:
                    clip_emb = self.extract_clip_emb(
                        x_ctx=x_ctx_uint8[:, :, 0, :, :w // self.fm.args.nb_cam].clone().to(torch.uint8),
                        text=text,
                    )

        video = torch.cat(all_chunks, dim=2)
        return video.float()
