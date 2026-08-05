"""Shared helpers for trainers built on frozen-OccRAE patch features.

``DeltaTokSharedMixin`` hosts the batch/feature plumbing used by both
``DeltaTokTrainer`` (tokenizer training) and ``DeltaTokFlowMatchingTrainer``
(flow matching over delta tokens). The batch/feature method bodies are copied
verbatim from ``occrae/deltatok_trainer.py``.

The host class must provide: ``self.cfg``, ``self.device``, ``self.is_master``,
``self.distributed``, ``self.autocast``, ``self._num_prefix_tokens``,
``self.train_loader``, a ``_save_current_checkpoint()`` method and the
``pointmap/depth/raymap_criterion`` attributes. ``_build_occ_rae`` sets
``self.occ_rae`` (frozen — never DDP-wrapped, never in the optimizer).
"""

import os
import shutil
import sys
import time

import torch

from occany.model.occ_rae import OccRAE
from occany.utils.slurm import slurm_time_remaining_sec

# Std floor for latent whitening. Delta tokens leave DeltaTok with per-token RMS 1
# over C channels, so a channel with std below this carries no signal; without the
# floor, 1/std would amplify its quantization noise to full scale.
_WHITEN_STD_EPS = 1e-3


class DeltaTokSharedMixin:

    def _end_of_epoch(self, epoch_wall_t0):
        """Per-epoch tail shared by both trainers: persist current.pth on rank 0,
        then maybe stop before the SLURM wall limit for a clean chain resume.
        ``epoch_wall_t0`` is the wall start of the epoch just finished."""
        if self.is_master:
            self._save_current_checkpoint()
            self._maybe_save_epoch_snapshot()
        self._maybe_exit_before_time_limit(epoch_wall_t0)

    def _maybe_save_epoch_snapshot(self):
        """Every ``save_ckpt_every_n_epochs`` epochs keep a permanent copy of the
        just-written current.pth as ``epoch_<N>.pth`` (0 = disabled). Rank 0 only;
        copies current.pth (always the last finite ckpt) so snapshots never hold
        NaN. ``global_epoch`` was already incremented, so it is the epoch count."""
        every = int(self.cfg.training.get("save_ckpt_every_n_epochs", 0))
        if every <= 0:
            return
        epoch = int(self.cfg.training.global_epoch)
        if epoch % every != 0:
            return
        src = os.path.join(self.cfg.training.vit_folder, "current.pth")
        if not os.path.isfile(src):
            return
        dst = os.path.join(self.cfg.training.vit_folder, f"epoch_{epoch}.pth")
        shutil.copyfile(src, dst)
        print(f">> saved periodic checkpoint: {dst}", flush=True)

    def _maybe_exit_before_time_limit(self, epoch_wall_t0):
        """Stop cleanly after the epoch checkpoint when the SLURM wall is near so
        a chained resume picks up current.pth instead of dying mid-epoch. Decide
        on rank 0 and broadcast so all ranks exit together. ``epoch_wall_t0`` is
        the start of the epoch just finished; its duration estimates the next."""
        if not bool(self.cfg.training.get("exit_before_time_limit", False)):
            return
        safety_min = float(self.cfg.training.get("time_limit_safety_min", 2.0))
        stop_run = False
        if self.is_master:
            remaining = slurm_time_remaining_sec()
            if remaining is None:
                # Flag is on but the limit is unknown (no scontrol?): warn once
                # so the silent no-op is visible, then keep training.
                if not getattr(self, "_time_limit_unknown_warned", False):
                    print(">> exit_before_time_limit is on but the SLURM time "
                          "limit could not be determined; the guard will not "
                          "fire.", flush=True)
                    self._time_limit_unknown_warned = True
            else:
                epoch_dur = time.time() - epoch_wall_t0          # est. of the next epoch's wall
                needed = epoch_dur + safety_min * 60.0
                if remaining < needed:
                    print(f">> {remaining / 60:.1f} min left in SLURM allocation < "
                          f"{needed / 60:.1f} min needed (epoch took {epoch_dur / 60:.1f} "
                          f"min + {safety_min:.1f} min safety); exiting cleanly after "
                          f"checkpoint for chain resume.", flush=True)
                    stop_run = True
        if self.distributed:
            stop_flag = torch.tensor([1 if stop_run else 0], device=self.device)
            torch.distributed.broadcast(stop_flag, src=0)
            stop_run = bool(stop_flag.item())
        if stop_run:
            if self.distributed:
                torch.distributed.barrier()
            sys.exit(0)

    def _set_train_loader_epoch(self, epoch: int):
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ds = getattr(self.train_loader, "dataset", None)
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

    def _normalize_batch(self, batch):
        """Convert a list-of-views batch (dust3r convention) to the dict format the
        rest of this trainer expects (with ``imgs``, ``gt_*``, ``num_cameras``...).

        ``get_data_loader`` over ``BaseSeqDatasetMultiView`` returns one collated
        view-dict per timestep/camera; ``num_cameras`` is recovered from the per-
        item ``timestep`` field, which is dense 0..k-1 (views are timestep-major,
        camera-minor).
        """
        if isinstance(batch, dict):
            return batch

        imgs = torch.stack([v["img"] for v in batch], dim=1).contiguous()  # (B, V, C, H, W)
        B, V = imgs.shape[:2]
        H, W = imgs.shape[-2:]

        timesteps_t = torch.stack([v["timestep"] for v in batch], dim=1)  # (B, V)
        item0_t = timesteps_t[0].tolist()
        num_unique_t = len(set(item0_t))
        num_cameras = V // num_unique_t if num_unique_t > 0 else 1

        out = {
            "imgs": imgs,
            "output_resolution_hw": (int(H), int(W)),
            "processed_root": list(batch[0]["dataset"]),
            "timesteps": timesteps_t,
            "num_cameras": num_cameras,
            "scene_name": list(batch[0]["scene_name"]),
            "frame_stems": list(zip(*[v["frame_id"] for v in batch])),
        }

        if "depthmap" in batch[0]:
            gt_depth = torch.stack([v["depthmap"] for v in batch], dim=1).float()
            gt_c2w = torch.stack([v["camera_pose"] for v in batch], dim=1).float()
            gt_intrinsics = torch.stack([v["camera_intrinsics"] for v in batch], dim=1).float()
            gt_raymap = torch.stack([v["gt_raymap"] for v in batch], dim=1).float()
            gt_pointmap = torch.stack([v["pts3d"] for v in batch], dim=1).float()
            if "valid_mask" in batch[0]:
                gt_mask = torch.stack([v["valid_mask"] for v in batch], dim=1)
            else:
                gt_mask = gt_depth > 0
            out.update({
                "gt_depth": gt_depth,
                "gt_intrinsics": gt_intrinsics,
                "gt_c2w": gt_c2w,
                "gt_raymap": gt_raymap,
                "gt_pointmap": gt_pointmap,
                "gt_mask": gt_mask,
            })
        return out

    def _extract_pair_feats(self, imgs, num_cameras=1, return_pairs=True, pair_t=None, gap=1):
        """Run OccRAE.encode and split into (x_prev, x) pair tensors plus shape metadata.

        With ``return_pairs=False`` the (x_prev, x) slots are returned as None —
        for callers (the flow trainer) that only need ``feats`` and would
        otherwise rebuild the same pair inside ``_encode_pair_deltas``.

        ``pair_t`` (B, n_pairs) encodes only the 2 timesteps each kept transition needs
        (exact: blocks 0..encode_layer are per-view local). tokens/feats then cover only
        those views and M = B*n_pairs, so feature-loss/eval callers must leave it None.

        Returns
        -------
        tokens : Tensor (B, V, N, C) — full OccRAE tokens (CLS + registers + patches).
        feats  : Tensor (B, T, num_cameras, P, C) — spatial-only patch features
                 indexed by timestep and camera. Used by the autoregressive eval
                 path.
        x_prev : Tensor (M, num_cameras, P, C) where M = B*(T-1) — spatial-only
                 features for frame_t, with the camera axis kept so DeltaTok can
                 alternate per-camera local and cross-camera global blocks.
        x      : Tensor (M, num_cameras, P, C) — spatial-only features for frame_{t+1}.
        H, W   : ints — image height / width (used to compute the rope grid).
        """
        prefix = self._num_prefix_tokens

        B, V_total, C_img, H, W = imgs.shape
        assert V_total % num_cameras == 0, (
            f"V={V_total} views not divisible by num_cameras={num_cameras}; "
            "a wrong/missing batch['num_cameras'] would silently mispair frames"
        )
        T = V_total // num_cameras

        if pair_t is not None:
            # Transition t spans timesteps t, t+gap; view v = t*num_cameras + cam.
            n_pairs = pair_t.shape[1]                                    # transitions kept per sequence
            steps = torch.stack([pair_t, pair_t + gap], dim=-1)          # (B, n_pairs, 2) timesteps per pair
            cams = torch.arange(num_cameras, device=imgs.device)         # (num_cameras,)
            views = (steps[..., None] * num_cameras + cams).reshape(B, -1)  # (B, n_pairs*2*num_cameras) views to keep
            seqs = torch.arange(B, device=imgs.device)[:, None]          # (B, 1) broadcast against `views`
            imgs = imgs[seqs, views]                                     # (B, n_pairs*2*num_cameras, 3, H, W)
            # Fold pairs into the batch axis: local-only blocks make B vs S immaterial.
            imgs = imgs.reshape(B * n_pairs, 2 * num_cameras, C_img, H, W)  # (M, 2*num_cameras, 3, H, W)
            T = 2

        with torch.no_grad(), self.autocast:
            latents = self.occ_rae.encode(imgs)
        tokens = latents["tokens"]  # (B, T*N_cam, N_tok, C)
        H_out, W_out = int(latents["H"]), int(latents["W"])

        feats = tokens[:, :, prefix:].contiguous()
        if self.cfg.training.dtype == "bfloat16":
            feats = feats.to(torch.bfloat16)

        B, _, P, C = feats.shape
        feats = feats.view(B, T, num_cameras, P, C)
        assert T >= 2, "DeltaTok pair training requires at least 2 timesteps per item"
        if not return_pairs:
            return tokens, feats, None, None, H_out, W_out
        x_prev = feats[:, :-1].reshape(B * (T - 1), num_cameras, P, C)
        x = feats[:, 1:].reshape(B * (T - 1), num_cameras, P, C)
        return tokens, feats, x_prev, x, H_out, W_out

    def _encode_pair_deltas(self, net, feats, height, width):
        """Encode all consecutive GT frame pairs into per-transition delta tokens.

        Factored out of ``DeltaTokTrainer._autoregressive_rollout`` so the flow
        trainer can reuse it with its frozen tokenizer.

        Args:
            net: unwrapped ``DeltaTokModule``.
            feats: spatial-only patch features (B, T, N, P, C) — B items,
                   T timesteps, N cameras, P patches per view, C channels.

        Returns:
            z: delta tokens (B, T-1, N, K, Cz) — K = num_delta_tokens tokens per
               camera per transition, Cz = the tokenizer's ``z_dim`` (= backbone C
               unless the target_channels bottleneck is on). The K axis is always
               present (even at K=1); every consumer is native-K.
        """
        B, T, N, P, C = feats.shape
        assert T >= 2

        x_prev = feats[:, :-1].reshape(B * (T - 1), N, P, C)  # (B*(T-1), N, P, C) frames 0..T-2
        x = feats[:, 1:].reshape(B * (T - 1), N, P, C)        # (B*(T-1), N, P, C) frames 1..T-1

        rope_local = net._compute_rope(height, width, feats.device, feats.dtype)
        rope_global = net._compute_global_rope(height, width, N, feats.device, feats.dtype)

        z = net.encode(x_prev, x, rope_local, rope_global)    # (B*(T-1), N, K, Cz) K delta tokens per camera
        K = z.shape[2]                                        # delta tokens per camera
        z = z.reshape(B, T - 1, N, K, z.shape[-1])            # (B, T-1, N, K, Cz) — Cz may be < backbone C (bottleneck)
        return z

    def _load_whiten_stats(self, num_channels):
        """Load delta-token whitening mean/std from ``model.whiten_stats`` (null -> identity).

        DeltaTok's LayerNorm normalizes each token across channels, leaving per-channel
        std free — so the flow's isotropic noise gives each channel a different SNR.
        Stats come from ``compute_deltatok_latent_stats.py`` in one of two layouts:
        (C,) pooled per-channel, or (K, C) per delta-slot (RAE-style per-element
        stats; camera axis pooled — a leading singleton axis is squeezed away).
        ``num_channels`` is C.
        """
        self._whiten_mean = None                              # None -> the pair below is the identity
        self._whiten_std = None
        path = self.cfg.model.get("whiten_stats", None)
        if not path:
            return

        path = str(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"model.whiten_stats not found: {path}")
        stats = torch.load(path, map_location="cpu", weights_only=False)

        mean = stats["mean"].float()                          # (C,) or (K, C) mean over the train split
        std = stats["std"].float()                            # matching std
        assert mean.shape == std.shape, (
            f"whiten_stats mean {tuple(mean.shape)} vs std {tuple(std.shape)} disagree"
        )
        if mean.dim() == 3 and mean.shape[0] == 1:
            mean, std = mean[0], std[0]                       # (K, C): singleton camera axis == per-slot stats
        assert mean.dim() in (1, 2) and mean.shape[-1] == num_channels, (
            f"whiten_stats shape {tuple(mean.shape)}, expected ({num_channels},) or "
            f"(K, {num_channels}) — computed for a different tokenizer"
        )
        if mean.dim() == 2:
            k_cfg = int(self.cfg.model.deltatok.get("num_delta_tokens", 1))
            assert int(mean.shape[0]) == k_cfg, (
                f"whiten_stats have K={int(mean.shape[0])} delta slots, tokenizer has "
                f"K={k_cfg} — computed for a different tokenizer"
            )

        # Floor before inverting; see _WHITEN_STD_EPS.
        n_floored = int((std < _WHITEN_STD_EPS).sum())
        raw_ratio = float(std.max() / std.min().clamp_min(1e-12))  # pre-floor: matches the stats script's [GATE]
        std = std.clamp_min(_WHITEN_STD_EPS)

        self._whiten_mean = mean.to(self.device)              # (C,) or (K, C); broadcasts over z's leading (B, T-1, N) dims
        self._whiten_std = std.to(self.device)
        self._whiten_stats_path = path                        # provenance -> saved into the flow ckpt

        if self.is_master:
            layout = "per-slot" if mean.dim() == 2 else "per-channel"
            print(f"[INFO] Whitening flow latents ({layout}) with: {path}")
            print(f"[INFO]   C={num_channels} n={stats.get('count')} "
                  f"std min={std.min():.4f} max={std.max():.4f} "
                  f"ratio={raw_ratio:.1f}x |mean|max={mean.abs().max():.4f}")
            if n_floored:
                print(f"[WARN]   {n_floored} channel(s) had std < {_WHITEN_STD_EPS}; floored")
            src = str(stats.get("deltatok_ckpt", ""))
            if src and src != str(self.cfg.model.deltatok_ckpt):
                # Wrong stats file = wrong moments; degrades silently rather than failing.
                print(f"[WARN]   stats are for a DIFFERENT tokenizer than cfg.model.deltatok_ckpt:\n"
                      f"[WARN]     stats: {src}\n"
                      f"[WARN]     cfg:   {self.cfg.model.deltatok_ckpt}")

    def _z_to_flow_latent(self, z):
        """Delta tokens (B, T-1, N, K, C) -> flow ViT latent (B, C, T-1, N, K).

        Whitens first when stats are loaded: channels are LAST here, so (C,) stats
        broadcast per channel and (N, K, C) stats per position. No stats -> plain
        permute (the default).
        """
        m = getattr(self, "_whiten_mean", None)               # getattr: DeltaTokTrainer hosts the mixin but never loads stats
        if m is not None:
            assert m.dim() == 1 or z.shape[3:] == m.shape, (  # (K, C) must match exactly; only the camera axis may broadcast
                f"z slots {tuple(z.shape[3:])} != whiten stats {tuple(m.shape)}"
            )
            dtype = z.dtype                                   # bind before rebinding z; fp32 stats promote the op
            z = ((z - m) / self._whiten_std).to(dtype)        # (B, T-1, N, K, C) per-channel/per-slot z-score
        return z.permute(0, 4, 1, 2, 3).contiguous()          # (B, C, T-1, N, K)

    def _flow_latent_to_z(self, x):
        """Flow ViT latent (B, C, T-1, N, K) -> delta tokens (B, T-1, N, K, C).

        Exact inverse of ``_z_to_flow_latent``, so sampled deltas return to the
        tokenizer's ORIGINAL space and MSEToken stays comparable across the ablation.
        """
        z = x.permute(0, 2, 3, 4, 1).contiguous()             # (B, T-1, N, K, C)
        m = getattr(self, "_whiten_mean", None)
        if m is not None:
            assert m.dim() == 1 or z.shape[3:] == m.shape, (  # same guard as _z_to_flow_latent
                f"z slots {tuple(z.shape[3:])} != whiten stats {tuple(m.shape)}"
            )
            z = (z * self._whiten_std + m).to(x.dtype)        # (B, T-1, N, K, C) back to the tokenizer's scale
        return z

    def _rollout_from_z(self, net, x0, z_seq, height, width, num_cameras):
        """Autoregressive decoder rollout from given per-transition delta tokens.

        Factored out of ``DeltaTokTrainer._autoregressive_rollout``: the decoder
        feeds its own previous prediction back, seeded with the GT first frame.
        ``z_seq`` may come from ``_encode_pair_deltas`` (GT deltas) or from a
        flow-model sample.

        Args:
            net: unwrapped ``DeltaTokModule``.
            x0: GT first-frame patch features (B, N, P, C).
            z_seq: delta tokens (B, T-1, N, K, Cz) — K per camera per transition,
                   Cz = the tokenizer's ``z_dim`` (decode up-projects when needed).

        Returns:
            x_hat: predicted patch features (B*(T-1), N, P, C) for t=1..T-1, in
                   the same layout as the teacher-forced ``x`` returned by
                   ``_extract_pair_feats``.
        """
        B, N, P, C = x0.shape
        num_transitions = z_seq.shape[1]  # T-1
        assert N == num_cameras
        assert z_seq.shape[0] == B and z_seq.shape[2] == N, (
            f"z_seq shape {tuple(z_seq.shape)} vs x0 {tuple(x0.shape)}"
        )

        rope_local = net._compute_rope(height, width, x0.device, x0.dtype)
        rope_global = net._compute_global_rope(height, width, N, x0.device, x0.dtype)

        # x_hat_prev tracks the per-camera "previous frame" the decoder feeds back.
        x_hat_prev = x0  # (B, N, P, C) GT first frame
        x_hats = []
        for t in range(num_transitions):
            z_t = z_seq[:, t].to(x0.dtype)  # (B, N, K, C) deltas for transition t -> t+1
            x_hat_t = net.decode(z_t, x_hat_prev, rope_local, rope_global)  # (B, N, P, C)
            x_hats.append(x_hat_t)
            x_hat_prev = x_hat_t

        return torch.stack(x_hats, dim=1).reshape(B * num_transitions, N, P, C)  # (B*(T-1), N, P, C)

    @torch.no_grad()
    def _compute_frame_losses(self, decoded, batch, view_slice, ray_conf, B, height, width):
        mask = batch["gt_mask"][:, view_slice].to(self.device)
        num_views = mask.shape[1]

        loss_pm, _ = self.pointmap_criterion(
            decoded["pointmap"][:, view_slice].float(),
            batch["gt_pointmap"][:, view_slice].to(self.device).float(),
            mask=mask
        )

        pred_d = decoded["depth"][:, view_slice].float().reshape(B * num_views, 1, height, width)
        gt_d = batch["gt_depth"][:, view_slice].to(self.device).float().reshape(B * num_views, 1, height, width)
        d_mask = mask.reshape(B * num_views, 1, height, width).float()
        loss_d, _ = self.depth_criterion(pred_d, gt_d, confidence=None, mask=d_mask)

        rc = ray_conf[:, view_slice] if ray_conf is not None else None
        loss_ray, _ = self.raymap_criterion(
            decoded["ray"][:, view_slice].float(),
            rc,
            batch["gt_c2w"][:, view_slice].to(self.device).float(),
            batch["gt_intrinsics"][:, view_slice].to(self.device).float(),
            batch["gt_raymap"][:, view_slice].to(self.device).float()
        )

        return loss_pm, loss_d, loss_ray

    def _decode_tokens(self, tokens, height, width, num_cameras=1):
        # One decode over all V views so they share view-0's (t0, cam0) frame, which is
        # the frame the GT pointmap uses (ref_view_strategy="first"). Decoding per-timestep
        # put each timestep in its own frame -> a constant ego-motion offset on multi-camera
        # pointmaps (docs/journal/deltatok_multicam_pointmap_eval_bug_2026-08-04.md).
        # num_cameras is now unused (decoder infers V from tokens); kept for caller parity.
        return self.occ_rae.decode({"tokens": tokens, "H": height, "W": width})

    def _reconstruct_full_tokens(self, tokens, x_hat, B, V, num_cameras=1):
        """Stitch DeltaTok-predicted spatial features back into full-token tensors.

        Timestep 0 keeps its ground-truth tokens (DeltaTok has no prediction for it);
        timesteps 1..T-1 use predicted spatial tokens with the original prefix tokens copied.
        For surround mode, timestep 0's N cameras are all GT context.
        """
        P, C = x_hat.shape[-2:]
        prefix = self._num_prefix_tokens

        if num_cameras > 1:
            T = V // num_cameras
            x_hat = x_hat.reshape(B, T - 1, num_cameras, P, C).to(tokens.dtype)
            tokens_5d = tokens.view(B, T, num_cameras, tokens.shape[2], tokens.shape[3])
            prefix_tokens = tokens_5d[:, 1:, :, :prefix]  # (B, T-1, N, prefix, C)
            predicted_full = torch.cat([prefix_tokens, x_hat], dim=3)  # (B, T-1, N, N_tok, C)
            first_timestep = tokens_5d[:, :1]  # (B, 1, N, N_tok, C)
            result = torch.cat([first_timestep, predicted_full], dim=1)  # (B, T, N, N_tok, C)
            return result.view(B, V, tokens.shape[2], tokens.shape[3])
        else:
            x_hat = x_hat.reshape(B, V - 1, P, C).to(tokens.dtype)
            prefix_tokens = tokens[:, 1:, :prefix]
            predicted_full = torch.cat([prefix_tokens, x_hat], dim=2)  # (B, V-1, N, C)
            first_frame = tokens[:, :1]  # (B, 1, N, C)
            return torch.cat([first_frame, predicted_full], dim=1)  # (B, V, N, C)

    def _make_deltatok_module(self, backbone):
        """Construct a DeltaTokModule from ``cfg.model.deltatok`` + backbone dims.

        Single source of truth for the constructor kwargs so the tokenizer
        trainer and the flow trainer (frozen tokenizer) can never drift apart.
        """
        # Local import: deltatok_trainer imports this module at top level.
        from occrae.deltatok_trainer import DeltaTokModule

        deltatok_cfg = self.cfg.model.get("deltatok", {})
        return DeltaTokModule(
            hidden_size=int(backbone.embed_dim),
            num_heads=int(backbone.num_heads),
            patch_size=int(backbone.patch_size),
            initializer_range=float(deltatok_cfg.get("initializer_range", 0.02)),
            num_hidden_layers=int(deltatok_cfg.get("num_hidden_layers", 12)),
            layer_scale_init=float(deltatok_cfg.get("layer_scale_init", 1e-5)),
            use_qk_norm=bool(deltatok_cfg.get("use_qk_norm", True)),
            use_gated_attn=bool(deltatok_cfg.get("use_gated_attn", True)),
            use_swiglu=bool(deltatok_cfg.get("use_swiglu", True)),
            use_rope_aug=bool(deltatok_cfg.get("use_rope_aug", False)),
            use_camera_rope=bool(deltatok_cfg.get("use_camera_rope", False)),
            mlp_ratio=int(deltatok_cfg.get("mlp_ratio", 4)),
            alt_start=int(deltatok_cfg.get("alt_start", 4)),
            num_delta_tokens=int(deltatok_cfg.get("num_delta_tokens", 1)),
            norm_affine=bool(deltatok_cfg.get("norm_affine", True)),
            z_norm=bool(deltatok_cfg.get("z_norm", True)),
            target_channels=int(deltatok_cfg.get("target_channels", 0)),
            bottleneck_mlp=bool(deltatok_cfg.get("bottleneck_mlp", False)),
        )

    def _build_occ_rae(self):
        self.occ_rae = OccRAE(
            weights_path=self.cfg.model.occany_recon_ckpt,
            device=str(self.device),
            encode_layer=int(self.cfg.model.encode_layer),
        )
        self.occ_rae.eval()
        self.occ_rae.requires_grad_(False)
        if self.cfg.training.dtype == "bfloat16":
            self.occ_rae.to(torch.bfloat16)
        if self.is_master:
            print(f"[INFO] Built shared OccRAE encoder at 518x518 ({self.cfg.training.dtype}), "
                  f"encode_layer={int(self.cfg.model.encode_layer)}")

        # Optional pretrained MAE-style image decoder for eval-time RGB viz.
        img_dec_cfg = self.cfg.model.get("img_decoder", None)
        ckpt_path = img_dec_cfg.get("ckpt_path") if img_dec_cfg is not None else None
        if ckpt_path:
            self.occ_rae.load_image_decoder(
                ckpt_path=str(ckpt_path),
                config_path=str(img_dec_cfg.get("config_path", "third_party/GLD/configs/decoder/ViTXL")),
                hidden_size=int(img_dec_cfg.get("hidden_size", 9216)),
                patch_size=int(img_dec_cfg.get("patch_size", 14)),
                use_ema=bool(img_dec_cfg.get("use_ema", True)),
            )
            if self.cfg.training.dtype == "bfloat16":
                self.occ_rae.img_decoder.to(torch.bfloat16)
            self._img_decoder_view_chunk = int(img_dec_cfg.get("view_chunk_size", 0))
        else:
            self._img_decoder_view_chunk = 0
