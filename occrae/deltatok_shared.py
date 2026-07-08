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
        item ``timestep`` field (views are camera-major within each timestep).
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

        # The vpt the sampler actually drew (carried per view); should equal num_cameras.
        vpt_field = batch[0].get("views_per_timestep")
        sampled_vpt = int(vpt_field[0]) if torch.is_tensor(vpt_field) else (
            int(vpt_field) if vpt_field is not None else None)

        out = {
            "imgs": imgs,
            "output_resolution_hw": (int(H), int(W)),
            "processed_root": list(batch[0]["dataset"]),
            "timesteps": timesteps_t,
            "num_cameras": num_cameras,
            "sampled_vpt": sampled_vpt,
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

    def _extract_pair_feats(self, imgs, num_cameras=1, return_pairs=True):
        """Run OccRAE.encode and split into (x_prev, x) pair tensors plus shape metadata.

        With ``return_pairs=False`` the (x_prev, x) slots are returned as None —
        for callers (the flow trainer) that only need ``feats`` and would
        otherwise rebuild the same pair inside ``_encode_pair_deltas``.

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
            z: delta tokens (B, T-1, N, K, C) — K = num_delta_tokens tokens per
               camera per transition. The K axis is always present (even at K=1);
               every consumer is native-K.
        """
        B, T, N, P, C = feats.shape
        assert T >= 2

        x_prev = feats[:, :-1].reshape(B * (T - 1), N, P, C)  # (B*(T-1), N, P, C) frames 0..T-2
        x = feats[:, 1:].reshape(B * (T - 1), N, P, C)        # (B*(T-1), N, P, C) frames 1..T-1

        rope_local = net._compute_rope(height, width, feats.device, feats.dtype)
        rope_global = net._compute_global_rope(height, width, N, feats.device, feats.dtype)

        z = net.encode(x_prev, x, rope_local, rope_global)    # (B*(T-1), N, K, C) K delta tokens per camera
        K = z.shape[2]                                        # delta tokens per camera
        z = z.reshape(B, T - 1, N, K, C)                      # (B, T-1, N, K, C) — K axis kept even at K=1
        return z

    @staticmethod
    def _z_to_flow_latent(z):
        """Delta tokens (B, T-1, N, K, C) -> flow ViT latent (B, C, T-1, N, K)."""
        return z.permute(0, 4, 1, 2, 3).contiguous()          # (B, C, T-1, N, K)

    @staticmethod
    def _flow_latent_to_z(x):
        """Flow ViT latent (B, C, T-1, N, K) -> delta tokens (B, T-1, N, K, C)."""
        return x.permute(0, 2, 3, 4, 1).contiguous()          # (B, T-1, N, K, C)

    def _rollout_from_z(self, net, x0, z_seq, height, width, num_cameras):
        """Autoregressive decoder rollout from given per-transition delta tokens.

        Factored out of ``DeltaTokTrainer._autoregressive_rollout``: the decoder
        feeds its own previous prediction back, seeded with the GT first frame.
        ``z_seq`` may come from ``_encode_pair_deltas`` (GT deltas) or from a
        flow-model sample.

        Args:
            net: unwrapped ``DeltaTokModule``.
            x0: GT first-frame patch features (B, N, P, C).
            z_seq: delta tokens (B, T-1, N, K, C) — K per camera per transition.

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
        if num_cameras <= 1:
            return self.occ_rae.decode({"tokens": tokens, "H": height, "W": width})

        B, V, N_tok, C = tokens.shape
        T = V // num_cameras
        tokens_by_t = tokens.view(B, T, num_cameras, N_tok, C)
        decoded_parts = []
        for t in range(T):
            decoded_parts.append(
                self.occ_rae.decode({"tokens": tokens_by_t[:, t], "H": height, "W": width})
            )
        decoded = {}
        for key in decoded_parts[0]:
            vals = [d[key] for d in decoded_parts]
            if isinstance(vals[0], torch.Tensor) and vals[0].dim() >= 2:
                decoded[key] = torch.cat(vals, dim=1)
            else:
                decoded[key] = vals[0]
        return decoded

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
