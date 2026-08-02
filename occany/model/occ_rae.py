"""OccRAE: Encode/Decode OccAny via cached backbone layer tokens.

Encode extracts backbone token state at a configurable layer.
Decode resumes the backbone from that layer through the DPT head, reproducing
the full OccAny output (pointmap, depth, depth_conf, ray, …).

Two cache points are supported for DA3-Giant:
  * encode_layer=12 (default): pre-fusion local cache at alt_start-1. Encode is
    a partial forward (patchify + local blocks 0..12), skipping all global
    blocks; decode resumes at alt_start=13 re-applying the camera-token swap.
  * encode_layer=18: post-fusion local cache (18 >= alt_start=13, 18 % 2 == 0).
    Encode runs the full backbone with export_feat_layers=[18] and snapshots
    the raw token state; decode resumes at 19 with the camera token already
    baked in (camera-token swap is a no-op).

In both cases x == local_x at the cached layer, so only a single tensor is
stored.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from occany.utils.io_da3 import load_da3_model_from_checkpoint


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class OccRAE(nn.Module):
    """Representation AutoEncoder wrapper around a DA3Wrapper model.

    Parameters
    ----------
    weights_path : str
        Path to the OccAny checkpoint (.pth).
    img_size : int
        DA3 backbone native image size (default 518).
    device : str
        Target device (e.g. ``"cuda"``).
    encode_layer : int
        Backbone layer index at which to capture tokens (default 18).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        img_size: int = 518,
        device: str = "cuda",
        encode_layer: int = 12,
        model_name: str = "da3-giant",
        da3_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        # Both supported cache points (12 pre-fusion, 18 post-fusion) are local
        # blocks at which x == local_x, so a single tensor suffices. Layer 12
        # also skips the global blocks during encode (partial forward).
        if model_name == "da3-giant" and encode_layer not in (12, 18):
            raise ValueError(
                f"OccRAE supports encode_layer in (12, 18) for da3-giant, got {encode_layer}."
            )
        self.encode_layer = encode_layer

        if weights_path is not None:
            model, checkpoint_args = load_da3_model_from_checkpoint(
                weights_path=weights_path,
                output_resolution=(img_size, img_size),
                semantic_feat_src=None,
                semantic_family=None,
                device=device,
            )
            self.model = model
            self.checkpoint_args = checkpoint_args
        else:
            from occany.model.model_da3 import DA3Wrapper
            model = DA3Wrapper(
                model_name=model_name,
                img_size=img_size,
                projection_features="pts3d_local,pts3d,rgb,conf",
            )
            if da3_pretrained_path is not None:
                from safetensors.torch import load_file
                state_dict = load_file(da3_pretrained_path)
                model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded pretrained weights from {da3_pretrained_path}")
            model.to(device).eval()
            model.requires_grad_(False)
            self.model = model
            self.checkpoint_args = None
            print(f"[INFO] Using {model_name} backbone")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
    ) -> Dict[str, object]:
        """Run the backbone through ``encode_layer`` and return cached tokens.

        Parameters
        ----------
        images : Tensor
            Shape ``(B, S, C, H, W)`` – normalised images.

        Returns
        -------
        dict with keys:
            ``tokens``  – ``(B, S, N, embed_dim)`` token state (float32, with cls).
            ``H``, ``W`` – original image height / width (int).
        """
        b, s, c, h, w = images.shape

        backbone = self.model._get_pretrained_backbone()
        is_pre_fusion = backbone.alt_start == -1 or self.encode_layer < backbone.alt_start

        if is_pre_fusion:
            # Partial forward: stop at encode_layer (a pre-fusion local block),
            # so global blocks at encode_layer+1..end never run. Returns the same
            # raw, un-normed token state the full forward would snapshot via
            # export_feat_layers=[encode_layer].
            x = self.model.encode_to_layer(images, self.encode_layer)  # (B, S, N, embed_dim)
        else:
            # Post-fusion cache (e.g. layer 18): full forward, snapshot token
            # state at encode_layer.
            _feats, aux_feats = self.model.model.backbone(
                images,
                cam_token=None,
                export_feat_layers=[self.encode_layer],
                ref_view_strategy="first",
            )
            # aux_feats[0] = (processed_feat, raw_state) where raw_state = (x, local_x)
            raw_state = aux_feats[0][1]
            x = raw_state[0]  # (B, S, N, embed_dim)

        return {"tokens": x, "H": h, "W": w}

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        latents: Dict[str, object],
        pose_from_depth_ray: bool = False,
        pose_from_cam_dec: bool = False,
        point_from_depth_and_pose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Decode cached tokens back to full OccAny output.

        Parameters
        ----------
        latents : dict
            Output of :meth:`encode` (``tokens``, ``H``, ``W``).
        pose_from_depth_ray, pose_from_cam_dec, point_from_depth_and_pose :
            Forwarded to ``DA3Wrapper._process_depth_output``.

        Returns
        -------
        dict – same keys as ``DA3Wrapper.inference_batch`` (pointmap, depth,
        depth_conf, ray, ray_conf, c2w, intrinsics, …).
        """
        x = latents["tokens"]
        h = latents["H"]
        w = latents["W"]
        start_layer = self.encode_layer + 1

        return self.model.inference_batch_from_layer(
            x=x,
            start_layer=start_layer,
            h=h,
            w=w,
            local_x=None,  # cached layer is local → x == local_x
            pose_from_depth_ray=pose_from_depth_ray,
            pose_from_cam_dec=pose_from_cam_dec,
            point_from_depth_and_pose=point_from_depth_and_pose,
        )

    # ------------------------------------------------------------------
    # Decode to multi-level features (for image decoder training)
    # ------------------------------------------------------------------

    @property
    def encoder_mean(self) -> torch.Tensor:
        """ImageNet mean as (1, 3, 1, 1) on the model's device."""
        device = next(self.parameters()).device
        return IMAGENET_MEAN.reshape(1, 3, 1, 1).to(device)

    @property
    def encoder_std(self) -> torch.Tensor:
        """ImageNet std as (1, 3, 1, 1) on the model's device."""
        device = next(self.parameters()).device
        return IMAGENET_STD.reshape(1, 3, 1, 1).to(device)

    def decode_to_features(
        self, latents: Dict[str, object], num_levels: Optional[int] = 3,
        requires_grad: bool = False, use_checkpoint: bool = False,
    ) -> list:
        """Run decode backbone and return multi-level features (CLS already stripped).

        Returns list of ``num_levels`` (shallowest) tensors, each ``(B*V, N_patches,
        feature_dim)``. num_levels=3 returns features at out_layers [19, 27, 33];
        num_levels=4 adds layer 39; ``num_levels=None`` returns every out_layer.
        ``requires_grad=True`` keeps the autograd graph so gradients flow back to
        ``latents['tokens']`` (frozen backbone still backprops); default False preserves
        no-grad behavior for existing callers. ``use_checkpoint=True`` recomputes each
        decode block in backward to cut activation memory (only meaningful with grad).
        """
        x = latents["tokens"]
        h, w = latents["H"], latents["W"]
        start_layer = self.encode_layer + 1

        with torch.set_grad_enabled(requires_grad):
            # Resuming at alt_start needs the camera-token swap (no-op otherwise);
            # local_x (2nd arg) stays the un-injected stream.
            x_inj = self.model._maybe_inject_camera_token(x, start_layer)  # (B, S, N, C)
            feats, _aux = self.model.model.backbone.forward_from_layer(
                x_inj, x,
                start_layer, h, w,
                ref_view_strategy="first",
                use_checkpoint=use_checkpoint,
            )
            return [feat for feat, _cam in feats[:num_levels]]

    # ------------------------------------------------------------------
    # Image decoder (MAE-style, trained in occrae/img_decoder_trainer.py)
    # ------------------------------------------------------------------

    def load_image_decoder(
        self,
        ckpt_path: str,
        config_path: str = "third_party/GLD/configs/decoder/ViTXL",
        hidden_size: int = 9216,
        patch_size: int = 14,
        base_image_size: Tuple[int, int] = (518, 518),
        use_ema: bool = True,
    ) -> "OccRAE":
        """Load a GeneralDecoder_Variable trained on 3-level OccRAE features.

        Mirrors the construction in ``occrae/img_decoder_trainer.py``. Defaults
        match ``configs/train_occrae_img_decoder.yaml``.

        Parameters
        ----------
        ckpt_path : str
            Path to a checkpoint saved by ImgDecoderTrainer (contains
            ``decoder`` and ``ema_decoder`` keys).
        use_ema : bool
            If True (default) load the EMA weights; fall back to live weights
            if EMA missing.
        """
        # The GLD decoder consumes decode-side features (layers 19/27/33),
        # produced by decode_to_features regardless of the cache layer, so any
        # supported encode_layer is valid here.
        if self.encode_layer not in (12, 18):
            raise ValueError(
                f"Image decoder expects encode_layer in (12, 18); got {self.encode_layer}."
            )

        from transformers import AutoConfig
        from stage1.decoders import GeneralDecoder_Variable

        dec_config = AutoConfig.from_pretrained(str(config_path))
        dec_config.hidden_size = int(hidden_size)
        dec_config.patch_size = int(patch_size)

        img_decoder = GeneralDecoder_Variable(dec_config, base_image_size=base_image_size)

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if use_ema and "ema_decoder" in ckpt:
            state_key = "ema_decoder"
        elif "decoder" in ckpt:
            state_key = "decoder"
            if use_ema:
                print("[WARN] 'ema_decoder' missing in checkpoint, falling back to 'decoder'.")
        else:
            raise KeyError(
                f"Checkpoint {ckpt_path} has no 'ema_decoder' or 'decoder' key. "
                f"Available: {list(ckpt.keys())}"
            )
        img_decoder.load_state_dict(ckpt[state_key])

        target_device = next(self.parameters()).device
        img_decoder = img_decoder.to(target_device).eval()
        img_decoder.requires_grad_(False)

        self.img_decoder = img_decoder
        print(
            f"[INFO] Loaded image decoder from {ckpt_path} (key='{state_key}', "
            f"epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')})"
        )
        return self

    @torch.no_grad()
    def decode_to_image(
        self,
        latents: Dict[str, object],
        view_chunk_size: int = 0,
    ) -> torch.Tensor:
        """Decode cached tokens to RGB images via the trained image decoder.

        Parameters
        ----------
        latents : dict
            Output of :meth:`encode` (``tokens``, ``H``, ``W``).
        view_chunk_size : int
            If > 0, process views in chunks of this size (memory parity with
            the trainer). 0 (default) decodes everything in one pass.

        Returns
        -------
        Tensor of shape ``(B, S, 3, H, W)`` in ``[0, 1]`` (un-normalised).
        """
        if getattr(self, "img_decoder", None) is None:
            raise RuntimeError("Image decoder not loaded. Call load_image_decoder() first.")

        tokens = latents["tokens"]
        h, w = int(latents["H"]), int(latents["W"])
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(0)
        b, s = tokens.shape[0], tokens.shape[1]

        feats = self.decode_to_features(
            {"tokens": tokens, "H": h, "W": w}, num_levels=3
        )
        z = torch.cat(feats, dim=-1)  # (B, S, N, D_total)
        z_flat = z.reshape(b * s, z.shape[-2], z.shape[-1])

        encoder_mean = self.encoder_mean.squeeze(0)  # (3,1,1)
        encoder_std = self.encoder_std.squeeze(0)

        chunks = z_flat.split(view_chunk_size, dim=0) if view_chunk_size > 0 else [z_flat]

        use_amp = z_flat.device.type == "cuda"
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else torch.autocast("cpu", enabled=False)
        )

        out_parts = []
        for chunk in chunks:
            with amp_ctx:
                logits = self.img_decoder(
                    chunk, input_size=(h, w), drop_cls_token=False
                ).logits
                rec = self.img_decoder.unpatchify(logits, (h, w))
            rec = rec.float() * encoder_std + encoder_mean
            out_parts.append(rec.clamp(0.0, 1.0))

        rec_all = torch.cat(out_parts, dim=0)
        return rec_all.view(b, s, 3, h, w)
