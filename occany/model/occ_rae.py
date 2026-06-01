"""OccRAE: Encode/Decode OccAny via cached backbone layer tokens.

Encode extracts backbone token state at a configurable layer (default 18).
Decode resumes the backbone from that layer through the DPT head, reproducing
the full OccAny output (pointmap, depth, depth_conf, ray, …).

Layer 18 is local attention in DA3-Giant (18 >= alt_start=13, 18 % 2 == 0),
so x == local_x after the block and only a single tensor needs to be stored.
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
        encode_layer: int = 18,
        model_name: str = "da3-giant",
        da3_pretrained_path: Optional[str] = None,
    ): 
        super().__init__()
        if encode_layer != 18 and model_name == "da3-giant":
            raise ValueError(
                f"OccRAE currently supports only encode_layer=18 for da3-giant, got {encode_layer}. "
                "Layer 18 is required because decode assumes local_x == x."
            )
        self.encode_layer = encode_layer

        if weights_path is not None:
            model, checkpoint_args = load_da3_model_from_checkpoint(
                weights_path=weights_path,
                output_resolution=(img_size, img_size),
                semantic_feat_src=None,
                semantic_family=None,
                device=device,
                is_gen_model=False,
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

        export_feat_layers = [self.encode_layer]

        ref_view_strategy = "first"
        _feats, aux_feats = self.model.model.backbone(
            images,
            cam_token=None,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=ref_view_strategy,
        )

        # aux_feats[0] = (processed_feat, raw_state)  where raw_state = (x, local_x)
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
            local_x=None,  # layer 18 is local → x == local_x
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

    @torch.no_grad()
    def decode_to_features(self, latents: Dict[str, object], num_levels: int = 3) -> list:
        """Run decode backbone and return multi-level features (CLS already stripped).

        Returns list of ``num_levels`` tensors, each ``(B*V, N_patches, feature_dim)``.
        Default num_levels=3 returns features at out_layers [19, 27, 33].
        """
        x = latents["tokens"]
        h, w = latents["H"], latents["W"]
        start_layer = self.encode_layer + 1

        feats, _aux = self.model.model.backbone.forward_from_layer(
            x, x,
            start_layer, h, w,
            ref_view_strategy="first",
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
        if self.encode_layer != 18:
            raise ValueError(
                f"Image decoder was trained with encode_layer=18; got {self.encode_layer}."
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
