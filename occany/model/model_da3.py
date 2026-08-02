from einops import rearrange

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import os
import numpy as np
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from occany.utils.helpers import convert_depth_to_point_cloud
from occany.model.must3r_blocks.head import SAM3Head


logger = logging.getLogger(__name__)


class DA3Wrapper(DepthAnything3):
    def __init__(self, img_size=518, projection_features='pts3d_local,pts3d,rgb,conf,sam3', **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.projection_features = projection_features
        self.head_sam = None  # Will be initialized by init_sam3_head if SAM3 distillation is enabled
        
        self.slice_layer_idx = None  # Layer index after which recon tokens are sliced in gen mode

        # Remove unused cam_enc and cam_dec modules to save memory
        self._remove_unused_modules()

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        # DepthAnything3 caches self.device; clear it after moving modules so
        # subsequent _get_model_device() calls reflect the real parameter device.
        self.device = None
        return module

    def _remove_unused_modules(self):
        """Remove unused cam_enc and cam_dec modules to save ~59M parameters."""
        if hasattr(self.model, 'cam_enc') and self.model.cam_enc is not None:
            del self.model.cam_enc
            self.model.cam_enc = None
        if hasattr(self.model, 'cam_dec') and self.model.cam_dec is not None:
            del self.model.cam_dec
            self.model.cam_dec = None

    def _get_backbone_wrapper(self):
        """Return the DA3 DinoV2 wrapper, unwrapping PEFT when needed."""
        backbone = self.model.backbone
        if hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'model'):
            backbone = backbone.base_model.model
        return backbone

    def _get_pretrained_backbone(self):
        """Return the underlying DinoVisionTransformer instance."""
        backbone = self._get_backbone_wrapper()
        pretrained_backbone = getattr(backbone, 'pretrained', None)
        if pretrained_backbone is None:
            raise RuntimeError('DA3 backbone does not expose a pretrained DinoV2 backbone')
        return pretrained_backbone

    def _log_backbone_metadata(self, prefix: str) -> Dict[str, object]:
        """Print active DA3 backbone metadata for easier debugging."""
        metadata = self.get_backbone_metadata()
        print(
            f"{prefix}: name={metadata['name']}, token_dim={metadata['token_dim']}, "
            f"feature_dim={metadata['feature_dim']}, out_layers={list(metadata['out_layers'])}, "
            f"alt_start={metadata['alt_start']}, total_layers={metadata['total_layers']}"
        )
        return metadata

    def set_slice_layer(self, layer_idx):
        """Set layer index where recon-token slicing is applied in gen mode."""
        self.slice_layer_idx = layer_idx
        print(f"Set slice_layer_idx to {layer_idx}")
        return self

    def set_alt_start(self, alt_start: int):
        """Change alt_start for backbone (camera_token already exists)."""
        backbone = self._get_backbone_wrapper()
        pretrained_backbone = self._get_pretrained_backbone()
        backbone.alt_start = alt_start
        pretrained_backbone.alt_start = alt_start
        print(f"Set backbone alt_start to {alt_start}")
        return self

    def get_backbone_metadata(self) -> Dict[str, object]:
        """Return runtime metadata for the currently loaded DA3 backbone."""
        backbone = self._get_backbone_wrapper()
        pretrained_backbone = self._get_pretrained_backbone()

        out_layers = tuple(int(layer_idx) for layer_idx in getattr(backbone, 'out_layers', ()))
        if len(out_layers) == 0:
            raise RuntimeError('DA3 backbone does not define out_layers')

        token_dim = getattr(pretrained_backbone, 'embed_dim', None)
        if token_dim is None:
            raise RuntimeError('DA3 pretrained backbone does not define embed_dim')

        blocks = getattr(pretrained_backbone, 'blocks', None)
        if blocks is None:
            raise RuntimeError('DA3 pretrained backbone does not expose transformer blocks')

        cat_token = bool(getattr(pretrained_backbone, 'cat_token', getattr(backbone, 'cat_token', False)))
        alt_start = getattr(pretrained_backbone, 'alt_start', getattr(backbone, 'alt_start', None))
        if alt_start is None:
            raise RuntimeError('DA3 backbone does not define alt_start')

        return {
            'name': getattr(backbone, 'name', pretrained_backbone.__class__.__name__),
            'token_dim': int(token_dim),
            'feature_dim': int(token_dim) * (2 if cat_token else 1),
            'out_layers': out_layers,
            'alt_start': int(alt_start),
            'total_layers': len(blocks),
            'num_heads': int(getattr(pretrained_backbone, 'num_heads', 16)),
            'cat_token': cat_token,
        }

    def init_sam3_head(self, img_size=518, embed_dim=256, patch_size=14, device=None, use_dpt_proj=False):
        """
        Initialize SAM3Head for distillation.
        
        Args:
            img_size: Input image size (default: 518 for DA3)
            embed_dim: SAM3 feature embedding dimension (default: 256)
            patch_size: Patch size for DinoV2 backbone (default: 14)
            device: Device to place the head on
            use_dpt_proj: Use DPT-style projection instead of Mlp (default: False)
            
        Returns:
            self for method chaining
        """
        print('Initializing SAM3Head for distillation...')
        print(f'  - img_size: {img_size}')
        print(f'  - embed_dim: {embed_dim}')
        print(f'  - patch_size: {patch_size}')
        print(f'  - use_dpt_proj: {use_dpt_proj}')

        backbone_metadata = self._log_backbone_metadata('[INFO] Active DA3 backbone for SAM3 head')
        out_layers = backbone_metadata['out_layers']
        feature_dim = backbone_metadata['feature_dim']

        if use_dpt_proj and len(out_layers) != 4:
            raise ValueError(
                f'SAM3 DPT projection expects exactly 4 backbone output layers, got {len(out_layers)}: {list(out_layers)}'
            )
        
        if use_dpt_proj:
            # DPT mode: use one feature width per exported layer.
            input_dims = tuple(feature_dim for _ in out_layers)
            print(f'  - input_dims: {input_dims}')
            
            self.head_sam = SAM3Head(
                input_dim=None,        # Not used in DPT mode
                input_dims=input_dims,
                img_size=img_size,
                embed_dim=embed_dim,
                patch_size=patch_size,
                use_dpt_proj=True,
            )
        else:
            # Current behavior: concatenate all 4 layers
            backbone_embed_dim = feature_dim * len(out_layers)
            print(f'  - backbone_embed_dim: {backbone_embed_dim}')
            
            self.head_sam = SAM3Head(
                input_dim=backbone_embed_dim,
                img_size=img_size,
                embed_dim=embed_dim,
                patch_size=patch_size,
                use_dpt_proj=False,
            )
        
        if device is not None:
            self.head_sam = self.head_sam.to(device)
        
        return self
    
    def forward_sam_features(self, backbone_features, img_shape):
        """Forward pass through SAM3Head to get features for distillation."""
        if self.head_sam is None:
            raise RuntimeError("SAM3Head not initialized. Call init_sam3_head() first.")
        
        # Extract main features from (feat, cam_token) tuples
        feats = [f[0] if isinstance(f, (list, tuple)) else f for f in backbone_features]
        B, T, N, C = feats[0].shape
        
        if self.head_sam.use_dpt_proj:
            # DPT mode: pass features separately as list
            # Flatten B and T for each layer: (B, T, N, C) -> (BT, N, C)
            feats_flat = [f.flatten(0, 1) for f in feats]  # List of 4 tensors, each [BT, N, C]
            sam_outputs = self.head_sam(feats_flat, img_shape)
        else:
            # Current Mlp mode: concatenate all features
            concat_feat = torch.cat(feats, dim=-1)  # (B, T, N, C*4)
            # SAM3Head expects list of [BT, N, C]
            sam_outputs = self.head_sam([concat_feat.flatten(0, 1)], img_shape)
        
        # Restore view dimension: (BT, ...) -> (B, T, ...)
        return tuple(out.reshape(B, T, *out.shape[1:]) for out in sam_outputs)

    
    def init_aux_branch(self, n_layers=6):
        """
        Initialize aux branch by duplicating the last n layers.
        The aux branch is frozen and takes input from layer (depth - n - 1).
        
        Args:
            n_layers: Number of final backbone layers to duplicate
        
        Returns:
            self for method chaining
        """
        import copy
        backbone_metadata = self.get_backbone_metadata()
        total_layers = backbone_metadata['total_layers']
        if n_layers <= 0:
            raise ValueError(f'n_layers must be positive, got {n_layers}')
        if n_layers > total_layers:
            raise ValueError(f'n_layers={n_layers} exceeds backbone depth={total_layers}')

        self.aux_n_layers = n_layers
        self.aux_input_layer_idx = total_layers - n_layers - 1
        
        # Get the pretrained backbone blocks
        backbone_blocks = self._get_pretrained_backbone().blocks
        
        # Duplicate last n layers
        self.aux_blocks = nn.ModuleList([
            copy.deepcopy(backbone_blocks[total_layers - n_layers + i])
            for i in range(n_layers)
        ])
        
        # Duplicate the head for aux predictions
        self.aux_head = copy.deepcopy(self.model.head)
        
        # Freeze all aux parameters
        for param in self.aux_blocks.parameters():
            param.requires_grad = False
        for param in self.aux_head.parameters():
            param.requires_grad = False
        
        print(f'Initialized aux branch with {n_layers} frozen layers (input from layer {self.aux_input_layer_idx})')
        return self
    
    def inference_batch_individual(self, images):
        """
        Per-image teacher forward pass producing pseudo depth.

        Runs the frozen shared backbone prefix (layers 0..aux_input_layer_idx)
        followed by the frozen copied tail (``aux_blocks``) and the frozen copied
        head (``aux_head``). All views are processed independently with S=1
        (single-image batches), so cross-view attention plays no role here and
        the output is suitable as a per-frame metric-depth teacher target.

        Args:
            images: (B, T, 3, H, W) input images.

        Returns:
            depth: (B, T, H, W) pseudo depth.
        """
        if not hasattr(self, 'aux_blocks'):
            raise RuntimeError("Aux branch not initialized. Call init_aux_branch() first.")

        B, T, C, H, W = images.shape
        # Reshape to per-image batches with S=1 so the teacher pass is purely
        # per-frame and never reorders or mixes views.
        images_per = images.reshape(B * T, 1, C, H, W)

        backbone = self._get_pretrained_backbone()
        out_layers = self.get_backbone_metadata()['out_layers']
        boundary = self.aux_input_layer_idx  # last prefix block index
        n_tail = len(self.aux_blocks)
        total_layers = boundary + 1 + n_tail

        # Patchify and prepare RoPE.
        x = backbone.prepare_tokens_with_masks(images_per)
        BS, S = x.shape[0], x.shape[1]
        pos, pos_nodiff = backbone._prepare_rope(BS, S, H, W, x.device)
        local_x = x

        collected = []  # list of (feat, camera_token) for out_layers
        for layer_idx in range(total_layers):
            if layer_idx <= boundary:
                blk = backbone.blocks[layer_idx]
            else:
                blk = self.aux_blocks[layer_idx - boundary - 1]

            if layer_idx < backbone.rope_start or backbone.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            # alt_start camera-token swap. With S=1 the src_token slice is
            # empty, so the concatenation reduces to the ref token.
            if backbone.alt_start != -1 and layer_idx == backbone.alt_start:
                ref_token = backbone.camera_token[:, :1].expand(BS, -1, -1)
                src_token = backbone.camera_token[:, 1:].expand(BS, S - 1, -1)
                cam_token = torch.cat([ref_token, src_token], dim=1)
                x = x.clone()
                x[:, :, 0] = cam_token

            if backbone.alt_start != -1 and layer_idx >= backbone.alt_start and layer_idx % 2 == 1:
                x = self._process_attention_aux(x, blk, "global", pos=g_pos)
            else:
                x = self._process_attention_aux(x, blk, "local", pos=l_pos)
                local_x = x

            if layer_idx in out_layers:
                out_x = torch.cat([local_x, x], dim=-1) if backbone.cat_token else x
                cam_token_out = out_x[:, :, 0]
                # Strip cls / register tokens, then norm following the same
                # cat_token convention used by get_intermediate_layers.
                patch_x = out_x[..., 1 + backbone.num_register_tokens:, :]
                if backbone.cat_token:
                    first_half = patch_x[..., :backbone.embed_dim]
                    second_half = backbone.norm(patch_x[..., backbone.embed_dim:])
                    normed = torch.cat([first_half, second_half], dim=-1)
                else:
                    normed = backbone.norm(patch_x)
                collected.append((normed, cam_token_out))

        if len(collected) != len(out_layers):
            raise RuntimeError(
                f"Teacher pass collected {len(collected)} feature maps but "
                f"backbone out_layers={list(out_layers)} expects {len(out_layers)}."
            )

        output = self._process_depth_output(
            feats=collected,
            h=H,
            w=W,
            device_type=images.device.type,
            head=self.aux_head,
        )

        depth = output["depth"]  # (B*T, 1, H, W)
        return depth.reshape(B, T, H, W)

    def _process_attention_aux(self, x, block, attn_type="global", pos=None, attn_mask=None):
        """
        Process attention in aux blocks. Mimics vision_transformer.py process_attention.
        """
        b, s, n = x.shape[:3]
        
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")
        
        x = block(x, pos=pos, attn_mask=attn_mask)
        

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        
        return x

    def forward(self, images=None, **kwargs):
        if images is None:
            raise ValueError("'images' must be provided")
        return self.inference_batch(images, **kwargs)


    def _process_ray_pose_estimation(
        self, ray: torch.Tensor, ray_conf: torch.Tensor, height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        try:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                ray,
                ray_conf.clone(),  # Clone to prevent in-place modification in compute_optimal_rotation_intrinsics_batch
                ray.shape[-3],
                ray.shape[-2],
            )
        except (ValueError, RuntimeError):
            # Degenerate rays (untrained model / sanity check) make the homography
            # ill-posed (<4 RANSAC inliers or non-convergent SVD); return identity
            # poses so eval/viz still runs instead of aborting the job.
            B, V = ray.shape[0], ray.shape[1]  # B batch, V views
            c2w = torch.eye(4, device=ray.device, dtype=torch.float32)[None, None, :3, :].expand(B, V, 3, 4).contiguous()  # (B, V, 3, 4)
            intrinsics = torch.eye(3, device=ray.device, dtype=torch.float32)[None, None].expand(B, V, 3, 3).contiguous()  # (B, V, 3, 3)
            return c2w, intrinsics
        pred_extrinsic = pred_extrinsic[:, :, :3, :]
        pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
        pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
        pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
        pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
        pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
        return pred_extrinsic, pred_intrinsic
        
    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
   
        pose_enc = self.cam_dec(feats[-1][1])
        c2w, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))

        return c2w, intrinsics

    def _process_depth_output(
        self,
        feats,
        h,
        w,
        device_type,
        pose_from_depth_ray=False,
        pose_from_cam_dec=False,
        point_from_depth_and_pose=False,
        images=None,
        head=None,  # Optional: use specific head (e.g., frozen aux_head)
    ):
        """
        Process depth head output and compute pointmap, pose, and SAM features.
        
        Args:
            feats: Features from backbone
            h, w: Image height and width
            device_type: Device type for autocast (e.g., 'cuda')
            pose_from_depth_ray: Whether to estimate pose from depth and ray
            pose_from_cam_dec: Whether to estimate pose from camera decoder
            
        Returns:
            Dict with pointmap, depth, depth_conf, ray, ray_conf, c2w, intrinsics, sam_feats
        """
        # Use provided head or default to main head
        depth_head = head if head is not None else self.model.head
        output = depth_head(feats, h, w, patch_start_idx=0)

        default_scale = 20
        # Extract depth and raymap from raw output
        depth = output["depth"]  # (B, T, H_proc, W_proc)
        depth = depth * default_scale
        depth_conf = output["depth_conf"]  # (B, T, H_proc, W_proc)

        # Predicted raymap from the model
        ray = output["ray"]  # (B, T, H_proc, W_proc, 6) with [ray_dirs(3), ray_origins(3)]
        ray_conf = output["ray_conf"]  # (B, T, H_proc, W_proc)
        ray[..., 3:] = ray[..., 3:] * default_scale  # scale ray origins

        c2w = None
        intrinsics = None
        if pose_from_depth_ray:
            # Cast to float32 to avoid torch.inverse() error with bf16/fp16
            with torch.autocast(device_type=device_type, enabled=False):
                c2w, intrinsics = self._process_ray_pose_estimation(ray.float(), ray_conf.float(), h, w)
    
        if pose_from_cam_dec:
            c2w, intrinsics = self._process_camera_estimation(feats, h, w, output)
         
        if point_from_depth_and_pose:
            assert intrinsics is not None and c2w is not None, "Intrinsics and c2w must be estimated to compute pointmap from depth and pose"
            pointmap = convert_depth_to_point_cloud(depth, intrinsics, c2w)
        else:    
            # DA3 paper: "We do not normalize d, so its magnitude preserves the projection scale.
            # Thus, a 3D point in world coordinates is simply P = t + D(u, v) · d"
            # where t = ray origin, D = depth, d = unnormalized ray direction
            pointmap = depth.unsqueeze(-1) * ray[..., :3] + ray[..., 3:]
            
        # Compute SAM features inside forward pass (important for DDP to avoid "marked ready twice" error)
        sam_feats = None
        if self.head_sam is not None:
            sam_feats = self.forward_sam_features(feats, (h, w))

        save_outputs = False
        if save_outputs:
            # Auto-incrementing folder logic for debug output saving
            base_output_dir = os.environ.get("DA3_DEBUG_OUTPUT", "debug_output/da3")
            os.makedirs(base_output_dir, exist_ok=True)
            
            # Find next available folder number (00000, 00001, ...)
            folder_idx = 0
            
            output_folder = os.path.join(base_output_dir, f"{folder_idx:05d}")
            os.makedirs(output_folder, exist_ok=True)
            
            # Prepare data for saving
            # Extract raw data (no scaling yet)
            batch_idx = 0
            pts3d_render = pointmap[batch_idx].detach().cpu()  # (T, H, W, 3)
            conf_render = depth_conf[batch_idx].detach().cpu()  # (T, H, W)
            gt_depths = depth[batch_idx].detach().cpu()  # (T, H, W)
            
            # Intrinsics and c2w
            if intrinsics is not None and c2w is not None:
                focal = torch.stack([intrinsics[batch_idx, :, 0, 0], intrinsics[batch_idx, :, 1, 1]], dim=-1).detach().cpu()  # (T, 2)
                c2w_save = c2w[batch_idx].detach().cpu()  # (T, 3, 4) camera-to-world
                
                # Create save dictionary
                save_dict = {
                    "pts3d": pts3d_render.numpy(),
                    "conf": conf_render.numpy(),
                    "gt_depths": gt_depths.numpy(),
                    "focal": focal[:, 0].numpy(),
                    "c2w": c2w_save.numpy(),
                }
                
                # Add colors if images are available
                if images is not None:
                    # images: (B, T, C, H, W) -> extract batch_idx and convert to (T, H, W, 3)
                    colors = images[batch_idx].detach().cpu()  # (T, C, H, W)
                    colors = colors.permute(0, 2, 3, 1)  # (T, H, W, C)
                    # ImageNet un-normalization to [0, 1], then map to [-1, 1]
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])
                    colors = (colors * std + mean) * 2.0 - 1.0
                    colors = colors.clamp(-1, 1)
                    save_dict["colors"] = colors.numpy()
                
                # Save to .npy file
                save_path = os.path.join(output_folder, "pts3d_render.npy")
                np.save(save_path, save_dict)
                print(f"Saved pts3d_render.npy to {save_path}")
              

        return {
            "pointmap": pointmap,
            "depth": depth,
            "depth_conf": depth_conf,
            "ray": ray,
            "ray_conf": ray_conf,
            "c2w": c2w,
            "intrinsics": intrinsics,
            "sam_feats": sam_feats,
        }

    def inference_batch(
        self,
        images,
        export_feat_layers=None,
        process_res=518,
        process_res_method="upper_bound_resize",
        save_outputs=False,
        save_prefix="da3",
        pose_from_depth_ray=False,
        pose_from_cam_dec=False,
        point_from_depth_and_pose=False,
    ):
        b, t, c, h, w = images.size()

        device = self._get_model_device()

        # Prepare export_feat_layers
        if export_feat_layers is None:
            export_feat_layers = list(self.get_backbone_metadata()['out_layers'])
        else:
            export_feat_layers = list(export_feat_layers)

        # Keep export layer order consistent with backbone traversal order.
        export_feat_layers = sorted(set(export_feat_layers))

        # Call the underlying model directly to get raw output including ray field
        # before any post-processing deletes it.
        # The DualDPT head outputs 'ray' (B, T, H, W, 6) with [ray_dirs(3), ray_origins(3)]
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        ref_view_strategy = "first"

        device_type = images.device.type
        feats, aux_feats = self.model.backbone(
            images,
            cam_token=None,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=ref_view_strategy
        )

        output = self._process_depth_output(
            feats=feats,
            h=h,
            w=w,
            device_type=device_type,
            pose_from_depth_ray=pose_from_depth_ray,
            pose_from_cam_dec=pose_from_cam_dec,
            point_from_depth_and_pose=point_from_depth_and_pose,
            images=images,
        )

        if aux_feats is not None:
            output["aux_feats"] = aux_feats

        return output

    @torch.no_grad()
    def encode_to_layer(self, images: torch.Tensor, encode_layer: int) -> torch.Tensor:
        """Partial backbone forward: patchify + run local blocks 0..encode_layer.

        Returns the same token state the full forward would snapshot via
        ``export_feat_layers=[encode_layer]`` (the raw, un-normed ``x``), but
        without running blocks ``encode_layer+1..end`` — which is the whole point,
        since those include the expensive global blocks (13/15/17/19/...).

        Only valid for a pre-fusion local layer (``encode_layer < alt_start``):
        every block run here is local attention, so there is nothing to replicate
        from the global regime — the camera-token swap (``i == alt_start``),
        reference-view selection (``i == alt_start - 1``), and global attention
        (``i >= alt_start and i % 2 == 1``) are all gated past this point. (The
        ref-view reorder is also a no-op under OccRAE's ``ref_view_strategy="first"``.)

        images : (B, S, 3, H, W)
        returns x : (B, S, N, C) — token state with cls/register tokens at the
                    front; equals ``local_x`` at this layer.
        """
        backbone = self._get_pretrained_backbone()
        if backbone.alt_start != -1 and encode_layer >= backbone.alt_start:
            raise ValueError(
                f"encode_to_layer only supports pre-fusion local layers "
                f"(< alt_start={backbone.alt_start}); got {encode_layer}. Layers "
                f">= alt_start need the camera-token/ref-view/global machinery."
            )
        B, S = images.shape[0], images.shape[1]
        H, W = images.shape[-2], images.shape[-1]
        x = backbone.prepare_tokens_with_masks(images)          # (B, S, N, C)
        pos, _pos_nodiff = backbone._prepare_rope(B, S, H, W, x.device)  # local rope (B, S, N, 2)
        for i in range(encode_layer + 1):
            l_pos = None if (i < backbone.rope_start or backbone.rope is None) else pos
            x = backbone.process_attention(x, backbone.blocks[i], "local", pos=l_pos)  # (B, S, N, C)
        return x

    def _maybe_inject_camera_token(self, x: torch.Tensor, start_layer: int) -> torch.Tensor:
        """Replicate the encode-path camera-token swap when resuming *at* alt_start.

        The full forward injects learned camera tokens into the CLS slot exactly
        at ``i == alt_start`` (vision_transformer.py: ``x[:, :, 0] = cam_token``)
        right before the first global block. The decode-resume helper
        ``forward_from_layer`` omits this, which is fine when caching post-fusion
        (layer 18 -> resume at 19, token already baked in) but wrong when caching
        a pre-fusion local layer (alt_start-1 -> resume at alt_start). This is a
        no-op unless ``start_layer == alt_start``, so the legacy layer-18 path is
        unaffected. Mirrors inference_batch_individual's swap (lines ~323-328).

        x : (B, S, N, C) cached token state with cls/register tokens at the front.
        """
        backbone = self._get_pretrained_backbone()
        if backbone.alt_start == -1 or start_layer != backbone.alt_start:
            return x
        B, S = x.shape[:2]
        ref_token = backbone.camera_token[:, :1].expand(B, -1, -1)   # (B, 1, C)
        src_token = backbone.camera_token[:, 1:].expand(B, S - 1, -1)  # (B, S-1, C)
        cam_token = torch.cat([ref_token, src_token], dim=1).to(x.dtype)  # (B, S, C)
        x = x.clone()
        x[:, :, 0] = cam_token  # overwrite cls slot per view -> (B, S, N, C)
        return x

    def inference_batch_from_layer(
        self,
        x: torch.Tensor,
        start_layer: int,
        h: int,
        w: int,
        local_x: torch.Tensor = None,
        export_feat_layers=None,
        pose_from_depth_ray=False,
        pose_from_cam_dec=False,
        point_from_depth_and_pose=False,
    ):
        """Resume inference from a cached backbone layer state.

        Args:
            x: (B, S, N, embed_dim) token state after ``start_layer - 1`` (with cls).
            start_layer: block index to resume from (e.g. 19).
            h, w: original image height and width.
            local_x: optional local-attention state.  When *None*, assumed
                      equal to *x* (valid when the previous layer was local attention).
            export_feat_layers: layers to export raw states from (default: out_layers).
            pose_from_depth_ray: estimate pose from depth/ray.
            pose_from_cam_dec: estimate pose from camera decoder.
            point_from_depth_and_pose: compute pointmap from depth+pose.

        Returns:
            Same dict as ``inference_batch``.
        """
        if local_x is None:
            local_x = x  # (B, S, N, C); keep the pre-injection local stream

        # Resuming at alt_start needs the camera-token swap the encode path does;
        # local_x stays un-injected (cam token only enters the global stream x).
        x = self._maybe_inject_camera_token(x, start_layer)  # (B, S, N, C)

        if export_feat_layers is None:
            export_feat_layers = list(self.get_backbone_metadata()['out_layers'])
        else:
            export_feat_layers = list(export_feat_layers)

        # Only keep export layers that are >= start_layer
        export_feat_layers = sorted(l for l in set(export_feat_layers) if l >= start_layer)

        device_type = x.device.type
        ref_view_strategy = "first"

        feats, aux_feats = self.model.backbone.forward_from_layer(
            x, local_x, start_layer, h, w,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=ref_view_strategy,
        )

        output = self._process_depth_output(
            feats=feats,
            h=h,
            w=w,
            device_type=device_type,
            pose_from_depth_ray=pose_from_depth_ray,
            pose_from_cam_dec=pose_from_cam_dec,
            point_from_depth_and_pose=point_from_depth_and_pose,
        )

        if aux_feats is not None:
            output["aux_feats"] = aux_feats

        return output
