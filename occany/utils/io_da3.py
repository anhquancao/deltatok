import argparse
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from occany.utils.checkpoint_io import save_on_master
from occany.utils.inference_helper import is_distill_source, uses_sam3_projection_features

def save_model(args, epoch, model, optimizer, loss_scaler, fname=None):
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    optim_state_dict = optimizer.state_dict()
    to_save = {
        'model': model.state_dict(),
        'optimizer': optim_state_dict,
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    
    saved_components = ['model', 'optimizer', 'scaler']
    
    print(f'>> Saving model to {checkpoint_path} ...')
    print(f'   - Saving: {", ".join(saved_components)}')
    save_on_master(to_save, checkpoint_path)


def load_model(args, chkpt_path, model, optimizer, loss_scaler):
    args.start_epoch = 0
    if chkpt_path is not None:
        checkpoint = torch.load(chkpt_path, map_location='cpu', weights_only=False)

        print("Resume checkpoint %s" % chkpt_path)
        
        print('   - Loading: model')
        model.load_state_dict(checkpoint['model'], strict=False)
        args.start_epoch = checkpoint['epoch'] + 1
        
        if 'optimizer' in checkpoint and optimizer is not None:
            optim_state_dict = checkpoint['optimizer']
            try:
                optimizer.load_state_dict(optim_state_dict)
                if 'scaler' in checkpoint and loss_scaler is not None:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')
            except ValueError as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("Starting with fresh optimizer state. Resetting to epoch 0.")
                args.start_epoch = 0
        else:
            print("No optimizer state found in checkpoint. Starting with fresh optimizer state.")


def _resolve_da3_model_name(hf_name: str) -> str:
    """Map a HuggingFace DA3 model name to the local config name.

    Examples:
        "depth-anything/DA3-LARGE"       -> "da3-large"
        "depth-anything/DA3-LARGE-1.1"   -> "da3-large"
        "depth-anything/DA3-GIANT-1.1"   -> "da3-giant"
        "da3-large"                      -> "da3-large"  (passthrough)
    """
    name = hf_name.strip()
    # Already a local config name
    if name.startswith("da3-"):
        return name
    # HF format: "depth-anything/DA3-GIANT-1.1" -> extract "DA3-GIANT"
    basename = name.split("/")[-1]
    # Remove trailing version suffixes like "-1.1"
    parts = basename.split("-")
    # Reconstruct: take "DA3" and the size part (LARGE, GIANT, etc.)
    if len(parts) >= 2 and parts[0].upper() == "DA3":
        size = parts[1].lower()
        return f"da3-{size}"
    # For metric models: "DA3METRIC-LARGE" -> "da3metric-large"
    return basename.lower()


def load_da3_model_from_checkpoint(
    weights_path: str,
    output_resolution: Tuple[int, int],
    semantic_feat_src: Optional[str],
    semantic_family: Optional[str],
    device: str,
    projection_features_override: Optional[str] = None,
) -> Tuple[Any, argparse.Namespace]:
    """Load DA3 model and initialize optional SAM3/gen heads from checkpoint metadata."""
    from occany.model.model_da3 import DA3Wrapper

    if not os.path.exists(weights_path):
        raise ValueError(f"Checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    if projection_features_override is not None:
        projection_features = projection_features_override
        print(f"[INFO] Using overridden DA3 projection features: {projection_features}")
    else:
        projection_features = getattr(checkpoint_args, "projection_features", "pts3d_local,pts3d,rgb,conf,sam3")
        print(f"[INFO] Using checkpoint DA3 projection features: {projection_features}")

    # Determine backbone architecture from checkpoint args (e.g. "depth-anything/DA3-GIANT-1.1")
    hf_model_name = getattr(checkpoint_args, "da3_model_name", "depth-anything/DA3-LARGE")
    da3_config_name = _resolve_da3_model_name(hf_model_name)

    model_input_size = output_resolution[0]
    print(f"[INFO] Building DA3 model locally (config={da3_config_name}, img_size={model_input_size})")
    model = DA3Wrapper(
        model_name=da3_config_name,
        img_size=model_input_size,
        projection_features=projection_features,
    ).to(device)

    needs_sam3_head_for_projection = uses_sam3_projection_features(projection_features)
    needs_sam3_head_for_semantic = semantic_family == "SAM3" and is_distill_source(semantic_feat_src)
    if needs_sam3_head_for_projection or needs_sam3_head_for_semantic:
        sam3_use_dpt_proj = getattr(checkpoint_args, "sam3_use_dpt_proj", False)
        sam3_reasons: List[str] = []
        if needs_sam3_head_for_projection:
            sam3_reasons.append("projection_features")
        if needs_sam3_head_for_semantic:
            sam3_reasons.append("semantic_distill")
        print(
            f"[INFO] Initializing DA3 SAM3 head "
            f"(use_dpt_proj={sam3_use_dpt_proj}, reason={'+'.join(sam3_reasons)})"
        )
        model.init_sam3_head(
            img_size=output_resolution[0],
            device=device,
            use_dpt_proj=sam3_use_dpt_proj,
        )

    model.load_state_dict(checkpoint["model"], strict=False)

    model.eval()
    model.requires_grad_(False)
    return model, checkpoint_args
