# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import numpy as np
from occany.model.sam3_model import Sam3ModelManager
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple


def split_distilled_sam_feats(
    sam_feats_img_and_raymap: Optional[List[torch.Tensor]],
    n_recon_views: int,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """Split concatenated SAM features into reconstruction and generated-view chunks."""
    if sam_feats_img_and_raymap is None or len(sam_feats_img_and_raymap) < 3:
        return None, None

    recon_feats = [feat[:, :n_recon_views] for feat in sam_feats_img_and_raymap[:3]]
    n_total_views = sam_feats_img_and_raymap[0].shape[1]
    if n_total_views <= n_recon_views:
        return recon_feats, None
    gen_feats = [feat[:, n_recon_views:] for feat in sam_feats_img_and_raymap[:3]]
    return recon_feats, gen_feats

    
def get_box_dict_for_view(data: Dict[str, Any], batch_idx: int, view_idx: int) -> Dict[str, Any]:
    """Safely fetch a per-view box dictionary with empty fallback."""
    empty_box_dict: Dict[str, Any] = {"boxes": [], "confidences": [], "labels": []}
    box_dicts = data.get("box_dicts")
    if box_dicts is None or batch_idx >= len(box_dicts):
        return empty_box_dict

    batch_box_dicts = box_dicts[batch_idx]
    if batch_box_dicts is None or view_idx >= len(batch_box_dicts):
        return empty_box_dict

    box_dict = batch_box_dicts[view_idx]
    if box_dict is None:
        return empty_box_dict

    return {
        "boxes": box_dict.get("boxes", []),
        "confidences": box_dict.get("confidences", []),
        "labels": box_dict.get("labels", []),
    }


def select_sam_feature_views(
    sam_feats: Optional[List[torch.Tensor]],
    view_ids: List[int],
    n_total_views: int,
    context: str = "",
) -> Optional[List[torch.Tensor]]:
    """Select view subsets from distilled SAM features for memory-efficient inference."""
    if sam_feats is None or len(view_ids) == 0:
        return None

    prefix = f"[{context}] " if context else ""
    selected_feats: List[torch.Tensor] = []
    for level_idx, feat in enumerate(sam_feats[:3]):
        if feat.dim() == 5:
            selected_feats.append(feat[:, view_ids])
            continue

        if feat.dim() == 4:
            if n_total_views <= 0 or feat.shape[0] % n_total_views != 0:
                print(
                    f"[WARNING] {prefix}Cannot subset SAM features at level {level_idx}: "
                    f"shape {tuple(feat.shape)} is incompatible with n_total_views={n_total_views}"
                )
                return sam_feats[:3]
            batch_size = feat.shape[0] // n_total_views
            reshaped_feat = feat.reshape(batch_size, n_total_views, *feat.shape[1:])
            selected = reshaped_feat[:, view_ids]
            selected_feats.append(selected.reshape(batch_size * len(view_ids), *feat.shape[1:]))
            continue

        print(
            f"[WARNING] {prefix}Cannot subset SAM features at level {level_idx}: "
            f"unsupported feature rank {feat.dim()}"
        )
        return sam_feats[:3]

    return selected_feats

    
def build_sam3_inference_state(
    sam_feats: Optional[List[torch.Tensor]],
    batch_idx: int,
    n_views: int,
    original_height: int,
    original_width: int,
    pos_enc: Optional[Any] = None,
    context: str = "",
) -> Optional[Dict[str, Any]]:
    """Build SAM3 inference state from distilled feature maps."""
    prefix = f"[{context}] " if context else ""
    if sam_feats is None:
        print(f"[WARNING] {prefix}SAM3 distilled features are missing")
        return None
    if len(sam_feats) < 3:
        print(f"[WARNING] {prefix}Expected at least 3 SAM3 feature levels, got {len(sam_feats)}")
        return None

    feat_levels = sam_feats[:3]
    if feat_levels[0].dim() == 5:
        feat_s0 = feat_levels[0][batch_idx]
        feat_s1 = feat_levels[1][batch_idx]
        feat_s2 = feat_levels[2][batch_idx]
    elif feat_levels[0].dim() == 4:
        start_idx = batch_idx * n_views
        end_idx = (batch_idx + 1) * n_views
        feat_s0 = feat_levels[0][start_idx:end_idx]
        feat_s1 = feat_levels[1][start_idx:end_idx]
        feat_s2 = feat_levels[2][start_idx:end_idx]
    else:
        print(f"[WARNING] {prefix}Unexpected SAM3 feature rank: {feat_levels[0].dim()}")
        return None

    if feat_s0.shape[0] != n_views or feat_s1.shape[0] != n_views or feat_s2.shape[0] != n_views:
        print(
            f"[WARNING] {prefix}SAM3 feature view count mismatch "
            f"(expected {n_views}, got {feat_s0.shape[0]}, {feat_s1.shape[0]}, {feat_s2.shape[0]})"
        )
        return None

    if pos_enc is None:
        from sam3.model.position_encoding import PositionEmbeddingSine

        pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True)

    pos_s0 = pos_enc(feat_s0).to(feat_s0.dtype)
    pos_s1 = pos_enc(feat_s1).to(feat_s1.dtype)
    pos_s2 = pos_enc(feat_s2).to(feat_s2.dtype)

    return {
        "backbone_out": {
            "backbone_fpn": [feat_s0, feat_s1, feat_s2],
            "vision_features": feat_s2,
            "vision_pos_enc": [pos_s0, pos_s1, pos_s2],
        },
        "original_height": original_height,
        "original_width": original_width,
    }


def infer_sam3_feats(sam3_imgs, original_height, original_width, device, sam3_resolution=1008):
    sam3_manager = Sam3ModelManager(
            resolution=sam3_resolution,
            confidence_threshold=0.5,
        )
    sam3_processor = sam3_manager.get_sam3(device)
    state = sam3_processor.forward(sam3_imgs, 
                               original_height=original_height, 
                               original_width=original_width)
    return state

def infer_semantic_from_classname_and_sam3_inference_state(
    prompts,
    prompt_to_class_mapping,
    sam3_inference_state, 
    ignore_ids=[],
    empty_class=0,
    device='cuda',
    confidence_threshold=0.5,
    sam3_resolution=1008,
    view_batch_size=4
):
    """Infer 2D semantics by querying SAM3 with text prompts and remapping to KITTI classes."""
    ignore_ids_set = set(ignore_ids)
    max_label_id = max([empty_class] + prompt_to_class_mapping)
    assert max_label_id <= torch.iinfo(torch.uint8).max, "Class indices must fit within torch.uint8"

    sam3_manager = Sam3ModelManager(
        resolution=sam3_resolution,
        confidence_threshold=confidence_threshold,
    )
    sam3_processor = sam3_manager.get_sam3(device)
    
    H = sam3_inference_state["original_height"]
    W = sam3_inference_state["original_width"]

    n_views = sam3_inference_state["backbone_out"]["vision_features"].shape[0]
    sem2d = torch.full((n_views, H, W), fill_value=empty_class, dtype=torch.uint8, device=device)
    
    # Prepare valid prompts and their corresponding class IDs
    valid_prompts = []
    prompt_to_class_id = []
    
    for prompt_idx, prompt in enumerate(prompts):
        class_id = prompt_to_class_mapping[prompt_idx]
        if class_id == empty_class or class_id in ignore_ids_set:
            continue
        valid_prompts.append(prompt)
        prompt_to_class_id.append(class_id)
    
    if len(valid_prompts) > 0:
        # Use batched prediction for all valid prompts and chunks of views to avoid OOM
        sam3_processor.reset_all_prompts(sam3_inference_state)
        
        num_prompts = len(valid_prompts)
        class_lookup = torch.tensor(prompt_to_class_id, device=device, dtype=torch.long)

        for i in tqdm(range(0, n_views, view_batch_size), desc="Inference 2D semantics (chunked)"):
            chunk_view_ids = list(range(i, min(i + view_batch_size, n_views)))
            
            mask_output = sam3_processor.predict_batched(
                prompts=valid_prompts, 
                state=sam3_inference_state, 
                image_ids=chunk_view_ids
            )
            
            if "masks" not in mask_output or len(mask_output["masks"]) == 0:
                continue
                
            masks = mask_output["masks"]  # [N, 1, H, W] boolean
            scores = mask_output["scores"]  # [N]
            task_indices = mask_output["prompt_indices"]  # [N]
            
            # Calculate relative view_id (within chunk) and prompt_idx for each prediction
            rel_view_ids = task_indices // num_prompts
            pred_prompt_indices = task_indices % num_prompts
            
            # Map back to absolute view_id
            chunk_view_ids_tensor = torch.tensor(chunk_view_ids, device=device, dtype=torch.long)
            pred_view_ids = chunk_view_ids_tensor[rel_view_ids]
            
            # Get label IDs
            pred_label_ids = class_lookup[pred_prompt_indices]
            
            # Iterate over each view that has predictions in this chunk
            unique_views = torch.unique(pred_view_ids)
            
            for view_id in unique_views.tolist():
                # Select predictions for this view
                view_mask_indices = (pred_view_ids == view_id)
                
                view_scores = scores[view_mask_indices]
                
                # Sort by confidence descending
                sorted_indices_local = torch.argsort(view_scores, descending=True)
                
                view_labels_sorted = pred_label_ids[view_mask_indices][sorted_indices_local]
                view_masks_sorted = masks[view_mask_indices][sorted_indices_local]
                
                for j in range(len(view_labels_sorted)):
                    label_id = view_labels_sorted[j]
                    if label_id.item() in ignore_ids_set:
                        continue
                    
                    mask = view_masks_sorted[j].squeeze(0) # [H, W]
                    
                    sem2d[view_id] = torch.where(mask & (sem2d[view_id] == empty_class), label_id, sem2d[view_id])
    
    return sem2d
