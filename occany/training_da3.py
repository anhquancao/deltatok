# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
import os
os.environ['OMP_NUM_THREADS'] = '3' # will affect the performance of pairwise prediction
os.environ['TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS'] = '1' # fix tensor.item() graph breaks

import argparse
import datetime
import json
import numpy as np
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
import logging
logger = logging.getLogger(__name__)

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# noqa: F401, needed when loading the model
from occany.datasets import get_data_loader  # noqa
from dust3r.inference import loss_of_one_batch, visualize_results, visualize_semantic  # noqa
from dust3r.inference_multiview import loss_of_one_batch as loss_of_one_batch_multiview_fast3r # noqa

from dust3r.losses import *  # noqa: F401, needed when loading the model
from occany.loss.losses import *  # noqa: F401, needed when loading the model
from occany.loss.losses_da3 import *  # noqa: F401, needed when loading the model

from occany.model.model_da3 import DA3Wrapper
from occany.da3_inference import loss_of_one_batch_occany_da3, loss_of_one_batch_occany_da3_gen
from occany.utils.helpers import depth2rgb
from dust3r.utils.geometry import geotrf
 
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention

import occany.utils.io_da3 as checkpoints
import occany.model.must3r_blocks.optimizer as optim
from PIL import Image
from occany.model.sam3_model import Sam3ModelManager


def get_args_parser():
    parser = argparse.ArgumentParser('DA3 training', add_help=False)
    
    # distillation
    parser.add_argument('--distill_model', default=None, type=str, help="distillation model (e.g., SAM3)")
    parser.add_argument('--distill_criterion', default=None, type=str, help="distill criterion")
    parser.add_argument('--distill_max_frames', default=10, type=int,
                        help="Max frames per sequence used for SAM3 distillation. "
                             "When T > K, K indices are sampled uniformly at random per batch. "
                             "-1 disables subsampling (use all frames). Default 10. "
                             "Ignored when --distill_k_schedule is not 'constant'.")
    parser.add_argument('--distill_k_schedule', default='constant', type=str,
                        choices=['constant', 'memory_budget'],
                        help="How to choose K (SAM3 distill frame count) per batch. "
                             "'constant': use --distill_max_frames as a fixed K. "
                             "'memory_budget': derive K from per-batch T (number of memory views "
                             "sampled by BatchedRandomSampler) via K = max(0, min(T, 2*(28 - T))). "
                             "The cap T=28 -> K=0 and the slope (one fewer T buys two more K) come "
                             "from the empirical (T, K) memory table for a 40 GiB GPU at 518x294; "
                             "see the slurm wrapper for the full table.")
    
    # fine-tuning
    parser.add_argument('--finetune_dual_dpt_only', default=False, action='store_true', help="Only finetune the DualDPT head, freeze backbone")

    parser.add_argument('--fine_tune_layers', type=str, default=None,
                        help="Comma-separated list of layer indices to fine-tune while freezing the rest of the backbone. "
                            "Example: '20,21,22,23' for DA3-LARGE or '34,35,36,37,38,39' for DA3-GIANT.")
    parser.add_argument('--freeze_head', default=False, action='store_true', help="Freeze the DualDPT head (Depth head)")

    # dataset
    parser.add_argument('--train_dataset', default='[None]', type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size per GPU")
    parser.add_argument('--accum_iter', default=1, type=int, help="Accumulate gradient iterations")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR', help='base learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--disable_lr_scheduler', action='store_true', default=False,
                        help='Disable per-iteration LR scheduler and use constant LR')
    parser.add_argument('--sam3_proj_lr_mult', type=float, default=10.0,
                        help='LR multiplier for SAM3Head proj parameters')
    parser.add_argument('--amp', choices=[False, "bf16", "fp16"], default=False, help="Use AMP for training")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False)
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--fixed_eval_set", action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')

    # distributed
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # logging / saving
    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency to save checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int, help='frequency to save checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int, help='frequency to print training info')
    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path to save output")

    # model settings
    parser.add_argument('--img_size', default=518, type=int, help='image size')
    parser.add_argument('--sam3_resolution', default=1008, type=int, help='SAM3 input image resolution')
    parser.add_argument('--sam_model', default='SAM2', type=str, help='SAM model version (SAM2 or SAM3)')
    parser.add_argument('--da3_model_name', default='depth-anything/DA3-LARGE-1.1', type=str,
                        help='Hugging Face model name/path for DA3 backbone initialization')
    parser.add_argument('--gen', action='store_true', default=False,
                        help='activate generation training with raymap prediction')
    
    # SAM3Head settings
    parser.add_argument('--sam3_use_dpt_proj', action='store_true', default=False,
                        help='Use DPT-style multi-level projection for SAM3Head instead of Mlp (default: False)')
    parser.add_argument('--loss_enc_feat', action='store_true', default=False, help='use encoder feature loss')
    parser.add_argument('--multiview', action='store_true', default=False, help='use multiview loss')
    parser.add_argument('--training_objective', type=str, default='pointmap', choices=['pointmap', 'depth_ray', 'pointmap_depth_ray', 'raymap'],
                        help='Training objective: "pointmap" uses pointmap+depth loss, "depth_ray" uses depth+raymap loss, "pointmap_depth_ray" uses pointmap+depth+raymap loss, "raymap" uses only raymap loss')

    # loss weights (used depending on training_objective)
    parser.add_argument('--lambda_depth', type=float, default=1.0, help='weight for depth loss (used when training_objective=depth_ray)')
    parser.add_argument('--lambda_raymap', type=float, default=1.0, help='weight for raymap loss (used when training_objective=depth_ray)')
    parser.add_argument('--lambda_pointmap', type=float, default=1.0, help='weight for pointmap loss (used when training_objective=pointmap)')
    parser.add_argument('--depth_lambda_c', type=float, default=1.0, help='confidence weight for depth loss')
    parser.add_argument('--depth_alpha', type=float, default=0.0, help='gradient loss weight for depth loss (0 for sparse lidar)')
    parser.add_argument('--raymap_lambda_c', type=float, default=1.0, help='confidence weight for raymap loss')
    parser.add_argument('--pointmap_lambda_c', type=float, default=1.0, help='confidence weight for pointmap loss (0 to disable, used in training only)')
    parser.add_argument('--loss_type', type=str, default='L2', choices=['L1', 'L2'], help='Loss type to use (L1 or L2) for pointmap and raymap losses')
    
    # Gen views settings (used when --gen is enabled)
    parser.add_argument('--projection_features', type=str, default='pts3d_local,pts3d,rgb,conf',
                        help='Comma-separated list of features to project for gen views: pts3d_local, pts3d, rgb, conf, sam')
    parser.add_argument('--gen_alt_start', type=int, default=None,
                        help='alt_start layer override for model_gen backbone. If unset, keep the default from the loaded DA3 variant (for example, 8 for DA3-LARGE and 13 for DA3-GIANT)')
    parser.add_argument('--pretrained_recon_model', type=str, default=None,
                        help='Path to pretrained reconstruction model checkpoint to load for gen views training')
    
    # Aux branch settings
    parser.add_argument('--aux_branch_layers', type=int, default=0,
                        help='Number of layers to duplicate for aux branch (0 to disable, e.g., 6 for last 6 layers)')
    parser.add_argument('--aux_metric_pseudo_supervision', action='store_true', default=False,
                        help='Training-only mode: use lidar up to 50m only to align frozen aux depth scale, then supervise main depth/pointmap with aux-derived pseudo targets')
    parser.add_argument('--da3_metric_model_name', type=str, default='depth-anything/DA3METRIC-LARGE',
                        help='HuggingFace name/path for the DA3 metric model used to predict sky masks during aux pseudo supervision')
    parser.add_argument('--sky_mask_threshold', type=float, default=0.3,
                        help='Threshold on the DA3 metric sky prediction; pixels with sky score < threshold are kept for supervision')
    parser.add_argument('--scale_inv_depth_loss', action='store_true', default=False,
                        help='Use scale-invariant depth loss with aux branch (legacy path; requires aux_branch_layers > 0)')
    parser.add_argument('--lambda_scale_inv_depth', type=float, default=1.0,
                        help='Weight for scale-invariant depth loss')
    parser.add_argument('--scale_inv_l1_weight', type=float, default=1.0,
                        help='Weight for L1 term in scale-invariant depth loss (set 0 for gradient-only)')
    parser.add_argument('--scale_inv_grad_alpha', type=float, default=1.0,
                        help='Weight for gradient term in scale-invariant depth loss')
    # InfiniDepth pseudo-supervision: train depth/pointmap against pseudo depth
    # produced by InfiniDepth_DepthSensor conditioned on the LiDAR projection
    # (full sensor depth + all valid LiDAR points as the dense prompt).
    parser.add_argument('--infinidepth_pseudo_supervision', action='store_true', default=False,
                        help='Training-only mode: use precomputed InfiniDepth pseudo-depth (sibling <stem>.infinidepth.png produced by extract_infinidepth_pseudo.py) as the pseudo-GT depth teacher; pseudo depth/pointmap supervise the depth and pointmap losses (mutually exclusive with --scale_inv_depth_loss and --aux_metric_pseudo_supervision; forbids --aux_branch_layers > 0; requires load_infinidepth_pseudo=True on each train dataset)')
    parser.add_argument('--infinidepth_depth_min', type=float, default=1e-3,
                        help='Min valid pseudo depth (m) for the pseudo-supervision range mask')
    parser.add_argument('--infinidepth_depth_max', type=float, default=150.0,
                        help='Max valid pseudo depth (m) for the pseudo-supervision range mask. Set just below InfiniDepth\'s 200m sky-sentinel and disparity-clamp tail (1/5e-3=200) to exclude outliers while keeping the longest reliable range.')
    # Dual supervision weighting (only active with --infinidepth_pseudo_supervision):
    # the lidar and pseudo branches are both computed and contribute additively,
    # each scaled by its own lambda on top of --lambda_pointmap / --lambda_depth.
    parser.add_argument('--lambda_pointmap_lidar', type=float, default=1.0,
                        help='Lidar-GT pointmap weight under --infinidepth_pseudo_supervision (multiplies --lambda_pointmap)')
    parser.add_argument('--lambda_pointmap_pseudo', type=float, default=1.0,
                        help='InfiniDepth pseudo-GT pointmap weight under --infinidepth_pseudo_supervision (multiplies --lambda_pointmap)')
    parser.add_argument('--lambda_depth_lidar', type=float, default=1.0,
                        help='Lidar-GT depth weight under --infinidepth_pseudo_supervision (multiplies --lambda_depth)')
    parser.add_argument('--lambda_depth_pseudo', type=float, default=1.0,
                        help='InfiniDepth pseudo-GT depth weight under --infinidepth_pseudo_supervision (multiplies --lambda_depth)')
    parser.add_argument('--lambda_feat_matching', type=float, default=1.0,
                        help='Weight for feature matching loss (used when --gen is enabled)')

    return parser



def load_distill_model(args, device):
    if args.distill_model is None:
        return None
    print('Loading distillation model: {:s}'.format(args.distill_model))
    if args.distill_model == 'SAM3':
        sam3_manager = Sam3ModelManager(resolution=args.sam3_resolution)
        distill_model = sam3_manager.get_sam3(device=device)
    else:
        if args.distill_model == 'SAM2_large':
            checkpoint_path = "./checkpoints/sam2.1_hiera_large.pt"
            config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:
            raise ValueError("Unknown distillation model: {:s}".format(args.distill_model))
        
        print('SAM2 model supports multiple resolutions including 512x512 and 768x768')
        distill_model = SAM2(
            model_cfg=str(config_path),
            sam2_checkpoint=str(checkpoint_path),
            device=device,
            image_size=args.img_size
        )
        distill_model.to(device)
        for param in distill_model.parameters():
            param.requires_grad = False
        distill_model.eval()
    return distill_model

def get_dtype(args):
    if args.amp:
        dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    else:
        dtype = torch.float32
    return dtype


def resolve_resume_checkpoint(args):
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f'Using explicit resume checkpoint: {args.resume}')
        return args.resume

    last_ckpt_fname = os.path.join(args.output_dir, 'checkpoint-last.pth')
    if os.path.isfile(last_ckpt_fname):
        print(f'Auto-resuming from: {last_ckpt_fname}')
        return last_ckpt_fname
    return None


def train(args):
    misc.init_distributed_mode_jz(args)

    toggle_memory_efficient_attention(enabled=True)

    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    logger.info(f"global_rank: {global_rank}, world_size: {world_size}")

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    last_ckpt_fname = resolve_resume_checkpoint(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print(f"training_mode: {'gen' if args.gen else 'recon'}")

    if args.scale_inv_depth_loss and args.aux_metric_pseudo_supervision:
        raise ValueError('--scale_inv_depth_loss and --aux_metric_pseudo_supervision are mutually exclusive')
    if args.scale_inv_depth_loss:
        if args.gen:
            raise ValueError('--scale_inv_depth_loss is only supported in reconstruction training')
        if args.aux_branch_layers <= 0:
            raise ValueError('--scale_inv_depth_loss requires --aux_branch_layers > 0')
    if args.aux_metric_pseudo_supervision:
        if args.gen:
            raise ValueError('--aux_metric_pseudo_supervision is only supported in reconstruction training')
        if args.aux_branch_layers <= 0:
            raise ValueError('--aux_metric_pseudo_supervision requires --aux_branch_layers > 0')
        if args.training_objective == 'raymap' or (args.lambda_depth <= 0 and args.lambda_pointmap <= 0):
            raise ValueError('--aux_metric_pseudo_supervision requires depth and/or pointmap supervision to be enabled')
    if args.infinidepth_pseudo_supervision:
        if args.scale_inv_depth_loss or args.aux_metric_pseudo_supervision:
            raise ValueError('--infinidepth_pseudo_supervision is mutually exclusive with --scale_inv_depth_loss and --aux_metric_pseudo_supervision')
        if args.gen:
            raise ValueError('--infinidepth_pseudo_supervision is only supported in reconstruction training')
        if args.aux_branch_layers > 0:
            raise ValueError('--infinidepth_pseudo_supervision must not be combined with --aux_branch_layers > 0 (no aux teacher tail is needed; keeping the aux branch wastes GPU memory)')
        if args.training_objective == 'raymap' or (args.lambda_depth <= 0 and args.lambda_pointmap <= 0):
            raise ValueError('--infinidepth_pseudo_supervision requires depth and/or pointmap supervision to be enabled')

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed mode requires CUDA but no GPU was detected.")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = args.cudnn_benchmark

    
    distill_model = None
    distill_criterion = None
    if args.distill_model is not None:
        distill_model = load_distill_model(args, device)
        distill_criterion = eval(args.distill_criterion)

    # DA3 metric (sky mask) is only needed for aux-metric pseudo supervision.
    # InfiniDepth pseudo-supervision uses precomputed `.infinidepth.png` files
    # where sky was already masked at extract time.
    da3_metric_model = None
    if args.aux_metric_pseudo_supervision:
        from depth_anything_3.api import DepthAnything3
        print(f"[INFO] Loading DA3 metric model for sky masking: {args.da3_metric_model_name}")
        da3_metric_model = DepthAnything3.from_pretrained(args.da3_metric_model_name)
        da3_metric_model = da3_metric_model.to(device)
        da3_metric_model.eval()
        da3_metric_model.requires_grad_(False)

    print(f"[INFO] Using DA3 model: {args.da3_model_name}")
    model = DA3Wrapper.from_pretrained(args.da3_model_name)
    model = model.to(device)
    backbone_metadata = model.get_backbone_metadata()
    print(
        f"[INFO] Loaded DA3 backbone: name={backbone_metadata['name']}, "
        f"token_dim={backbone_metadata['token_dim']}, feature_dim={backbone_metadata['feature_dim']}, "
        f"out_layers={list(backbone_metadata['out_layers'])}, alt_start={backbone_metadata['alt_start']}"
    )
    
    fine_tune_layers = None
    if args.fine_tune_layers is not None:
        fine_tune_layers = [int(x.strip()) for x in args.fine_tune_layers.split(',')]

    if fine_tune_layers is not None and args.aux_branch_layers > 0:
        if len(fine_tune_layers) != args.aux_branch_layers:
            raise ValueError(
                "aux_branch_layers must match the number of fine_tune_layers "
                f"({args.aux_branch_layers} vs {len(fine_tune_layers)})"
            )
        # The aux teacher tail is a frozen deep copy of the final
        # `aux_branch_layers` backbone blocks. The trainable tail must match
        # exactly those final blocks so the teacher and student tails stay
        # aligned with the same shared frozen prefix.
        total_layers = backbone_metadata['total_layers']
        expected_layers = list(range(total_layers - args.aux_branch_layers, total_layers))
        if sorted(fine_tune_layers) != expected_layers:
            raise ValueError(
                f"fine_tune_layers must equal the final {args.aux_branch_layers} backbone "
                f"layers {expected_layers}, got {sorted(fine_tune_layers)}"
            )

    # Selective layer fine-tuning
    if fine_tune_layers is not None:
        print(f'Selective fine-tuning: freezing backbone except layers {fine_tune_layers}...')
        # Freeze entire backbone first
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        # Unfreeze specified layers
        for layer_idx in fine_tune_layers:
            block = model.model.backbone.pretrained.blocks[layer_idx]
            for param in block.parameters():
                param.requires_grad = True
        
    # Apply head freezing
    if args.freeze_head:
        print('Freezing DualDPT head...')
        for param in model.model.head.parameters():
            param.requires_grad = False
    else:
        # Keep DualDPT head trainable by default unless specified otherwise
        for param in model.model.head.parameters():
            param.requires_grad = True

    if fine_tune_layers is not None or args.freeze_head:
        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Trainable parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)')
    
    # Apply fine-tuning mode: only train DualDPT head if specified
    elif args.finetune_dual_dpt_only:
        print('Freezing backbone and only training DualDPT head and Camera Encoder...')
        # Freeze backbone (DinoV2)
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        # Freeze camera encoder if exists
        if model.model.cam_enc is not None:
            for param in model.model.cam_enc.parameters():
                param.requires_grad = True
        # Freeze camera decoder if exists  
        if model.model.cam_dec is not None:
            for param in model.model.cam_dec.parameters():
                param.requires_grad = False
        # Freeze GS head if exists
        if model.model.gs_head is not None:
            for param in model.model.gs_head.parameters():
                param.requires_grad = False
        if model.model.gs_adapter is not None:
            for param in model.model.gs_adapter.parameters():
                param.requires_grad = False
        # Keep DualDPT head trainable (model.model.head)
        for param in model.model.head.parameters():
            param.requires_grad = True
        print('DualDPT head and Camera Encoder parameters will be trained, backbone is frozen')

    model_without_ddp = model

    # Initialize aux branch for either legacy scale-invariant loss or aux pseudo supervision.
    if args.aux_branch_layers > 0 and not args.gen and (args.aux_metric_pseudo_supervision or args.scale_inv_depth_loss):
        print(f'Initializing aux branch with {args.aux_branch_layers} frozen layers...')
        model.init_aux_branch(n_layers=args.aux_branch_layers)
        print(f'  - Aux input layer idx: {model.aux_input_layer_idx}')
        print(f'  - Aux blocks: {len(model.aux_blocks)}')
    elif args.aux_branch_layers > 0:
        print(
            f'Aux branch NOT initialized: requires recon mode (gen={args.gen}) and '
            f'(aux_metric_pseudo_supervision={args.aux_metric_pseudo_supervision} '
            f'or scale_inv_depth_loss={args.scale_inv_depth_loss})'
        )
    
    # If using SAM3 distillation, initialize SAM3Head and load pretrained SAM3 neck weights
    if args.distill_model == 'SAM3':
        # Initialize SAM3Head for distillation
        model_without_ddp.init_sam3_head(img_size=args.img_size, device=device, use_dpt_proj=args.sam3_use_dpt_proj)
        
        print('Loading pretrained SAM3 neck weights into model.head_sam.neck...')
        # distill_model is Sam3ProcessorWrapper, the actual SAM3 model is at distill_model.model
        # The SAM3 neck is at: distill_model.model.backbone.vision_backbone.convs
        # Our SAM3Head neck is at: model.head_sam.neck.convs
        sam3_neck_state = {}
        for name, param in distill_model.model.backbone.vision_backbone.named_parameters():
            if name.startswith('convs.'):
                sam3_neck_state[name] = param.data.clone()
        
        # Also copy buffers if any
        for name, buf in distill_model.model.backbone.vision_backbone.named_buffers():
            if name.startswith('convs.'):
                sam3_neck_state[name] = buf.clone()
        
        # Load into our neck
        neck_load_status = model_without_ddp.head_sam.neck.load_state_dict(sam3_neck_state, strict=False)
        print('SAM3 neck load_state_dict status:', neck_load_status)
    
        # Freeze the neck parameters
        for param in model_without_ddp.head_sam.neck.parameters():
            param.requires_grad = False
        print('Froze model.head_sam.neck parameters')

    # Initialize variables for gen views (encoder is now handled within DA3Wrapper)
    gen_input_encoder = None
    model_recon = None  # Will be set if pretrained_recon_model is specified
    if args.gen:
        
        # Load pretrained reconstruction model if specified
        # Create dual models: model_recon (frozen) for reconstruction, model_gen (trainable) for generation
        if args.pretrained_recon_model is not None:
            print(f'Loading pretrained reconstruction checkpoint from: {args.pretrained_recon_model}')
            checkpoint = torch.load(args.pretrained_recon_model, map_location=device, weights_only=False)
            model_state = checkpoint.get('model', checkpoint)
            
            # Delete unwanted keys from checkpoint
            to_delete = ['aux_head', 'aux_blocks', 'cam_dec', 'cam_enc']
            for k in list(model_state.keys()):
                if any(prefix in k for prefix in to_delete):
                    del model_state[k]

            
            # Parse fine_tune_layers for selective backbone tuning
            fine_tune_layers = None
            if args.fine_tune_layers is not None:
                fine_tune_layers = [int(x.strip()) for x in args.fine_tune_layers.split(',')]
            
            # === model_recon: Frozen reconstruction model ===
            print('Creating model_recon (frozen)...')
            model_recon = DA3Wrapper.from_pretrained(args.da3_model_name).to(device)
            
            # Initialize SAM3 head BEFORE loading checkpoint so that SAM3 weights can be loaded
            if 'sam3' in getattr(args, 'projection_features', ''):
                model_recon.init_sam3_head(img_size=args.img_size, device=device, use_dpt_proj=args.sam3_use_dpt_proj)
            
            # Load checkpoint (includes SAM3 head weights if sam3 was initialized above)
            load_status_recon = model_recon.load_state_dict(model_state, strict=False)
            print(f'model_recon load status: {load_status_recon}')
            
            # Freeze all parameters
            for param in model_recon.parameters():
                param.requires_grad = False
            model_recon.eval()
            print('model_recon: Frozen (no gradients)')
            
            # === model_gen: Trainable generation model wrapped with encoders ===
            print('Creating model_gen via DA3Wrapper.from_pretrained...')
            model_gen = DA3Wrapper.from_pretrained(
                args.da3_model_name,
                img_size=args.img_size,
                projection_features=getattr(args, 'projection_features', 'pts3d_local,pts3d,rgb,conf'),
            )

            # Initialize SAM3 head BEFORE loading checkpoint so that DPTProj/Mlp weights are loaded.
            # This is required when --freeze_head is OFF (no head sharing from model_recon).
            if args.distill_model is not None and args.distill_model.upper() == 'SAM3':
                model_gen.init_sam3_head(img_size=args.img_size, device=device, use_dpt_proj=args.sam3_use_dpt_proj)

            # Initialize model_gen weights from model_recon
            load_status_gen = model_gen.load_state_dict(model_state, strict=False)
            print(f'model_gen load status (backbone/head): {load_status_gen}')
            
            # Override alt_start for gen mode only when explicitly requested.
            if args.gen_alt_start is not None:
                print(f'Setting model_gen alt_start = {args.gen_alt_start}')
                model_gen.set_alt_start(args.gen_alt_start)
            
            model_gen.init_gen_encoders()
            
            # Move to device AFTER init_gen_encoders() to ensure encoders are on CUDA
            model_gen = model_gen.to(device)
            
            # Share frozen heads from model_recon to save GPU memory (~350M parameters)
            if args.freeze_head:
                # Share DualDPT head from model_recon (already frozen)
                print('Sharing DualDPT head from model_recon to model_gen (saves ~300M params)...')
                del model_gen.model.head
                model_gen.model.head = model_recon.model.head
                
                # Share SAM3 head from model_recon if it exists
                if model_recon.head_sam is not None:
                    print('Sharing SAM3 head from model_recon to model_gen (saves ~50M params)...')
                    if model_gen.head_sam is not None:
                        del model_gen.head_sam
                    model_gen.head_sam = model_recon.head_sam
            
            # Selective fine-tuning for model_gen
            if args.fine_tune_layers is not None:
                fine_tune_layers_gen = [int(x.strip()) for x in args.fine_tune_layers.split(',')]
                print(f'Selective fine-tuning (model_gen): freezing backbone except layers {fine_tune_layers_gen}...')

                # Set slice layer for memory optimization in generation mode
                if len(fine_tune_layers_gen) > 0:
                    model_gen.set_slice_layer(max(fine_tune_layers_gen))

                for param in model_gen.model.backbone.parameters():
                    param.requires_grad = False
                for layer_idx in fine_tune_layers_gen:
                    block = model_gen.model.backbone.pretrained.blocks[layer_idx]
                    for param in block.parameters():
                        param.requires_grad = True

                # Share frozen backbone blocks from model_recon to save memory
                if model_recon is not None:
                    total_layers = len(model_gen.model.backbone.pretrained.blocks)
                    # Use the gen-specific fine-tuned layers list
                    ft_layers = fine_tune_layers_gen if fine_tune_layers_gen is not None else []
                    frozen_layers = [i for i in range(total_layers) if i not in ft_layers]
                    print(f'Sharing {len(frozen_layers)} frozen backbone blocks from model_recon to model_gen...')
                    
                    # Use setattr with string index to replace blocks without shifting indices
                    # nn.ModuleList uses string keys internally (e.g., '0', '1', '2', ...)
                    blocks_gen = model_gen.model.backbone.pretrained.blocks
                    blocks_recon = model_recon.model.backbone.pretrained.blocks
                    for layer_idx in frozen_layers:
                        # Replace model_gen's block with reference to model_recon's block
                        setattr(blocks_gen, str(layer_idx), blocks_recon[layer_idx])
                    
                    # Estimate memory savings
                    block_params = sum(p.numel() for p in blocks_recon[frozen_layers[0]].parameters())
                    # Assuming 4 bytes per param (float32) for the shared weights
                    savings_mb = block_params * len(frozen_layers) * 4 / (1024 * 1024)
                    import gc; gc.collect(); torch.cuda.empty_cache()
                    print(f'  Memory saved by sharing {len(frozen_layers)} backbone blocks: ~{savings_mb:.1f} MB')


            # Note: DualDPT head freezing is handled above via sharing or needs explicit freeze
            if not args.freeze_head:
                # Head is not shared, so we need to explicitly set trainability
                pass  # Head remains trainable by default
            
            # Ensure raymap encoder is trainable
            for param in model_gen.gen_input_encoder.parameters():
                param.requires_grad = True
            
            # Replace main model with model_gen
            model = model_gen
            model_without_ddp = model_gen
            
            # Parameters from model_gen.gen_input_encoder
            # are now part of model.parameters() and will be picked up by optimizer.
            # We don't need separate variables for encoders anymore.
            gen_input_encoder = None
        
    if args.distributed:
        # Check if model has any trainable parameters before wrapping with DDP
        has_trainable_params = any(p.requires_grad for p in model.parameters())
        if has_trainable_params:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=False,
                gradient_as_bucket_view=True)
            model_without_ddp = model.module
        else:
            print('Skipping DDP wrapping for model (no trainable parameters)')
        
        # Raymap encoders are now part of model (DA3Wrapper)
        # Single DDP wrapping for the whole model covers them
        pass
    
    # Checkpoint references
    gen_input_encoder_without_ddp = None
        

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
   
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {}
    for dataset in args.test_dataset.split('+'):
        testset = build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        name_testset = dataset.split('(')[0]
        data_loader_test[name_testset] = testset

    # Create DA3-specific criteria using PointmapLoss from losses_da3.py
    from occany.loss.losses_da3 import PointmapLoss, DepthLosses, RaymapLoss, ScaleInvariantDepthLoss
    
    # Log training objective
    print(f'>> Training objective: {args.training_objective}')
    if args.training_objective == 'depth_ray':
        print(f'   - Using depth loss (lambda={args.lambda_depth}) + raymap loss (lambda={args.lambda_raymap})')
        print('   - Pointmap loss is DISABLED')
    elif args.training_objective == 'pointmap_depth_ray':
        print(f'   - Using pointmap loss (lambda={args.lambda_pointmap})')
        print(f'   - Using depth loss (lambda={args.lambda_depth})')
        print(f'   - Using raymap loss (lambda={args.lambda_raymap})')
        print('   - Pointmap computed from depth + ray (point_from_depth_ray=True)')
    elif args.training_objective == 'raymap':
        print(f'   - Using raymap loss (lambda={args.lambda_raymap})')
        print('   - Depth loss is DISABLED')
        print('   - Pointmap loss is DISABLED')
    else:  # pointmap
        print(f'   - Using pointmap loss (lambda={args.lambda_pointmap})')
        print(f'   - Using depth loss (lambda={args.lambda_depth})')
        print('   - Raymap loss is DISABLED')
    if args.aux_metric_pseudo_supervision:
        print('   - Aux metric pseudo supervision is ENABLED for training')
        print('   - Lidar is used only up to 50m to align frozen aux depth scale')
        print('   - Training depth/pointmap use aux-derived pseudo targets; evaluation still uses direct lidar GT')
    elif args.infinidepth_pseudo_supervision:
        print('   - InfiniDepth pseudo supervision is ENABLED for training (dual GT mode)')
        print(f'   - Pseudo depth loaded from precomputed sibling `.infinidepth.png` files; range [{args.infinidepth_depth_min}, {args.infinidepth_depth_max}] m')
        print(f'   - Dual losses: lidar GT (lambda_pointmap_lidar={args.lambda_pointmap_lidar}, lambda_depth_lidar={args.lambda_depth_lidar}) + InfiniDepth pseudo GT (lambda_pointmap_pseudo={args.lambda_pointmap_pseudo}, lambda_depth_pseudo={args.lambda_depth_pseudo})')
        print('   - Evaluation still uses direct lidar GT')
    elif args.scale_inv_depth_loss:
        print('   - Legacy scale-invariant depth loss is ENABLED')
        print(f'   - lambda_scale_inv_depth={args.lambda_scale_inv_depth}, scale_inv_l1_weight={args.scale_inv_l1_weight}, scale_inv_grad_alpha={args.scale_inv_grad_alpha}')
    
    # Training pointmap criterion: with confidence awareness (lambda_c > 0)
    print(f'>> Creating pointmap criterion for training with lambda_c={args.pointmap_lambda_c}, loss_type={args.loss_type}')
    pointmap_criterion_train = PointmapLoss(reduction="mean", lambda_c=args.pointmap_lambda_c,
                                            gt_scale=False, loss_type=args.loss_type).to(device)
    # Testing pointmap criterion: without confidence awareness (lambda_c=0) for fair evaluation
    print('>> Creating pointmap criterion for testing (no confidence weighting, GT scale, loss_type=L2)')
    pointmap_criterion_test = PointmapLoss(reduction="mean", lambda_c=0.0,
                                           gt_scale=True, loss_type="L2").to(device)
    
    # Create depth and raymap criteria (always active)
    depth_detach_conf = False
    # Depth criterion is always L1. Set depth_lambda_c=0 to disable confidence weighting.
    print(f'>> Creating depth criterion with lambda_c={args.depth_lambda_c}, alpha={args.depth_alpha}, detach_confidence={depth_detach_conf}, loss_type=L1')
    depth_criterion = DepthLosses(lambda_c=args.depth_lambda_c, alpha=args.depth_alpha,
                                  detach_confidence=depth_detach_conf, gt_scale=True).to(device)
    print('>> Creating depth criterion for testing (at GT scale, no confidence weighting, loss_type=L1)')
    depth_criterion_test = DepthLosses(lambda_c=0.0, alpha=args.depth_alpha,
                                       detach_confidence=depth_detach_conf, gt_scale=True).to(device)

    print(f'>> Creating raymap criterion with lambda_c={args.raymap_lambda_c}, loss_type={args.loss_type}')
    # Raymap criterion for training (at GT scale to avoid unstable division by near zero)
    raymap_criterion = RaymapLoss(lambda_c=args.raymap_lambda_c,
                                  gt_scale=True, loss_type=args.loss_type).to(device)
    # Raymap criterion for testing (at GT scale, no confidence weighting, loss_type=L2)
    print('>> Creating raymap criterion for testing (at GT scale, no confidence weighting, loss_type=L2)')
    raymap_criterion_test = RaymapLoss(lambda_c=0.0,
                                       gt_scale=True, loss_type="L2").to(device)

    scale_inv_depth_criterion = None
    if args.scale_inv_depth_loss:
        print(
            f'>> Creating scale-invariant depth loss criterion '
            f'(l1_weight={args.scale_inv_l1_weight}, grad_alpha={args.scale_inv_grad_alpha})'
        )
        scale_inv_depth_criterion = ScaleInvariantDepthLoss(
            alpha=args.scale_inv_grad_alpha,
            l1_weight=args.scale_inv_l1_weight,
        ).to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    

    param_groups = []
    param_groups += optim.get_parameter_groups(model_without_ddp, 0, args.weight_decay)

    sam3_proj_params = {
        param for name, param in model_without_ddp.named_parameters()
        if name.startswith("head_sam.proj.")
    }
    if sam3_proj_params:
        print(f"[INFO] Applying SAM3 proj LR multiplier: {args.sam3_proj_lr_mult}")
        updated_param_groups = []
        for group in param_groups:
            group_params = group["params"]
            sam3_group_params = [param for param in group_params if param in sam3_proj_params]
            other_group_params = [param for param in group_params if param not in sam3_proj_params]
            if other_group_params:
                updated_param_groups.append({**group, "params": other_group_params})
            if sam3_group_params:
                updated_param_groups.append({
                    **group,
                    "params": sam3_group_params,
                    "lr_scale": group.get("lr_scale", 1.0) * args.sam3_proj_lr_mult,
                })
        param_groups = updated_param_groups

    if args.disable_lr_scheduler:
        for group in param_groups:
            group["lr"] = args.lr * group.get("lr_scale", 1.0)
        print("[INFO] LR scheduler disabled; using constant per-group LR.")
    
    if misc.is_main_process():
        total_params = sum(p.numel() for p in model_without_ddp.parameters())
        trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f'>> Total parameters: {total_params/1e6:.2f}M')
        print(f'>> Trainable parameters: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.1f}%)')
        print(f'>> Frozen parameters: {frozen_params/1e6:.2f}M ({100*frozen_params/total_params:.1f}%)')
        
        if isinstance(model_without_ddp, DA3Wrapper) and model_without_ddp.gen_input_encoder is not None:
            print("Parameter breakdown for model_gen encoder:")
            gen_enc_params = sum(p.numel() for p in model_without_ddp.gen_input_encoder.parameters())
            print(f'>> GenInputEncoder: {gen_enc_params/1e6:.2f}M')

    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            gathered_test_stats = {}
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            for test_name, testset in data_loader_test.items():

                if test_name not in test_stats:
                    continue

                if getattr(testset.dataset.dataset, 'strides', None) is not None:
                    original_test_name = test_name.split('_stride')[0]
                    if original_test_name not in gathered_test_stats.keys():
                        gathered_test_stats[original_test_name] = []
                    gathered_test_stats[original_test_name].append(test_stats[test_name])

                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            if len(gathered_test_stats) > 0:
                for original_test_name, stride_stats in gathered_test_stats.items():
                    if len(stride_stats) > 1:
                        stride_stats = {k: np.mean([x[k] for x in stride_stats]) for k in stride_stats[0]}
                        log_stats.update({original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()})
                        if args.wandb:
                            log_dict = {original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()}
                            log_dict.update({'epoch': epoch})
                            wandb.log(log_dict)

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname):
        checkpoints.save_model(
            args=args,
            epoch=epoch,
            model=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            fname=fname,
        )

    checkpoints.load_model(
        args=args,
        chkpt_path=last_ckpt_fname,
        model=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None



    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Test on multiple datasets
        new_best = False
        new_pose_best = False
        already_saved = False
        if (epoch >= args.start_epoch and args.eval_freq > 0 and epoch % args.eval_freq == 0) or args.eval_only:
            torch.cuda.empty_cache()
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                if args.eval_only:
                    log_writer = None
                
                stats = test_one_epoch(model=model,
                                       pointmap_criterion=pointmap_criterion_test,
                                       data_loader=testset,
                                       device=device, epoch=epoch,
                                       distill_model=distill_model,
                                       distill_criterion=distill_criterion,
                                       log_writer=log_writer, args=args, prefix=test_name,
                                       depth_criterion=depth_criterion_test,
                                       raymap_criterion=raymap_criterion_test,
                                       model_recon=model_recon)
                test_stats[test_name] = stats
            
            # Synchronize all processes after test phase before starting training
            # This prevents hangs when main process does extra visualization work
            if args.distributed:
                torch.distributed.barrier()
            torch.cuda.empty_cache()

        # Train
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            distill_model=distill_model,
            distill_criterion=distill_criterion,
            log_writer=log_writer,
            args=args,
            depth_criterion=depth_criterion,
            raymap_criterion=raymap_criterion,
            pointmap_criterion=pointmap_criterion_train,
            scale_inv_depth_criterion=scale_inv_depth_criterion,
            model_recon=model_recon,
            da3_metric_model=da3_metric_model)

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)
        if args.eval_only and args.epochs <= 1:
            exit(0)

        # Save the 'last' checkpoint
        if global_rank == 0 and epoch >= args.start_epoch:
            save_model(epoch, 'last')
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch, str(epoch))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)

    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not test)

    sampler = getattr(loader, 'sampler', None)
    sampler_name = type(sampler).__name__ if sampler is not None else 'None'
    print(f'[INFO] {split} sampler: {sampler_name}')

    print(f"{split} dataset length: ", len(loader))
    return loader


def _log_viz_sample(
        batch_result, batch_idx, epoch, epoch_step, output_dir, log_writer, tb_prefix,
        extra_panels=None, is_raymap=None, view_order=None, max_depth=50.0):
    """Shared visualization helper for train_one_epoch and test_one_epoch.

    Renders a side-by-side image: [GT image | pred depth | *extra_panels]
    and writes it to disk (and optionally TensorBoard).

    Args:
        extra_panels:  list of (T, H, W, 3) float tensors in [0, 255], already in
                       the desired view order (pre-sorted by the caller if needed).
        is_raymap:     list of T bools — draws a red border on those views across
                       all panels (top+bottom everywhere, left on first, right on last).
        view_order:    list of T indices to reorder the internally extracted gt_img
                       and pred_depth columns; extra_panels must be pre-sorted by
                       the caller to match.
        max_depth:     depth colorization range for the predicted depth panel.
        epoch_step:    integer x-axis value passed to TensorBoard
                       (e.g. int(epoch_f * 1000) for train or 1000 * epoch for test).
    """
    gt_imgs    = batch_result['combined_gt']
    pred_depth = batch_result['combined_preds']['depth']
    T = len(gt_imgs)

    pred_depth_b = pred_depth[batch_idx].detach().cpu()                    # (T, H, W)
    gt_img_b = torch.stack(
        [gt_imgs[t]['img'][batch_idx] for t in range(T)]
    ).permute(0, 2, 3, 1).detach().cpu()                                   # (T, H, W, 3)

    if view_order is not None:
        idx = torch.tensor(view_order)
        pred_depth_b = pred_depth_b[idx]
        gt_img_b = gt_img_b[idx]

    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
    gt_img_b = (gt_img_b * _std + _mean).clamp(0, 1) * 255.0

    frame_id = gt_imgs[0]['label'][batch_idx]

    pred_depth_color = torch.stack([torch.from_numpy(
        depth2rgb(pred_depth_b[t].numpy(), min_depth=0.1, max_depth=max_depth))
        for t in range(T)])                                                 # (T, H, W, 3)

    all_panels = [gt_img_b, pred_depth_color] + (extra_panels or [])

    # Red borders for raymap views: top+bottom on every panel, left on first, right on last
    if is_raymap is not None:
        bw, sw = 2, 5
        red = torch.tensor([255.0, 0.0, 0.0])
        for t, is_ray in enumerate(is_raymap):
            if is_ray:
                for panel in all_panels:
                    panel[t, :bw] = red
                    panel[t, -bw:] = red
                all_panels[0][t, :, :sw] = red   # left border on first panel
                all_panels[-1][t, :, -sw:] = red  # right border on last panel

    cols = [torch.cat([p[t] for t in range(T)], dim=0) for p in all_panels]
    combined_np = torch.cat(cols, dim=1).numpy()

    if log_writer is not None:
        log_writer.add_image(
            f'{tb_prefix}/{frame_id}', combined_np / 255.0, epoch_step, dataformats='HWC')

    save_dir = os.path.join(output_dir, *tb_prefix.split('/'))
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(combined_np.astype(np.uint8)).save(
        os.path.join(save_dir, f'{frame_id}_epoch{epoch}_concat.jpg'))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args, distill_model, distill_criterion, log_writer=None,
                    depth_criterion=None, raymap_criterion=None, pointmap_criterion=None,
                    scale_inv_depth_criterion=None,
                    model_recon=None, da3_metric_model=None):

    assert torch.backends.cuda.matmul.allow_tf32 == True
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()
    
    if args.distributed:
        torch.distributed.barrier()

    n_train_draw = 0
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            if not args.disable_lr_scheduler:
                misc.adjust_learning_rate(optimizer, epoch_f, args)
        dtype = get_dtype(args)
        # Use loss_of_one_batch_occany_da3_gen in generation mode.
        if args.gen:
            
            batch_result = loss_of_one_batch_occany_da3_gen(
                views=batch, 
                model=model,
                device=device,
                model_recon=model_recon,
                dtype=dtype,
                distill_criterion=distill_criterion,
                distill_model=distill_model,
                sam_model=args.sam_model,
                pointmap_criterion=pointmap_criterion,
                depth_criterion=depth_criterion,
                raymap_criterion=raymap_criterion,
                lambda_depth=args.lambda_depth,
                lambda_raymap=args.lambda_raymap,
                lambda_pointmap=args.lambda_pointmap,
                pose_from_depth_ray=False,
                projection_features=getattr(args, 'projection_features', 'pts3d_local,pts3d,rgb,conf'),
                lambda_feat_matching=args.lambda_feat_matching,
                distill_max_frames=args.distill_max_frames,
                distill_k_schedule=args.distill_k_schedule,
            )
        else:

            # Determine lambda values based on training_objective
            # depth_ray: use raymap + depth loss
            # pointmap: use pointmap + depth loss
            # pointmap_depth_ray: use pointmap + depth + raymap loss (pointmap from depth+ray)
            # raymap: use only raymap loss
            if args.training_objective == 'depth_ray':
                lambda_depth_train = args.lambda_depth
                lambda_raymap_train = args.lambda_raymap
                lambda_pointmap_train = 0.0  # disable pointmap loss for depth_ray
            elif args.training_objective == 'pointmap_depth_ray':
                lambda_depth_train = args.lambda_depth
                lambda_raymap_train = args.lambda_raymap  # enable raymap loss for pointmap_depth_ray
                lambda_pointmap_train = args.lambda_pointmap
            elif args.training_objective == 'raymap':
                lambda_depth_train = 0.0  # disable depth loss for raymap
                lambda_raymap_train = args.lambda_raymap
                lambda_pointmap_train = 0.0  # disable pointmap loss for raymap
            else:  # pointmap
                lambda_depth_train = args.lambda_depth  # enable depth loss for pointmap
                lambda_raymap_train = 0.0  # disable raymap loss for pointmap
                lambda_pointmap_train = args.lambda_pointmap
            
            batch_result = loss_of_one_batch_occany_da3(views=batch, 
                                                    model=model,
                                                    device=device,
                                                    dtype=dtype,
                                                    distill_criterion=distill_criterion,
                                                    distill_model=distill_model,
                                                    sam_model=args.sam_model,
                                                    depth_criterion=depth_criterion,
                                                    raymap_criterion=raymap_criterion,
                                                    pointmap_criterion=pointmap_criterion,
                                                    lambda_depth=lambda_depth_train,
                                                    lambda_raymap=lambda_raymap_train,
                                                    lambda_pointmap=lambda_pointmap_train,
                                                    pose_from_depth_ray=False,
                                                    scale_inv_depth_criterion=scale_inv_depth_criterion,
                                                    lambda_scale_inv_depth=args.lambda_scale_inv_depth,
                                                    aux_metric_pseudo_supervision=args.aux_metric_pseudo_supervision,
                                                    da3_metric_model=da3_metric_model,
                                                    sky_mask_threshold=args.sky_mask_threshold,
                                                    infinidepth_pseudo_supervision=args.infinidepth_pseudo_supervision,
                                                    infinidepth_depth_min=args.infinidepth_depth_min,
                                                    infinidepth_depth_max=args.infinidepth_depth_max,
                                                    lambda_pointmap_lidar=args.lambda_pointmap_lidar,
                                                    lambda_pointmap_pseudo=args.lambda_pointmap_pseudo,
                                                    lambda_depth_lidar=args.lambda_depth_lidar,
                                                    lambda_depth_pseudo=args.lambda_depth_pseudo,
                                                    distill_max_frames=args.distill_max_frames,
                                                    distill_k_schedule=args.distill_k_schedule)
        

        loss, loss_details = batch_result['loss']  # criterion returns two values
        
        loss_value = float(loss)
    
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            for k, v in loss_details.items():
                print("{}: {}".format(k, v))
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Visualize pseudo-depth supervision labels (aux_metric or infinidepth pseudo supervision)
        if (not args.gen
                and (args.aux_metric_pseudo_supervision or args.infinidepth_pseudo_supervision)
                and misc.is_main_process() and n_train_draw < 10):
            pseudo_depth_vis = batch_result.get('pseudo_depth')
            pseudo_mask_vis  = batch_result.get('pseudo_supervision_mask')
            if pseudo_depth_vis is not None and pseudo_mask_vis is not None:
                bs = pseudo_depth_vis.shape[0]
                for batch_idx in range(bs):
                    if n_train_draw >= 10:
                        break
                    n_train_draw += 1
                    T = pseudo_depth_vis.shape[1]
                    pseudo_depth_b = pseudo_depth_vis[batch_idx].detach().cpu()  # (T, H, W)
                    pseudo_mask_b  = pseudo_mask_vis[batch_idx].detach().cpu()   # (T, H, W)
                    pseudo_depth_color = torch.stack([torch.from_numpy(
                        depth2rgb(pseudo_depth_b[t].numpy(),
                                  valid_mask=pseudo_mask_b[t].numpy(),
                                  min_depth=0.1, max_depth=80))
                        for t in range(T)])
                    _log_viz_sample(
                        batch_result, batch_idx, epoch, int(epoch_f * 1000),
                        args.output_dir, log_writer, tb_prefix='train_pseudo_depth',
                        extra_panels=[pseudo_depth_color], max_depth=80.0)

        del loss
        del batch
        del batch_result

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        # Only perform all_reduce for logging if distributed training is active
        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            if args.distributed:
                loss_value_reduce = misc.all_reduce_mean(loss_value)  # All ranks must execute this
            else:
                loss_value_reduce = loss_value
            if log_writer is not None:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_lr', lr, epoch_1000x)
                log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
                for name, val in loss_details.items():
                    log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module,
                   pointmap_criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, distill_model, distill_criterion, log_writer=None, prefix='test',
                   depth_criterion=None, raymap_criterion=None,
                   model_recon=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Always pin the eval loader to epoch 0 so each eval run sees the same samples
    # in the same order — needed for stable visualization (only rank 0 visualizes,
    # so the rank-0 shard must be stable across epochs) and for comparable metrics.
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(0)

    n_draw = 0
    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):


    
        dtype = get_dtype(args)
        # Use loss_of_one_batch_occany_da3_gen in generation mode.
        if args.gen:
            batch_result = loss_of_one_batch_occany_da3_gen(
                views=batch, 
                model=model,
                device=device,
                model_recon=model_recon,
                dtype=dtype,
                distill_criterion=distill_criterion,
                distill_model=distill_model,
                sam_model=args.sam_model,
                pointmap_criterion=pointmap_criterion,
                depth_criterion=depth_criterion,
                raymap_criterion=raymap_criterion,
                lambda_depth=1.0,
                lambda_raymap=0.0,
                lambda_pointmap=1.0,
                pose_from_depth_ray=True,
                projection_features=getattr(args, 'projection_features', 'pts3d_local,pts3d,rgb,conf'),
                lambda_feat_matching=getattr(args, 'lambda_feat_matching', 1.0),
                distill_max_frames=getattr(args, 'distill_max_frames', -1),
                distill_k_schedule=getattr(args, 'distill_k_schedule', 'constant'),
            )
        else:
            batch_result = loss_of_one_batch_occany_da3(views=batch,
                                                    model=model,
                                                    pointmap_criterion=pointmap_criterion,
                                                    device=device,
                                                    distill_criterion=distill_criterion,
                                                    distill_model=distill_model,
                                                    sam_model=args.sam_model,
                                                    depth_criterion=depth_criterion,
                                                    raymap_criterion=raymap_criterion,
                                                    lambda_depth=1.0,
                                                    lambda_raymap=1.0,
                                                    lambda_pointmap=1.0,
                                                    pose_from_depth_ray=True,
                                                    aux_metric_pseudo_supervision=False,
                                                    distill_max_frames=getattr(args, 'distill_max_frames', -1),
                                                    distill_k_schedule=getattr(args, 'distill_k_schedule', 'constant'))

        loss_tuple = batch_result['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)
       
        
        if misc.is_main_process() and n_draw < 8:
            gt_key = "combined_gt"
            bs = batch_result['combined_preds']['rgb'].shape[0]
            for batch_idx in range(bs):
                if n_draw >= 8:
                    break
                n_draw += 1
                T = len(batch_result[gt_key])

                # Collect per-view data and sort by timestep
                timestep    = [batch_result[gt_key][t]['timestep'][batch_idx] for t in range(T)]
                is_raymap   = [batch_result[gt_key][t]['is_raymap'][batch_idx] for t in range(T)]
                gt_pts3d    = torch.stack([batch_result[gt_key][t]['pts3d'][batch_idx] for t in range(T)])  # (T, H, W, 3)
                gt_c2w      = torch.stack([batch_result[gt_key][t]['camera_pose'][batch_idx] for t in range(T)])
                valid_mask  = [batch_result[gt_key][t]['valid_mask'][batch_idx].detach().cpu().numpy() for t in range(T)]

                sorted_idx  = sorted(range(T), key=lambda i: timestep[i])
                is_raymap   = [is_raymap[i] for i in sorted_idx]
                gt_pts3d    = gt_pts3d[sorted_idx]
                gt_c2w      = gt_c2w[sorted_idx]
                valid_mask  = [valid_mask[i] for i in sorted_idx]

                # GT depth from pts3d in camera space
                gt_w2c = torch.linalg.inv(gt_c2w)
                gt_pts3d_local = geotrf(gt_w2c, gt_pts3d).detach().cpu()  # (T, H, W, 3)
                gt_depth_color = torch.stack([torch.from_numpy(
                    depth2rgb(gt_pts3d_local[t, :, :, 2].numpy(),
                              valid_mask=valid_mask[t], min_depth=0.1, max_depth=50))
                    for t in range(T)])

                _log_viz_sample(
                    batch_result, batch_idx, epoch, 1000 * epoch,
                    args.output_dir, log_writer, tb_prefix=f'{prefix}_combined_preds',
                    extra_panels=[gt_depth_color], is_raymap=is_raymap,
                    view_order=sorted_idx)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}


    for name, val in results.items():
        if log_writer is not None:
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)
        else:
            print(f"{prefix}_{name}: {val}")
    return results
