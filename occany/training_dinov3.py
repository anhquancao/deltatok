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

from occany.model.model_dinov3 import Dinov3Wrapper
from occany.da3_inference import loss_of_one_batch_occany_da3
from occany.utils.helpers import depth2rgb
from dust3r.utils.geometry import geotrf

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention

import occany.utils.io_da3 as checkpoints
import occany.model.must3r_blocks.optimizer as optim
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3 recon training', add_help=False)

    # fine-tuning
    parser.add_argument('--finetune_dual_dpt_only', default=False, action='store_true',
                        help="Only finetune the DualDPT head, freeze backbone")
    parser.add_argument('--fine_tune_layers', type=str, default=None,
                        help="Backbone block indices to fine-tune while freezing the rest. "
                             "Accepts a comma-separated list, a range, or a mix: "
                             "'26,27,28,29,30,31', '11-31', or '0,5,11-13'.")
    parser.add_argument('--freeze_head', default=False, action='store_true',
                        help="Freeze the DualDPT head (Depth head)")

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

    # model settings (DINOv3)
    parser.add_argument('--dinov3_config', type=str, required=True,
                        help="Path to DINOv3 backbone YAML (e.g. occany/configs/dinov3/occany_dinov3_vith16plus.yaml).")
    parser.add_argument('--dinov3_weights', type=str, required=True,
                        help="Path to DINOv3 pretrained .pth checkpoint.")
    parser.add_argument('--multiview', action='store_true', default=False, help='use multiview loss')
    parser.add_argument('--training_objective', type=str, default='pointmap',
                        choices=['pointmap', 'depth_ray', 'pointmap_depth_ray', 'raymap'],
                        help='Training objective: "pointmap" uses pointmap+depth loss, '
                             '"depth_ray" uses depth+raymap loss, '
                             '"pointmap_depth_ray" uses pointmap+depth+raymap loss, '
                             '"raymap" uses only raymap loss')

    # loss weights (used depending on training_objective)
    parser.add_argument('--lambda_depth', type=float, default=1.0,
                        help='weight for depth loss (used when training_objective=depth_ray)')
    parser.add_argument('--lambda_raymap', type=float, default=1.0,
                        help='weight for raymap loss (used when training_objective=depth_ray)')
    parser.add_argument('--lambda_pointmap', type=float, default=1.0,
                        help='weight for pointmap loss (used when training_objective=pointmap)')
    parser.add_argument('--depth_lambda_c', type=float, default=1.0,
                        help='confidence weight for depth loss')
    parser.add_argument('--depth_alpha', type=float, default=0.0,
                        help='gradient loss weight for depth loss (0 for sparse lidar)')
    parser.add_argument('--raymap_lambda_c', type=float, default=1.0,
                        help='confidence weight for raymap loss')
    parser.add_argument('--pointmap_lambda_c', type=float, default=1.0,
                        help='confidence weight for pointmap loss (0 to disable, used in training only)')
    parser.add_argument('--loss_type', type=str, default='L2', choices=['L1', 'L2'],
                        help='Loss type to use (L1 or L2) for pointmap and raymap losses')

    # InfiniDepth pseudo-supervision: train depth/pointmap against pseudo depth
    # produced by InfiniDepth_DepthSensor conditioned on the LiDAR projection
    # (full sensor depth + all valid LiDAR points as the dense prompt).
    parser.add_argument('--infinidepth_pseudo_supervision', action='store_true', default=False,
                        help='Training-only mode: use precomputed InfiniDepth pseudo-depth (sibling '
                             '<stem>.infinidepth.png produced by extract_infinidepth_pseudo.py) as the '
                             'pseudo-GT depth teacher; pseudo depth/pointmap supervise the depth and '
                             'pointmap losses (requires load_infinidepth_pseudo=True on each train dataset)')
    parser.add_argument('--infinidepth_depth_min', type=float, default=1e-3,
                        help='Min valid pseudo depth (m) for the pseudo-supervision range mask')
    parser.add_argument('--infinidepth_depth_max', type=float, default=150.0,
                        help='Max valid pseudo depth (m) for the pseudo-supervision range mask. '
                             "Set just below InfiniDepth's 200m sky-sentinel and disparity-clamp tail "
                             '(1/5e-3=200) to exclude outliers while keeping the longest reliable range.')
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
                        help='InfiniDepth pseudo-GT pseudo weight under --infinidepth_pseudo_supervision (multiplies --lambda_depth)')

    return parser


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
    print("training_mode: recon")

    if args.infinidepth_pseudo_supervision:
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

    print(f"[INFO] Building Dinov3Wrapper from config={args.dinov3_config}, weights={args.dinov3_weights}")
    model = Dinov3Wrapper(args.dinov3_config, weights_path=args.dinov3_weights)
    model = model.to(device)
    backbone_metadata = model.get_backbone_metadata()
    print(
        f"[INFO] Loaded DINOv3 backbone: name={backbone_metadata['name']}, "
        f"token_dim={backbone_metadata['token_dim']}, feature_dim={backbone_metadata['feature_dim']}, "
        f"out_layers={list(backbone_metadata['out_layers'])}, alt_start={backbone_metadata['alt_start']}"
    )

    def _parse_layer_spec(spec: str) -> list[int]:
        out: set[int] = set()
        for tok in spec.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if '-' in tok:
                lo, hi = tok.split('-', 1)
                lo_i, hi_i = int(lo), int(hi)
                if lo_i > hi_i:
                    raise ValueError(f"invalid range {tok!r}: lo > hi")
                out.update(range(lo_i, hi_i + 1))
            else:
                out.add(int(tok))
        return sorted(out)

    fine_tune_layers = None
    if args.fine_tune_layers is not None:
        fine_tune_layers = _parse_layer_spec(args.fine_tune_layers)

    # Selective layer fine-tuning
    if fine_tune_layers is not None:
        print(f'Selective fine-tuning: freezing backbone except layers {fine_tune_layers}...')
        # Freeze entire backbone first
        for param in model.backbone.vit.parameters():
            param.requires_grad = False
        # Unfreeze specified layers
        for layer_idx in fine_tune_layers:
            block = model.backbone.vit.blocks[layer_idx]
            for param in block.parameters():
                param.requires_grad = True

    # Apply head freezing
    if args.freeze_head:
        print('Freezing DualDPT head...')
        for param in model.head.parameters():
            param.requires_grad = False
    else:
        # Keep DualDPT head trainable by default unless specified otherwise
        for param in model.head.parameters():
            param.requires_grad = True

    if fine_tune_layers is not None or args.freeze_head:
        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Trainable parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)')

    # Apply fine-tuning mode: only train DualDPT head if specified
    elif args.finetune_dual_dpt_only:
        print('Freezing backbone and only training DualDPT head...')
        for param in model.backbone.vit.parameters():
            param.requires_grad = False
        # Keep DualDPT head trainable (model.head)
        for param in model.head.parameters():
            param.requires_grad = True
        print('DualDPT head parameters will be trained, backbone is frozen')

    model_without_ddp = model

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

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {}
    for dataset in args.test_dataset.split('+'):
        testset = build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        name_testset = dataset.split('(')[0]
        data_loader_test[name_testset] = testset

    # Create DA3-specific criteria using PointmapLoss from losses_da3.py
    from occany.loss.losses_da3 import PointmapLoss, DepthLosses, RaymapLoss

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
    if args.infinidepth_pseudo_supervision:
        print('   - InfiniDepth pseudo supervision is ENABLED for training (dual GT mode)')
        print(f'   - Pseudo depth loaded from precomputed sibling `.infinidepth.png` files; range [{args.infinidepth_depth_min}, {args.infinidepth_depth_max}] m')
        print(f'   - Dual losses: lidar GT (lambda_pointmap_lidar={args.lambda_pointmap_lidar}, lambda_depth_lidar={args.lambda_depth_lidar}) + InfiniDepth pseudo GT (lambda_pointmap_pseudo={args.lambda_pointmap_pseudo}, lambda_depth_pseudo={args.lambda_depth_pseudo})')
        print('   - Evaluation still uses direct lidar GT')

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

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = []
    param_groups += optim.get_parameter_groups(model_without_ddp, 0, args.weight_decay)

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
                                       log_writer=log_writer, args=args, prefix=test_name,
                                       depth_criterion=depth_criterion_test,
                                       raymap_criterion=raymap_criterion_test)
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
            log_writer=log_writer,
            args=args,
            depth_criterion=depth_criterion,
            raymap_criterion=raymap_criterion,
            pointmap_criterion=pointmap_criterion_train)

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
                    args, log_writer=None,
                    depth_criterion=None, raymap_criterion=None, pointmap_criterion=None):

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
                                                distill_criterion=None,
                                                distill_model=None,
                                                depth_criterion=depth_criterion,
                                                raymap_criterion=raymap_criterion,
                                                pointmap_criterion=pointmap_criterion,
                                                lambda_depth=lambda_depth_train,
                                                lambda_raymap=lambda_raymap_train,
                                                lambda_pointmap=lambda_pointmap_train,
                                                pose_from_depth_ray=False,
                                                infinidepth_pseudo_supervision=args.infinidepth_pseudo_supervision,
                                                infinidepth_depth_min=args.infinidepth_depth_min,
                                                infinidepth_depth_max=args.infinidepth_depth_max,
                                                lambda_pointmap_lidar=args.lambda_pointmap_lidar,
                                                lambda_pointmap_pseudo=args.lambda_pointmap_pseudo,
                                                lambda_depth_lidar=args.lambda_depth_lidar,
                                                lambda_depth_pseudo=args.lambda_depth_pseudo)

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

        # Visualize pseudo-depth supervision labels (infinidepth pseudo supervision)
        if (args.infinidepth_pseudo_supervision
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
                   args, log_writer=None, prefix='test',
                   depth_criterion=None, raymap_criterion=None):
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
        batch_result = loss_of_one_batch_occany_da3(views=batch,
                                                model=model,
                                                pointmap_criterion=pointmap_criterion,
                                                device=device,
                                                distill_criterion=None,
                                                distill_model=None,
                                                depth_criterion=depth_criterion,
                                                raymap_criterion=raymap_criterion,
                                                lambda_depth=1.0,
                                                lambda_raymap=1.0,
                                                lambda_pointmap=1.0,
                                                pose_from_depth_ray=True)

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
