# Abstract class for the trainer
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio
from einops import rearrange
import math

import torch.distributed as dist
import torchvision.utils as vutils

# TensorBoard imports the real TensorFlow (slow, spams cuFFT/cuDNN/cuBLAS
# factory errors) whenever it is installed, unless the `tensorboard.compat.notf`
# marker module is importable — register it so TB uses its TF-free stub.
import sys
import types
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
from torch.utils.tensorboard import SummaryWriter



class DualOptimizer:
    def __init__(self, muon_optimizer=None, adamw_optimizer=None):
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer

    @property
    def param_groups(self):
        groups = []
        if self.muon_optimizer is not None:
            groups.extend(self.muon_optimizer.param_groups)
        if self.adamw_optimizer is not None:
            groups.extend(self.adamw_optimizer.param_groups)
        return groups

    def step(self, closure=None):
        loss = None
        if self.muon_optimizer is not None:
            loss = self.muon_optimizer.step(closure=closure)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.step(closure=closure)
        return loss

    def zero_grad(self, set_to_none=True):
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": None if self.muon_optimizer is None else self.muon_optimizer.state_dict(),
            "adamw": None if self.adamw_optimizer is None else self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if "muon" in state_dict or "adamw" in state_dict:
            if self.muon_optimizer is not None and state_dict.get("muon") is not None:
                self.muon_optimizer.load_state_dict(state_dict["muon"])
            if self.adamw_optimizer is not None and state_dict.get("adamw") is not None:
                self.adamw_optimizer.load_state_dict(state_dict["adamw"])
            return

        if self.muon_optimizer is not None:
            self.muon_optimizer.load_state_dict(state_dict)

class Trainer(object):
    """ Abstract class for each trainer """

    vit = None
    ae = None
    optim = None
    sampler = None
    writer = None
    sae = None

    def __init__(self, args):
        """ Initializes the Trainer class.
         :param
            args: Configuration object containing various hyperparameters and settings, see main.py
        """
        self.args = args

        # Initialize logging writer (TensorBoard)
        is_master = getattr(self.args, "is_master", getattr(self, "is_master", True))
        is_debug = getattr(args, "debug", False)
        writer_log = getattr(self.args, "writer_log", "")
        
        if is_master and not is_debug and writer_log != "":
            self.writer = SummaryWriter(log_dir=writer_log)

    def get_network(self, archi):
        """ Placeholder method to get a neural network architecture. """
        pass

    def get_resume_checkpoint_path(self):
        if not self.args.resume:
            return None

        ckpt = os.path.join(self.args.vit_folder, "current.pth")
        if os.path.isfile(ckpt):
            return ckpt

        return None

    def get_pretrained_checkpoint_path(self):
        ckpt = getattr(self.args, "pretrained_ckpt", None)
        if not ckpt:
            return None

        ckpt = os.path.expanduser(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt}")

        return ckpt

    def get_model_checkpoint_path(self):
        ckpt = self.get_resume_checkpoint_path()
        if ckpt is not None:
            return ckpt

        return self.get_pretrained_checkpoint_path()

    def transformer_size(self, size):
        """ Returns the transformer configuration based on the specified size.
            :param:
                size -> str: Transformer size (tiny, small, base, large, xlarge, 2b).
            :returns
                (hidden_dim, depth, heads) -> tuple: Transformer settings.
        """

        if size == "tiny":
            hidden_dim, depth, heads = 384, 6, 6
        elif size == "small":
            hidden_dim, depth, heads = 384, 12, 6
        elif size == "base":
            hidden_dim, depth, heads = 768, 12, 12
        elif size == "large":
            hidden_dim, depth, heads = 1024, 24, 16
        elif size == "xlarge":
            hidden_dim, depth, heads = 1152, 28, 16
        elif size == "xxlarge":
            hidden_dim, depth, heads = 1536, 28, 16
        elif size.lower() in ["2b", "giant", "xxxl"]:
            hidden_dim, depth, heads = 2048, 32, 16
        else:
            hidden_dim, depth, heads = 768, 12, 12
            if self.args.is_master:
                print("Size of the transformer not understood, initialize a Base VIT")

        return hidden_dim, depth, heads

    def log_add_img(self, names, img, iteration, nrow=None):
        """ Logs an image to TensorBoard.
            :param:
               names     -> str: Tag name for logging.
               img       -> Tensor: Image tensor to be logged.
               iteration -> int: Global step for TensorBoard logging.
        """
        if self.writer is None:
            return
        
        if img.dim() == 5:
            b, c, t, h, w = img.size()
            img = rearrange(img, 'b c t h w -> (b t) c h w', b=b, t=t, c=c, h=h, w=w).contiguous()
        if img.size(0) > 8:
            idx = torch.linspace(0, img.size(0) - 1, steps=8).long()
            img = img[idx]
        b, c, h, w = img.size()

        img = vutils.make_grid(img, nrow=nrow if nrow is not None else min(10, len(img)), padding=2, normalize=False)
        img = (img + 1) / 2 # Scale image from [-1,1] to [0,1]
        img = torch.clip(img * 255, 0, 255).to(torch.uint8) # Convert to 8-bit format

        self.writer.add_image(tag=names, img_tensor=img, global_step=iteration)

    def log_add_txt(self, names, txt, iteration):
        """ Logs a text to TensorBoard.
            :param:
               names     -> str: Tag name for logging.
               txt       -> str: Text content.
               iteration -> int: Global step for TensorBoard logging.
        """
        if self.writer is None:
            return
        self.writer.add_text(tag=names, text_string=txt, global_step=iteration)

    def log_add_scalar(self, names, scalar, iteration):
        """ Logs a scalar value to TensorBoard.
            :param:
               names     -> str: Tag name for logging.
               scalar    -> (float or dict): Scalar value(s) to log.
               iteration -> int: Global step for TensorBoard logging.
        """
        if self.writer is None:
            return
        if isinstance(scalar, dict):
            self.writer.add_scalars(main_tag=names, tag_scalar_dict=scalar, global_step=iteration)
        else:
            self.writer.add_scalar(tag=names, scalar_value=scalar, global_step=iteration)

    def log_add_vid(self, names, vid, iteration):
        """ Logs an video to TensorBoard.
            :param:
               names     -> str: Tag name for logging.
               img       -> Tensor: Image tensor to be logged.
               iteration -> int: Global step for TensorBoard logging.
        """
        if self.writer is None:
            return
        b, c, t, h, w = vid.size()

        vid = rearrange(vid, 'b c t h w -> b t c h w', b=b, t=t, c=c, h=h, w=w).contiguous()
        vid = (vid + 1) / 2 # Scale image from [-1,1] to [0,1]
        vid = torch.clip(vid * 255, 0, 255).to(torch.uint8)  # Convert to 8-bit format
        
        self.writer.add_video(names,
                              vid,  # Shape: (1, T, C, H, W)
                              global_step=iteration,
                              fps=1)

    def get_optim(self, net, lr, mode="AdamW", **kwargs):
        """ Returns an optimizer for the given network.
            :param:
                net      -> (nn.Module or list): Model or list of models whose parameters will be optimized.
                lr       -> float: Learning rate.
                mode     -> str: Optimizer type ("AdamW", "Adam", "SGD").
                **kwargs -> Additional optimizer parameters.

            :return:
                Optimizer: Configured optimizer.
        """

        # Extract parameters from network(s)
        if isinstance(net, list):
            params = []
            for n in net:
                params += list(n.parameters())
        else:
            params = list(net.parameters())

        # Choose an optimizer type
        if mode == "adamw":
            optimizer = optim.AdamW(params, lr=lr, **kwargs)
        elif mode == "adam":
            optimizer = optim.Adam(params, lr=lr, **kwargs)
        elif mode == "sgd":
            optimizer = optim.SGD(params, lr=lr, **kwargs)
        elif mode == "muon":
            from torch.optim import Muon
            muon_lr = kwargs.pop("muon_lr", 1e-4)
            muon_momentum = kwargs.pop("muon_momentum", 0.95)

            muon_params = [p for p in params if p.ndim == 2 and p.requires_grad]
            adam_params = [p for p in params if p.ndim != 2 and p.requires_grad]

            muon_optim = Muon(muon_params, lr=muon_lr, momentum=muon_momentum) if len(muon_params) > 0 else None
            adamw_optim = optim.AdamW(adam_params, lr=lr, **kwargs) if len(adam_params) > 0 else None

            optimizer = DualOptimizer(muon_optim, adamw_optim)
        else:
            optimizer = None

        if optimizer is not None:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        # Resume training from checkpoint if applicable
        ckpt = self.get_resume_checkpoint_path()
        if ckpt is not None:
            # Restore optimizer state only for training flows.
            if getattr(self.args, "test_only", False) or getattr(self.args, "visualize_traj_only", False):
                return optimizer

            # Restore optimizer state
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            if 'optimizer_state_dict' in checkpoint.keys() and checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for group in optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])

        return optimizer

    @staticmethod
    def get_loss(mode="cross_entropy", **kwargs):
        """Static method to return a loss function based on the specified mode. """

        if mode == "l1":
            return nn.L1Loss(**kwargs)
        elif mode == "l2":
            return nn.MSELoss(**kwargs)
        elif mode == "cross_entropy":
            return nn.CrossEntropyLoss(**kwargs)
        
        return nn.MSELoss(**kwargs)

    def train_one_epoch(self, epoch):
        """ Placeholder method for training the model for one epoch. """
        return

    def fit(self):
        """ Placeholder method for the main training loop. """
        pass

    @staticmethod
    def all_gather(obj, reduce="mean"):
        """ Static method to gather objects from all processes and reduce them.
            :param:
               obj -> any:The object to gather.
               reduce -> str: The reduction operation to apply. Options are "mean", "sum", or "none".
            :returns:
               obj -> torch.Tensor: The reduced object.
        """
        world_size = dist.get_world_size()
        tensor_list = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather_object(tensor_list, obj)
        obj = torch.FloatTensor(tensor_list)
        if reduce == "mean":
            obj = obj.mean()
        elif reduce == "sum":
            obj = obj.sum()
        elif reduce == "none":
            pass
        else:
            raise NameError("reduction not known")

        return obj

    def save_network(self, model, path, optimizer=None, iter=None, global_epoch=None, ema_state=None,
                     extra=None):
        """ Save the state of the model, including the iteration,
            the optimizer state and the current epoch
            :params:
                model        -> nn.Module: The model to save.
                path         -> str: The path where the model state will be saved.
                optimizer    -> torch.optim.Optimizer: (optional) The optimizer whose state should be saved.
                iter         -> int: The current iteration number.
                global_epoch -> int: The current global epoch number.
                extra        -> dict: (optional) extra provenance keys merged into the saved dict.
        """
        state_dict = model.module.state_dict() if self.args.is_multi_gpus else model.state_dict()

        # Back up any existing file first so a crash mid-save can't truncate it.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            root, ext = os.path.splitext(path)
            os.replace(path, f"{root}_backup{ext}")
        torch.save({'iter': iter,
                'global_epoch': global_epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
                'ema_state': ema_state,
                **(extra or {})
                },
               path)

    def adapt_learning_rate(self, mode=None):
        """ Method to adapt the learning rate based on the current iteration.
            This method implements a linear warmup for the first few iterations and a cosine decay for the last 1
        """
        if self.args.iter < self.args.warm_up:  # linear warmup updates
            if self.args.warm_up > 0:
                scale = self.args.iter / self.args.warm_up
            else:
                scale = 1.0
        # cosine learning rate decay for the last 10% iterations
        elif (self.args.iter > (self.args.max_iter - (.1 * self.args.max_iter))) and mode=="constant_then_cosine":
            decay_step = .1 * self.args.max_iter
            decay = (self.args.iter - (self.args.max_iter - decay_step)) / decay_step
            cosine_decay = (1 + np.cos(np.pi * decay))
            scale = max(0.5 * cosine_decay, 0.01)
        elif mode=="cosine":
            progress = (self.args.iter - self.args.warm_up) / (self.args.max_iter - self.args.warm_up)
            progress = min(max(progress, 0.0), 1.0)
            scale = max(0.5 * (1 + math.cos(math.pi * progress)), 0.01)
        else: # basic lr for the rest of the time
            scale = 1.0

        for group in self.optim.param_groups:
            base_lr = group.get('initial_lr', group['lr'])
            group['lr'] = base_lr * scale
