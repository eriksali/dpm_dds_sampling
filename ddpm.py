import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
##from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from core import Smooth
import argparse
from torchvision.utils import save_image

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from torchvision import transforms, datasets
import yaml
import timm
import logging
import time 
import datetime 
import glob
from tkinter import E

import blobfile as bf

import numpy as np
import tqdm
import math
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from models.diffusion import Model
##from models.improved_diffusion.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from models.guided_diffusion.unet import SuperResModel as GuidedDiffusion_SRModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
##from evaluate.fid_score import calculate_fid_given_paths

from sampler import NoiseScheduleVP, model_wrapper, DPM_Solver

import torchvision.utils as tvu
from torchvision.utils import save_image

from models.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
 
CIFAR10_DATA_DIR = "data/cifar10"
##IMAGENET_DATA_DIR = "../_data/datasets/tiny-imagenet/val"
IMAGENET_DATA_DIR = "../_data/datasets/imagenet/val"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, default=0.25, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=100, help="the number of Monte Carlo samples for selection")
    parser.add_argument("--N", type=int, default=1000, help="the number of Monte Carlo samples to use for estimation")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.0001, help="failure probability")
    parser.add_argument("--outfile", type=str, default="results/imagenet/sigma_0.25", help="output file")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=False,
        default="",
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach ('generalized'(DDIM) or 'ddpm_noisy'(DDPM) or 'dpmsolver' or 'dpmsolver++')",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--base_samples",
        type=str,
        default="fid_stats/fid_stats_cifar10_train_pytorch.npz",
        help="base samples for upsampling, *.npz",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--dpm_solver_order", type=int, default=3, help="order of dpm-solver"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--fixed_class", type=int, default=None, help="fixed class label for conditional sampling"
    )
    parser.add_argument(
        "--dpm_solver_atol", type=float, default=0.0078, help="atol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_rtol", type=float, default=0.05, help="rtol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_method",
        type=str,
        default="singlestep",
        help="method of dpm_solver ('adaptive' or 'singlestep' or 'multistep' or 'singlestep_fixed'",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="dpm_solver",
        help="type of dpm_solver ('dpm_solver' or 'taylor'",
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--lower_order_final", action="store_true", default=False)
    parser.add_argument("--thresholding", action="store_true", default=False)
    
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--port", type=str, default="12355")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    '''if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)'''

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.backends.cudnn.benchmark = True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_data_for_worker(base_samples, batch_size, cond_class):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if cond_class:
            label_arr = obj["arr_1"]
    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if cond_class:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if cond_class:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == 'discrete':
        num_steps = num_diffusion_timesteps  # Number of discrete betas
        step_size = 0.01  # Change this based on your requirements
        betas = np.array([0.0001 + i * step_size for i in range(num_steps)], dtype=np.float64)

    elif beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, rank=None):
        self.args = args
        self.config = config
        if rank is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = rank
            self.rank = rank
        self.device = device

        self.model_var_type = config.model.var_type
        self.model_mean_type = 'start_x'
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        print("self.num_timesteps = betas.shape[0]*************************************\n", self.num_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1].cpu().numpy(), self.posterior_variance[1:].cpu().numpy())
        )
        self.posterior_mean_coef1 = (
            betas* torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        ##self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        ##self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        ##self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)


    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        print("448=========================\n")
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out
        
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        print("476=========================\n")
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape).cuda() * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).cuda()
            * noise
        )

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = self._extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = mean_pred + nonzero_mask.cpu() * sigma * noise.cpu()
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        print("556=========================\n")
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]



    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        ##print("x_t.device==============================\n", x_t.device)
        ##print("self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)==============================\n", self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape).device)
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t.cpu()
            - pred_xstart
        ) / self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _p_sample(x, t):
        """
        One step of revese process sampling - Algorithm 2 from the paper
        """
        ## alpha_t = alphas.gather(-1, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        
        ##sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.gather(-1, t)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod_double
        
        # Get mean x[t - 1] conditioned at x[t] - see eq. (11) in the paper
        ##model_mean = torch.sqrt(1.0 / alpha_t) * (x - (1 - alpha_t) * denoise(x, t) / sqrt_one_minus_alphas_cumprod_t)
        model_mean = torch.sqrt(1.0 / alpha_t) * (x - (1 - alpha_t) * noise_pred_fn(x, t) / sqrt_one_minus_alphas_cumprod_t)
        # Get variance of x[t - 1]
        model_var = posterior_variance.gather(-1, t)
        # Samples for the normal distribution with given mean and variance
        return model_mean + torch.sqrt(model_var) * torch.randn(1)

    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod_double, t, x.shape
        )
        ## sqrt_recip_alphas_t = extract(= torch.sqrt(1.0 / alphas), t, x.shape)
        sqrt_recip_alphas_t = self.extract(torch.sqrt(1.0 / self.alphas), t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:, None,None,None].cuda() * (
            x - betas_t[:, None,None,None].cuda() * model(x, t) / sqrt_one_minus_alphas_cumprod_t[:, None,None,None].cuda()
        )

        posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise[:, None,None,None].cuda()
        
    '''@torch.no_grad()
    def p_sample(model, x, t, t_index):
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod_double, t, x.shape
        )
        ## sqrt_recip_alphas_t = extract(= torch.sqrt(1.0 / alphas), t, x.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise'''

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(model, shape):
        ##device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            img = p_sample(model, img, torch.full((b,), i, device=self.device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(model, image_size, batch_size=16, channels=3):
        return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


    '''timesteps = 1000

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)'''

    def extract(self, a, t, x_shape):
        t = torch.tensor(t)
        batch_size = t.shape
        ##print(a)
        ##print(t)
        out = a.gather(-1, t.cuda())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


            
    def p_sample_(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        ##x=x[0]
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        '''print("out[mean].shape = ", out["mean"].device)
        print("nonzero_mask.shape = ", nonzero_mask.device)
        print("out[log_variance].shape = ", out["log_variance"].device)
        print("noise.shape = ", noise.device)'''
        sample = out["mean"].cpu() + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise.cpu()
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        ##print("x[0].shape = ", x[0].shape)
        ##print("x.shape[:2] = ", x.shape[:2])
        ##print("x.shape = ", x.shape)
        B, C = x.shape[:2]
        ##print("(B,) = ", (B,))

        ##t = torch.tensor([t] * x.shape[0]).cuda()
        
        assert t.shape == (B,)
        ##model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_output = model(x, t, **model_kwargs)
        model_output = torch.split(model_output, 3, dim=1)[0]
        print("model_output.shape=========================\n", model_output.shape)
        print("x.shape=========================\n", x.shape)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            'fixedlarge': (
                np.append(self.posterior_variance[1].cpu().numpy(), self.betas[1:].cpu().numpy()),
                np.log(np.append(self.posterior_variance[1].cpu().numpy(), self.betas[1:].cpu().numpy())),
            ),
            
            'fixedsmall': (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        print("886=========================\n")
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        print("888=========================\n")
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == 'start_x':
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output



        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    @staticmethod
    def _extract_into_tensor(arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        ##res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        if isinstance(arr, torch.Tensor) and isinstance(timesteps, torch.Tensor):
            res = torch.tensor(arr.cpu().numpy()[timesteps.cpu()]).float()
        else:
            res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        print("res=========================\n", res)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        print("944=========================\n")
        return (  # (xprev - coef2*x_t) / coef1
            self._extract_into_tensor(1.0 / self.posterior_mean_coef1.cpu().numpy(), t.cpu(), x_t.cpu().shape) * xprev.cpu()
            - self._extract_into_tensor(
                self.posterior_mean_coef2.cpu().numpy() / self.posterior_mean_coef1.cpu().numpy(), t.cpu(), x_t.cpu().shape
            )
            * x_t.cpu()
        )
        
    def sample(self):
        if self.config.model.model_type == 'improved_ddpm':
            model = ImprovedDDPM_Model(
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                use_checkpoint=self.config.model.use_checkpoint,
                num_heads=self.config.model.num_heads,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm
            )
        elif self.config.model.model_type == "guided_diffusion":
            if self.config.model.is_upsampling:
                model = GuidedDiffusion_SRModel(
                    image_size=self.config.model.large_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
            else:
                model = GuidedDiffusion_Model(
                ##model = UNetModel(
                    image_size=self.config.model.image_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
        else:
            model = Model(self.config)

        model = model.to(self.rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}

        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = os.path.expanduser(self.config.model.ckpt_dir)
            states = torch.load(
                ckpt_dir,
                map_location=map_location
            )
            # states = {f"module.{k}":v for k, v in states.items()}
            if self.config.model.model_type == 'improved_ddpm' or self.config.model.model_type == 'guided_diffusion':
                model.load_state_dict(states, strict=True)
                if self.config.model.use_fp16:
                    model.convert_to_fp16()
            else:
                # TODO: FIXME
                # model = torch.nn.DataParallel(model)
                # model.load_state_dict(states[0], strict=True)
                model.load_state_dict(states, strict=True)

            if self.config.model.ema: # for celeba 64x64 in DDIM
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None

            if self.config.sampling.cond_class and not self.config.model.is_upsampling:
                classifier = GuidedDiffusion_Classifier(
                    image_size=self.config.classifier.image_size,
                    in_channels=self.config.classifier.in_channels,
                    model_channels=self.config.classifier.model_channels,
                    out_channels=self.config.classifier.out_channels,
                    num_res_blocks=self.config.classifier.num_res_blocks,
                    attention_resolutions=self.config.classifier.attention_resolutions,
                    channel_mult=self.config.classifier.channel_mult,
                    use_fp16=self.config.classifier.use_fp16,
                    num_head_channels=self.config.classifier.num_head_channels,
                    use_scale_shift_norm=self.config.classifier.use_scale_shift_norm,
                    resblock_updown=self.config.classifier.resblock_updown,
                    pool=self.config.classifier.pool
                )
                ckpt_dir = os.path.expanduser(self.config.classifier.ckpt_dir)
                states = torch.load(
                    ckpt_dir,
                    map_location=map_location,
                )
                # states = {f"module.{k}":v for k, v in states.items()}
                classifier = classifier.to(self.rank)
                # classifier = DDP(classifier, device_ids=[self.rank])
                classifier.load_state_dict(states, strict=True)
                if self.config.classifier.use_fp16:
                    classifier.convert_to_fp16()
                    # classifier.module.convert_to_fp16()
            else:
                classifier = None
        else:
            classifier = None
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            if self.rank == 0:
                print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=map_location))

        model.eval()

        if self.args.fid:
            if not os.path.exists(os.path.join(self.args.exp, "fid.npy")):
                self.sample_fid(model, classifier=classifier)
                torch.distributed.barrier()
                if self.rank == 0:
                    print("Begin to compute FID...")
                    fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=8)
                    print("FID: {}".format(fid))
                    np.save(os.path.join(self.args.exp, "fid"), fid)
        elif self.args.interpolation:
             self.sample_interpolation(model)
        elif self.args.sequence:
             self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, classifier=None):
        config = self.config
        total_n_samples = config.sampling.fid_total_samples
        world_size = torch.cuda.device_count()
        if total_n_samples % config.sampling.batch_size != 0:
            raise ValueError("Total samples for sampling must be divided exactly by config.sampling.batch_size, but got {} and {}".format(total_n_samples, config.sampling.batch_size))
        if len(glob.glob(f"{self.args.image_folder}/*.png")) == total_n_samples:
            return
        else:
            n_rounds = total_n_samples // config.sampling.batch_size // world_size
        img_id = self.rank * total_n_samples // world_size

        if self.config.model.is_upsampling:
            base_samples_total = load_data_for_worker(self.args.base_samples, config.sampling.batch_size, config.sampling.cond_class)

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                if self.config.model.is_upsampling:
                    base_samples = next(base_samples_total)
                else:
                    base_samples = None

                x, classes = self.sample_image(x, model, classifier=classifier, base_samples=base_samples)

                # end.record()
                # torch.cuda.synchronize()
                # t_list.append(start.elapsed_time(end))
                x = inverse_data_transform(config, x)
                for i in range(x.shape[0]):
                    if classes is None:
                        path = os.path.join(self.args.image_folder, f"{img_id}.png")
                    else:
                        path = os.path.join(self.args.image_folder, f"{img_id}_{int(classes.cpu()[i])}.png")
                    tvu.save_image(x.cpu()[i], path)
                    img_id += 1
        # # Remove the time evaluation of the first batch, because it contains extra initializations
        # print('time / batch', np.mean(t_list[1:]) / 1000., 'std', np.std(t_list[1:]) / 1000.)

    def sample_sequence(self, model, classifier=None):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, classifier=classifier)

        x = [inverse_data_transform(config, y) for y in x]
        print("x######=======================\n", x)
        ###print("x.shape=======================\n", x.shape) list without shape

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_image(self, x, model, last=False, classifier=None, base_samples=None):
        ##assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale
        if self.config.sampling.cond_class:
            if self.args.fixed_class is None:
                classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
            else:
                classes = torch.randint(low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)).to(x.device)
        else:
            classes = None
        
        if base_samples is None:
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = generalized_steps(x, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = ddpm_steps(x, seq, model_fn, self.betas, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type in ["dpmsolver", "dpmsolver++"]:
            from dpm_solver.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver
            def model_fn(x, t, **model_kwargs):
                print("t*********************************************\n", t)
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            dpm_solver = DPM_Solver(
                model_fn_continuous,
                noise_schedule,
                algorithm_type=self.args.sample_type,
                correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
            )
            x = dpm_solver.sample(
                x,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                order=self.args.dpm_solver_order,
                skip_type=self.args.skip_type,
                method=self.args.dpm_solver_method,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
                solver_type=self.args.dpm_solver_type,
                atol=self.args.dpm_solver_atol,
                rtol=self.args.dpm_solver_rtol,
            )
            # x = x.cpu()
        else:
            raise NotImplementedError
        return x, classes
       
    def test(self):
        pass
        
class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class DiffusionModel(nn.Module):
        
    def __init__(self, args, classifier_name="beit"):
        super().__init__()
        self.args = args
        
        args, config = parse_args_and_config()
        self.args = args
        self.config = config        
        
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("../_data/pretrained/imagenet/256x256_diffusion_uncond.pt")
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 

        # Load the BEiT model
        classifier = timm.create_model('beit_large_patch16_512', pretrained=True)
        classifier.eval().cuda()

        self.classifier = classifier

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()

    def forward(self, x, t):
        print("t*********************************************\n", t)
        
        ##x_in = x * 2 -1
        ##x_in = self.scale*(x_in)

        ##t = torch.tensor([self.t]*10)
        ##imgs = self.diffusion.sample(x_in, self.model)
        ##imgs = self.sampling(x, t[0].item())
        ##self.diffusion.sample_sequence(self.model)
        ##print("x.shape=========================================================\n", x.shape)
        ##x = self.sampling(x_in, t)
        x = self.sampling(x, t)
        ##imgs = inverse_data_transform(config, x)
        imgs = inverse_data_transform(self.config, x)
        ##print("imgs.shape", x)
        ##print("out.shape=========================================================\n", out)
        ###############imgs = inverse_data_transform(self.config, x)
        ##imgs = torch.nn.functional.interpolate(imgs, (224, 224), mode='bicubic', antialias=True)
        imgs = torch.nn.functional.interpolate(x, (512, 512), mode='bicubic', antialias=True) 
                             
        ##imgs = torch.tensor(imgs).cuda()
        imgs = imgs.clone().detach().cuda()

            
        with torch.no_grad():
            out = self.classifier(imgs)

        
        from torchvision.utils import save_image
        output_dir = "samples/imagenet/"
        import os
        os.makedirs(output_dir, exist_ok=True)
        for i, sample in enumerate(imgs):
            save_image(sample, os.path.join(output_dir, f"sample_{i}.png"))
               

        ##return out.logits
        return out
   
    def sampling(self, x_start, t):
        assert x_start.ndim == 4, x_start.ndim
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        '''with torch.no_grad():
            ##if multistep:
            ##out = x_t_start
            for i in range(t)[::-1]:
                print(i)
                
                x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
                t_batch = torch.tensor([i] * len(x_start)).cuda()
                out = self.diffusion.sample_image(
                ##out = self.diffusion.ddim_sample(
                    x_t_start,
                    self.model,
                    ##out,
                    ##t_batch,
                    ##clip_denoised=True
                )'''##['sample']
        
        with torch.no_grad():
            ##out = self.diffusion.sample(
            out = self.diffusion.sample_image(
            ##out = self.diffusion.ddim_sample(
                x_t_start,
                self.model,
                ##t_batch,
                ##clip_denoised=True
            )##['pred_xstart']

        return out
       
     
def certify(model, dataloader, args, config):
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a
        ##real_sigma = model.diffusion.noise_schedule.marginal_std(t)
        ##print("a------------------------------------------------------------------")
        ##print(a)

    ##t = torch.tensor(t * args.batch_size)
    ##t = torch.tensor(t)
    print(t)
    print("--------------------------------")

    # Define the smoothed classifier 
    ##smoothed_classifier = Smooth(model, 10, args.sigma, t)
    smoothed_classifier = Smooth(model, 1000, args.sigma, t)

    f = open(args.outfile, 'w')
    ##print("idx       \t label     \t prediction\t radius    \t correct   \t time", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    ##total_num = 0
    correct = 0
    index = 0
    # Iterate through the dataset
    for inputs, labels in dataloader:
        for i in range(len(inputs)):
            if i % args.skip != 0:
                continue

            (x, label) = inputs[i], labels[i].item()
            print("label#########################33\n", label)

            before_time = time.time()
            prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
            after_time = time.time()
            print("prediction********************\n", prediction)

            correct = int(prediction == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            ##total_num += 1

            ##print("{}   \t\t {}\t\t\t {}\t\t\t {:.3}\t\t {}\t\t\t {}".format(
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                index, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
            index += 1
    

def sample(args, config):

    # load model
    print('starting the model and loader...')
    model = DiffusionModel(args, config)
    ##model = DiffusionModel()
    model = model.eval().to(config.device)

    # load dataset
    ##dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=IMAGENET_DATA_DIR, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    print("dataset loaded ===========================================================")
    # load dataset
    ##dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    '''import deeplake
    ##ds = deeplake.load("hub://activeloop/tiny-imagenet-validation")
    ds = deeplake.load("../_data/datasets/tiny-imagenet-200/val")     
    dataloader = ds.pytorch(num_workers=0, batch_size=args.batch_size, shuffle=False)'''


    certify(model, dataloader, args, config)   

def launch():
    args, config = parse_args_and_config()
    sample(args, config)
    ##sample_pretrained_model(args, config)

     
if __name__ == "__main__":
    launch()


