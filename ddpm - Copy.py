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
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from models.guided_diffusion.unet import SuperResModel as GuidedDiffusion_SRModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

from sampler import NoiseScheduleVP, model_wrapper, DPM_Solver

import torchvision.utils as tvu
from torchvision.utils import save_image
 
 
CIFAR10_DATA_DIR = "data/cifar10"
##IMAGENET_DATA_DIR = "../_data/datasets/tiny-imagenet/val"
IMAGENET_DATA_DIR = "../_data/datasets/imagenet/val"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, default=0.50, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=10, help="the number of Monte Carlo samples for selection")
    parser.add_argument("--N", type=int, default=10, help="the number of Monte Carlo samples to use for estimation")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.0001, help="failure probability")
    parser.add_argument("--outfile", type=str, default="results/imagenet/sigma_0.50", help="output file")

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
        ##self.model_mean_type = 'start_x'
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

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
            
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)


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
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape).cuda() * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).cuda()
            * noise
        )


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
        ##timesteps = torch.tensor(timesteps)
        timesteps = timesteps.clone().detach()
        ##res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        res = arr.to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def sample_image(self, x, model, last=True, classifier=None, base_samples=None):
        assert last
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
                print("t=============================================\n", t)
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

class DiffusionModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        
        args, config = parse_args_and_config()
        self.args = args
        self.config = config
        
        if self.config.model.model_type == "guided_diffusion":
            model = GuidedDiffusion_Model(
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

        
        diffusion = Diffusion(args, config)
        
        ##print("Loading checkpoint {}".format(ckpt))
        print("Loading checkpoint from pretrained/model-790000.ckpt")

        '''model.load_state_dict(
            ##torch.load("../_data/pretrained/cifar10/model-790000.ckpt")
            ##torch.load("../_data/pretrained/imagenet/imagenet64_uncond_100M_1500K.pt") 128x128_classifier.pt
            torch.load("../_data/pretrained/imagenet/128x128_diffusion.pt")
        )'''

        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = os.path.expanduser(self.config.model.ckpt_dir)
            states = torch.load(
                ckpt_dir,
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
                ) 
                classifier.load_state_dict(states, strict=True)
                if self.config.classifier.use_fp16:
                    classifier.convert_to_fp16()
                    # classifier.module.convert_to_fp16()
            else:
                classifier = None

        self.classifier_sampling = classifier 
        
        model.eval()
        ##model.eval()

        self.model = model 
        self.diffusion = diffusion 

        # Load the BEiT model
        classifier_ = timm.create_model('beit_large_patch16_512', pretrained=True)
        classifier_.eval().cuda()

        self.classifier = classifier_

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()

        '''classifier.load_state_dict(
            torch.load("pretrained/256x256_classifier.pt")
            ##torch.load("pretrained/imagenet64_uncond_100M_1500K.pt")
        )'''

        ##self.classifier.eval()
    def forward(self, x, t):
            
        x_in = x * 2 -1
        x = self.sampling(x_in, t)
        ##x = self.diffusion.sample_image(x_in, self.model, t)
        print("-----------------------tttttttttt")

        x = inverse_data_transform(self.config, x)
        x = x.clone().detach()

        imgs = torch.nn.functional.interpolate(x, (512, 512), mode='bicubic', antialias=True)   
        
        from torchvision.utils import save_image
        output_dir = "samples/imagenet/"
        import os
        os.makedirs(output_dir, exist_ok=True)
        for i, sample in enumerate(imgs):
            save_image(sample, os.path.join(output_dir, f"sample_{i}.png"))
               
               
        ##imgs = torch.tensor(imgs).cuda()
        imgs = imgs.clone().detach().cuda()

            
        with torch.no_grad():
            out = self.classifier(imgs)


        ##return out.logits
        return out
    
    def sampling(self, x_start, t):
        assert x_start.ndim == 4, x_start.ndim
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
        with torch.no_grad():
            out = self.diffusion.sample_image(
                x_t_start,
                self.model,
                ##self.classifier_sampling,#################################################################################################################
                ##self.classifier,
                ##clip_denoised=True
            )

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

    print(t)
    print("--------------------------------")

    # Define the smoothed classifier 
    smoothed_classifier = Smooth(model, 1000, args.sigma, t)

    f = open(args.outfile, 'w')
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

            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                index, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
            index += 1
    

    
def sample(args, config):

    # load model
    print('starting the model and loader...')
    model = DiffusionModel(args, config)
    model = model.eval().to(config.device)

    # load dataset
    ##dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=IMAGENET_DATA_DIR, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("dataset loaded ===========================================================")


    certify(model, dataloader, args, config)   

def launch():
    args, config = parse_args_and_config()
    sample(args, config)
    
if __name__ == "__main__":
    launch()


