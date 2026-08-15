#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tuning script for Stable Video Diffusion with support for LoRA."""
from collections import OrderedDict
import argparse
import random
import logging
import math
import os
import gc
import cv2
import yaml
import shutil
import pyiqa
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm as tqdma
from tqdm import tqdm
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version, is_wandb_available, load_image

from peft import LoraConfig
from diffusers.training_utils import cast_training_params

from geonvs.seva.lr_scheduler import get_scheduler_
from geonvs.seva.modules.diffusionmodules.loss_weighting import get_loss_weighting
from geonvs.seva.modules.diffusionmodules.sigma_sampling import DiscreteSampling
from geonvs.seva.modules.diffusionmodules.denoiser import DiscreteDenoiserM
from geonvs.seva.modules.conditioner import CLIPConditioner
from geonvs.seva.seva_diffusers import SEVASpatialTemporalModel, create_samplers
from geonvs.seva.sampling import DDPMDiscretization, DiscreteDenoiser, append_dims

from geonvs.data import build_dataset
from torch.utils.data import DistributedSampler
from geonvs.data.view_sampler.step_tracker import StepTracker
from geonvs.data.collate_fn import collate_with_gaussian_padding
from geonvs.utils.visualizer import pca_latents, export_to_gif


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.1")

logger = get_logger(__name__, log_level="INFO")

# Backup torch load
original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    """Force weights_only=False (needed for accelerate.load_state on optimizer states)."""
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


@torch.no_grad()
def tensor_to_vae_latent(t, vae, mask=None, training_mode: bool = True):
    batch_size = t.shape[0]
    video_length = t.shape[1]

    if mask is not None:
        # last component of channel dimension is the mask, 
        if training_mode:
            latents_ = vae.encode(t[mask]).latent_dist.sample() * vae.config.scaling_factor
        else:
            latents_ = vae.encode(t[mask]).latent_dist.mean * vae.config.scaling_factor
        latents_ = F.pad(latents_, (0, 0, 0, 0, 0, 1), value=1.0)
        latents = latents_.new_zeros(batch_size, video_length, *latents_.shape[1:])
        latents[mask] = latents_
    else:
        t = rearrange(t, "b f c h w -> (b f) c h w")
        if training_mode:
            latents = vae.encode(t).latent_dist.sample()
        else:
            latents = vae.encode(t).latent_dist.mean
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        latents = latents * vae.config.scaling_factor

    return latents


@torch.no_grad()
def latent_to_vae_tensor(latents, vae, num_frames):
    latents = latents / vae.config.scaling_factor
    frames = vae.decode(latents, num_frames).sample.clamp(0.0, 1.0)
    return frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--base_folder",
        default="datasets",
        type=str,
        help="Root folder that contains the dataset directories (e.g. datasets/dl3dv_low).",
    )
    parser.add_argument(
        "--train_dataset",
        default="dl3dv",
        type=str,
    )
    parser.add_argument(
        "--valid_dataset",
        default="dl3dv",
        type=str,
    )
    parser.add_argument(
        "--lrm_model",
        default="depthsplat",
        type=str,
    )
    parser.add_argument(
        "--highres",
        action="store_true",
    )
    parser.add_argument(
        "--gs_adapter_config",
        type=str,
        default=None,
        required=False,
        help="Path to gs adapter config file.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="stabilityai/stable-virtual-camera",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=384,  # 576, 384
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,  # 576, 384
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--do_validation",
        action="store_true",
    )
    parser.add_argument(
        "--skip_baseline_validation",
        action="store_true",
        help=(
            "Skip the extra adapter-disabled (base model) sampling pass during validation."
            " Halves validation time; the val_metric/*/[input|novel]_seva comparison logs are omitted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use. This is 8 GPU standard",
    )
    parser.add_argument(
        "--learning_rate_adapter",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use. This is 8 GPU standard",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="lambdalinear",
        help=(
            'The scheduler type to use. Choose between ["lambdalinear", "warmup_cosine", "warmup_cosine2", "linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=1000,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=4000,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=20,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
            
    # read gs adapter config
    use_gs_adapter, add_lora = False, False
    adapter_config = {}
    if args.gs_adapter_config:
        with open(args.gs_adapter_config, 'r') as f:
            adapter_config = yaml.safe_load(f)
        use_gs_adapter = adapter_config['use_gs_adapter']
        add_lora = adapter_config['add_lora']
        if accelerator.is_main_process:
            shutil.copy(args.gs_adapter_config, args.output_dir)
            
    # possible gs adapter module names
    gs_adapter_module_names = ["multi_fusion", "input_fusion", "feat_decoder", "gsattn"]
        
    # LRM model, only needed for the on-the-fly datasets (dl3dvo / re10ko).
    # Imported lazily so the LRM stack (and its extra CUDA extensions) is not
    # required when training with precomputed Gaussians (dl3dv / re10k).
    if (args.train_dataset[-1] == 'o' or args.valid_dataset[-1] == 'o'):
        from baselines.lrm_models import get_lrm_model
        lrm_model = get_lrm_model(args.train_dataset, args.lrm_model)
    else:
        lrm_model = None
    
    # model defined here
    feature_extractor = CLIPConditioner()
    unet = SEVASpatialTemporalModel(
        attn_mode="xformers" if args.enable_xformers_memory_efficient_attention else "naive",
        use_gs_adapter=use_gs_adapter,
        gs_adapter_config=adapter_config,
        lrm_model=lrm_model,
        num_frames=args.num_frames,  # newly added
        max_resolution=(args.height, args.width),  # newly added
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
    )
    
    # for evaluation
    DISCRETIZATION = DDPMDiscretization()
    test_options = {
        "guider_types": 1,
        "log_snr_shift": 2.4,
        "cfg": 2.0,
        "camera_scale": 2.0,
        "num_steps": 50,
        "cfg_min": 1.2,
        "encoding_t": 1,
        "decoding_t": 1,
        "beta_linear_start": 5e-6,
        "step_int_features": [49]
    }
    
    # load pretrained weight here
    weight_path = hf_hub_download(
        repo_id=args.pretrained_model_path, filename="model.safetensors"  # for 1.1 version, modelv1.1.safetensors
    )
    _ = hf_hub_download(
        repo_id=args.pretrained_model_path, filename="config.yaml"
    )
    state_dict = load_safetensors(
        weight_path,
        device='cpu',
    )
    missing, unexpected = unet.load_state_dict(state_dict, strict=False, assign=True)

    # Freeze vae and image_encoder
    feature_extractor.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # For training, sigma sampling module and denoiser
    sigma_sampler = DiscreteSampling(num_idx=1000)
    denoiser_training = DiscreteDenoiserM(
        scaling_method='eps',
        num_idx=1000
    )
    loss_weight_fn = get_loss_weighting(loss_weighting="seva")

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    # Move image_encoder and vae to gpu and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    feature_extractor.to(accelerator.device, dtype=weight_dtype)
    denoiser_training.to(accelerator.device, dtype=weight_dtype)
    
    # Add LoRA adapters to unet
    if add_lora:
        '''
        target_module_names = []
        highlevel_module_names = ["transformer_blocks", "time_mix_blocks"]   # ["time_mix_blocks"]
        specific_module_names = ["to_q", "to_k", "to_v", "to_out.0"]
        for name, module in unet.named_modules():
            for module_name_h in highlevel_module_names:
                for module_name_l in specific_module_names:
                    if module_name_h in name and module_name_l in name:
                        target_module_names.append(name)

        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=target_module_names,
        )
        '''
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config, adapter_name='gs_lora')
    
    # resume if provided
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            resume_path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_path = dirs[-1] if len(dirs) > 0 else None

        if resume_path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        resume_path = os.path.join(args.output_dir, resume_path)
    else:
        resume_path = None
    
    # add learnable parameters
    if use_gs_adapter:
        net_gs_adapter_module_names = set()
        for name, param in unet.named_parameters():
            for module_name in gs_adapter_module_names:
                if name.startswith(module_name):
                    net_gs_adapter_module_names.add(module_name)
                    param.requires_grad_(True)
                    break
        net_gs_adapter_module_names = list(net_gs_adapter_module_names)
    else:
        net_gs_adapter_module_names = []
    
    if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )
        args.learning_rate_adapter = (
            args.learning_rate_adapter * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_lrs = []
    for name, param in unet.named_parameters(): 
        if param.requires_grad:
            if any(name.startswith(module_name) for module_name in net_gs_adapter_module_names):
                params_lrs.append({'params': param, 'lr': args.learning_rate_adapter})
            else:
                params_lrs.append({'params': param, 'lr': args.learning_rate})

    optimizer = optimizer_cls(
        params_lrs,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open(os.path.join(args.output_dir, 'params_freeze.txt'), 'w')
        rec_txt2 = open(os.path.join(args.output_dir, 'params_train.txt'), 'w')
        total_params = 0
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
                total_params += para.numel()
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    
    # Custom dataset
    step_tracker = StepTracker()
    train_dataset = build_dataset(args, args.train_dataset, 'train', step_tracker)
    test_dataset = build_dataset(args, args.valid_dataset, 'test', step_tracker)
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_with_gaussian_padding,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_with_gaussian_padding,
    )
    test_dataloader = enumerate(iter(test_dataloader))
    # for it, v in tqdm(enumerate(train_dataset)):
    #     continue
    # import pdb; pdb.set_trace()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Scheduler
    lr_scheduler = get_scheduler_(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SEVA", config=vars(args))
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Total trainable parameters = {total_params}")
    global_step = 0
    first_epoch = 0
    resume_step = 0
    if resume_path:
        # load state
        torch.load = safe_torch_load
        accelerator.load_state(resume_path)
        torch.load = original_torch_load
        global_step = int(resume_path.split("-")[1])
        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        # Initialize step tracker with resumed step
        step_tracker.set_step(global_step)

    # Only show the progress bar once on each machine.
    progress_bar = tqdma(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # validation metric objects, created lazily on first validation and reused
    render_metrics = None
    
    ## skip logic, use active_dataloader instead of train_dataloader
    # if resume_path and epoch == first_epoch:
    #     active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
    # else:
    #     active_dataloader = train_dataloader

    for epoch in range(first_epoch, args.num_train_epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        train_svd_loss, train_feat_loss = 0.0, 0.0
        for step, batch in enumerate(train_dataloader):
            
            # skip bad batch
            if batch == {}:
                continue
            
            # Skip steps until we reach the resumed step
            if resume_path and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            #TODO: need consistent type casting method
            # if use_gs_adapter and accelerator.mixed_precision in ['fp16', 'bf16']: unet.gsattn.float()

            with accelerator.accumulate(unet):
                
                # pixel space and mask
                pixel_values = batch["image"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                indices_mask = batch['indices_mask'].to(accelerator.device, non_blocking=True)
                extrinsics = batch['extrinsics'].to(accelerator.device, non_blocking=True)
                intrinsics = batch['intrinsics'].to(accelerator.device, non_blocking=True)
                
                # vae encoding to convert to latent space
                latents = tensor_to_vae_latent(pixel_values, vae, training_mode=True)
                latents_masked = F.pad(latents[indices_mask], (0, 0, 0, 0, 0, 1), value=1.0)
                conditional_latents = latents_masked.new_zeros(*latents.shape[:2], *latents_masked.shape[1:])
                conditional_latents[indices_mask] = latents_masked
                
                # Sample noise that we'll add to the latents
                noise = randn_tensor(latents.shape, generator=generator, device=accelerator.device, dtype=weight_dtype)
                bsz = latents.shape[0]
                
                # sampling sigma and make noisy latents
                sigmas = sigma_sampler(bsz).to(latents)
                noisy_latents = latents + noise * append_dims(sigmas, latents.ndim)
                
                # get all the denoising coefficients here
                c_skip, c_out, c_in, timesteps = denoiser_training(latents, sigmas)
                
                # Merge input with noise (input for unet)
                x, mask = conditional_latents.split((noisy_latents.shape[2], 1), dim=2)
                noisy_latents = noisy_latents * (1 - mask) + x * mask
                inp_noisy_latents = noisy_latents * c_in
                
                # concat with plucker inputs
                concat_tensor = batch['concat'].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                inp_noisy_latents = torch.cat([inp_noisy_latents, concat_tensor], dim=2)

                # Get the text embedding for conditioning.
                clip_features = feature_extractor(pixel_values[indices_mask])
                encoder_hidden_states = clip_features.new_zeros(*indices_mask.shape, clip_features.shape[-1])
                encoder_hidden_states[indices_mask] = clip_features
                encoder_hidden_states = encoder_hidden_states.sum(dim=1) / indices_mask.sum(dim=1, keepdim=True)
                encoder_hidden_states = encoder_hidden_states[:, None]  # (bsz, 1, C)
                
                dense_sample = batch['plucker'].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                
                ## GAUSSIAN DEFINE ##
                if use_gs_adapter:
                    if 'gaussians' in batch:
                        gaussians = batch['gaussians'].to(accelerator.device, non_blocking=True)
                        near, far = batch['near'].to(accelerator.device, non_blocking=True), batch['far'].to(accelerator.device, non_blocking=True)
                        scale_invariant = False
                        e3nn = False
                    else:
                        batch_input = {
                            'original_size': (args.height, args.width),
                            'pixel_values': pixel_values,
                            'indices_mask': indices_mask,
                            'intrinsics': intrinsics,
                            'extrinsics': extrinsics,
                        }
                        gaussians, near, far = unet.lrm_model(batch_input, accelerator.device, non_blocking=True)
                        scale_invariant = unet.lrm_model.scale_invariant
                        e3nn = unet.lrm_model.use_e3nn
                else:
                    gaussians = None
                    near, far = None, None
                    scale_invariant = False
                    e3nn = False

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents
                model_pred, feature_loss = unet(
                    sample=inp_noisy_latents, 
                    input_pixels=pixel_values[indices_mask],
                    input_mask=indices_mask,
                    dense_sample=dense_sample,
                    timestep=timesteps, 
                    encoder_hidden_states=encoder_hidden_states,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    near=near,
                    far=far,
                    gaussians=gaussians,
                    scale_invariant=scale_invariant,
                    e3nn=e3nn,
                    training_mode=True,
                    use_gs_adapter=use_gs_adapter,
                )

                # Denoise the latents
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                if 'seva' in loss_weight_fn.name:
                    weighting = loss_weight_fn(
                        sigma=append_dims(sigmas, latents.ndim),
                        mask=indices_mask,
                        c2w=extrinsics,
                        min_weight=0.0
                    )
                else:
                    weighting = loss_weight_fn(
                        sigma=append_dims(sigmas, latents.ndim)
                    )

                # MSE loss (to prevent overflow, use float)
                svd_loss = torch.mean(
                    (weighting.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                svd_loss = svd_loss.mean()
                loss = svd_loss + feature_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_svd_loss = accelerator.gather(
                    svd_loss.repeat(args.per_gpu_batch_size)).mean()
                train_svd_loss += avg_svd_loss.item() / args.gradient_accumulation_steps
                if isinstance(feature_loss, torch.Tensor):
                    avg_feat_loss = accelerator.gather(
                        feature_loss.repeat(args.per_gpu_batch_size)).mean()
                    train_feat_loss += avg_feat_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:  # this is used to resolve unstable learning of pointnet models
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_tracker.set_step(global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_svd_loss": train_svd_loss, "train_feat_loss": train_feat_loss}, step=global_step)
                train_svd_loss, train_feat_loss = 0.0, 0.0
                
                # save checkpoints!
                # if global_step == 10:
                if global_step % args.checkpointing_steps == 0:

                    if accelerator.is_main_process:
                        
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    
                    if accelerator.is_main_process:
                        
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        if add_lora:
                            unwrapped_unet.save_lora_adapter(
                                save_directory=save_path,
                                adapter_name="gs_lora",
                                safe_serialization=True,
                            )  # save the LoRA weights to 'pytorch_lora_weights.safetensors'
                        
                        if use_gs_adapter:
                            torch.save(
                                {
                                    name: getattr(unwrapped_unet, name).state_dict()
                                    for name in net_gs_adapter_module_names
                                },
                                os.path.join(save_path, "gs_adapter_weights.pth")
                            )
                        
                        logger.info(f"Saved state to {save_path}")
                        
                if accelerator.is_main_process:
                    # sample images!
                    if (
                        ((global_step % args.validation_steps == 0)
                        or (global_step == 1)) and args.do_validation
                    ):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        
                        # The models need unwrapping because for compatibility in distributed training mode.
                        unwrapped_vae = accelerator.unwrap_model(vae)
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_feature_extractor = accelerator.unwrap_model(feature_extractor)
                        # if use_gs_adapter and accelerator.mixed_precision in ['fp16', 'bf16']:
                        #     unwrapped_unet.gsattn.float()

                        # denoiser and sampler
                        DENOISER = DiscreteDenoiser(discretization=DISCRETIZATION, num_idx=1000, device=accelerator.device)
                        eval_sampler = create_samplers(
                            test_options['guider_types'],
                            DISCRETIZATION,
                            args.num_frames,
                            test_options['num_steps'],
                            test_options['cfg_min'],
                            device=accelerator.device,
                            abort_event=None,
                        )[0]
                        
                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")
                        os.makedirs(val_save_dir, exist_ok=True)

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ):
                            for val_img_idx in range(args.num_validation_images):
                                
                                # skip if there is empty batch
                                while True:
                                    test_idx, test_batch = next(test_dataloader)
                                    if test_batch != {}: break
                                num_frames = args.num_frames
                
                                # first, convert images to latent space.
                                indices_mask = test_batch['indices_mask'].to(accelerator.device)
                                pixel_values = test_batch["image"].to(accelerator.device, dtype=weight_dtype)
                                c_replace = tensor_to_vae_latent(pixel_values, unwrapped_vae, indices_mask, training_mode=False)[0]
                                uc_replace = torch.zeros_like(c_replace)
                                
                                # Get the text embedding for conditioning.
                                clip_features = unwrapped_feature_extractor(pixel_values[indices_mask])
                                c_crossattn = clip_features.new_zeros(*indices_mask.shape, clip_features.shape[-1])
                                c_crossattn[indices_mask] = clip_features
                                c_crossattn = c_crossattn.sum(dim=1) / indices_mask.sum(dim=1, keepdim=True)
                                c_crossattn = c_crossattn[:, None]  # (bsz, 1, C)
                                c_crossattn = repeat(c_crossattn, '1 1 C -> b 1 C', b=num_frames)
                                uc_crossattn = torch.zeros_like(c_crossattn)
                                
                                c_dense_vector = test_batch['plucker'][0].to(accelerator.device, dtype=weight_dtype)
                                uc_dense_vector = c_dense_vector
                                
                                c_concat = test_batch['concat'][0].to(accelerator.device, dtype=weight_dtype)
                                uc_concat = torch.cat(
                                    [c_dense_vector.new_zeros(num_frames, 1, *c_dense_vector.shape[-2:]), c_dense_vector], 1
                                )
                                
                                ## GAUSSIAN DEFINE ##
                                if use_gs_adapter:
                                    if 'gaussians' in test_batch:
                                        gaussians = test_batch['gaussians'].to(accelerator.device)
                                        near, far = test_batch['near'].to(accelerator.device), test_batch['far'].to(accelerator.device)
                                        scale_invariant = False
                                        e3nn = False
                                    else:
                                        batch_input = {
                                            'original_size': (args.height, args.width),
                                            'pixel_values': pixel_values,
                                            'indices_mask': indices_mask,
                                            'intrinsics': test_batch['intrinsics'].to(accelerator.device),
                                            'extrinsics': test_batch['extrinsics'].to(accelerator.device),
                                        }
                                        gaussians, near, far = unwrapped_unet.lrm_model(batch_input, accelerator.device)
                                        scale_invariant = unet.lrm_model.scale_invariant
                                        e3nn = unet.lrm_model.use_e3nn
                                else:
                                    gaussians = None
                                    near, far = None, None
                                    scale_invariant = False
                                    e3nn = False
                                
                                c = {
                                    "crossattn": c_crossattn,  # Mean of Input CLIP features : <B, 1, d=1024>, repeated B times (Conditional)
                                    "mask": indices_mask.flatten(0, 1),  # viewpoint mask : <B>
                                    "replace": c_replace,  # Latent vector all zeros except for input frames, <B, C + 1, H/8, W/8> (Conditional)
                                    "concat": c_concat,  # plucker coordinates concatenated with input masks, <B, 7, H/8, W/8> (Conditional)
                                    "dense_vector": c_dense_vector,  # plucker coordinates, <B, 6, H/8, W/8> (Conditional)
                                    "extrinsics": test_batch['extrinsics'].to(accelerator.device),
                                    "intrinsics": test_batch['intrinsics'].to(accelerator.device),
                                    "near": near,
                                    "far": far,
                                    "gaussians": gaussians,
                                }
                                uc = {
                                    "crossattn": uc_crossattn,  # Zero tensor, <B, 1, d=1024>, repeated B times (Unconditional)
                                    "mask": indices_mask.flatten(0, 1),  # viewpoint mask : <B, L>
                                    "replace": uc_replace,  # Zero tensor, <B, C + 1, H/8, W/8> (Unconditional)
                                    "concat": uc_concat,  # plucker coordinates concatenated with zero masks, <B, 7, H/8, W/8> (Unconditional)
                                    "dense_vector": uc_dense_vector,  # plucker coordinates, <B, 6, H/8, W/8> (Unconditional)
                                    "extrinsics": test_batch['extrinsics'].to(accelerator.device),
                                    "intrinsics": test_batch['intrinsics'].to(accelerator.device),
                                    "near": near,
                                    "far": far,
                                    "gaussians": gaussians,
                                }
                                
                                additional_model_inputs = {"num_frames": num_frames, "scale_invariant": scale_invariant, "e3nn": e3nn}
                                additional_sampler_inputs = {
                                    "c2w": test_batch['extrinsics'][0].to(accelerator.device, dtype=weight_dtype),
                                    "K": test_batch['intrinsics'][0].to(accelerator.device, dtype=weight_dtype),
                                    "input_frame_mask": indices_mask[0],
                                }
                                shape = (num_frames, 4, args.height // 8, args.width // 8)
                                randn = randn_tensor(shape, generator=generator, device=accelerator.device, dtype=weight_dtype)
                                
                                with torch.no_grad():
                                    
                                    # model inference with gs-adapter and lora
                                    additional_model_inputs["use_gs_adapter"] = use_gs_adapter
                                    samples_z, samples_feat = eval_sampler(
                                        lambda input, sigma, c, log_int_features: DENOISER(
                                            unwrapped_unet,
                                            input,
                                            sigma,
                                            c,
                                            log_int_features,
                                            **additional_model_inputs,
                                        ),
                                        randn.clone(),
                                        scale=test_options["cfg"],
                                        cond=c,
                                        uc=uc,
                                        step_int_features=test_options["step_int_features"],
                                        generator=generator,
                                        verbose=True,
                                        **additional_sampler_inputs,
                                    )
                                    
                                    # model inference without gs-adapter and lora (original model)
                                    if not args.skip_baseline_validation:
                                        additional_model_inputs["use_gs_adapter"] = False
                                        if add_lora: unwrapped_unet.disable_adapter(net_gs_adapter_module_names)
                                        samples_z_ori, samples_feat_ori = eval_sampler(
                                            lambda input, sigma, c, log_int_features: DENOISER(
                                                unwrapped_unet,
                                                input,
                                                sigma,
                                                c,
                                                log_int_features,
                                                **additional_model_inputs,
                                            ),
                                            randn.clone(),
                                            scale=test_options["cfg"],
                                            cond=c,
                                            uc=uc,
                                            step_int_features=[],
                                            generator=generator,
                                            verbose=True,
                                            **additional_sampler_inputs,
                                        )
                                        if add_lora: unwrapped_unet.enable_adapter(net_gs_adapter_module_names)
                                    else:
                                        samples_z_ori = None

                                # convert to pixel space
                                pred_video_frames = latent_to_vae_tensor(samples_z, unwrapped_vae, num_frames)
                                pred_video_frames_ori = latent_to_vae_tensor(samples_z_ori, unwrapped_vae, num_frames) if samples_z_ori is not None else None

                                out_file_iv = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_input_{test_idx}.gif",
                                )
                                out_file_nv = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_novel_{test_idx}.gif",
                                )
                                pred_video_frames_iv = pred_video_frames[indices_mask.flatten(0, 1)]
                                gt_video_frames_iv = pixel_values[indices_mask]
                                pred_video_frames_nv = pred_video_frames[~indices_mask.flatten(0, 1)]
                                gt_video_frames_nv = pixel_values[~indices_mask]
                                if pred_video_frames_ori is not None:
                                    pred_video_frames_iv_ori = pred_video_frames_ori[indices_mask.flatten(0, 1)]
                                    pred_video_frames_nv_ori = pred_video_frames_ori[~indices_mask.flatten(0, 1)]
                                video_frames_iv, video_frames_nv = [], []
                                for i, frame in enumerate(pred_video_frames_iv):
                                    frame = np.uint8(frame.float().permute(1, 2, 0).cpu().numpy() * 255)
                                    frame_gt = np.uint8(gt_video_frames_iv[i].float().permute(1, 2, 0).cpu().numpy() * 255)
                                    if pred_video_frames_ori is not None:
                                        frame_ori = np.uint8(pred_video_frames_iv_ori[i].float().permute(1, 2, 0).cpu().numpy() * 255)
                                        video_frames_iv.append(np.hstack([frame, frame_ori, frame_gt]))
                                    else:
                                        video_frames_iv.append(np.hstack([frame, frame_gt]))
                                for i, frame in enumerate(pred_video_frames_nv):
                                    frame = np.uint8(frame.float().permute(1, 2, 0).cpu().numpy() * 255)
                                    frame_gt = np.uint8(gt_video_frames_nv[i].float().permute(1, 2, 0).cpu().numpy() * 255)
                                    if pred_video_frames_ori is not None:
                                        frame_ori = np.uint8(pred_video_frames_nv_ori[i].float().permute(1, 2, 0).cpu().numpy() * 255)
                                        video_frames_nv.append(np.hstack([frame, frame_ori, frame_gt]))
                                    else:
                                        video_frames_nv.append(np.hstack([frame, frame_gt]))

                                baseline_tag = " / seva" if pred_video_frames_ori is not None else ""
                                description_iv_tag = f"val_input_{test_idx} ours{baseline_tag} / gt"
                                description_nv_tag = f"val_novel_{test_idx} ours{baseline_tag} / gt"
                                for key in samples_feat.keys():
                                    if '/feature' in key:
                                        feat_pca_iv = pca_latents(samples_feat[key][indices_mask.flatten(0, 1)], (args.height, args.width))
                                        feat_pca_nv = pca_latents(samples_feat[key][~indices_mask.flatten(0, 1)], (args.height, args.width))
                                        for i, frame in enumerate(video_frames_iv):
                                            video_frames_iv[i] = np.hstack([frame, feat_pca_iv[i]])
                                        for i, frame in enumerate(video_frames_nv):
                                            video_frames_nv[i] = np.hstack([frame, feat_pca_nv[i]])
                                        description_iv_tag += f" / {key.replace('/feature', '')}"
                                        description_nv_tag += f" / {key.replace('/feature', '')}"
                                    elif 'color' in key:
                                        color_log = F.interpolate(samples_feat[key], size=(args.height, args.width), mode='bilinear')
                                        color_iv, color_nv = color_log[indices_mask.flatten(0, 1)], color_log[~indices_mask.flatten(0, 1)]
                                        color_iv = np.uint8(color_iv.permute(0, 2, 3, 1).float().cpu().numpy() * 255)
                                        color_nv = np.uint8(color_nv.permute(0, 2, 3, 1).float().cpu().numpy() * 255)
                                        for i, frame in enumerate(video_frames_iv):
                                            video_frames_iv[i] = np.hstack([frame, color_iv[i]])
                                        for i, frame in enumerate(video_frames_nv):
                                            video_frames_nv[i] = np.hstack([frame, color_nv[i]])
                                        description_iv_tag += f" / gs_rendered"
                                        description_nv_tag += f" / gs_rendered"
                                    elif 'uncertainty' in key:
                                        uc_log = F.interpolate(samples_feat[key].float(), size=(args.height, args.width), mode='bilinear')
                                        uc_log = (uc_log - uc_log.min()) / (uc_log.max() - uc_log.min() + 1e-8)
                                        uc_log = uc_log.expand(-1, 3, -1, -1)
                                        uc_iv, uc_nv = uc_log[indices_mask.flatten(0, 1)], uc_log[~indices_mask.flatten(0, 1)]
                                        uc_iv = np.uint8(uc_iv.permute(0, 2, 3, 1).cpu().numpy() * 255)
                                        uc_nv = np.uint8(uc_nv.permute(0, 2, 3, 1).cpu().numpy() * 255)
                                        for i, frame in enumerate(video_frames_iv):
                                            video_frames_iv[i] = np.hstack([frame, uc_iv[i]])
                                        for i, frame in enumerate(video_frames_nv):
                                            video_frames_nv[i] = np.hstack([frame, uc_nv[i]])
                                        description_iv_tag += f" / gs_uncertainty"
                                        description_nv_tag += f" / gs_uncertainty"
                                    else:
                                        raise Exception(f"Unsupported key pattern: {key}")
                                export_to_gif(video_frames_iv, out_file_iv, 8)
                                export_to_gif(video_frames_nv, out_file_nv, 8)
                                
                                # measure psnr / ssim / lpips of input / novel frames
                                # (metric objects are created once and reused across
                                # validations to avoid reloading the LPIPS network)
                                if render_metrics is None:
                                    render_metrics = {
                                        "psnr": pyiqa.create_metric('psnr').to(accelerator.device),
                                        "ssim": pyiqa.create_metric('ssim').to(accelerator.device),
                                        "lpips": pyiqa.create_metric('lpips').to(accelerator.device),
                                    }
                                for key in render_metrics.keys():
                                    metric_logs = {
                                        f"val_metric/{key}/input_ours": render_metrics[key](pred_video_frames_iv, gt_video_frames_iv).mean().item(),
                                        f"val_metric/{key}/novel_ours": render_metrics[key](pred_video_frames_nv, gt_video_frames_nv).mean().item(),
                                    }
                                    if pred_video_frames_ori is not None:
                                        metric_logs[f"val_metric/{key}/input_seva"] = render_metrics[key](pred_video_frames_iv_ori, gt_video_frames_iv).mean().item()
                                        metric_logs[f"val_metric/{key}/novel_seva"] = render_metrics[key](pred_video_frames_nv_ori, gt_video_frames_nv).mean().item()
                                    accelerator.log(metric_logs, step=global_step)
                                                 
                                if args.report_to == 'wandb':
                                    accelerator.log({description_iv_tag: wandb.Video(os.path.join(os.getcwd(), out_file_iv), format='gif')}, step=global_step)
                                    accelerator.log({description_nv_tag: wandb.Video(os.path.join(os.getcwd(), out_file_nv), format='gif')}, step=global_step)

                        del DENOISER
                        del eval_sampler
                        gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        if add_lora:
            unwrapped_unet.save_lora_adapter(
                save_directory=args.output_dir,
                adapter_name="gs_lora",
                safe_serialization=True,
            )  # save the LoRA weights to 'pytorch_lora_weights.safetensors'
        
        if use_gs_adapter:
            torch.save(
                {
                    name: getattr(unwrapped_unet, name).state_dict()
                    for name in net_gs_adapter_module_names
                },
                os.path.join(args.output_dir, "gs_adapter_weights.pth")
            )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
