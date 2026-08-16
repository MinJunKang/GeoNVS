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
from omegaconf import OmegaConf

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
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version, is_wandb_available, load_image

from peft import LoraConfig
from diffusers.training_utils import cast_training_params

from geonvs.data import build_dataset
from torch.utils.data import DistributedSampler
from geonvs.data.view_sampler.step_tracker import StepTracker
from geonvs.data.collate_fn import collate_with_gaussian_padding
from geonvs.utils.visualizer import pca_latents, export_to_gif

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.utils.torch_utils import randn_tensor
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from geonvs.seva.lr_scheduler import get_scheduler_
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from geonvs.camctrl.modules.pose_adaptor import CameraPoseEncoder
from geonvs.camctrl.modules.unet import UNetSpatioTemporalConditionModelPoseCond
from geonvs.camctrl.camctrl_diffusers import CameraCtrlModel
from geonvs.camctrl.camctrl_pipeline import StableVideoDiffusionPipelinePoseCond


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
def tensor_to_vae_latent(t, vae, mask=None):
    batch_size = t.shape[0]
    video_length = t.shape[1]

    if mask is not None:
        # last component of channel dimension is the mask, 
        latents_ = vae.encode(t[mask]).latent_dist.sample() * vae.config.scaling_factor
        latents_ = F.pad(latents_, (0, 0, 0, 0, 0, 1), value=1.0)
        latents = latents_.new_zeros(batch_size, video_length, *latents_.shape[1:])
        latents[mask] = latents_
    else:
        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
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
        "--model_config",
        type=str,
        default="configs/svd_cameractrl.yaml",
        help="Path to model config file.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
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
            " Halves validation time; the val_metric/*/[input|novel]_camctrl comparison logs are omitted."
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
        default=200000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use. This is 8 GPU standard",
    )
    parser.add_argument(
        "--learning_rate_adapter",
        type=float,
        default=1e-4,
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
        default="constant",
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
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
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
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.seed)
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
        
    # model defined here
    model_configs = OmegaConf.load(args.model_config)
    feature_extractor = CLIPImageProcessor.from_pretrained(
        model_configs["svd_pretrained_path"], subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_configs["svd_pretrained_path"], subfolder="image_encoder"
    )
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        model_configs["svd_pretrained_path"], subfolder="scheduler"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        model_configs["svd_pretrained_path"], subfolder="vae"
    )
    unet = UNetSpatioTemporalConditionModelPoseCond.from_pretrained(
        model_configs["svd_pretrained_path"], subfolder="unet",
        down_block_types=model_configs['down_block_types'], up_block_types=model_configs['up_block_types']
    )
    pose_encoder = CameraPoseEncoder(**model_configs['pose_encoder_kwargs'])
    unet.set_pose_cond_attn_processor(
        enable_xformers=(model_configs["enable_xformers_memory_efficient_attention"] and is_xformers_available()), 
        **model_configs['attention_processor_kwargs']
    )
    main_model = CameraCtrlModel(
        unet=unet,
        pose_encoder=pose_encoder,
        num_frames=args.num_frames,
        max_resolution=(args.height, args.width),
        use_gs_adapter=use_gs_adapter,
        gs_adapter_config=adapter_config,
    )
    
    # load pretrained weight here
    state_dict = torch.load(model_configs["pretrained_model_path"], map_location=unet.device, weights_only=True)
    pose_encoder_state_dict = state_dict['pose_encoder_state_dict']
    pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
    assert len(pose_encoder_m) == 0 and len(pose_encoder_u) == 0
    attention_processor_state_dict = state_dict['attention_processor_state_dict']
    _, attention_processor_u = unet.load_state_dict(attention_processor_state_dict, strict=False, assign=True)
    assert len(attention_processor_u) == 0
    
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_encoder.requires_grad_(False)
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Freeze the pose_encoder parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)
    
    # Move image_encoder and vae to gpu and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    pose_encoder.to(accelerator.device, dtype=weight_dtype)
    main_model.to(accelerator.device, dtype=weight_dtype)
    
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

    # load weights if provided
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
        for name, param in main_model.named_parameters():
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
        cast_training_params(main_model, dtype=torch.float32)
        
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
    for name, param in main_model.named_parameters():
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
        for name, para in main_model.named_parameters():
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
    train_dataset = build_dataset(args, args.train_dataset, 'train', step_tracker, plucker_convention='camctrl', input_num=[1])
    test_dataset = build_dataset(args, args.valid_dataset, 'test', step_tracker, plucker_convention='camctrl', input_num=[1])
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
    # for it, v in enumerate(tqdm(train_dataloader)):
    #     pass
    # import pdb; pdb.set_trace()
    # for it, v in enumerate(tqdm(train_dataset)):
    #     import pdb; pdb.set_trace()
    #     if v == {}: print(it)
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
        # accelerate advances the wrapped scheduler once per process per
        # optimizer step, so step counts are pre-multiplied to land on the
        # requested number. Curve-shape arguments such as num_cycles are not
        # step counts and must not be scaled; each schedule keeps its own.
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    pipeline = StableVideoDiffusionPipelinePoseCond(
        vae=vae,
        image_encoder=image_encoder,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        main_model=main_model
    )
    pipeline.set_progress_bar_config(disable=True)
    
    # Prepare everything with our `accelerator`.
    main_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        main_model, optimizer, lr_scheduler, train_dataloader
    )
    
    # attribute handling for models using DDP
    if isinstance(main_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        main_model = main_model.module
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
        accelerator.init_trackers("CAMCTRL", config=vars(args))
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
    
    for epoch in range(first_epoch, args.num_train_epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        main_model.train()
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
            # if use_gs_adapter and accelerator.mixed_precision in ['fp16', 'bf16']: main_model.gsattn.float()
            
            with accelerator.accumulate(main_model):
                
                # pixel space and mask
                pixel_values = batch["image"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                extrinsics = batch['extrinsics'].to(accelerator.device, non_blocking=True)
                intrinsics = batch['intrinsics'].to(accelerator.device, non_blocking=True)
                indices_mask = batch['indices_mask'].to(accelerator.device, non_blocking=True)
                pose_embedding = batch['plucker'].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                
                # vae encoding to convert to latent space
                latents = tensor_to_vae_latent(pixel_values, vae)
                
                # Sample noise that we'll add to the latents
                bsz = latents.shape[0]
                rnd_normal = randn_tensor([bsz, 1, 1, 1, 1], device=accelerator.device, dtype=weight_dtype, generator=generator)
                sigma = (rnd_normal * model_configs['P_std'] + model_configs['P_mean']).exp()
                noise = randn_tensor(latents.shape, generator=generator, device=accelerator.device, dtype=weight_dtype) * sigma
                noisy_latents = latents + noise
                
                # Get the preconditioning parameters
                c_skip = 1 / (sigma ** 2 + 1)
                c_out = -sigma / (sigma ** 2 + 1) ** 0.5
                c_in = 1 / (sigma ** 2 + 1) ** 0.5
                c_noise = (sigma.log() / 4).reshape([bsz])

                # Get the loss weighting
                weighting = (sigma ** 2 + 1) / sigma ** 2
                
                # encode conditioning image latent
                conditional_pixels = pixel_values[indices_mask]
                conditioning_rnd_normal = randn_tensor([bsz, 1, 1, 1], device=accelerator.device, dtype=weight_dtype, generator=generator)
                conditioning_sigma = (conditioning_rnd_normal * model_configs['condition_image_noise_std'] + model_configs['condition_image_noise_mean']).exp()
                conditioning_pixels = randn_tensor(conditional_pixels.shape, generator=generator, device=accelerator.device, dtype=weight_dtype) * conditioning_sigma + conditional_pixels
                with torch.no_grad():
                    conditioning_latents = vae.encode(conditioning_pixels).latent_dist.mode()
                conditioning_latents = repeat(conditioning_latents, 'b c h w -> b f c h w', f=args.num_frames)
                
                # merge latents
                input_latents = torch.cat([c_in * noisy_latents, conditioning_latents], dim=2)
                
                # encode image latent using clip
                conditioning_images_clip = _resize_with_antialiasing(pixel_values[indices_mask], (224, 224))
                conditioning_images_clip = (conditioning_images_clip + 1.0) / 2.0
                conditioning_images_clip = feature_extractor(
                    images=conditioning_images_clip.float(),
                    do_normalize=True,
                    do_center_crop=False,
                    do_resize=False,
                    do_rescale=False,
                    return_tensors="pt"
                ).pixel_values.to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                encoder_hidden_states = image_encoder(conditioning_images_clip).image_embeds.unsqueeze(1)
                
                # get additional time ids
                noise_aug_strength = conditioning_sigma[:, 0, 0, 0]       # [bsz, ]
                add_time_ids = [[model_configs['fps'], model_configs['motion_bucket_id'], strength] for strength in noise_aug_strength]
                add_time_ids = torch.tensor(add_time_ids, device=accelerator.device, dtype=weight_dtype)
                
                ## GAUSSIAN DEFINE ##
                if use_gs_adapter and 'gaussians' in batch:
                    gaussians = batch['gaussians'].to(accelerator.device, non_blocking=True)
                    near, far = batch['near'].to(accelerator.device, non_blocking=True), batch['far'].to(accelerator.device, non_blocking=True)
                    scale_invariant = False
                    e3nn = False
                else:
                    gaussians = None
                    near, far = None, None
                    scale_invariant = True
                    e3nn = True
                
                target = latents
                model_pred, feature_loss = main_model(
                    sample=input_latents, 
                    input_mask=indices_mask,
                    timestep=c_noise,
                    added_time_ids=add_time_ids,
                    pose_embedding=pose_embedding,
                    encoder_hidden_states=encoder_hidden_states,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    near=near,
                    far=far,
                    gaussians=gaussians,
                    input_pixels=pixel_values[indices_mask],
                    scale_invariant=scale_invariant,
                    e3nn=e3nn,
                    training_mode=True,
                    use_gs_adapter=use_gs_adapter,
                )
                
                # Denoise the latents
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                
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
                optimizer.zero_grad(set_to_none=True)
                
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
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
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
                    if accelerator.is_main_process:
                        
                        unwrapped_model = accelerator.unwrap_model(main_model)
                        if add_lora:
                            unwrapped_unet = unwrapped_model.unet
                            unwrapped_unet.save_lora_adapter(
                                save_directory=save_path,
                                adapter_name="gs_lora",
                                safe_serialization=True,
                            )  # save the LoRA weights to 'pytorch_lora_weights.safetensors'
                        
                        if use_gs_adapter:
                            torch.save(
                                {
                                    name: getattr(unwrapped_model, name).state_dict()
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
                        num_frames = args.num_frames
                            
                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")
                        os.makedirs(val_save_dir, exist_ok=True)
                        
                        # if use_gs_adapter and accelerator.mixed_precision in ['fp16', 'bf16']: pipeline.main_model.gsattn.float()
                        
                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ):
                            for val_img_idx in range(args.num_validation_images):
                                
                                # skip if there is empty batch
                                while True:
                                    test_idx, test_batch = next(test_dataloader)
                                    if test_batch != {}: break
                                    
                                extrinsics = test_batch['extrinsics'].to(accelerator.device)
                                intrinsics = test_batch['intrinsics'].to(accelerator.device)
                                
                                # first, convert images to latent space.
                                indices_mask = test_batch['indices_mask'].to(accelerator.device)
                                pixel_values = test_batch["image"].to(accelerator.device, dtype=weight_dtype)
                                pose_embedding = test_batch['plucker'].to(accelerator.device, dtype=weight_dtype)
                                conditioning_images = pixel_values[indices_mask]
                                
                                ## GAUSSIAN DEFINE ##
                                if use_gs_adapter and 'gaussians' in test_batch:
                                    gaussians = test_batch['gaussians'].to(accelerator.device)
                                    near, far = test_batch['near'].to(accelerator.device), test_batch['far'].to(accelerator.device)
                                    scale_invariant = False
                                    e3nn = False
                                else:
                                    gaussians = None
                                    near, far = None, None
                                    scale_invariant = True
                                    e3nn = True
                                
                                with torch.no_grad():
                                    
                                    # model inference with gs-adapter and lora
                                    pred_video_frames, int_feats = pipeline(
                                        image=conditioning_images,
                                        pose_embedding=pose_embedding,
                                        extrinsics=extrinsics,
                                        intrinsics=intrinsics,
                                        near=near,
                                        far=far,
                                        gaussians=gaussians,
                                        scale_invariant=scale_invariant,
                                        e3nn=e3nn,
                                        use_gs_adapter=use_gs_adapter,
                                        height=args.height,
                                        width=args.width,
                                        num_frames=num_frames,
                                        num_inference_steps=model_configs["num_inference_steps"],
                                        min_guidance_scale=model_configs["min_guidance_scale"],
                                        max_guidance_scale=model_configs["max_guidance_scale"],
                                        generator=generator,
                                        step_int_features=[model_configs["num_inference_steps"] - 1],
                                        output_type='pt'
                                    )
                                    
                                    # model inference without gs-adapter and lora (original model)
                                    if not args.skip_baseline_validation:
                                        if add_lora: pipeline.main_model.disable_adapter(net_gs_adapter_module_names)
                                        pred_video_frames_ori, int_feats_ori = pipeline(
                                            image=conditioning_images,
                                            pose_embedding=pose_embedding,
                                            extrinsics=extrinsics,
                                            intrinsics=intrinsics,
                                            near=near,
                                            far=far,
                                            gaussians=gaussians,
                                            scale_invariant=scale_invariant,
                                            e3nn=e3nn,
                                            use_gs_adapter=False,
                                            height=args.height,
                                            width=args.width,
                                            num_frames=num_frames,
                                            num_inference_steps=model_configs["num_inference_steps"],
                                            min_guidance_scale=model_configs["min_guidance_scale"],
                                            max_guidance_scale=model_configs["max_guidance_scale"],
                                            generator=generator,
                                            output_type='pt'
                                        )
                                        if add_lora: pipeline.main_model.enable_adapter(net_gs_adapter_module_names)
                                    else:
                                        pred_video_frames_ori = None
                                pred_video_frames = pred_video_frames[0]
                                pred_video_frames_ori = pred_video_frames_ori[0] if pred_video_frames_ori is not None else None

                                out_file_iv = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_input_{test_idx}.gif",
                                )
                                out_file_nv = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_novel_{test_idx}.gif",
                                )
                                pred_video_frames_iv = pred_video_frames[indices_mask.flatten(0, 1)]
                                gt_video_frames_iv = (pixel_values[indices_mask] + 1) / 2
                                pred_video_frames_nv = pred_video_frames[~indices_mask.flatten(0, 1)]
                                gt_video_frames_nv = (pixel_values[~indices_mask] + 1) / 2
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

                                baseline_tag = " / cameractrl" if pred_video_frames_ori is not None else ""
                                description_iv_tag = f"val_input_{test_idx} ours{baseline_tag} / gt"
                                description_nv_tag = f"val_novel_{test_idx} ours{baseline_tag} / gt"
                                for key in int_feats.keys():
                                    if '/feature' in key:
                                        feat_pca_iv = pca_latents(int_feats[key][indices_mask].float(), (args.height, args.width))
                                        feat_pca_nv = pca_latents(int_feats[key][~indices_mask].float(), (args.height, args.width))
                                        for i, frame in enumerate(video_frames_iv):
                                            video_frames_iv[i] = np.hstack([frame, feat_pca_iv[i]])
                                        for i, frame in enumerate(video_frames_nv):
                                            video_frames_nv[i] = np.hstack([frame, feat_pca_nv[i]])
                                        description_iv_tag += f" / {key.replace('/feature', '')}"
                                        description_nv_tag += f" / {key.replace('/feature', '')}"
                                    elif 'color' in key:
                                        color_log = F.interpolate(int_feats[key], size=(args.height, args.width), mode='bilinear')
                                        color_iv, color_nv = color_log[indices_mask.flatten(0, 1)], color_log[~indices_mask.flatten(0, 1)]
                                        color_iv = np.uint8(color_iv.permute(0, 2, 3, 1).float().cpu().numpy() * 255)
                                        color_nv = np.uint8(color_nv.permute(0, 2, 3, 1).float().cpu().numpy() * 255)
                                        for i, frame in enumerate(video_frames_iv):
                                            video_frames_iv[i] = np.hstack([frame, color_iv[i]])
                                        for i, frame in enumerate(video_frames_nv):
                                            video_frames_nv[i] = np.hstack([frame, color_nv[i]])
                                        description_iv_tag += f" / gs_rendered"
                                        description_nv_tag += f" / gs_rendered"
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
                                        metric_logs[f"val_metric/{key}/input_camctrl"] = render_metrics[key](pred_video_frames_iv_ori, gt_video_frames_iv).mean().item()
                                        metric_logs[f"val_metric/{key}/novel_camctrl"] = render_metrics[key](pred_video_frames_nv_ori, gt_video_frames_nv).mean().item()
                                    accelerator.log(metric_logs, step=global_step)
                                                 
                                if args.report_to == 'wandb':
                                    accelerator.log({description_iv_tag: wandb.Video(os.path.join(os.getcwd(), out_file_iv), format='gif')}, step=global_step)
                                    accelerator.log({description_nv_tag: wandb.Video(os.path.join(os.getcwd(), out_file_nv), format='gif')}, step=global_step)
                            
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
        main_model = main_model.to(torch.float32)

        unwrapped_model = accelerator.unwrap_model(main_model)
        if add_lora:
            unwrapped_unet = unwrapped_model.unet
            unwrapped_unet.save_lora_adapter(
                save_directory=args.output_dir,
                adapter_name="gs_lora",
                safe_serialization=True,
            )  # save the LoRA weights to 'pytorch_lora_weights.safetensors'
        
        if use_gs_adapter:
            torch.save(
                {
                    name: getattr(unwrapped_model, name).state_dict()
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