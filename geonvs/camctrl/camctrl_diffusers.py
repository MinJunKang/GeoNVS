

import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.utils import logging
from diffusers.utils import (
    MIN_PEFT_VERSION, 
    check_peft_version
)
from peft.tuners.tuners_utils import BaseTunerLayer

from geonvs.decoder.decoder import Gaussians
from geonvs.decoder.gaussians import build_covariance
from geonvs.decoder.decoder_splatting_cuda import DecoderSplattingCUDA
from geonvs.adapter_core.gsadapter import FusionLayer, GSAdapter, GSDecoder
from geonvs.adapter_core.dpt import MultiScaleFusionBlock


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class CameraCtrlModel(nn.Module):
    
    def __init__(
        self,
        unet,
        pose_encoder,
        num_frames: int = 14,  # 14 or 25 frames
        max_resolution: tuple[int, int] = (576, 576),
        use_gs_adapter: bool = False,
        gs_adapter_config: dict = {},
    ):
        super().__init__()
        self.scale_factor = 8
        self.unet = unet
        self.pose_encoder = pose_encoder
        self.num_frames = num_frames
        
        ## Gaussian Adapter ## 
        self.use_gs_adapter = use_gs_adapter
        self.gs_adapter_config = gs_adapter_config
        self.multi_fusion = None
        self.input_fusion = None
        self.feat_decoder = None
        self.gsattn = None
        self.input_layer_indices = []
        if self.use_gs_adapter:
            # For gaussian splat
            self.decoder = DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])
            self.gsattn = GSAdapter(
                gs_decoder=self.decoder,
                gs_channel_dim=gs_adapter_config['gs_channel_dim'],
                **gs_adapter_config['attn_layer']
            )
            self.input_layer_indices = gs_adapter_config['gs_layer_indices']
            self.gs_layer_feature_indices = gs_adapter_config['gs_layer_feature_indices']
            feature_dims = [unet.block_out_channels[i] for i in self.gs_layer_feature_indices]

            self.multi_fusion = MultiScaleFusionBlock(
                gs_channel_dim=gs_adapter_config['gs_channel_dim'],
                feature_dims=feature_dims,
                **gs_adapter_config['multi_scale_layer']
            )
            self.input_fusion = FusionLayer(
                channel_dim=gs_adapter_config['gs_channel_dim'], 
                h=max_resolution[0] // self.scale_factor,
                w=max_resolution[1] // self.scale_factor,
                **gs_adapter_config['fusion_layer']
            )
            self.feat_decoder = GSDecoder(
                gs_channel_dim=gs_adapter_config['gs_channel_dim'],
                feature_dims=feature_dims
            )
        
    def disable_adapter(self, gs_adapter_module_names):
        check_peft_version(min_version=MIN_PEFT_VERSION)
        if not self.unet._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        
        for name, module in self.unet.named_modules():
            if any(name.startswith(module_name) for module_name in gs_adapter_module_names):
                continue
            
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = True
        
    def enable_adapter(self, gs_adapter_module_names):
        check_peft_version(min_version=MIN_PEFT_VERSION)
        if not self.unet._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        
        for name, module in self.unet.named_modules():
            if any(name.startswith(module_name) for module_name in gs_adapter_module_names):
                continue
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = False
            
    def forward(
        self,
        sample: torch.Tensor,  # latent vector
        input_mask: torch.Tensor,
        timestep: torch.Tensor,
        added_time_ids: torch.Tensor,
        pose_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        extrinsics: torch.Tensor,  # camera to world
        intrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        gaussians: Optional[torch.Tensor] = None,
        input_pixels: Optional[torch.Tensor] = None,  # raw pixel values of input viewpoint
        log_int_features: bool = False,
        training_mode: bool = True,
        scale_invariant: bool = False,
        e3nn: bool = False,
        use_gs_adapter: bool = False,
    ):
        use_gs_adapter = use_gs_adapter and self.use_gs_adapter and gaussians is not None
        
        # Pose Embedding
        if training_mode:
            pose_embedding = self.pose_encoder(pose_embedding)
        
        ## Gaussian Splats ##
        gs_modules = {
            "multi_fusion": self.multi_fusion,
            "input_decoder": self.feat_decoder,
            "input_fusion": self.input_fusion,
            "input_layer_indices": self.input_layer_indices,
            "gsattn": self.gsattn
        }
        if use_gs_adapter:
            height, width = int(sample.shape[-2] * self.scale_factor), int(sample.shape[-1] * self.scale_factor)

            # The Gaussians and cameras are constant across a sampling loop, so
            # the top-k index/weight rasterization is computed once per scene
            # and reused for every denoising step (inputs matched by identity).
            cache_key = (gaussians, extrinsics, intrinsics, near, far,
                         sample.shape[-2], sample.shape[-1], scale_invariant, e3nn)
            cache = getattr(self, "_gs_batch_cache", None)
            if (
                not training_mode
                and cache is not None
                and all(a is b for a, b in zip(cache[0][:5], cache_key[:5]))
                and cache[0][5:] == cache_key[5:]
            ):
                gs_batch = cache[1]
                gaussian_param = gs_batch["gaussian_param"]
                intrinsics_scale = gs_batch["intrinsics"]
            else:
                num_harmonics = gaussians.shape[-1] - 47
                means, rotations, scales, opacities, fishers, harmonics = torch.split(gaussians, [3, 4, 3, 1, 36, num_harmonics], dim=-1)
                covariances = build_covariance(scales, rotations, 'wxyz').type_as(gaussians)

                with torch.no_grad():
                    gaussian_param = Gaussians(
                        means=means,
                        rotations=rotations,
                        scales=scales,
                        covariances=covariances,
                        opacities=opacities[..., 0],
                        fishers=rearrange(fishers, 'b n (p q) -> b n p q', p=6, q=6),
                        harmonics=rearrange(harmonics, 'b n (c m) -> b n c m', c=3)
                    )
                    intrinsics_scale = intrinsics.clone()
                    intrinsics_scale[..., 0, :] /= width
                    intrinsics_scale[..., 1, :] /= height

                    # valid location sampling (top-k gaussian)
                    rendered = self.decoder(
                        gaussians=gaussian_param,
                        extrinsics=extrinsics.float(),
                        intrinsics=intrinsics_scale.float(),
                        near=near.float(),
                        far=far.float(),
                        image_shape=(sample.shape[-2], sample.shape[-1]),
                        scale_invariant=scale_invariant,
                        render_gsindex=True,
                        e3nn=e3nn,
                        max_contrib_per_pixel=self.gs_adapter_config['knum']
                    )

                    gs_batch = {
                        "gaussian_param": gaussian_param,
                        "extrinsics": extrinsics,
                        "intrinsics": intrinsics_scale,
                        "near": near,
                        "far": far,
                        "index": rendered.gsindex.long(),
                        "weight": rendered.gsweight,
                        "ori_shape": (height, width),
                    }
                self._gs_batch_cache = (cache_key, gs_batch) if not training_mode else None
        else:
            gs_batch = {}
                
        noise_pred, auxilary_output = self.unet(
            sample=sample,
            input_mask=input_mask,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            pose_features=pose_embedding,
            input_pixels=input_pixels,
            log_int_features=log_int_features,
            training_mode=training_mode,
            scale_invariant=scale_invariant,
            e3nn=e3nn,
            use_gs_adapter=use_gs_adapter,
            gs_batch=gs_batch,
            gs_modules=gs_modules,
        )
        if log_int_features and use_gs_adapter:
            
            with torch.no_grad():
                rendered_log = self.decoder(
                    gaussians=gaussian_param,
                    extrinsics=extrinsics.float(),
                    intrinsics=intrinsics_scale.float(),
                    near=near.float(),
                    far=far.float(),
                    image_shape=(height, width),
                    scale_invariant=scale_invariant,
                    render_color=True,
                    e3nn=e3nn
                )
            
            # only support single batch logging
            auxilary_output += [rearrange(rendered_log.color[0:1], 'b v c h w -> (b v) c h w')]
        
        return noise_pred, auxilary_output