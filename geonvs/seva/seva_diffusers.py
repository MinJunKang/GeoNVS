import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin, PeftAdapterMixin
from diffusers.utils import logging
from diffusers.models.modeling_utils import ModelMixin

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.utils import (
    MIN_PEFT_VERSION, 
    check_peft_version
)
from peft.tuners.tuners_utils import BaseTunerLayer

from geonvs.seva.modules.layers import (
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    timestep_embedding,
)
from geonvs.seva.modules.transformer import MultiviewTransformer
from geonvs.seva.sampling import (
    EulerEDMSampler,
    MultiviewCFG,
    MultiviewTemporalCFG,
    VanillaCFG,
)
from geonvs.decoder.decoder import Gaussians
from geonvs.decoder.gaussians import build_covariance
from geonvs.decoder.decoder_splatting_cuda import DecoderSplattingCUDA
from geonvs.adapter_core.gsadapter import FusionLayer, GSAdapter, GSDecoder
from geonvs.adapter_core.dpt import MultiScaleFusionBlock


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def create_samplers(
    guider_types: int | list[int],
    discretization,
    num_frames: list[int] | None,
    num_steps: int,
    cfg_min: float = 1.0,
    device: str | torch.device = "cuda",
    abort_event: threading.Event | None = None,
):
    guider_mapping = {
        0: VanillaCFG,
        1: MultiviewCFG,
        2: MultiviewTemporalCFG,
    }
    samplers = []
    if not isinstance(guider_types, (list, tuple)):
        guider_types = [guider_types]
    for i, guider_type in enumerate(guider_types):
        if guider_type not in guider_mapping:
            raise ValueError(
                f"Invalid guider type {guider_type}. Must be one of {list(guider_mapping.keys())}"
            )
        guider_cls = guider_mapping[guider_type]
        guider_args = ()
        if guider_type > 0:
            guider_args += (cfg_min,)
            if guider_type == 2:
                assert num_frames is not None
                guider_args = (num_frames[i], cfg_min)
        guider = guider_cls(*guider_args)

        sampler = EulerEDMSampler(
            discretization=discretization,
            guider=guider,
            num_steps=num_steps,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=999.0,
            s_noise=1.0,
            verbose=True,
            device=device,
            abort_event=abort_event,
        )
        samplers.append(sampler)
    return samplers
    
   
    
class SEVASpatialTemporalModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 11,  # 4 + 7
        model_channels: int = 320,
        out_channels: int = 4,
        num_frames: int = 21,
        max_resolution: tuple[int, int] = (576, 576),
        num_res_blocks: int = 2,
        attention_resolutions: list[int] = [4, 2, 1],
        channel_mult: list[int] = [1, 2, 4, 4],
        num_head_channels: int = 64,
        transformer_depth: list[int] = [1, 1, 1, 1],
        context_dim: int = 1024,
        dense_in_channels: int = 6,
        dropout: float = 0.0,
        unflatten_names: list[str] = ["middle_ds8", "output_ds4", "output_ds2"],
        attn_mode: Literal["naive", "xformers"] = "naive",
        use_gs_adapter: bool = False,
        gs_adapter_config: dict = {},
        lrm_model = None,
    ):
        super().__init__()
        
        self.scale_factor = 8
        self.num_frames = num_frames
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_head_channels = num_head_channels
        
        ## Gaussian Adapter ## 
        self.use_gs_adapter = use_gs_adapter
        self.gs_adapter_config = gs_adapter_config
        self.input_layer_feature_dims = []
        if self.use_gs_adapter:
            # For gaussian splat
            self.decoder = DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])
            self.gsattn = GSAdapter(
                gs_decoder=self.decoder,
                gs_channel_dim=gs_adapter_config['gs_channel_dim'],
                **gs_adapter_config['attn_layer']
            )
            self.lrm_model = lrm_model
            self.input_layer_indices = gs_adapter_config['gs_layer_indices']

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, padding=1)
                )
            ]
        )
            
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels  # 320
        ds = 1
        # channel_mult = [1, 2, 4, 4]
        # ds : 1 -> 2 -> 4 -> 8
        # ch : 320 -> 640 -> 1280 -> 1280
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                
                '''
                Sequential[
                    Sequential[
                        ResBlock : ch -> mult * model_channels
                        ch <- mult * model_channels
                        MultiviewTransformer
                    ] * num_res_blocks
                    Downsample
                ] * (len(channel_mult) - 1)
                ResBlock * num_res_blocks
                
                '''
                
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * model_channels,
                        dense_in_channels=dense_in_channels,
                        dropout=dropout,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            unflatten_names=unflatten_names,
                            attn_mode=attn_mode,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch
                
        if self.use_gs_adapter:
            for iidx in self.input_layer_indices:
                self.input_layer_feature_dims.append(input_block_chans[iidx])

        num_heads = ch // num_head_channels
        dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=dense_in_channels,
                dropout=dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=transformer_depth[-1],
                context_dim=context_dim,
                unflatten_names=unflatten_names,
                attn_mode=attn_mode,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=dense_in_channels,
                dropout=dropout,
            ),
        )
        self._feature_size += ch

        output_block_chans = []
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=model_channels * mult,
                        dense_in_channels=dense_in_channels,
                        dropout=dropout,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            unflatten_names=unflatten_names,
                            attn_mode=attn_mode,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch
                output_block_chans.append(ch)
        
        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, out_channels, 3, padding=1),
        )
        
        if use_gs_adapter:
            self.multi_fusion = MultiScaleFusionBlock(
                gs_channel_dim=gs_adapter_config['gs_channel_dim'],
                feature_dims=self.input_layer_feature_dims,
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
                feature_dims=self.input_layer_feature_dims
            )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    def disable_adapter(self, gs_adapter_module_names):
        check_peft_version(min_version=MIN_PEFT_VERSION)
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        
        for name, module in self.named_modules():
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
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        
        for name, module in self.named_modules():
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
        input_mask: torch.Tensor,  # mask for input
        dense_sample: torch.Tensor,  # plucker
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        extrinsics: torch.Tensor,  # camera to world
        intrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        gaussians: Optional[torch.Tensor] = None,
        input_pixels: Optional[torch.Tensor] = None,  # raw pixel values of input viewpoint
        log_int_features: bool = False,
        training_mode: bool = True,
        num_frames: Optional[int] = None,
        scale_invariant: bool = False,
        e3nn: bool = False,
        use_gs_adapter: bool = False,
    ):
        ndim = sample.ndim
        bsz = sample.shape[0]
        num_frames = num_frames or self.num_frames
        use_gs_adapter = use_gs_adapter and self.use_gs_adapter and gaussians is not None
        
        # timestep embedding
        t_emb = timestep_embedding(timestep, self.model_channels).to(dtype=sample.dtype)
        t_emb = self.time_embed(t_emb)  # [B, C]
        
        if ndim == 5:  # training time
            t_emb = t_emb.repeat_interleave(num_frames, dim=0)  # [B * N, C]
            sample = sample.flatten(0, 1)
            input_mask = input_mask.flatten(0, 1)
            dense_sample = dense_sample.flatten(0, 1)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        
        ## Gaussian Splats ##
        if use_gs_adapter:
            height, width = int(sample.shape[-2] * self.scale_factor), int(sample.shape[-1] * self.scale_factor)

            # The Gaussians and cameras are constant across a sampling loop, so
            # the top-k index/weight rasterization (and everything derived from
            # it) is computed once per scene and reused for every denoising
            # step. Inputs are matched by tensor identity, so any new batch
            # recomputes from scratch.
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
        
        h = sample
        hs = []
        intermediate_hs = []
        feature_loss = 0.0
        for idx, module in enumerate(self.input_blocks):
            h = module(
                h,
                emb=t_emb,
                context=encoder_hidden_states,
                dense_emb=dense_sample,
                num_frames=num_frames,
            )
            
            hs.append(h)
        
        if use_gs_adapter:
            hs_augmentation = [hs[iidx] for iidx in self.input_layer_indices]

            # multi scale feature fusion
            fused_feature = self.multi_fusion(hs_augmentation)[0]

            # feature aware gs attention
            updated_h, feature_rec_loss = self.gsattn(
                feature=fused_feature,
                gs_batch=gs_batch, 
                input_mask=input_mask,
                scale_invariant=scale_invariant,
                e3nn=e3nn
            )
            
            if log_int_features:
                intermediate_hs += [fused_feature.detach()]
                
            # feature fusion
            input_h, novel_h = fused_feature[input_mask], fused_feature[~input_mask]
            novel_h = self.input_fusion(novel_h, updated_h)
            fused_feature = torch.empty_like(fused_feature)
            fused_feature[input_mask] = input_h
            fused_feature[~input_mask] = novel_h

            # compute intermediate feature loss
            feature_loss = feature_loss + feature_rec_loss
            
            if log_int_features:
                merged_h = torch.empty_like(fused_feature)
                merged_h[input_mask] = input_h
                merged_h[~input_mask] = updated_h
                intermediate_hs += [fused_feature.detach(), merged_h.detach()]
                
            # simple upsampling
            hs_gs_augmentation = self.feat_decoder(fused_feature, hs_augmentation)

            # During training every sample in the batch is conditional (there is
            # no unconditional branch), so the GS-augmented features replace the
            # skip features for the whole batch. At sampling time the batch is
            # the CFG pair (uncond, cond); the GS features are injected only
            # into the conditional half so that the unconditional branch stays
            # consistent with what the base model saw during training.
            gs_cls_guidance = not training_mode
            for iidx in self.input_layer_indices:
                if gs_cls_guidance:
                    hs_gs_aug_feat = hs_gs_augmentation.popleft()
                    hs_gs_feats = torch.split(hs_gs_aug_feat, num_frames, dim=0)  # (uncond, cond)
                    hs_diff_feats = torch.split(hs[iidx], num_frames, dim=0)  # (uncond, cond)
                    hs[iidx] = torch.cat([hs_diff_feats[0], hs_gs_feats[1]], dim=0)
                else:
                    hs[iidx] = hs_gs_augmentation.popleft()

        h = self.middle_block(
            h,
            emb=t_emb,
            context=encoder_hidden_states,
            dense_emb=dense_sample,
            num_frames=num_frames,
        )
        
        for idx, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=encoder_hidden_states,
                dense_emb=dense_sample,
                num_frames=num_frames,
            )
        h = h.type(sample.dtype)
        sample_out = self.out(h)
        
        # reshape back to original shape
        if ndim == 5:
            sample_out = sample_out.reshape(bsz, num_frames, *sample_out.shape[1:])
            
        # log gaussian rendered color output
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
                    render_uncertainty=True,
                    e3nn=e3nn
                )

            # only support single batch logging; the sampler expects the color
            # render second-to-last and the uncertainty render last
            intermediate_hs += [rearrange(rendered_log.color[0:1], 'b v c h w -> (b v) c h w')]
            intermediate_hs += [rearrange(rendered_log.uncertainty[0:1], 'b v h w -> (b v) 1 h w')]
        
        if not training_mode:
            return sample_out, intermediate_hs
        
        return sample_out, feature_loss
    
    
    
# SGMWrapper for SEVASpatialTemporalModel, only used for inference
class SGMWrapper(nn.Module):
    def __init__(self, weight_dtype, lrm_model=None, *args, **kwargs):
        super().__init__()
        self.module = SEVASpatialTemporalModel(*args, **kwargs)
        self.lrm_model = lrm_model
        self.weight_dtype = weight_dtype
        self.module.to(self.weight_dtype)

    def forward(
        self, 
        sample: torch.Tensor, 
        timestep: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        dense_sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        return self.module(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            dense_sample=dense_sample,
            **kwargs,
        )