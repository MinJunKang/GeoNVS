

import gc
import os
import sys
from math import isqrt
from pathlib import Path
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch_scatter import scatter_mean, scatter_max
from safetensors import safe_open
from safetensors.torch import load_file

## Fisher Metrics
from geonvs.decoder.decoder import Gaussians
from geonvs.decoder.gaussians import build_covariance
from geonvs.decoder.fisher_metric import run_fisher
from geonvs.decoder.decoder_splatting_cuda import DecoderSplattingCUDA


def get_num(p):
    return p.split("_")[-1].removesuffix(".json")


def quat_multiply_wxyz(q1, q2):
    """Hamilton product of two quaternions in wxyz order (broadcastable)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def load_safetensor_file(filepath: Path):
    """Read the header of a safetensors file.

    Args:
        file: The safetensors file to read.

    Returns:
        The header of the safetensors file.
    """
    tensors = load_file(filepath)
    
    # Read the header
    with safe_open(filepath, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    
    return tensors, metadata


def get_lrm_model(dataset_path, modelname):
    if 'depthsplat' in modelname:
        model = DepthSplatModel(dataset_path, modelname, return_full_gs="geonvs" in modelname)
    elif modelname == 'mvsplat360':
        model = MVSplat360Model(dataset_path, modelname, return_full_gs=False)
    elif 'mvsplat' in modelname:
        model = MVSplatModel(dataset_path, modelname, return_full_gs="geonvs" in modelname)
    elif 'hisplat' in modelname:
        model = HiSplatModel(dataset_path, modelname, return_full_gs="geonvs" in modelname)
    elif 'vggt' in modelname or 'pi3' in modelname:
        model = GaussianModelWrapper(dataset_path, modelname, return_full_gs="geonvs" in modelname)
    elif 'da3' in modelname:
        model = DepthAnything3Model(dataset_path, modelname, return_full_gs="geonvs" in modelname)
    elif modelname in ['cameractrl', 'motionctrl', 'viewcrafter', 'wancameractrl']:
        model = None
    else:
        raise ValueError(f"Unknown model name: {modelname}")
    return model


class GeometryModel(nn.Module):
    def __init__(self, dataset_path, model_name, return_full_gs=False):
        super().__init__()
        self.scale_factor = 16
        self.use_e3nn = False
        self.scale_invariant = False
        self.dataset_path = dataset_path
        self.return_full_gs = return_full_gs
        self.decoder = DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])
        self.decoder.eval()

    def release_memory(self):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    def get_bound(
        self,
        bound,
        num_views: int,
    ):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
    
    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)
    
    def voxelization_with_fusion(self, pts3d, gs_feat, voxel_size=0.01, conf=None):
        b = pts3d.shape[0]
        
        pts_vox, feats_vox = [], []
        for b_i in range(b):
            voxel_indices = (pts3d[b_i] / voxel_size).round().int()  # [G, 3]
            unique_voxels, inverse_indices, counts = torch.unique(
                voxel_indices, dim=0, return_inverse=True, return_counts=True
            )
            # Aggregate per voxel
            voxel_pts = scatter_mean(
                pts3d[b_i], inverse_indices, dim=0
            )  # [num_unique_voxels, 3]
            
            voxel_feats = scatter_mean(
                gs_feat[b_i], inverse_indices, dim=0
            )  # [num_unique_voxels, feat_dim]
            pts_vox.append(voxel_pts)
            feats_vox.append(voxel_feats)

        max_voxels = max(f.shape[0] for f in feats_vox)
        agg_feats = self.pad_tensor_list(
            feats_vox, (max_voxels,), value=-1e10
        )
        agg_pts = self.pad_tensor_list(
            pts_vox, (max_voxels,), -1e4
        )
        
        gaussians_vox = torch.cat([agg_pts, agg_feats], dim=-1)  # [B, num_voxels, 3 + feat_dim]

        return gaussians_vox

    @torch.no_grad()
    def forward(self, batch, device="cuda", non_blocking=False):
        """Training-side entry point (dl3dvo / re10ko datasets): selects the
        input views with `indices_mask`, runs `forward_geometry`, and returns
        (packed_gaussians, nears, fars) with nears/fars shaped (B, V_total)
        as expected by the training loop."""
        batch_size, num_viewpoints = batch['pixel_values'].shape[:2]

        gaussian_tensors = []
        for i in range(batch_size):
            mask_b = batch['indices_mask'][i]
            # NOTE: pixel_values stay in [-1, 1]; forward_geometry (and each
            # subclass) handles the [0, 1] conversion / intrinsics
            # normalization internally.
            sub_batch = {
                'original_size': batch['original_size'],
                'pixel_values': batch['pixel_values'][i][mask_b][None]
                    .to(device, non_blocking=non_blocking).float(),
                'extrinsics': batch['extrinsics'][i][mask_b][None]
                    .to(device, non_blocking=non_blocking).float(),
                'intrinsics': batch['intrinsics'][i][mask_b][None]
                    .to(device, non_blocking=non_blocking).float(),
            }
            gaussians_b, _, _ = self.forward_geometry(sub_batch, device)
            gaussian_tensors.append(gaussians_b)
        gaussians = torch.cat(gaussian_tensors, dim=0)

        nears = repeat(
            self.get_bound('near', num_viewpoints).to(device), 'v -> b v', b=batch_size
        )
        fars = repeat(
            self.get_bound('far', num_viewpoints).to(device), 'v -> b v', b=batch_size
        )
        return gaussians, nears, fars

    @torch.no_grad()
    def forward_geometry(self, batch, device="cuda", normalize_intrinsics=True):
        downsample_factor = batch.get('downsample_factor', 1)
        allow_h = int((batch['original_size'][0] / downsample_factor) // self.scale_factor * self.scale_factor)
        allow_w = int((batch['original_size'][1] / downsample_factor) // self.scale_factor * self.scale_factor)
        batch_size, num_viewpoints = batch['pixel_values'].shape[:2]
        nears = self.get_bound("near", batch_size).to(device)
        fars = self.get_bound("far", batch_size).to(device)
        input_images = F.interpolate(rearrange(batch['pixel_values'], 'b v c h w -> (b v) c h w'), size=(allow_h, allow_w), mode='bilinear', align_corners=True)
        input_images = rearrange(input_images, '(b v) c h w -> b v c h w', b=batch_size, v=num_viewpoints)
        input_extrinsics = batch['extrinsics']
        input_intrinsics_scale = batch['intrinsics'].clone()
        if normalize_intrinsics:
            input_intrinsics_scale[:, :, 0, :] /= batch['original_size'][1]
            input_intrinsics_scale[:, :, 1, :] /= batch['original_size'][0]
        
        # if given a single image, give the same input twice
        if num_viewpoints == 1:
            input_extrinsics = repeat(input_extrinsics, 'b 1 p q -> b v p q', v=2)
            input_intrinsics_scale = repeat(input_intrinsics_scale, 'b 1 p q -> b v p q', v=2)
            input_images = repeat(input_images, 'b 1 c h w -> b v c h w', v=2)
        
        return input_images, input_extrinsics, input_intrinsics_scale, nears, fars
        
    @torch.no_grad()
    def forward_render(self, batch, device="cuda"):
        assert "gaussians" in batch, "Gaussians not found in batch for rendering."
        assert "near" in batch, "Near bound not found in batch for rendering."
        assert "far" in batch, "Far bound not found in batch for rendering."
        gaussians_tensor = batch['gaussians']
        nears = batch['near']
        fars = batch['far']
        batch_size = gaussians_tensor.shape[0]
        
        num_harmonics = gaussians_tensor.shape[-1] - 47
        means, rotations, scales, opacities, fishers, harmonics = torch.split(gaussians_tensor, [3, 4, 3, 1, 36, num_harmonics], dim=-1)
        covariances = build_covariance(scales, rotations, 'wxyz').type_as(gaussians_tensor)
        
        gaussian_param = Gaussians(
            means=means,
            rotations=rotations,
            scales=scales,
            covariances=covariances,
            opacities=opacities[..., 0],
            fishers=rearrange(fishers, 'b n (p q) -> b n p q', p=6, q=6),
            harmonics=rearrange(harmonics, 'b n (c m) -> b n c m', c=3)
        )
        
        render_tensors, render_dtensors = [], []
        for i in range(batch_size):
            intrinsics_scale = batch['intrinsics'][i].clone()
            intrinsics_scale[:, 0, :] /= batch['original_size'][1]
            intrinsics_scale[:, 1, :] /= batch['original_size'][0]
            
            output_render = self.decoder.forward(
                gaussians=gaussian_param,
                extrinsics=batch['extrinsics'][i][None],
                intrinsics=intrinsics_scale[None],
                near=nears[i][None],
                far=fars[i][None],
                image_shape=batch['original_size'],
                scale_invariant=self.scale_invariant,
                render_color=True,
                render_depth=True,
                e3nn=self.use_e3nn,
            )
            # torchvision.utils.save_image(output_render.color[0][0], 'ex.png')
            # torchvision.utils.save_image(output_render.color[0][2], 'ex2.png')
            # import pdb; pdb.set_trace()
            rendered_colors = output_render.color[0]
            rendered_depths = output_render.depth[0]
            render_tensors.append(rendered_colors)
            render_dtensors.append(rendered_depths)
        renders = torch.cat(render_tensors, dim=0)
        drenders = torch.cat(render_dtensors, dim=0)
        
        outputs = {
            "rendered_images": renders,
            "rendered_depths": drenders,
        }
        if self.return_full_gs:
            outputs.update({
                "gaussians": gaussians_tensor,
                "nears": nears,
                "fars": fars,
                "scale_invariant": self.scale_invariant,
                "use_e3nn": self.use_e3nn,
            })
        return outputs

class DepthSplatModel(GeometryModel):

    def __init__(self, dataset_path, model_name, return_full_gs=False):
        super().__init__(dataset_path, model_name, return_full_gs)
        dataset_name = "dl3dv"  # we use dl3dv pretrained model for all depthsplat models
        self.use_e3nn = True
        self.scale_invariant = True
        self.model_name = "depthsplat"
        
        if 'dl3dv' in dataset_name:
            # checkpoint = 'pretrained_weights/depthsplat-gs-base-dl3dv-256x448-randview2-6-d94d996f.pth'
            # DEPTHSPLAT_CKPT lets you swap in another DepthSplat checkpoint
            # (upstream re-released this file; see tools/scripts/download_weights.sh)
            checkpoint = os.environ.get(
                'DEPTHSPLAT_CKPT',
                'pretrained_weights/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f40abc4f.pth')
        else:
            checkpoint = 'pretrained_weights/depthsplat-gs-large-re10k-256x256-view2-8ee2ec2c.pth'

        if 'base' in checkpoint:
            monovit_type = 'vitb'
        elif 'small' in checkpoint:
            monovit_type = 'vits'
        elif 'large' in checkpoint:
            monovit_type = 'vitl'
        else:
            raise ValueError()
        
        ## DepthSplat
        from baselines.lrm.depthsplat.src.model.encoder import get_encoder, EncoderCfg
        from baselines.lrm.depthsplat.src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
        from baselines.lrm.depthsplat.src.model.encoder.visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg
        
        encoder_cfg = EncoderCfg(
            name='depthsplat',
            d_feature=128,
            num_depth_candidates=128,
            num_surfaces=1,
            gaussians_per_pixel=1,
            visualizer=EncoderVisualizerDepthSplatCfg(
                num_samples=8,
                min_resolution=256,
                export_ply=False
            ),
            gaussian_adapter=GaussianAdapterCfg(
                gaussian_scale_min=0.5,
                gaussian_scale_max=15,
                sh_degree=2,
            ),
            unimatch_weights_path='pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth',
            downscale_factor=4,
            multiview_trans_attn_split=2, 
            costvolume_unet_feat_dim=128, 
            costvolume_unet_channel_mult=[1, 1, 1],
            costvolume_unet_attn_res=[4], 
            depth_unet_feat_dim=32, 
            depth_unet_attn_res=[16], 
            depth_unet_channel_mult=[1, 1, 1, 1, 1], 
            shim_patch_size=4 if dataset_name == 're10k' else 16,
            num_scales=1 if 'small' in checkpoint else 2,
            upsample_factor=2 if dataset_name == 're10k' or 'small' in checkpoint else 4,
            lowest_feature_resolution=4 if dataset_name == 're10k' else 8,
            depth_unet_channels=128, 
            grid_sample_disable_cudnn=False, 
            large_gaussian_head=False, 
            color_large_unet=False, 
            init_sh_input_img=True, 
            feature_upsampler_channels=64, 
            gaussian_regressor_channels=64,
            supervise_intermediate_depth=False, 
            return_depth=True, 
            train_depth_only=False, 
            monodepth_vit_type=monovit_type,
            local_mv_match=2,
        )
        if dataset_name == 'dl3dv':
            self.near = 1.0
            self.far = 200.0
        else:
            self.near = 0.5
            self.far = 100.0
        self.sh_degree = encoder_cfg.gaussian_adapter.sh_degree
        self.encoder, self.encoder_visualizer = get_encoder(encoder_cfg)
        
        # load checkpoint here
        pretrained_model = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        pretrained_model_corr = {}
        for key, value in pretrained_model.items():
            pretrained_model_corr[key.replace('encoder.', '')] = value
        self.encoder.load_state_dict(pretrained_model_corr)
        self.encoder.eval()
    
    @torch.no_grad()
    def forward_geometry(self, batch, device="cuda"):
        
        input_images, input_c2ws, input_intrinsics, nears, fars = super().forward_geometry(batch, device)

        gaussian_tensors = []
        for input_images_b, input_c2ws_b, input_intrinsics_b, nears_b, fars_b in zip(input_images, input_c2ws, input_intrinsics, nears, fars):
            
            num_viewpoints, _, height, width = input_images_b.shape
            nears_b = repeat(nears_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            fars_b = repeat(fars_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            
            context_input = {
                "extrinsics": input_c2ws_b[None],
                "intrinsics": input_intrinsics_b[None],
                "image": (input_images_b[None] + 1) / 2,  # to [0, 1]
                "near": nears_b,
                "far": fars_b
            }
            # prediction
            visualization_dump = {}
            gaussians = self.encoder(
                context_input,
                0,
                deterministic=False,
                visualization_dump=visualization_dump,
            )
            if isinstance(gaussians, dict):
                gaussians = gaussians['gaussians']
            gaussians.rotations = visualization_dump['rotations'][..., [3, 0, 1, 2]]
            gaussians.scales = visualization_dump['scales']
            
            # measure fisher metrics
            fishers = run_fisher(
                gaussians=gaussians, 
                extrinsics=input_c2ws_b[None], 
                intrinsics=input_intrinsics_b[None], 
                near=nears_b,
                far=fars_b,
                image_shape=(height, width),
                scale_invariant=self.scale_invariant,
                e3nn=self.use_e3nn,
                device=device
            )
            gaussians.fishers = fishers[None]
            
            # to tensor
            gaussians_tensor = torch.cat([
                    gaussians.means,  # 3
                    gaussians.rotations,  # 4
                    gaussians.scales,  # 3
                    gaussians.opacities[..., None],  # 1
                    rearrange(gaussians.fishers, 'b g p q -> b g (p q)'),  # 36
                    rearrange(gaussians.harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 3 * (n ** 2)
                ], dim=-1
            )
            
            # voxelization
            if batch.get("voxel_size", 0) > 0:
                gaussians_tensor = self.voxelization_with_fusion(
                    gaussians_tensor[..., :3],
                    gaussians_tensor[..., 3:],
                    voxel_size=batch.get("voxel_size", 0)
                )
                
            gaussian_tensors.append(gaussians_tensor)
            
            self.release_memory()
        gaussians = torch.cat(gaussian_tensors, dim=0)
        return gaussians, nears, fars
    
    
class MVSplatModel(GeometryModel):
    
    def __init__(self, dataset_path, model_name, return_full_gs=False):
        super().__init__(dataset_path, model_name, return_full_gs)
        self.use_e3nn = True
        self.scale_invariant = True
        self.model_name = "mvsplat"
        checkpoint = 'pretrained_weights/mvsplat_re10k.ckpt'
        
        ## MVSplat
        from baselines.lrm.mvsplat.src.model.encoder import get_encoder, EncoderCfg
        from baselines.lrm.mvsplat.src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
        from baselines.lrm.mvsplat.src.model.encoder.encoder_costvolume import OpacityMappingCfg
        from baselines.lrm.mvsplat.src.model.encoder.visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
        
        encoder_cfg = EncoderCfg(
            name='costvolume',
            d_feature=128,
            num_depth_candidates=128, 
            num_surfaces=1, 
            opacity_mapping=OpacityMappingCfg(
                initial=0.0,
                final=0.0,
                warm_up=1
            ),
            visualizer=EncoderVisualizerCostVolumeCfg(
                num_samples=8,
                min_resolution=256,
                export_ply=False
            ),
            gaussian_adapter=GaussianAdapterCfg(
                gaussian_scale_min=0.5,
                gaussian_scale_max=15,
                sh_degree=4,
            ),
            num_context_views=2,  # only used for training
            gaussians_per_pixel=1, 
            unimatch_weights_path='checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth', 
            downscale_factor=4, 
            shim_patch_size=4, 
            multiview_trans_attn_split=2, 
            costvolume_unet_feat_dim=128, 
            costvolume_unet_channel_mult=[1, 1, 1], 
            costvolume_unet_attn_res=[4], 
            depth_unet_feat_dim=32, 
            depth_unet_attn_res=[16], 
            depth_unet_channel_mult=[1, 1, 1, 1, 1], 
            wo_depth_refine=False, 
            wo_cost_volume=False, 
            wo_backbone_cross_attn=False, 
            wo_cost_volume_refine=False, 
            use_epipolar_trans=False,
            mode='test',
        )
        self.near = 1.0
        self.far = 100.0
        self.sh_degree = encoder_cfg.gaussian_adapter.sh_degree
        self.encoder, self.encoder_visualizer = get_encoder(encoder_cfg)
        
        # load checkpoint here
        pretrained_model = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        pretrained_model_corr = {}
        for key, value in pretrained_model.items():
            pretrained_model_corr[key.replace('encoder.', '')] = value
        self.encoder.load_state_dict(pretrained_model_corr)
        self.encoder.eval()
        
    @torch.no_grad()
    def forward_geometry(self, batch, device="cuda"):
        
        input_images, input_c2ws, input_intrinsics, nears, fars = super().forward_geometry(batch, device)

        gaussian_tensors = []
        for input_images_b, input_c2ws_b, input_intrinsics_b, nears_b, fars_b in zip(input_images, input_c2ws, input_intrinsics, nears, fars):
            
            num_viewpoints, _, height, width = input_images_b.shape
            nears_b = repeat(nears_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            fars_b = repeat(fars_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            
            context_input = {
                "extrinsics": input_c2ws_b[None],
                "intrinsics": input_intrinsics_b[None],
                "image": (input_images_b[None] + 1) / 2,  # to [0, 1]
                "near": nears_b,
                "far": fars_b
            }
            # prediction
            visualization_dump = {}
            gaussians = self.encoder(
                context_input,
                0,
                deterministic=False,
                visualization_dump=visualization_dump,
            )
                
            # convert xyzw to wxyz
            gaussians.rotations = visualization_dump['rotations'][..., [3, 0, 1, 2]]
            gaussians.scales = visualization_dump['scales']
            
            # measure fisher metrics
            fishers = run_fisher(
                gaussians=gaussians, 
                extrinsics=input_c2ws_b[None], 
                intrinsics=input_intrinsics_b[None], 
                near=nears_b,
                far=fars_b,
                image_shape=(height, width),
                scale_invariant=self.scale_invariant,
                e3nn=self.use_e3nn,
                device=device
            )
            gaussians.fishers = fishers[None]
            
            # to tensor
            gaussians_tensor = torch.cat([
                    gaussians.means,  # 3
                    gaussians.rotations,  # 4
                    gaussians.scales,  # 3
                    gaussians.opacities[..., None],  # 1
                    rearrange(gaussians.fishers, 'b g p q -> b g (p q)'),  # 36
                    rearrange(gaussians.harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 3 * (n ** 2)
                ], dim=-1
            )
            
            # voxelization
            if batch.get("voxel_size", 0) > 0:
                gaussians_tensor = self.voxelization_with_fusion(
                    gaussians_tensor[..., :3],
                    gaussians_tensor[..., 3:],
                    voxel_size=batch.get("voxel_size", 0)
                )
                
            gaussian_tensors.append(gaussians_tensor)
            
            self.release_memory()
        gaussians = torch.cat(gaussian_tensors, dim=0)
        return gaussians, nears, fars
    
    
class HiSplatModel(GeometryModel):
    
    def __init__(self, dataset_path, model_name, return_full_gs=False):
        super().__init__(dataset_path, model_name, return_full_gs)
        self.use_e3nn = True
        self.scale_invariant = False
        self.model_name = "hisplat"
        checkpoint = 'pretrained_weights/hisplat_re10k.ckpt'
        
        ## MVSplat
        from baselines.lrm.hisplat.src.model.decoder import get_decoder, DecoderCfg
        from baselines.lrm.hisplat.src.model.encoder import get_encoder, EncoderCfg
        from baselines.lrm.hisplat.src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
        from baselines.lrm.hisplat.src.model.encoder.encoder_costvolume_pyramid import OpacityMappingCfg
        from baselines.lrm.hisplat.src.model.encoder.visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
        
        encoder_cfg = EncoderCfg(
            name='costvolume_pyramid',
            d_feature=128,
            num_depth_candidates=128, 
            num_surfaces=1, 
            visualizer=EncoderVisualizerCostVolumeCfg(
                num_samples=8,
                min_resolution=256,
                export_ply=False
            ),
            gaussian_adapter=GaussianAdapterCfg(
                gaussian_scale_min=0.5,
                gaussian_scale_max=15,
                sh_degree=4,
                use_xy_sin=True
            ),
            opacity_mapping=OpacityMappingCfg(
                initial=0.0,
                final=0.0,
                warm_up=1
            ),
            gaussians_per_pixel=1, 
            unimatch_weights_path='checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth', 
            downscale_factor=4, 
            shim_patch_size=4, 
            multiview_trans_attn_split=2, 
            costvolume_unet_feat_dim=128, 
            costvolume_unet_channel_mult=[1, 1, 1], 
            costvolume_unet_attn_res=[4], 
            depth_unet_feat_dim=32, 
            depth_unet_attn_res=[16], 
            depth_unet_channel_mult=[1, 1, 1, 1, 1], 
            num_context_views=2,  # only used for training
            mode='test',
        )
        self.near = 1.0
        self.far = 100.0
        self.sh_degree = encoder_cfg.gaussian_adapter.sh_degree
        decoder_cfg = DecoderCfg(name='splatting_cuda', background_color=[0.0, 0.0, 0.0])
        self.encoder, self.encoder_visualizer = get_encoder(encoder_cfg, get_decoder(decoder_cfg))
        
        # load checkpoint here
        pretrained_model = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        pretrained_model_corr = {}
        for key, value in pretrained_model.items():
            pretrained_model_corr[key[len('encoder.'):]] = value
        self.encoder.load_state_dict(pretrained_model_corr)
        self.encoder.eval()
        
    @torch.no_grad()
    def forward_geometry(self, batch, device="cuda"):
        
        input_images, input_c2ws, input_intrinsics, nears, fars = super().forward_geometry(batch, device)

        gaussian_tensors = []
        for input_images_b, input_c2ws_b, input_intrinsics_b, nears_b, fars_b in zip(input_images, input_c2ws, input_intrinsics, nears, fars):
            
            num_viewpoints, _, height, width = input_images_b.shape
            nears_b = repeat(nears_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            fars_b = repeat(fars_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            
            context_input = {
                "extrinsics": input_c2ws_b[None],
                "intrinsics": input_intrinsics_b[None],
                "image": (input_images_b[None] + 1) / 2,  # to [0, 1]
                "near": nears_b,
                "far": fars_b
            }
            # prediction
            gaussians_dict, results_dict = self.encoder(
                context_input,
                0,
                deterministic=False,
            )
            gaussians = gaussians_dict[f"stage2"]["gaussians"]
            
            # convert xyzw to wxyz
            gaussians.rotations = gaussians_dict[f"stage2"]["rotations"][..., [3, 0, 1, 2]]  # to wxyz
            gaussians.scales = gaussians_dict[f"stage2"]["scales"]
            
            # measure fisher metrics
            fishers = run_fisher(
                gaussians=gaussians, 
                extrinsics=input_c2ws_b[None], 
                intrinsics=input_intrinsics_b[None], 
                near=nears_b,
                far=fars_b,
                image_shape=(height, width),
                scale_invariant=self.scale_invariant,
                e3nn=self.use_e3nn,
                device=device
            )
            gaussians.fishers = fishers[None]
            
            # to tensor
            gaussians_tensor = torch.cat([
                    gaussians.means,  # 3
                    gaussians.rotations,  # 4
                    gaussians.scales,  # 3
                    gaussians.opacities[..., None],  # 1
                    rearrange(gaussians.fishers, 'b g p q -> b g (p q)'),  # 36
                    rearrange(gaussians.harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 3 * (n ** 2)
                ], dim=-1
            )
            
            # voxelization
            if batch.get("voxel_size", 0) > 0:
                gaussians_tensor = self.voxelization_with_fusion(
                    gaussians_tensor[..., :3],
                    gaussians_tensor[..., 3:],
                    voxel_size=batch.get("voxel_size", 0)
                )
                
            gaussian_tensors.append(gaussians_tensor)
            
            self.release_memory()
        gaussians = torch.cat(gaussian_tensors, dim=0)
        return gaussians, nears, fars
    
    
class MVSplat360Model(GeometryModel):
    
    def __init__(self, dataset_path, model_name, return_full_gs=False):
        super().__init__(dataset_path, model_name, return_full_gs)
        self.model_name = "mvsplat360"
        checkpoint = "pretrained_weights/dl3dv_480p.ckpt"
        
        ## MVSplat
        from baselines.lrm.mvsplat360.src.model.encoder import get_encoder, EncoderCfg
        from baselines.lrm.mvsplat360.src.model.decoder import get_decoder, DecoderCfg
        from baselines.lrm.mvsplat360.src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
        from baselines.lrm.mvsplat360.src.model.encoder.encoder_costvolume import OpacityMappingCfg
        from baselines.lrm.mvsplat360.src.model.encoder.visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
        
        # define encoder
        encoder_cfg = EncoderCfg(
            name="costvolume",
            d_feature=128,
            num_depth_candidates=128,
            num_surfaces=1,
            visualizer=EncoderVisualizerCostVolumeCfg(
                num_samples=8,
                min_resolution=256,
                export_ply=False
            ),
            gaussian_adapter=GaussianAdapterCfg(
                gaussian_scale_min=0.5,
                gaussian_scale_max=15.0,
                sh_degree=4,
                feature_sh_degree=2,
                n_feature_channels=4
            ),
            opacity_mapping=OpacityMappingCfg(
                initial=0.0,
                final=0.0,
                warm_up=1
            ),
            gaussians_per_pixel=1,
            unimatch_weights_path=None,
            downscale_factor=4,
            shim_patch_size=16,
            multiview_trans_attn_split=2,
            costvolume_unet_feat_dim=128,
            costvolume_unet_channel_mult=[1, 1, 1],
            costvolume_unet_attn_res=[4],
            depth_unet_feat_dim=32,
            depth_unet_channel_mult=[1, 1, 1, 1, 1],
            depth_unet_attn_res=[16],
            wo_depth_refine=False,
            wo_cost_volume=False,
            wo_backbone_cross_attn=False,
            wo_cost_volume_refine=False,
            use_epipolar_trans=False,
            legacy_2views=False,
            use_legacy_unimatch_backbone=False,
            grid_sample_disable_cudnn=True,
            costvolume_nearest_n_views=3,
            multiview_trans_nearest_n_views=3,
            fit_ckpt=True,
            mode='test'
        )
        self.near = 1.0
        self.far = 100.0
        self.dec_chunk_size = 30
        self.sh_degree = encoder_cfg.gaussian_adapter.sh_degree
        self.encoder, self.encoder_visualizer = get_encoder(encoder_cfg)
        
        # define decoder
        decoder_cfg = DecoderCfg(
            name="splatting_cuda",
            scale_factor=None,
            background_color=[0.0, 0.0, 0.0]
        )
        self.decoder = get_decoder(decoder_cfg)
        self.decoder.eval()
        
        # load checkpoint here
        pretrained_model = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        pretrained_encoder_ckpts = {}
        for k, v in pretrained_model.items():
            if k.startswith("encoder."):
                pretrained_encoder_ckpts[k[len("encoder."):]] = v
        self.encoder.load_state_dict(pretrained_encoder_ckpts)
        self.encoder.eval()
        
    @torch.no_grad()
    def forward_geometry(self, batch, device="cuda"):
        
        input_images, input_c2ws, input_intrinsics, nears, fars = super().forward_geometry(batch, device)

        gaussian_tensors = []
        for input_images_b, input_c2ws_b, input_intrinsics_b, nears_b, fars_b in zip(input_images, input_c2ws, input_intrinsics, nears, fars):
            
            num_viewpoints, _, height, width = input_images_b.shape
            nears_b = repeat(nears_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            fars_b = repeat(fars_b.to(dtype=input_images_b.dtype), '-> 1 v', v=num_viewpoints)
            
            context_input = {
                "extrinsics": input_c2ws_b[None],
                "intrinsics": input_intrinsics_b[None],
                "image": (input_images_b[None] + 1) / 2,  # to [0, 1]
                "near": nears_b,
                "far": fars_b
            }
            # prediction
            visualization_dump = {}
            gaussians = self.encoder(
                context_input,
                0,
                deterministic=False,
                visualization_dump=visualization_dump,
            )
                
            # convert xyzw to wxyz
            gaussians.rotations = visualization_dump['rotations'][..., [3, 0, 1, 2]]
            gaussians.scales = visualization_dump['scales']
            
            # to tensor
            gaussians_tensor = torch.cat([
                    gaussians.means,  # 3
                    gaussians.rotations,  # 4
                    gaussians.scales,  # 3
                    gaussians.opacities[..., None],  # 1
                    rearrange(gaussians.feature_harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 4 * (n ** 2)
                    rearrange(gaussians.harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 3 * (n ** 2)
                ], dim=-1
            )
            
            # voxelization
            if batch.get("voxel_size", 0) > 0:
                gaussians_tensor = self.voxelization_with_fusion(
                    gaussians_tensor[..., :3],
                    gaussians_tensor[..., 3:],
                    voxel_size=batch.get("voxel_size", 0)
                )
                
            gaussian_tensors.append(gaussians_tensor)
            
            self.release_memory()
        gaussians = torch.cat(gaussian_tensors, dim=0)
        return gaussians, nears, fars
    
    @torch.no_grad()
    def forward_render(self, batch, device="cuda"):
        assert "gaussians" in batch, "Gaussians not found in batch for rendering."
        assert "near" in batch, "Near bound not found in batch for rendering."
        assert "far" in batch, "Far bound not found in batch for rendering."
        gaussians_tensor = batch['gaussians']
        nears = batch['near']
        fars = batch['far']
        batch_size = gaussians_tensor.shape[0]
        
        num_harmonics = gaussians_tensor.shape[-1] - 47
        means, rotations, scales, opacities, feat_harmonics, harmonics = torch.split(gaussians_tensor, [3, 4, 3, 1, 36, num_harmonics], dim=-1)
        covariances = build_covariance(scales, rotations, 'wxyz').type_as(gaussians_tensor)
        
        gaussian_param = Gaussians(
            means=means,
            rotations=rotations,
            scales=scales,
            covariances=covariances,
            opacities=opacities[..., 0],
            harmonics=rearrange(harmonics, 'b n (c m) -> b n c m', c=3),
            feature_harmonics=rearrange(feat_harmonics, 'b n (c m) -> b n c m', c=4)
        )
        
        for i in range(batch_size):
            intrinsics_scale = batch['intrinsics'][i].clone()
            intrinsics_scale[:, 0, :] /= batch['original_size'][1]
            intrinsics_scale[:, 1, :] /= batch['original_size'][0]
            
            output_render = self.decoder.forward(
                gaussians=gaussian_param,
                extrinsics=batch['extrinsics'][i][None],
                intrinsics=intrinsics_scale[None],
                near=nears[i][None],
                far=fars[i][None],
                image_shape=batch['original_size']
            )
            # torchvision.utils.save_image(output_render.color[0][0], 'ex.png')
            # torchvision.utils.save_image(output_render.color[0][2], 'ex2.png')
            # import pdb; pdb.set_trace()
            if i == 0: output = output_render
            else:
                for attr in ["color", "depth", "feature", "mask"]:
                    if getattr(output_render, attr) is not None:
                        setattr(
                            output,
                            attr,
                            torch.cat(
                                (getattr(output, attr), getattr(output_render, attr)),
                                dim=0,
                            ),
                        )
        return output
    
    
class DepthAnything3Model(GeometryModel):

    def __init__(self, dataset_path, model_name, return_full_gs=False):
        super().__init__(dataset_path, model_name, return_full_gs)

        # Depth-Anything-3 is vendored under baselines/lrm/Depth-Anything-3
        # as a src-layout package. It is normally resolved through an editable
        # install (`pip install -e baselines/lrm/Depth-Anything-3`); the
        # sys.path insert below makes the vendored copy importable (and take
        # precedence) even when the editable install is missing or points at
        # another checkout.
        _repo_root = Path(__file__).resolve().parents[2]
        da3_src = str(_repo_root / "baselines" / "lrm" / "Depth-Anything-3" / "src")
        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1")
        self.model.eval()

        self.use_e3nn = False
        self.scale_invariant = False
        self.model_name = "depthanything3"
        # DA3-GIANT predicts SH of degree 2 (configs/da3-giant.yaml);
        # refreshed from the actual harmonics dim in forward_geometry.
        self.sh_degree = 2

        dataset_name = 'dl3dv' if 'dl3dv' in str(dataset_path) else 're10k'
        if dataset_name == 'dl3dv':
            self.near = 1.0
            self.far = 200.0
        else:
            self.near = 0.5
            self.far = 100.0

    @torch.no_grad()
    def forward_geometry(self, batch, device="cuda"):
        """
        Run Depth-Anything-3 (gaussian branch) on the input views and pack the
        predicted 3D gaussians into the [means3, rot4(wxyz), scale3, opacity1,
        fisher36, harmonics] channel layout used by the rest of the pipeline.

        Unlike the other subclasses, this does NOT go through
        `super().forward_geometry`: DA3's `InputProcessor` performs its own
        preprocessing (resize to patch-14 multiples + ImageNet normalization)
        and expects raw RGB images, *pixel-unit* intrinsics and *w2c*
        extrinsics (verified in `depth_anything_3/api.py`:
        `_normalize_extrinsics` inverts them to obtain c2ws, and
        `gs_renderer.render_3dgs` documents its extrinsics as w2c), whereas
        the base preprocessing resizes to /16 multiples, keeps images in
        [-1, 1] and normalizes intrinsics.

        DA3 predicts the gaussians in its own (first-camera-centered,
        median-normalized) world frame; `inference()` only re-aligns
        depth/extrinsics to the input cameras, not the gaussians. We therefore
        replicate `inference()` with its private helpers so we can keep the
        *predicted* camera poses and map the gaussians back into the original
        input-camera world frame with a Sim(3) (umeyama) alignment, undoing
        the internal `pose_scale` rescaling applied by DA3's `gs_adapter`.
        """
        from depth_anything_3.model.utils.transform import mat_to_quat
        from depth_anything_3.utils.pose_align import align_poses_umeyama
        from depth_anything_3.utils.sh_helpers import rotate_sh

        batch_size, num_viewpoints = batch['pixel_values'].shape[:2]
        height, width = int(batch['original_size'][0]), int(batch['original_size'][1])
        nears = self.get_bound("near", batch_size).to(device)
        fars = self.get_bound("far", batch_size).to(device)

        # make sure the DA3 backbone sits on the requested device
        self.model.to(device)
        self.model.device = torch.device(device)

        gaussian_tensors = []
        for i in range(batch_size):
            input_images = batch['pixel_values'][i].to(device).float()    # (V, 3, H, W) in [-1, 1]
            input_c2ws = batch['extrinsics'][i].to(device).float()        # (V, 4, 4) c2w
            input_intrinsics = batch['intrinsics'][i].to(device).float()  # (V, 3, 3) pixel units

            # if given a single image, give the same input twice
            if input_images.shape[0] == 1:
                input_images = repeat(input_images, '1 c h w -> v c h w', v=2)
                input_c2ws = repeat(input_c2ws, '1 p q -> v p q', v=2)
                input_intrinsics = repeat(input_intrinsics, '1 p q -> v p q', v=2)
            num_views = input_images.shape[0]

            # DA3 inputs: raw uint8 RGB + pixel intrinsics + w2c extrinsics
            images_np = [
                np.clip((img.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5, 0, 255).astype(np.uint8)
                for img in input_images
            ]
            w2cs_np = np.linalg.inv(input_c2ws.cpu().numpy().astype(np.float64)).astype(np.float32)
            intrinsics_np = input_intrinsics.cpu().numpy()

            # --- replicate DepthAnything3.inference() (api.py) so that the
            # *predicted* extrinsics stay available for gaussian alignment
            imgs_cpu, exts_in, ixts_in = self.model._preprocess_inputs(
                images_np, w2cs_np, intrinsics_np
            )
            imgs_t, ex_t, in_t = self.model._prepare_model_inputs(imgs_cpu, exts_in, ixts_in)
            ex_t_norm = self.model._normalize_extrinsics(ex_t.clone())
            raw_output = self.model._run_model_forward(
                imgs_t, ex_t_norm, in_t, [], infer_gs=True
            )
            gs = raw_output["gaussians"]  # depth_anything_3.specs.Gaussians, batch=1
            pred_w2cs = raw_output["extrinsics"][0].detach().float().cpu().numpy()  # (V, 3|4, 4) w2c

            # --- Sim(3): predicted (gaussian) frame -> original input frame.
            # 1) gs_adapter rescaled the predicted-frame scene by pose_scale
            #    (umeyama scale between ex_t_norm and the predicted poses,
            #    clamped to [1/3, 3]); replicate it so it can be undone.
            try:
                _, _, pose_scale = align_poses_umeyama(
                    ex_t_norm[0].float().cpu().numpy(), pred_w2cs
                )
                pose_scale = float(np.clip(pose_scale, 1.0 / 3.0, 3.0))
            except Exception:
                pose_scale = 1.0
            # 2) full Sim(3) from the raw predicted frame to the original
            #    input cameras (same call/params as api.py's alignment).
            rot_np, trans_np, scale = align_poses_umeyama(
                w2cs_np,
                pred_w2cs,
                ransac=num_views >= 10,
                random_state=42,
            )
            # the gaussians live at pose_scale * (raw predicted frame), so:
            eff_scale = float(scale) / pose_scale
            rot = torch.from_numpy(rot_np).to(device).float()      # (3, 3)
            trans = torch.from_numpy(trans_np).to(device).float()  # (3,)

            means = gs.means.to(device).float()  # (1, g, 3)
            means = eff_scale * torch.einsum('ij,bgj->bgi', rot, means) + trans
            scales = gs.scales.to(device).float() * eff_scale  # (1, g, 3)
            rotations = gs.rotations.to(device).float()  # (1, g, 4) wxyz (world_quat_wxyz)
            rot_quat = mat_to_quat(rot)[..., [3, 0, 1, 2]]  # xyzw -> wxyz
            rotations = quat_multiply_wxyz(rot_quat.expand_as(rotations), rotations)
            harmonics = gs.harmonics.to(device).float()  # (1, g, 3, d_sh)
            d_sh = harmonics.shape[-1]
            self.sh_degree = isqrt(d_sh) - 1
            if self.sh_degree > 0:
                harmonics = rotate_sh(harmonics, rot[None, None, None])

            # opacities are plain "(b, g)" or view-dependent SH "(b, g, 1, d_sh)"
            # (specs.Gaussians). The vendored render_3dgs only consumes plain
            # (b, g) opacities, so for the SH case take the view-independent DC
            # band and squash it with a sigmoid (the SH coefficients are
            # linear-activated when conf_dim > 1, mirroring render_3dgs's
            # sigmoid handling of raw coefficients in its non-SH branch).
            # DA3-GIANT (conf_dim=1) takes the plain path.
            opacities = gs.opacities.to(device).float()
            if opacities.dim() == 4:
                opacities = (opacities[..., 0, 0] * 0.28209479177387814).sigmoid()
            elif opacities.dim() == 3:
                opacities = opacities[..., 0]

            # measure fisher metrics against the GT input cameras
            nears_b = repeat(nears[i].to(dtype=means.dtype), '-> 1 v', v=num_views)
            fars_b = repeat(fars[i].to(dtype=means.dtype), '-> 1 v', v=num_views)
            intrinsics_scale = input_intrinsics.clone()
            intrinsics_scale[:, 0, :] /= width
            intrinsics_scale[:, 1, :] /= height
            gaussians_param = Gaussians(
                means=means,
                harmonics=harmonics,
                opacities=opacities,
                covariances=build_covariance(scales, rotations, 'wxyz').type_as(means),
                rotations=rotations,
                scales=scales,
            )
            fishers = run_fisher(
                gaussians=gaussians_param,
                extrinsics=input_c2ws[None],
                intrinsics=intrinsics_scale[None],
                near=nears_b,
                far=fars_b,
                image_shape=(height, width),
                scale_invariant=self.scale_invariant,
                e3nn=self.use_e3nn,
                device=device,
            )

            # to tensor
            gaussians_tensor = torch.cat([
                    means,  # 3
                    rotations,  # 4 (wxyz)
                    scales,  # 3
                    opacities[..., None],  # 1
                    rearrange(fishers[None], 'b g p q -> b g (p q)'),  # 36
                    rearrange(harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 3 * (n ** 2)
                ], dim=-1
            )
            # DA3's forward runs under torch.inference_mode(); clone so the
            # packed tensor is a regular tensor usable in autograd graphs.
            gaussians_tensor = gaussians_tensor.clone()

            # voxelization
            if batch.get("voxel_size", 0) > 0:
                gaussians_tensor = self.voxelization_with_fusion(
                    gaussians_tensor[..., :3],
                    gaussians_tensor[..., 3:],
                    voxel_size=batch.get("voxel_size", 0)
                )

            gaussian_tensors.append(gaussians_tensor)

            if torch.cuda.is_available():
                self.release_memory()
        gaussians = torch.cat(gaussian_tensors, dim=0)
        return gaussians, nears, fars


class GaussianModelWrapper(GeometryModel):
    def __init__(self, dataset_path, model_name, return_full_gs=False):
        super().__init__(dataset_path, model_name, return_full_gs)
        self.near = 0.01
        self.far = 100.0
        self.gaussians_prior = None
        self.gaussians_scene = ""
        if model_name.endswith("vggt_iv"):
            self.model_name = "vggt_iv"
        elif model_name.endswith("vggt_av"):
            self.model_name = "vggt_av"
        elif model_name.endswith("pi3_iv"):
            self.model_name = "pi3_iv"
        elif model_name.endswith("pi3_av"):
            self.model_name = "pi3_av"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @torch.no_grad()
    def forward_geometry(self, batch, device="cuda"):
        batch_size = batch['pixel_values'].shape[0]
        scene_config_path = batch['scene_config_path']
        scene_name = Path(scene_config_path).parent.stem
        
        if self.gaussians_scene != scene_name:
            num_inputs = int(get_num(scene_config_path)) if get_num(scene_config_path).isdigit() else get_num(scene_config_path)
            gaussian_path = Path(scene_config_path).parent / self.model_name
            gaussian_pt_path = [p for p in sorted(gaussian_path.glob('*.safetensors')) if int(p.stem.split("_")[0]) == num_inputs][0]
            gaussian_pt, _ = load_safetensor_file(str(gaussian_pt_path))
            self.gaussians_prior = gaussian_pt
            self.gaussians_scene = scene_name
            
        nears, fars = self.gaussians_prior['near'].float().to(device), self.gaussians_prior['far'].float().to(device)
        gaussian_tensors = self.gaussians_prior['gaussians'].float().to(device)
        
        # voxelization
        if batch.get("voxel_size", 0) > 0:
            num_points_before = gaussian_tensors.shape[1]
            gaussian_tensors = self.voxelization_with_fusion(
                gaussian_tensors[..., :3],
                gaussian_tensors[..., 3:],
                voxel_size=batch.get("voxel_size", 0)
            )
            num_points_after = gaussian_tensors.shape[1]
            # with open("voxelization_log.txt", 'a') as f:
            #     voxel_reduction_rate = num_points_after / num_points_before
            #     f.write(f"Voxelized with size: {batch['voxel_size']}\n")
            #     f.write(f"Voxelization rate: {voxel_reduction_rate:.2%}\n")
            print("Voxelized with size:", batch.get("voxel_size", 0))
            print(f"Voxelization reduced points from {num_points_before} to {num_points_after}.")
        gaussian_tensors = repeat(gaussian_tensors, "1 g c -> b g c", b=batch_size)
        nears = repeat(nears, "1 -> b", b=batch_size)
        fars = repeat(fars, "1 -> b", b=batch_size)
            
        return gaussian_tensors, nears, fars