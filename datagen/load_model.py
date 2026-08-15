
import pyiqa
import numpy as np
from tqdm import tqdm
from random import randint
from einops import repeat, rearrange
import torchvision
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

## gaussian decoder / fisher metrics — shared with the training code (../decoder)
import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from geonvs.decoder.decoder import Gaussians
from geonvs.decoder.decoder_splatting_cuda import DecoderSplattingCUDA
from geonvs.decoder.fisher_metric import run_fisher
from geonvs.decoder.fisher_renderer import pool_fisher_cuda

## For VGGT + 3DGS
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from pi3.models.pi3 import Pi3
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

from scene_custom import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, EdgeAwareLogL1
from utils.general_utils import get_expon_lr_func
from geometry import align_to_ground_truth, visualize_pcd, project_pc, compute_co_vis_masks


try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False


def release_memory():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    
    
class VGGtModel(object):
    
    def __init__(self, dataset_name, measure_confidence, device, pretrained_model='facebook/VGGT-1B'):
        self.model_name = 'vggt'
        self.device = device
        self.measure_confidence = measure_confidence
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = VGGT.from_pretrained(pretrained_model).to(device)
        self.model.eval()
        
        self.decoder = DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])
        self.decoder.to(device)
        self.decoder.eval()
        
        self.near = 0.01
        self.far = 100.0
        self.depth_threshold = 0.05  # larger value will construct more generous co-vis mask
        
        ## hyperparameters
        self.sh_degree = 3
        self.resolution = -1
        
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        
        self.iterations = 30_000
        self.training_args = {
            "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "position_lr_max_steps": 30_000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.005,
            "rotation_lr": 0.001,
            "percent_dense": 0.01,
        }
        
        self.lambda_dssim = 0.2
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.pp_optimizer = True
        self.end_iteration = 3000
        self.prune_iteration = 1000
        self.prune_percent = 0.7
        
        self.depth_lossfn = EdgeAwareLogL1()
        self.render_metrics = {
            "psnr": pyiqa.create_metric('psnr').to(self.device),
            "ssim": pyiqa.create_metric('ssim').to(self.device),
            "lpips": pyiqa.create_metric('lpips').to(self.device),
        }
        
    def get_bound(
        self,
        bound,
        num_views: int,
    ):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
        
    def prediction(self, batch, include_target=True):
        
        num_input_viewpoints = len(batch['input_images'])
        num_target_viewpoints = len(batch['target_images'])
        H, W = batch['input_images'].shape[-2:]
        
        # point cloud prior from VGGt (for scale consistency, use all the available viewpoints and prune target viewpoints)
        if include_target:
            images_all = torch.cat([batch['input_images'], batch['target_images']], dim=0)
            extrinsics_all = torch.cat([batch['input_extrinsics'], batch['target_extrinsics']], dim=0)
            intrinsics_all = torch.cat([batch['input_intrinsics'], batch['target_intrinsics']], dim=0)
        else:
            if num_input_viewpoints == 1:
                images_all = torch.cat([batch['input_images'], batch['input_images']], dim=0)
                extrinsics_all = torch.cat([batch['input_extrinsics'], batch['input_extrinsics']], dim=0)
                intrinsics_all = torch.cat([batch['input_intrinsics'], batch['input_intrinsics']], dim=0)
            else:
                images_all = batch['input_images']
                extrinsics_all = batch['input_extrinsics']
                intrinsics_all = batch['input_intrinsics']

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_vgg = images_all[None].type(self.dtype)
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_vgg)
            
            intrinsics = intrinsics_all.clone()
            intrinsics[..., 0, :] *= W
            intrinsics[..., 1, :] *= H
            
            # predict cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic_ = pose_encoding_to_extri_intri(pose_enc, images_vgg.shape[-2:])[0]  # world to camera
            extrinsic = torch.zeros_like(extrinsics_all)
            extrinsic[..., :3, :] = extrinsic_[0].clone()
            extrinsic[..., 3, 3] = 1.0
            
            # predict depth
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images_vgg, ps_idx)
            
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            aligned_depth_map = align_to_ground_truth(extrinsics_all, extrinsic) * depth_map
            point_map = unproject_depth_map_to_point_map(aligned_depth_map.squeeze(0), 
                                                         extrinsics_all[..., :3, :], 
                                                         intrinsics)
            point_map = torch.as_tensor(point_map, dtype=torch.float).to(self.device)
            
            # confidence aware ranking and overlapping mask (parallize in future)
            color_map = images_vgg[0, :].permute(0, 2, 3, 1).float()
            images_np = color_map.cpu().numpy()
            depthmaps = aligned_depth_map.squeeze().cpu().numpy()
            intrinsics_np = intrinsics.cpu().numpy()
            extrinsics_w2c = extrinsics_all.cpu().numpy()
            confidence = depth_conf[0]
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            
            # compute co-vis masks here (GPU-accelerated)
            avg_conf_scores = torch.mean(confidence[:num_input_viewpoints], dim=(1, 2))
            sorted_conf_indices = torch.argsort(avg_conf_scores, descending=True)
            overlapping_masks = compute_co_vis_masks(
                sorted_conf_indices=sorted_conf_indices, 
                depthmaps=aligned_depth_map.squeeze()[:num_input_viewpoints], 
                pointmaps=point_map[:num_input_viewpoints], 
                camera_intrinsics=intrinsics[:num_input_viewpoints], 
                extrinsics_w2c=extrinsics_all[:num_input_viewpoints], 
                image_sizes=(H, W), 
                depth_threshold=self.depth_threshold, 
                device_=self.device
            )
            overlapping_masks = ~overlapping_masks
            
            # release memory
            release_memory()
            
        # train gaussian splats
        inputs = {
            'images': images_np[:num_input_viewpoints],
            'depths': depthmaps[:num_input_viewpoints],
            'intrinsics': intrinsics_np[:num_input_viewpoints],
            'extrinsics': extrinsics_w2c[:num_input_viewpoints],
            'pts3d': point_map[:num_input_viewpoints],
            'color3d': color_map[:num_input_viewpoints],
            'confidence': confidence[:num_input_viewpoints],
            'overlapping_masks': overlapping_masks
        }
        
        # pc = {
        #     'pts3d': point_map.reshape(-1, 3),
        # }
        # depth_prj, index_prj, _, _ = project_pc(
        #     w2c=batch['input_extrinsics'], 
        #     intrinsic=batch['input_intrinsics'], 
        #     pc=pc, 
        #     image_size=batch['original_size'], 
        #     device_=self.device
        # )
        # depth_prj = (depth_prj - depth_prj.min()) / (depth_prj.max() - depth_prj.min())
        
        # torchvision.utils.save_image(depth_prj[0], 'ex.png')
        # plt.imsave('ex2.png', inputs['images'][0])
        
        # import pdb; pdb.set_trace()
        
        
        visualization_dump = {} if self.measure_confidence else None
        gaussians = self.subscene_training(inputs, visualization_dump, silent=True)
        
        if gaussians == None: return None, None
        if visualization_dump is not None:
            gaussians.rotations = visualization_dump['rotations']
            gaussians.scales = visualization_dump['scales']
        
        # after training, construct the visible input mask
        # pc = {
        #     'pts3d': gaussians.means.reshape(-1, 3)
        # }
        # depth_prj, index_prj, _, _ = project_pc(
        #     w2c=batch['input_extrinsics'], 
        #     intrinsic=batch['input_intrinsics'], 
        #     pc=pc, 
        #     image_size=batch['original_size'], 
        #     device_=self.device
        # )
        
        # measure fisher metrics
        if self.measure_confidence:
            fisher_scores, fishers = run_fisher(
                return_score=True,
                gaussians=gaussians, 
                extrinsics=batch['input_extrinsics'].inverse()[None], 
                intrinsics=batch['input_intrinsics'][None], 
                near=self.get_bound("near", num_input_viewpoints)[None].to(self.device),
                far=self.get_bound("far", num_input_viewpoints)[None].to(self.device),
                scale_invariant=False,
                e3nn=False,
                image_shape=batch['original_size'],
                device=self.device
            )
            gaussians.fishers = fishers[None]
            gaussians.fisher_scores = fisher_scores[None]
            
        # novel view synthesis
        with torch.no_grad():
            output = self.decoder.forward(
                gaussians=gaussians,
                extrinsics=batch['target_extrinsics'].inverse()[None],
                intrinsics=batch['target_intrinsics'][None],
                near=self.get_bound("near", num_target_viewpoints)[None].to(self.device),
                far=self.get_bound("far", num_target_viewpoints)[None].to(self.device),
                image_shape=batch['original_size'],
                scale_invariant=False,
                render_color=True,
                e3nn=False,
            )
            rendered_colors = output.color[0]
        
        # compute metrics
        metrics = {"psnr": None, "ssim": None, "lpips": None}
        if 'target_images' in batch:
            target_images = F.interpolate(batch['target_images'], size=batch['original_size'], mode='bilinear', align_corners=True)
            for metric in metrics.keys():
                score = self.render_metrics[metric](rendered_colors, target_images)
                metrics[metric] = score.cpu().numpy()
               
        # torchvision.utils.save_image(rendered_colors[metrics['psnr'].argmax()], 'best.png') 
        # torchvision.utils.save_image(rendered_colors[metrics['psnr'].argmin()], 'worst.png') 
        # torchvision.utils.save_image(batch['target_images'][metrics['psnr'].argmax()], 'best_gt.png') 
        # torchvision.utils.save_image(batch['target_images'][metrics['psnr'].argmin()], 'worst_gt.png')
        # print(metrics['psnr'].max())
        # print(metrics['psnr'].min())
        # print(metrics['psnr'].mean())
        # print(metrics['psnr'].var())
        # print(np.median(metrics['psnr']))
            
        # release memory
        release_memory()
        
        return gaussians, metrics
        
    def prepare_confidence(self, confidence, mask, scale=(0.1, 1.0)):
        """
        Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
        
        Args:
            confidence_path (str): Path to the .npy confidence file.
            device (str): Device to load the tensor onto.
            scale (tuple): Desired range for the learning rate modifiers.
        
        Returns:
            torch.Tensor: Learning rate modifiers.
        """
        # Load and normalize
        normalized_confidence = torch.sigmoid(confidence[mask])

        # Invert confidence and scale to desired range
        inverted_confidence = 1.0 - normalized_confidence
        min_scale, max_scale = scale
        lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
        
        return lr_modifiers[..., None]
        
    @torch.enable_grad()
    def subscene_training(self, inputs, visualization_dump=None, silent=True):
        
        first_iter = 0
        gaussians = GaussianModel(self.sh_degree, self.device)
        
        # per-point optimizer
        scene = Scene(self.resolution, gaussians, inputs)
        if self.pp_optimizer:
            confidence_lr = self.prepare_confidence(inputs['confidence'], inputs['overlapping_masks'], scale=(1.0, 100.0))
            gaussians.training_setup_pp(self.training_args, confidence_lr)
        else:
            gaussians.training_setup(self.training_args)
            
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        ema_loss_for_log = 0.0
        depth_l1_weight = get_expon_lr_func(self.depth_l1_weight_init, self.depth_l1_weight_final, max_steps=self.iterations)
        
        if silent:
            progress_bar = range(first_iter, self.iterations)
        else:
            progress_bar = tqdm(range(first_iter, self.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, self.iterations + 1):
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            
            # Render
            bg = torch.rand((3), device="cuda") if self.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, bg, self.convert_SHs_python, self.compute_cov3D_python)
            image, invdepth = render_pkg["render"], render_pkg["depth"]

            # Compute loss
            gt_image = viewpoint_cam.original_image.to(self.device)
            gt_depth = viewpoint_cam.original_depth.to(self.device)
            valid_mask = gt_depth > 0
            gt_invdepth = torch.where(valid_mask, 1.0 / gt_depth, 0.0)
            
            Ldl = self.depth_lossfn(invdepth.permute(1,2,0), gt_invdepth.permute(1,2,0), gt_image.permute(1,2,0), valid_mask.permute(1,2,0))
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value) + depth_l1_weight(iteration) * Ldl
            loss.backward()
            iter_end.record()
            
            with torch.no_grad():
                
                # Progress bar
                if not silent:
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0 and not silent:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if (iteration == self.iterations) and not silent:
                    progress_bar.close()
                
                # uncertainty aware pruning : https://github.com/j-alex-hanson/gaussian-splatting-pup
                if iteration == self.prune_iteration and self.prune_percent > 0.0:
                    N_gaussians = gaussians.get_xyz.shape[0]
                    with torch.enable_grad():
                        fishers = torch.zeros(N_gaussians, 6, 6, device=self.device).float()
                        for view_idx, view in enumerate(scene.getTrainCameras()):
                            # compute fisher
                            pool_fisher_cuda(view, gaussians, background, fishers=fishers, e3nn=False)
                    fishers_log_dets = torch.linalg.slogdet(fishers + 1e-8 * torch.eye(6, device=self.device))[1]
                    gaussians.prune_gaussians(self.prune_percent, fishers_log_dets)
                    if len(gaussians.get_xyz) == 0:
                        return None
                    
                # Optimizer step
                if iteration < self.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                # Save Model and Finish
                if iteration == self.end_iteration:
                    break
                
        # gaussian model
        gaussian_output = Gaussians(
            means=gaussians.get_xyz[None].detach(),
            covariances=gaussians.get_covariance_mat()[None].detach(),
            harmonics=rearrange(gaussians.get_features[None].detach(), 'b n d c -> b n c d'),
            opacities=gaussians.get_opacity[None, ..., 0].detach(),
        )
        
        if visualization_dump is not None:
            visualization_dump['scales'] = gaussians.get_scaling[None].detach()
            visualization_dump['rotations'] = gaussians.get_rotation[None].detach()
        
        return gaussian_output
        
        
        
class PI3Model(object):
    
    def __init__(self, dataset_name, measure_confidence, device):
        self.model_name = 'pi3'
        self.device = device
        self.measure_confidence = measure_confidence
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(device)
        self.model.eval()
        
        self.decoder = DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])
        self.decoder.to(device)
        self.decoder.eval()
        
        self.near = 0.01
        self.far = 100.0
        self.depth_threshold = 0.05  # larger value will construct more generous co-vis mask
        
        ## hyperparameters
        self.sh_degree = 3
        self.resolution = -1
        
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        
        self.iterations = 30_000
        self.training_args = {
            "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "position_lr_max_steps": 30_000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.005,
            "rotation_lr": 0.001,
            "percent_dense": 0.01,
        }
        
        self.lambda_dssim = 0.2
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.pp_optimizer = True
        self.end_iteration = 3000
        self.prune_iteration = 1000
        self.prune_percent = 0.7
        
        self.depth_lossfn = EdgeAwareLogL1()
        self.render_metrics = {
            "psnr": pyiqa.create_metric('psnr').to(self.device),
            "ssim": pyiqa.create_metric('ssim').to(self.device),
            "lpips": pyiqa.create_metric('lpips').to(self.device),
        }
        
    def get_bound(
        self,
        bound,
        num_views: int,
    ):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
        
    def prediction(self, batch, include_target=True):
        
        num_input_viewpoints = len(batch['input_images'])
        num_target_viewpoints = len(batch['target_images'])
        H, W = batch['input_images'].shape[-2:]
        
        # point cloud prior from VGGt (for scale consistency, use all the available viewpoints and prune target viewpoints)
        if include_target:
            images_all = torch.cat([batch['input_images'], batch['target_images']], dim=0)
            extrinsics_all = torch.cat([batch['input_extrinsics'], batch['target_extrinsics']], dim=0)
            intrinsics_all = torch.cat([batch['input_intrinsics'], batch['target_intrinsics']], dim=0)
        else:
            if num_input_viewpoints == 1:
                images_all = torch.cat([batch['input_images'], batch['input_images']], dim=0)
                extrinsics_all = torch.cat([batch['input_extrinsics'], batch['input_extrinsics']], dim=0)
                intrinsics_all = torch.cat([batch['input_intrinsics'], batch['input_intrinsics']], dim=0)
            else:
                images_all = batch['input_images']
                extrinsics_all = batch['input_extrinsics']
                intrinsics_all = batch['input_intrinsics']

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_vgg = images_all[None].type(self.dtype)
                results = self.model(images_vgg)
                
            intrinsics = intrinsics_all.clone()
            intrinsics[..., 0, :] *= W
            intrinsics[..., 1, :] *= H
            
            # predict cameras
            extrinsic_ = results['camera_poses']  # world to camera
            extrinsic = torch.zeros_like(extrinsics_all)
            extrinsic[..., :3, :] = extrinsic_[0, :, :3].clone()
            extrinsic[..., 3, 3] = 1.0
            
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            scale = align_to_ground_truth(extrinsics_all, extrinsic)
            aligned_depth_map = results['local_points'][0, ..., -1:] * scale
            point_map = unproject_depth_map_to_point_map(aligned_depth_map, 
                                                         extrinsics_all[..., :3, :], 
                                                         intrinsics)
            point_map = torch.as_tensor(point_map, dtype=torch.float).to(self.device)
            
            
            # confidence aware ranking and overlapping mask (parallize in future)
            color_map = images_vgg[0, :].permute(0, 2, 3, 1).float()
            images_np = color_map.cpu().numpy()
            depthmaps = aligned_depth_map.squeeze().cpu().numpy()
            intrinsics_np = intrinsics.cpu().numpy()
            extrinsics_w2c = extrinsics_all.cpu().numpy()
            confidence = results['conf'][0].squeeze(-1)
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            
            # compute co-vis masks here (GPU-accelerated)
            avg_conf_scores = torch.mean(confidence[:num_input_viewpoints], dim=(1, 2))
            sorted_conf_indices = torch.argsort(avg_conf_scores, descending=True)
            overlapping_masks = compute_co_vis_masks(
                sorted_conf_indices=sorted_conf_indices, 
                depthmaps=aligned_depth_map.squeeze()[:num_input_viewpoints], 
                pointmaps=point_map[:num_input_viewpoints], 
                camera_intrinsics=intrinsics[:num_input_viewpoints], 
                extrinsics_w2c=extrinsics_all[:num_input_viewpoints], 
                image_sizes=(H, W), 
                depth_threshold=self.depth_threshold, 
                device_=self.device
            )
            overlapping_masks = ~overlapping_masks
            
            # release memory
            release_memory()
            
        # train gaussian splats
        inputs = {
            'images': images_np[:num_input_viewpoints],
            'depths': depthmaps[:num_input_viewpoints],
            'intrinsics': intrinsics_np[:num_input_viewpoints],
            'extrinsics': extrinsics_w2c[:num_input_viewpoints],
            'pts3d': point_map[:num_input_viewpoints],
            'color3d': color_map[:num_input_viewpoints],
            'confidence': confidence[:num_input_viewpoints],
            'overlapping_masks': overlapping_masks
        }
        
        # pc = {
        #     'pts3d': point_map.reshape(-1, 3),
        # }
        # depth_prj, index_prj, _, _ = project_pc(
        #     w2c=batch['input_extrinsics'], 
        #     intrinsic=batch['input_intrinsics'], 
        #     pc=pc, 
        #     image_size=batch['original_size'], 
        #     device_=self.device
        # )
        # depth_prj = (depth_prj - depth_prj.min()) / (depth_prj.max() - depth_prj.min())
        # depth_prj2 = (inputs['depths'] - inputs['depths'].min()) / (inputs['depths'].max() - inputs['depths'].min())
        
        # torchvision.utils.save_image(depth_prj[0], 'ex.png')
        # plt.imsave('ex3.png', depth_prj2[0])
        # plt.imsave('ex2.png', inputs['images'][0])
        
        # import pdb; pdb.set_trace()
        
        
        visualization_dump = {} if self.measure_confidence else None
        gaussians = self.subscene_training(inputs, visualization_dump, silent=True)
        
        if gaussians == None: return None, None
        if visualization_dump is not None:
            gaussians.rotations = visualization_dump['rotations']
            gaussians.scales = visualization_dump['scales']
        
        # after training, construct the visible input mask
        # pc = {
        #     'pts3d': gaussians.means.reshape(-1, 3)
        # }
        # depth_prj, index_prj, _, _ = project_pc(
        #     w2c=batch['input_extrinsics'], 
        #     intrinsic=batch['input_intrinsics'], 
        #     pc=pc, 
        #     image_size=batch['original_size'], 
        #     device_=self.device
        # )
        
        # measure fisher metrics
        if self.measure_confidence:
            fisher_scores, fishers = run_fisher(
                return_score=True,
                gaussians=gaussians, 
                extrinsics=batch['input_extrinsics'].inverse()[None], 
                intrinsics=batch['input_intrinsics'][None], 
                near=self.get_bound("near", num_input_viewpoints)[None].to(self.device),
                far=self.get_bound("far", num_input_viewpoints)[None].to(self.device),
                scale_invariant=False,
                e3nn=False,
                image_shape=batch['original_size'],
                device=self.device
            )
            gaussians.fishers = fishers[None]
            gaussians.fisher_scores = fisher_scores[None]
            
        # novel view synthesis
        with torch.no_grad():
            output = self.decoder.forward(
                gaussians=gaussians,
                extrinsics=batch['target_extrinsics'].inverse()[None],
                intrinsics=batch['target_intrinsics'][None],
                near=self.get_bound("near", num_target_viewpoints)[None].to(self.device),
                far=self.get_bound("far", num_target_viewpoints)[None].to(self.device),
                image_shape=batch['original_size'],
                scale_invariant=False,
                render_color=True,
                e3nn=False,
            )
            rendered_colors = output.color[0]
        
        # compute metrics
        metrics = {"psnr": None, "ssim": None, "lpips": None}
        if 'target_images' in batch:
            target_images = F.interpolate(batch['target_images'], size=batch['original_size'], mode='bilinear', align_corners=True)
            for metric in metrics.keys():
                score = self.render_metrics[metric](rendered_colors.clip(0, 1), target_images.clip(0, 1))
                metrics[metric] = score.cpu().numpy()
                
        # torchvision.utils.save_image(rendered_colors[1], 'ex_.png')
        # torchvision.utils.save_image(batch['target_images'][1], 'ex2_.png')
        # import pdb; pdb.set_trace()
               
        # torchvision.utils.save_image(rendered_colors[metrics['psnr'].argmax()], 'best.png') 
        # torchvision.utils.save_image(rendered_colors[metrics['psnr'].argmin()], 'worst.png') 
        # torchvision.utils.save_image(batch['target_images'][metrics['psnr'].argmax()], 'best_gt.png') 
        # torchvision.utils.save_image(batch['target_images'][metrics['psnr'].argmin()], 'worst_gt.png')
        # print(metrics['psnr'].max())
        # print(metrics['psnr'].min())
        # print(metrics['psnr'].mean())
        # print(metrics['psnr'].var())
        # print(np.median(metrics['psnr']))
            
        # release memory
        release_memory()
        
        return gaussians, metrics
        
    def prepare_confidence(self, confidence, mask, scale=(0.1, 1.0)):
        """
        Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
        
        Args:
            confidence_path (str): Path to the .npy confidence file.
            device (str): Device to load the tensor onto.
            scale (tuple): Desired range for the learning rate modifiers.
        
        Returns:
            torch.Tensor: Learning rate modifiers.
        """
        # Load and normalize
        normalized_confidence = torch.sigmoid(confidence[mask])

        # Invert confidence and scale to desired range
        inverted_confidence = 1.0 - normalized_confidence
        min_scale, max_scale = scale
        lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
        
        return lr_modifiers[..., None]
        
    @torch.enable_grad()
    def subscene_training(self, inputs, visualization_dump=None, silent=True):
        
        first_iter = 0
        gaussians = GaussianModel(self.sh_degree, self.device)
        
        # per-point optimizer
        scene = Scene(self.resolution, gaussians, inputs)
        if self.pp_optimizer:
            confidence_lr = self.prepare_confidence(inputs['confidence'], inputs['overlapping_masks'], scale=(1.0, 100.0))
            gaussians.training_setup_pp(self.training_args, confidence_lr)
        else:
            gaussians.training_setup(self.training_args)
            
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        ema_loss_for_log = 0.0
        depth_l1_weight = get_expon_lr_func(self.depth_l1_weight_init, self.depth_l1_weight_final, max_steps=self.iterations)
        
        if silent:
            progress_bar = range(first_iter, self.iterations)
        else:
            progress_bar = tqdm(range(first_iter, self.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, self.iterations + 1):
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            
            # Render
            bg = torch.rand((3), device="cuda") if self.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, bg, self.convert_SHs_python, self.compute_cov3D_python)
            image, invdepth = render_pkg["render"], render_pkg["depth"]

            # Compute loss
            gt_image = viewpoint_cam.original_image.to(self.device)
            gt_depth = viewpoint_cam.original_depth.to(self.device)
            valid_mask = gt_depth > 0
            gt_invdepth = torch.where(valid_mask, 1.0 / gt_depth, 0.0)
            
            Ldl = self.depth_lossfn(invdepth.permute(1,2,0), gt_invdepth.permute(1,2,0), gt_image.permute(1,2,0), valid_mask.permute(1,2,0))
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value) + depth_l1_weight(iteration) * Ldl
            loss.backward()
            iter_end.record()
            
            with torch.no_grad():
                
                # Progress bar
                if not silent:
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0 and not silent:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if (iteration == self.iterations) and not silent:
                    progress_bar.close()
                
                # uncertainty aware pruning : https://github.com/j-alex-hanson/gaussian-splatting-pup
                if iteration == self.prune_iteration and self.prune_percent > 0.0:
                    N_gaussians = gaussians.get_xyz.shape[0]
                    with torch.enable_grad():
                        fishers = torch.zeros(N_gaussians, 6, 6, device=self.device).float()
                        for view_idx, view in enumerate(scene.getTrainCameras()):
                            # compute fisher
                            pool_fisher_cuda(view, gaussians, background, fishers=fishers, e3nn=False)
                    fishers_log_dets = torch.linalg.slogdet(fishers + 1e-8 * torch.eye(6, device=self.device))[1]
                    gaussians.prune_gaussians(self.prune_percent, fishers_log_dets)
                    if len(gaussians.get_xyz) == 0:
                        return None
                    
                # Optimizer step
                if iteration < self.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                # Save Model and Finish
                if iteration == self.end_iteration:
                    break
                
        # gaussian model
        gaussian_output = Gaussians(
            means=gaussians.get_xyz[None].detach(),
            covariances=gaussians.get_covariance_mat()[None].detach(),
            harmonics=rearrange(gaussians.get_features[None].detach(), 'b n d c -> b n c d'),
            opacities=gaussians.get_opacity[None, ..., 0].detach(),
        )
        
        if visualization_dump is not None:
            visualization_dump['scales'] = gaussians.get_scaling[None].detach()
            visualization_dump['rotations'] = gaussians.get_rotation[None].detach()
        
        return gaussian_output
    
    
class MapAnythingModel(object):
    
    def __init__(self, dataset_name, measure_confidence, device):
        self.model_name = 'mapanything'
        self.device = device
        self.measure_confidence = measure_confidence
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = MapAnything.from_pretrained("facebook/map-anything").to(device)
        self.model.eval()
        
        self.decoder = DecoderSplattingCUDA(background_color=[0.0, 0.0, 0.0])
        self.decoder.to(device)
        self.decoder.eval()
        
        self.near = 0.01
        self.far = 100.0
        self.depth_threshold = 0.05  # larger value will construct more generous co-vis mask
        
        ## hyperparameters
        self.sh_degree = 3
        self.resolution = -1
        
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        
        self.iterations = 30_000
        self.training_args = {
            "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "position_lr_max_steps": 30_000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.005,
            "rotation_lr": 0.001,
            "percent_dense": 0.01,
        }
        
        self.lambda_dssim = 0.2
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.pp_optimizer = True
        self.end_iteration = 3000
        self.prune_iteration = 1000
        self.prune_percent = 0.7
        
        self.depth_lossfn = EdgeAwareLogL1()
        self.render_metrics = {
            "psnr": pyiqa.create_metric('psnr').to(self.device),
            "ssim": pyiqa.create_metric('ssim').to(self.device),
            "lpips": pyiqa.create_metric('lpips').to(self.device),
        }
        
    def get_bound(
        self,
        bound,
        num_views: int,
    ):
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
        
    def prediction(self, batch, include_target=True):
        
        num_input_viewpoints = len(batch['input_images'])
        num_target_viewpoints = len(batch['target_images'])
        H, W = batch['input_images'].shape[-2:]
            
        input_data = []
        for i in range(num_input_viewpoints):
            img = np.uint8(batch['input_images'][i].permute(1, 2, 0).cpu().numpy() * 255)
            intrinsics_ = batch['input_intrinsics'][i].cpu().numpy()
            extrinsics_ = torch.linalg.inv(batch['input_extrinsics'][i]).cpu().numpy()
            intrinsics_[0, :] *= W
            intrinsics_[1, :] *= H
            input_data.append({
                "img": img,
                "intrinsics": intrinsics_,
                "camera_poses": extrinsics_,
                "is_metric_scale": torch.tensor([True], device=self.device),
            })
        if include_target:
            for i in range(num_target_viewpoints):
                img = np.uint8(batch['target_images'][i].permute(1, 2, 0).cpu().numpy() * 255)
                intrinsics_ = batch['target_intrinsics'][i].cpu().numpy()
                extrinsics_ = torch.linalg.inv(batch['target_extrinsics'][i]).cpu().numpy()
                intrinsics_[0, :] *= W
                intrinsics_[1, :] *= H
                input_data.append({
                    "img": img,
                    "intrinsics": intrinsics_,
                    "camera_poses": extrinsics_,
                    "is_metric_scale": torch.tensor([False], device=self.device),
                })
        else:
            if num_input_viewpoints == 1:
                for i in range(num_input_viewpoints):
                    img = np.uint8(batch['input_images'][i].permute(1, 2, 0).cpu().numpy() * 255)
                    intrinsics_ = batch['input_intrinsics'][i].cpu().numpy()
                    extrinsics_ = torch.linalg.inv(batch['input_extrinsics'][i]).cpu().numpy()
                    intrinsics_[0, :] *= W
                    intrinsics_[1, :] *= H
                    input_data.append({
                        "img": img,
                        "intrinsics": intrinsics_,
                        "camera_poses": extrinsics_,
                        "is_metric_scale": torch.tensor([False], device=self.device),
                    })

        with torch.no_grad():
            
            processed_views = preprocess_inputs(input_data)
            results = self.model.infer(
                processed_views,
                memory_efficient_inference=False,
                use_amp=True, 
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,  
                apply_confidence_mask=False,
                confidence_percentile=10,
                ignore_calibration_inputs=False,
                ignore_pose_inputs=False,
                ignore_pose_scale_inputs=False,
                # ignore_calibration_inputs=True,
                # ignore_pose_inputs=True,
                # ignore_pose_scale_inputs=True,
            )
            num_predictions = len(input_data)
            
            depth_map, images_map, conf_map = [], [], []
            intrinsics_all, extrinsics_all, extrinsics = [], [], []
            for i in range(num_predictions):
                depth_map.append(results[i]['depth_z'])
                images_map.append(results[i]['img_no_norm'])
                conf_map.append(results[i]['conf'])
                intrinsics_all.append(torch.tensor(input_data[i]['intrinsics'][None], device=self.device))
                extrinsics_all.append(torch.linalg.inv(torch.tensor(input_data[i]['camera_poses'][None], device=self.device)))
                extrinsics.append(torch.linalg.inv(results[i]['camera_poses']))
            depth_map = torch.cat(depth_map, dim=0)
            color_map = torch.cat(images_map, dim=0)
            intrinsics = torch.cat(intrinsics_all, dim=0)
            extrinsics_all = torch.cat(extrinsics_all, dim=0)
            extrinsics = torch.cat(extrinsics, dim=0)
            confidence = torch.cat(conf_map, dim=0)
            
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            # scale = align_to_ground_truth(extrinsics_all, extrinsics)
            # aligned_depth_map = depth_map * scale
            aligned_depth_map = depth_map
            point_map = unproject_depth_map_to_point_map(aligned_depth_map, 
                                                         extrinsics_all[..., :3, :], 
                                                         intrinsics)
            point_map = torch.as_tensor(point_map, dtype=torch.float).to(self.device)
            
            # confidence aware ranking and overlapping mask (parallize in future)
            images_np = color_map.cpu().numpy()
            depthmaps = aligned_depth_map.squeeze().cpu().numpy()
            intrinsics_np = intrinsics.cpu().numpy()
            extrinsics_w2c = extrinsics_all.cpu().numpy()
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            
            # compute co-vis masks here (GPU-accelerated)
            avg_conf_scores = torch.mean(confidence[:num_input_viewpoints], dim=(1, 2))
            sorted_conf_indices = torch.argsort(avg_conf_scores, descending=True)
            overlapping_masks = compute_co_vis_masks(
                sorted_conf_indices=sorted_conf_indices, 
                depthmaps=aligned_depth_map.squeeze()[:num_input_viewpoints], 
                pointmaps=point_map[:num_input_viewpoints], 
                camera_intrinsics=intrinsics[:num_input_viewpoints], 
                extrinsics_w2c=extrinsics_all[:num_input_viewpoints], 
                image_sizes=(H, W), 
                depth_threshold=self.depth_threshold, 
                device_=self.device
            )
            overlapping_masks = ~overlapping_masks
            
            # release memory
            release_memory()
            
        # train gaussian splats
        inputs = {
            'images': images_np[:num_input_viewpoints],
            'depths': depthmaps[:num_input_viewpoints],
            'intrinsics': intrinsics_np[:num_input_viewpoints],
            'extrinsics': extrinsics_w2c[:num_input_viewpoints],
            'pts3d': point_map[:num_input_viewpoints],
            'color3d': color_map[:num_input_viewpoints],
            'confidence': confidence[:num_input_viewpoints],
            'overlapping_masks': overlapping_masks
        }
        
        # pc = {
        #     'pts3d': point_map.reshape(-1, 3),
        # }
        # depth_prj, index_prj, _, _ = project_pc(
        #     w2c=batch['input_extrinsics'], 
        #     intrinsic=batch['input_intrinsics'], 
        #     pc=pc, 
        #     image_size=batch['original_size'], 
        #     device_=self.device
        # )
        # depth_prj = (depth_prj - depth_prj.min()) / (depth_prj.max() - depth_prj.min())
        # depth_prj2 = (inputs['depths'] - inputs['depths'].min()) / (inputs['depths'].max() - inputs['depths'].min())
        
        # torchvision.utils.save_image(depth_prj[0], 'ex.png')
        # plt.imsave('ex3.png', depth_prj2[0])
        # plt.imsave('ex2.png', inputs['images'][0])
        
        # import pdb; pdb.set_trace()
        
        
        visualization_dump = {} if self.measure_confidence else None
        gaussians = self.subscene_training(inputs, visualization_dump, silent=True)
        
        if gaussians == None: return None, None
        if visualization_dump is not None:
            gaussians.rotations = visualization_dump['rotations']
            gaussians.scales = visualization_dump['scales']
        
        # after training, construct the visible input mask
        # pc = {
        #     'pts3d': gaussians.means.reshape(-1, 3)
        # }
        # depth_prj, index_prj, _, _ = project_pc(
        #     w2c=batch['input_extrinsics'], 
        #     intrinsic=batch['input_intrinsics'], 
        #     pc=pc, 
        #     image_size=batch['original_size'], 
        #     device_=self.device
        # )
        
        # measure fisher metrics
        if self.measure_confidence:
            fisher_scores, fishers = run_fisher(
                return_score=True,
                gaussians=gaussians, 
                extrinsics=batch['input_extrinsics'].inverse()[None], 
                intrinsics=batch['input_intrinsics'][None], 
                near=self.get_bound("near", num_input_viewpoints)[None].to(self.device),
                far=self.get_bound("far", num_input_viewpoints)[None].to(self.device),
                scale_invariant=False,
                e3nn=False,
                image_shape=batch['original_size'],
                device=self.device
            )
            gaussians.fishers = fishers[None]
            gaussians.fisher_scores = fisher_scores[None]
            
        # novel view synthesis
        with torch.no_grad():
            output = self.decoder.forward(
                gaussians=gaussians,
                extrinsics=batch['target_extrinsics'].inverse()[None],
                intrinsics=batch['target_intrinsics'][None],
                near=self.get_bound("near", num_target_viewpoints)[None].to(self.device),
                far=self.get_bound("far", num_target_viewpoints)[None].to(self.device),
                image_shape=batch['original_size'],
                render_color=True,
                scale_invariant=False,
                e3nn=False,
            )
            rendered_colors = output.color[0]
        
        # compute metrics
        metrics = {"psnr": None, "ssim": None, "lpips": None}
        if 'target_images' in batch:
            target_images = F.interpolate(batch['target_images'], size=batch['original_size'], mode='bilinear', align_corners=True)
            for metric in metrics.keys():
                score = self.render_metrics[metric](rendered_colors, target_images)
                metrics[metric] = score.cpu().numpy()
                
        # torchvision.utils.save_image(rendered_colors[1], 'ex.png')
        # torchvision.utils.save_image(batch['target_images'][1], 'ex2.png')
        # import pdb; pdb.set_trace()
               
        # torchvision.utils.save_image(rendered_colors[metrics['psnr'].argmax()], 'best.png') 
        # torchvision.utils.save_image(rendered_colors[metrics['psnr'].argmin()], 'worst.png') 
        # torchvision.utils.save_image(batch['target_images'][metrics['psnr'].argmax()], 'best_gt.png') 
        # torchvision.utils.save_image(batch['target_images'][metrics['psnr'].argmin()], 'worst_gt.png')
        # print(metrics['psnr'].max())
        # print(metrics['psnr'].min())
        # print(metrics['psnr'].mean())
        # print(metrics['psnr'].var())
        # print(np.median(metrics['psnr']))
            
        # release memory
        release_memory()
        
        return gaussians, metrics
        
    def prepare_confidence(self, confidence, mask, scale=(0.1, 1.0)):
        """
        Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
        
        Args:
            confidence_path (str): Path to the .npy confidence file.
            device (str): Device to load the tensor onto.
            scale (tuple): Desired range for the learning rate modifiers.
        
        Returns:
            torch.Tensor: Learning rate modifiers.
        """
        # Load and normalize
        normalized_confidence = torch.sigmoid(confidence[mask])

        # Invert confidence and scale to desired range
        inverted_confidence = 1.0 - normalized_confidence
        min_scale, max_scale = scale
        lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
        
        return lr_modifiers[..., None]
        
    @torch.enable_grad()
    def subscene_training(self, inputs, visualization_dump=None, silent=True):
        
        first_iter = 0
        gaussians = GaussianModel(self.sh_degree, self.device)
        
        # per-point optimizer
        scene = Scene(self.resolution, gaussians, inputs)
        if self.pp_optimizer:
            confidence_lr = self.prepare_confidence(inputs['confidence'], inputs['overlapping_masks'], scale=(1.0, 100.0))
            gaussians.training_setup_pp(self.training_args, confidence_lr)
        else:
            gaussians.training_setup(self.training_args)
            
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        ema_loss_for_log = 0.0
        depth_l1_weight = get_expon_lr_func(self.depth_l1_weight_init, self.depth_l1_weight_final, max_steps=self.iterations)
        
        if silent:
            progress_bar = range(first_iter, self.iterations)
        else:
            progress_bar = tqdm(range(first_iter, self.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, self.iterations + 1):
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            
            # Render
            bg = torch.rand((3), device="cuda") if self.random_background else background
            render_pkg = render(viewpoint_cam, gaussians, bg, self.convert_SHs_python, self.compute_cov3D_python)
            image, invdepth = render_pkg["render"], render_pkg["depth"]

            # Compute loss
            gt_image = viewpoint_cam.original_image.to(self.device)
            gt_depth = viewpoint_cam.original_depth.to(self.device)
            valid_mask = gt_depth > 0
            gt_invdepth = torch.where(valid_mask, 1.0 / gt_depth, 0.0)
            
            Ldl = self.depth_lossfn(invdepth.permute(1,2,0), gt_invdepth.permute(1,2,0), gt_image.permute(1,2,0), valid_mask.permute(1,2,0))
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value) + depth_l1_weight(iteration) * Ldl
            loss.backward()
            iter_end.record()
            
            with torch.no_grad():
                
                # Progress bar
                if not silent:
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0 and not silent:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if (iteration == self.iterations) and not silent:
                    progress_bar.close()
                
                # uncertainty aware pruning : https://github.com/j-alex-hanson/gaussian-splatting-pup
                if iteration == self.prune_iteration and self.prune_percent > 0.0:
                    N_gaussians = gaussians.get_xyz.shape[0]
                    with torch.enable_grad():
                        fishers = torch.zeros(N_gaussians, 6, 6, device=self.device).float()
                        for view_idx, view in enumerate(scene.getTrainCameras()):
                            # compute fisher
                            pool_fisher_cuda(view, gaussians, background, fishers=fishers, e3nn=False)
                    fishers_log_dets = torch.linalg.slogdet(fishers + 1e-8 * torch.eye(6, device=self.device))[1]
                    gaussians.prune_gaussians(self.prune_percent, fishers_log_dets)
                    if len(gaussians.get_xyz) == 0:
                        return None
                    
                # Optimizer step
                if iteration < self.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                # Save Model and Finish
                if iteration == self.end_iteration:
                    break
                
        # gaussian model
        gaussian_output = Gaussians(
            means=gaussians.get_xyz[None].detach(),
            covariances=gaussians.get_covariance_mat()[None].detach(),
            harmonics=rearrange(gaussians.get_features[None].detach(), 'b n d c -> b n c d'),
            opacities=gaussians.get_opacity[None, ..., 0].detach(),
        )
        
        if visualization_dump is not None:
            visualization_dump['scales'] = gaussians.get_scaling[None].detach()
            visualization_dump['rotations'] = gaussians.get_rotation[None].detach()
        
        return gaussian_output