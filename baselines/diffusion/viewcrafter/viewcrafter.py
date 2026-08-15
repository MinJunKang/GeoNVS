

import torch
import numpy as np
import torchvision
import os
import copy
import glob
from einops import repeat, rearrange
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from baselines.diffusion.viewcrafter.utils.pvd_utils import *
from baselines.diffusion.viewcrafter.utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from baselines.diffusion.viewcrafter.extern.dust3r.dust3r.inference import inference, load_model
from baselines.diffusion.viewcrafter.extern.dust3r.dust3r.utils.image import center_crop_pil_image, _resize_pil_image, resize_pil_image, ImgNorm
from baselines.diffusion.viewcrafter.extern.dust3r.dust3r.image_pairs import make_pairs
from baselines.diffusion.viewcrafter.extern.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from baselines.diffusion.viewcrafter.extern.dust3r.dust3r.utils.device import to_numpy
from roma import rotmat_to_unitquat, rigid_points_registration



class ViewCrafter:
    def __init__(self, opts, ckpt_path, model_path, device):
        self.opts = opts
        self.ckpt_path = ckpt_path  # diffusion model checkpoint path
        self.model_path = model_path  # dust3r model path
        self.device = device
        seed_everything(self.opts.seed)
        
        # initialize models
        self.setup_dust3r()
        self.setup_diffusion()      
        self.to_pil = torchvision.transforms.ToPILImage()
        
    def setup_diffusion(self):
        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.ckpt_path)
        model.to(self.device).eval()
        self.diffusion = model
        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.model_path, self.device)
    
    def preprocess_images(self, images):
        images_out = []
        for idx, img in enumerate(images):
            images_pil = self.to_pil(img.cpu())
            images_pil = resize_pil_image(images_pil, target_width=self.opts.width, target_height=self.opts.height)
            img_ori = images_pil
            images_out.append(dict(img=ImgNorm(images_pil)[None], true_shape=np.int32(
                [images_pil.size[::-1]]), idx=len(images_out), instance=str(len(images_out)), img_ori=ImgNorm(img_ori)[None], ))
            
        return images_out
        
    def run_dust3r(self, input_images, clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False):
        
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None

        return images, view_masks
    
    def run_render(self, pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views,renderer,self.device,nbv=False)
        return render_results, viewmask

    def split_into_chunks(self, lst_len, chunk_size, overlap):
        chunks = []
        start = 0
        while start < lst_len:
            end = start + chunk_size
            chunks.append([start, end])
            start += chunk_size - overlap
            if end >= lst_len:
                break
        return chunks
    
    def get_current_chunk(self, batch_input, start_idx, end_idx, chunk_size):
        n_border_patch = 0
        if batch_input is None:
            return None, n_border_patch

        assert (
            batch_input.shape[0] == 1
        ), "currently only used for test time, use batch_size=1"

        batch_input = rearrange(batch_input, "b c t h w -> b t c h w")
        cur_gs_batch = batch_input[0, start_idx:end_idx]

        if cur_gs_batch.shape[0] != chunk_size:
            n_border_patch = chunk_size - cur_gs_batch.shape[0]
            cur_gs_batch = torch.cat(
                (
                    cur_gs_batch,
                    repeat(
                        cur_gs_batch[-1:],
                        "b ... -> (b n) ...",
                        n=n_border_patch,
                    ),
                ),
                dim=0,
            )
        cur_gs_batch = rearrange(cur_gs_batch, "t c h w -> c t h w")
        return cur_gs_batch.unsqueeze(0), n_border_patch
    
    def run_diffusion(self, renderings, input_index=0):
        n_frames = renderings.shape[0]
        prompts = [self.opts.prompt]
        chunk_ses = self.split_into_chunks(n_frames, self.opts.video_length, 0)
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        cond_img = videos[:, :, input_index] #bchw
        outputs = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for (s_idx, e_idx) in chunk_ses:
                cur_color_batch, n_border_patch = self.get_current_chunk(
                    videos, s_idx, e_idx, self.opts.video_length
                )
                self.noise_shape[2] = min(self.opts.video_length, cur_color_batch.shape[2])
                batch_samples = image_guided_synthesis(self.diffusion, prompts, cur_color_batch, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, cond_img)
                rendered_rgb = batch_samples[0, 0, :, :self.opts.video_length - n_border_patch]
                rendered_rgb = torch.clamp(rendered_rgb, -1., 1.)
                outputs.append(rendered_rgb)
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def align_to_ground_truth(self, poses_gt, poses_pred):
        # pose is camera to world
        
        # compute camera centers
        camera_center_gt = poses_gt[..., :3, 3:]
        camera_center_pred = poses_pred[..., :3, 3:]
        
        # compute scale difference
        scale = rigid_points_registration(camera_center_pred[..., 0], camera_center_gt[..., 0], compute_scaling=True)[-1]
        
        if torch.isnan(scale).any() or scale < 1e-6:
            norm_gt = torch.norm(camera_center_gt[..., 0], dim=-1)
            norm_pred = torch.norm(camera_center_pred[..., 0], dim=-1)
            
            valid_mask = norm_pred > 1e-6
            if valid_mask.any():
                scale = (norm_gt[valid_mask] / norm_pred[valid_mask]).mean()
            else:
                scale = torch.tensor(1.0, device=poses_gt.device)
        
        return scale
    
    def pose_distance(self, target_c2ws, source_c2ws):
        """
        :param reference_pose: Nx4x4 torch array, reference frame camera-to-world pose (not extrinsic matrix!)
        :param measurement_pose: Mx4x4 torch array, measurement frame camera-to-world pose (not extrinsic matrix!)
        :return combined_measure: float, combined pose distance measure
        :return R_measure: float, rotation distance measure
        :return t_measure: float, translation distance measure
        """
        rel_pose = torch.einsum('nij,mjk->nmik', torch.linalg.inv(target_c2ws), source_c2ws)
        R = rel_pose[..., :3, :3]
        t = rel_pose[..., :3, 3]
        batch_R_trace = torch.einsum('...ii', R)
        R_measure = torch.sqrt(2 * (1 - batch_R_trace.clip(max=3.0) / 3))
        t_measure = torch.norm(t, dim=-1)
        combined_measure = torch.sqrt(t_measure ** 2 + R_measure ** 2)
        return combined_measure

    def nvs_single_view_custom(self, images_init, input_indices, sampled_indices, intrinsics, target_extrinsics):
        # get camera trajectory of the input frames
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        shape = images_init[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        
        scale_x = principal_points[0, 0] / intrinsics[..., 0, 2]
        scale_y = principal_points[0, 1] / intrinsics[..., 1, 2]
        intrinsics_new = intrinsics.clone()
        intrinsics_new[..., 0, :] *= scale_x[..., None]
        intrinsics_new[..., 1, :] *= scale_y[..., None]
        new_focals = torch.maximum(intrinsics_new[..., 0, 0], intrinsics_new[..., 1, 1])[..., None]
        principal_points_new = torch.stack([intrinsics_new[..., 0, 2], intrinsics_new[..., 1, 2]], dim=-1)

        c2ws = world_to_kth(poses=c2ws, k=0)
        c2ws_target = world_to_kth(poses=target_extrinsics, k=0)
        scale = self.align_to_ground_truth(c2ws_target[sampled_indices], c2ws)
        camera_traj, num_views = generate_traj(c2ws_target, H, W, new_focals, principal_points_new, self.device)

        # estimate pcd again using only one ref image
        overlapped_masks = torch.isin(sampled_indices, input_indices)
        net_input_indices = torch.nonzero(overlapped_masks, as_tuple=True)[0].item()
        images_ref = [images_init[net_input_indices], copy.deepcopy(images_init[net_input_indices])]
        focals_ref = torch.cat([new_focals[input_indices], new_focals[input_indices]], dim=0)
        principal_points_ref = torch.cat([principal_points_new[input_indices], principal_points_new[input_indices]], dim=0)
        c2ws_ref = torch.cat([c2ws_target[input_indices], c2ws_target[input_indices]], dim=0)
        images_ref[1]['idx'] = 1
        self.run_dust3r(input_images=images_ref)
        depth_ref = self.scene.get_depthmaps(raw=True, clip_thred=self.opts.dpt_trd) * scale
        pcd_ref = self.scene.get_pts3d_from_depthmaps2(depth_ref, focals_ref, principal_points_ref, c2ws_ref)[0].detach()
        img_ref = np.array(self.scene.imgs)[0]
        masks = None

        render_results, viewmask = self.run_render([pcd_ref], [img_ref], masks, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0, 3, 1, 2), size=(self.opts.height, self.opts.width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        render_results[0] = (images_init[net_input_indices]['img_ori'][0].permute(1, 2, 0) + 1) / 2

        diffusion_results = self.run_diffusion(render_results)
        return diffusion_results

    def nvs_sparse_view_custom(self, images_init, input_indices, sampled_indices, intrinsics, target_extrinsics):
        # get camera trajectory of the input frames
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        shape = images_init[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        
        scale_x = principal_points[0, 0] / intrinsics[..., 0, 2]
        scale_y = principal_points[0, 1] / intrinsics[..., 1, 2]
        intrinsics_new = intrinsics.clone()
        intrinsics_new[..., 0, :] *= scale_x[..., None]
        intrinsics_new[..., 1, :] *= scale_y[..., None]
        new_focals = torch.maximum(intrinsics_new[..., 0, 0], intrinsics_new[..., 1, 1])[..., None]
        principal_points_new = torch.stack([intrinsics_new[..., 0, 2], intrinsics_new[..., 1, 2]], dim=-1)
        
        c2ws = world_to_kth(poses=c2ws, k=0)
        c2ws_target = world_to_kth(poses=target_extrinsics, k=0)
        scale = self.align_to_ground_truth(c2ws_target[sampled_indices], c2ws)
        camera_traj, num_views = generate_traj(c2ws_target, H, W, new_focals, principal_points_new, self.device)
        
        # find overlapped indices
        overlapped_masks = torch.isin(sampled_indices, input_indices)
        net_input_indices = torch.nonzero(overlapped_masks, as_tuple=True)[0]
        depth_ref = self.scene.get_depthmaps(raw=True, clip_thred=self.opts.dpt_trd) * scale
        pcd_ref = self.scene.get_pts3d_from_depthmaps2(depth_ref, new_focals[sampled_indices], principal_points_new[sampled_indices], c2ws_target[sampled_indices])
        pcd_ref = [p.detach() for i, p in enumerate(pcd_ref) if i in net_input_indices]
        
        if len(images_init) == 2:
            masks = None
        else:
            ## masks for cleaner point cloud
            self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = [mask for i, mask in enumerate(self.scene.get_masks()) if i in net_input_indices]
            depths = [dpt for i, dpt in enumerate(self.scene.get_depthmaps()) if i in net_input_indices]
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depths]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
            masks = to_numpy(masks_new)
        imgs = np.array(self.scene.imgs)[net_input_indices.tolist()]

        render_results, viewmask = self.run_render(pcd_ref, imgs, masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(self.opts.height, self.opts.width), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        # divide sequence into number of input images
        # target_indices = torch.tensor([i for i in range(len(render_results)) if i not in sampled_indices.tolist()], device=self.device)
        # pose_distmatrix = self.pose_distance(c2ws_target[target_indices], c2ws_target[sampled_indices])
        # input_index = torch.argmin(pose_distmatrix.mean(dim=0), dim=0) # find nearest input
        diffusion_results = self.run_diffusion(render_results, input_index=input_indices[0])
        return diffusion_results