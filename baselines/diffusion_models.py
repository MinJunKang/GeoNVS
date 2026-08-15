

import gc
import os
import math
from omegaconf import OmegaConf
import numpy as np
import torchvision
from pathlib import Path
import importlib
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from PIL import Image

# genfusion imports
from baselines.diffusion.genfusion.lvdm.models.samplers.ddim import DDIMSampler
from baselines.diffusion.genfusion.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond

# cameractrl imports (GeoNVS keeps this helper in seva.geometry; it is
# byte-identical to SVC's diffusions/cameractrl/geometry.py)
from geonvs.seva.geometry import get_plucker_coordinates


def get_diffusion_model(dataset_path, modelname, weight_dtype, device,
                        camctrl_gs_config=None, camctrl_weight_path=None):
    if modelname == 'mvsplat360':
        model = MVSplat360Model(dataset_path, modelname, weight_dtype, device)
    elif "difix3d" in modelname:
        # difix3d_{lrm_model_name}
        model = DifiX3DModel(dataset_path, modelname, weight_dtype, device)
    elif "genfusion" in modelname:
        # genfusion_{lrm_model_name}
        model = GenFusionModel(dataset_path, modelname, weight_dtype, device)
    elif modelname == "viewcrafter":
        model = ViewCrafterModel(dataset_path, modelname, weight_dtype, device)
    elif modelname == "cameractrl":
        model = CameraCtrlModel(dataset_path, modelname, weight_dtype, device)
    elif "geonvs_cameractrl" in modelname:
        model = CameraCtrlGeoNVSModel(
            dataset_path, modelname, weight_dtype, device,
            gs_adapter_config=camctrl_gs_config, weight_path=camctrl_weight_path,
        )
    elif modelname == "motionctrl":
        model = MotionCtrlModel(dataset_path, modelname, weight_dtype, device)
    else:
        raise ValueError(f"Unknown model name: {modelname}")
    return model



class MVSplat360Model(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        checkpoint = "pretrained_weights/dl3dv_480p.ckpt"
        
        # define refiner here
        from baselines.diffusion.mvsplat360.src.model.refiner import get_refiner, RefinerCfg
        
        # define refiner
        refiner_cfg = RefinerCfg(
            name="svd",
            config_path="baselines/diffusion/mvsplat360/config/model/refiner/diffusion_configs/svd_wo_mb_fps.yaml",
            ckpt_path=None,
            use_same_noise_fwd=False,
            verbose=True,
            fps_id=None,
            fit_ckpt=True,
            cond_aug=0.02,
            motion_bucket_id=None,
            svd_clip_cond_type="average",
            test_first_stage=False,
            gs_feat_scale_factor=0.25,
            en_and_decode_n_samples_a_time=1,
            weight_train_gs_feat_via_enc=0.01,
            train_gs_feat_enc_type="target_gt",
            svd_num_frames=21,
            svd_num_steps=25,
            test_time_attn_num_splits=None,
            n_feature_channels=4
        )
        self.refiner = get_refiner(refiner_cfg)
        
        # load checkpoint here
        pretrained_model = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        pretrained_refiner_ckpts = {}
        for k, v in pretrained_model.items():
            if k.startswith("refiner."):
                pretrained_refiner_ckpts[k[len("refiner."):]] = v
        self.refiner.load_state_dict(pretrained_refiner_ckpts)
        self.refiner.to(device)
        self.refiner.eval()
        
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
        return cur_gs_batch.unsqueeze(0), n_border_patch

    def forward(self, batch, device="cuda"):
        
        h, w = batch["input_images"].shape[-2:]
        target_c2ws = batch['extrinsics']
        
        b, target_num_viewpoints = target_c2ws.shape[:2]
        n_svd_frames = self.refiner.cfg.svd_num_frames
        gs_feat_scale_factor = self.refiner.cfg.gs_feat_scale_factor
        
        all_gs_color = None if batch["rendered_results"] is None else batch["rendered_results"].color
        all_gs_feat = None if batch["rendered_results"] is None else batch["rendered_results"].feature
        
        chunk_ses = self.split_into_chunks(
            target_num_viewpoints, n_svd_frames, 0
        )
        randn = torch.randn((b * n_svd_frames, 4,
                             int(h * gs_feat_scale_factor),
                             int(w * gs_feat_scale_factor)), device=device, dtype=self.weight_dtype)
        
        rgb_refined_candidates = []
        for (s_idx, e_idx) in chunk_ses:
            cur_gs_batch, n_border_patch = self.get_current_chunk(
                all_gs_color, s_idx, e_idx, n_svd_frames
            )
            cur_gsfeat_batch, _ = self.get_current_chunk(
                all_gs_feat, s_idx, e_idx, n_svd_frames
            )
            refiner_input = {
                "context_views": batch["input_images"][None].to(dtype=self.weight_dtype),
                "gs_rendered_views": cur_gs_batch.to(dtype=self.weight_dtype),
                "gs_rendered_features": cur_gsfeat_batch.to(dtype=self.weight_dtype),
                "randn": randn.clone(),
            }
            with torch.inference_mode(), torch.autocast(device):
                refiner_output = self.refiner.test_step(refiner_input)
            rgb_refined = rearrange(
                refiner_output["samples_cfg"], "(b v) ... -> b v ...", b=b
            )[0]
            
            rgb_refined_candidates.append(
                rgb_refined[: n_svd_frames - n_border_patch]
            )
        rgb_refined_candidates = torch.cat(rgb_refined_candidates, dim=0)
        
        return rgb_refined_candidates



def _patch_diffusers_for_difix():
    """Restore the old diffusers symbols that `nvidia/difix_ref`'s remote code expects.

    Difix3D was written against diffusers 0.25.1. Newer diffusers renamed a few
    names and moved the PEFT adapter helpers out of `ModelMixin`; nothing is
    overridden here, symbols are only added when missing, so models that already
    inherit `PeftAdapterMixin` (all of GeoNVS's) are unaffected.
    """
    import sys

    import diffusers.models.embeddings as _emb
    if not hasattr(_emb, "PositionNet"):  # renamed to GLIGENTextBoundingboxProjection
        _emb.PositionNet = _emb.GLIGENTextBoundingboxProjection

    import diffusers.loaders as _loaders
    if not hasattr(_loaders, "FromOriginalVAEMixin"):  # renamed to FromOriginalModelMixin
        _loaders.FromOriginalVAEMixin = _loaders.FromOriginalModelMixin

    # diffusers.models.unet_2d_blocks moved under diffusers.models.unets
    import diffusers.models.unets.unet_2d_blocks as _unet_2d_blocks
    sys.modules.setdefault("diffusers.models.unet_2d_blocks", _unet_2d_blocks)

    # add_adapter() and friends moved from ModelMixin into PeftAdapterMixin
    from diffusers.loaders import PeftAdapterMixin
    from diffusers.models.modeling_utils import ModelMixin
    for name in (
        "_hf_peft_config_loaded",
        "add_adapter",
        "set_adapter",
        "set_adapters",
        "enable_adapters",
        "disable_adapters",
        "active_adapters",
        "delete_adapters",
    ):
        if not hasattr(ModelMixin, name) and hasattr(PeftAdapterMixin, name):
            setattr(ModelMixin, name, getattr(PeftAdapterMixin, name))


class DifiX3DModel(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        
        # define model here.
        # Difix3D upstream pins diffusers==0.25.1; the `nvidia/difix_ref` remote
        # code (loaded via trust_remote_code) is written against that old API,
        # so the shim below restores the handful of symbols it expects on the
        # newer diffusers that GeoNVS requires.
        _patch_diffusers_for_difix()
        from baselines.diffusion.difix3d.src.pipeline_difix import DifixPipeline

        self.model = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        self.model.to(device)
        
        # pytorch tensor to pil image
        self.to_pil = torchvision.transforms.ToPILImage()
        self.prompt = "remove degradation"
        
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
        
    def forward(self, batch, device="cuda"):
        assert "rendered_results" in batch, "need regression model before diffusion"
        rendered_colors = batch["rendered_results"]["rendered_images"]
        
        input_mask = batch["mask"]
        video_length = len(rendered_colors)
        input_extrinsics = batch["extrinsics"][0][input_mask]
        target_extrinsics = batch["extrinsics"][0]
        
        dist_matrix = self.pose_distance(target_extrinsics, input_extrinsics)
        ref_indices = dist_matrix.argmin(dim=-1)
        
        output_results = []
        for i in range(video_length):
            target_noisy_img = self.to_pil(rendered_colors[i].cpu())
            ref_img = self.to_pil((batch["input_images"][ref_indices[i]].cpu() + 1) / 2)
            with torch.inference_mode(), torch.autocast(device):
                output_image = self.model(self.prompt, image=target_noisy_img, ref_image=ref_img, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
            output_image = torch.tensor(np.array(output_image)).permute(2,0,1).to(device)
            output_results.append(output_image)
        output = torch.stack(output_results, dim=0) / 255.0  # normalize to [0,1]
        
        return output


class GenFusionModel(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        # Upstream GenFusion releases this file as "epoch=59-step=34000.ckpt"
        # (DL3DV model); it is renamed at download time, see
        # tools/scripts/download_weights_all.sh.
        checkpoint = "pretrained_weights/genfusion_dl3dv.ckpt"
        config_path = "baselines/diffusion/genfusion/generation_infer.yaml"

        # define model here
        # requires open-clip-torch==2.12.0, whereas ours are open-clip-torch==2.32.0
        # requires transformers==4.46.2, whereas ours are transformers==4.49.0
        #
        # The numbers reported in the paper were measured in GenFusion's own
        # environment (open-clip-torch==2.12.0 / transformers==4.46.2), not in
        # the shared one this repository installs. GenFusion still runs here
        # because encode_with_vision_transformer() tolerates the `input_patchnorm`
        # attribute that open_clip >= 2.26 removed, but the conditioner is not
        # bit-identical across those versions and the results are lower:
        #
        #                   this env        paper
        #   DL3DV-10 P=6    10.94 PSNR      13.11 PSNR
        #   RE10K    P=3    20.58 PSNR      22.69 PSNR
        #
        # Recreate the upstream environment to reproduce the reported numbers:
        #   pip install open-clip-torch==2.12.0 transformers==4.46.2
        model_config = OmegaConf.load(config_path)
        self.model = self.instantiate_from_config(model_config)
        
        # load ckpt
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except Exception:
                ## rename the keys for 256x256 model
                new_pl_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_pl_sd[k] = v

                for k in list(new_pl_sd.keys()):
                    if "framestride_embed" in k:
                        new_key = k.replace("framestride_embed", "fps_embedding")
                        new_pl_sd[new_key] = new_pl_sd[k]
                        del new_pl_sd[k]
                self.model.load_state_dict(new_pl_sd, strict=True)
        else:
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict["module"].keys():
                new_pl_sd[key[16:]] = state_dict["module"][key]
            self.model.load_state_dict(new_pl_sd)
        self.model.eval()
        self.model.perframe_ae = True
        self.model = self.model.to(dtype=weight_dtype, device=device)
        self.unconditional_guidance_scale = 3.2  # in genfusion code
        self.diffusion_resize_height = 320
        self.diffusion_resize_width = 512
        self.n_generate_frames = 21
        
    def instantiate_from_config(self, config):
        if not "target" in config:
            if config == "__is_first_stage__":
                return None
            elif config == "__is_unconditional__":
                return None
            raise KeyError("Expected key `target` to instantiate.")
        return self.get_obj_from_str(config["target"])(**config.get("params", dict()))
    
    def get_obj_from_str(self, string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    
    def get_latent_z(self, videos):
        with torch.no_grad(), torch.cuda.amp.autocast():
            z = self.model.encode_first_stage(videos)
        return z

    def image_guided_synthesis(
        self,
        rgb_video,
        depth_video,
        ref_frames,
        noise_shape,
        init_rgb=None,
        init_depth=None,
        n_samples=1,
        ddim_steps=50,
        ddim_eta=1.0,
        unconditional_guidance_scale=1.0,
        cfg_img=None,
        fs=None,
        text_input=False,
        multiple_cond_cfg=False,
        loop=False,
        interp=False,
        timestep_spacing="uniform",
        guidance_rescale=0.0,
        **kwargs,
    ):
        ddim_sampler = (
            DDIMSampler(self.model)
            if not multiple_cond_cfg
            else DDIMSampler_multicond(self.model)
        )

        batch_size = noise_shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=self.model.device)

        # img = videos[:,:,0] #bchw
        # img_emb = model.embedder(img) ## blc
        # img_emb = model.image_proj_model(img_emb)
        with torch.no_grad(), torch.cuda.amp.autocast():
            ref_frames = rearrange(ref_frames, "b c l h w -> (b l) c h w")
            img_emb = self.model.embedder(ref_frames)  ## (b l) c
            img_emb = self.model.image_proj_model(img_emb)
            img_emb = rearrange(img_emb, "(b t) l c -> b (t l) c", b=1)
            cond = {"c_crossattn": [torch.cat([img_emb], dim=1)]}

            if self.model.model.conditioning_key == "hybrid":
                videos = torch.cat([rgb_video, depth_video], dim=1)

                img_cat_cond = self.get_latent_z(videos).half().detach()  # b c t h w
                cond["c_concat"] = [img_cat_cond]  # b c 1 h w

            init_latent_z = None

            if init_rgb is not None and init_depth is not None:
                init_latent_z = self.get_latent_z(
                    torch.cat([init_rgb, init_depth], dim=1)
                )

            if unconditional_guidance_scale != 1.0:
                uc_img_emb = self.model.embedder(torch.zeros_like(ref_frames))  ## b l c
                uc_img_emb = self.model.image_proj_model(uc_img_emb)
                uc_img_emb = rearrange(uc_img_emb, "(b t) l c -> b (t l) c", b=1)
                uc = {"c_crossattn": [torch.cat([uc_img_emb], dim=1)]}
                if self.model.model.conditioning_key == "hybrid":
                    uc["c_concat"] = [img_cat_cond]
            else:
                uc = None

            kwargs.update({"unconditional_conditioning_img_nonetext": None})

            # z0 = img_cat_cond#None
            z0 = None  # None
            cond_mask = None

            results = []
            for _ in range(n_samples):
                if z0 is not None:
                    cond_z0 = z0.clone()
                    kwargs.update({"clean_cond": True})
                else:
                    cond_z0 = None
                if ddim_sampler is not None:
                    # x_t = ddim_sampler.stochastic_encode(img_cat_cond, ddim_steps,use_original_steps=True
                    x_t = init_latent_z

                    samples, _ = ddim_sampler.sample(
                        S=ddim_steps,
                        conditioning=cond,
                        batch_size=batch_size,
                        shape=noise_shape[1:],
                        verbose=False,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        cfg_img=cfg_img,
                        mask=cond_mask,
                        x0=cond_z0,
                        x_T=x_t,
                        fs=fs,
                        timestep_spacing=timestep_spacing,
                        precision=16,
                        guidance_rescale=guidance_rescale,
                        **kwargs,
                    )

                batch_images = self.model.decode_first_stage(samples, return_depth=True)

                results.append(batch_images)

            results = torch.stack(results)
            return results.permute(1, 0, 2, 3, 4, 5)
    
    def rgb_preprocess(self, artifact_rgb):
        artifact_rgb = (artifact_rgb - 0.5) * 2
        artifact_rgb = F.interpolate(
            artifact_rgb,
            size=(
                artifact_rgb.shape[2],
                self.diffusion_resize_height,
                self.diffusion_resize_width,
            ),
            mode="trilinear",
            align_corners=False,
        )
        return artifact_rgb
    
    def depth_preprocess(self, artifact_depth):
        epsilon = 1e-10
        valid_mask = artifact_depth > 0
        disparity = torch.zeros_like(artifact_depth)
        disparity[valid_mask] = 1.0 / (artifact_depth[valid_mask] + epsilon)
        valid_disparities = torch.masked_select(disparity, valid_mask)

        if valid_disparities.numel() > 0:
            disp_min = valid_disparities.min()
            disp_max = valid_disparities.max()
            normalized_disparity = torch.zeros_like(disparity)
            normalized_disparity[valid_mask] = (disparity[valid_mask] - disp_min) / (
                disp_max - disp_min
            )
        else:
            print("Warning: No valid depth values found")
            normalized_disparity = torch.zeros_like(disparity)
        normalized_disparity = (normalized_disparity - 0.5) * 2
        normalized_disparity = F.interpolate(
            normalized_disparity,
            size=(
                normalized_disparity.shape[2],
                self.diffusion_resize_height,
                self.diffusion_resize_width,
            ),
            mode="nearest",
        )
        return normalized_disparity
    
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
    
    def forward(self, batch, device="cuda"):
        assert "rendered_results" in batch, "need regression model before diffusion"
        rendered_colors = batch["rendered_results"]["rendered_images"]
        rendered_depths = batch["rendered_results"]["rendered_depths"]
        rendered_colors = rearrange(rendered_colors, "t c h w -> 1 c t h w").to(dtype=self.weight_dtype)
        rendered_depths = rearrange(rendered_depths, "t h w -> 1 1 t h w").to(dtype=self.weight_dtype)
        ref_frames = rearrange(batch["input_images"], "t c h w -> 1 c t h w").to(dtype=self.weight_dtype)
        H, W = batch["input_images"].shape[-2:]
        n_frames = rendered_colors.shape[2]
        chunk_ses = self.split_into_chunks(
            n_frames, self.n_generate_frames, 0
        )
        
        # preprocess
        rendered_colors = self.rgb_preprocess(rendered_colors)
        rendered_depths = self.depth_preprocess(rendered_depths)
        ref_frames = self.rgb_preprocess((ref_frames + 1) / 2.0)
        
        init_rgb = None
        init_depth = None
        rgb_refined, depth_refined = [], []
        with torch.inference_mode(), torch.autocast(device):
            
            for (s_idx, e_idx) in chunk_ses:
                cur_color_batch, n_border_patch = self.get_current_chunk(
                    rendered_colors, s_idx, e_idx, self.n_generate_frames
                )
                cur_depth_batch, _ = self.get_current_chunk(
                    rendered_depths, s_idx, e_idx, self.n_generate_frames
                )
                
                _, _, n_frames, h, w = cur_color_batch.shape  # (1, 3, n_frames, h, w)
                h = h // 16 * 2
                w = w // 16 * 2
                noise_shape = [1, 4, n_frames, h, w]

                batch_samples = self.image_guided_synthesis(
                    cur_color_batch,
                    cur_depth_batch,
                    ref_frames,
                    noise_shape,
                    init_rgb,
                    init_depth,
                    n_samples=1,
                    ddim_steps=25,
                    ddim_eta=1.0,
                    unconditional_guidance_scale=self.unconditional_guidance_scale,
                    cfg_img=None,
                    fs=30,
                    guidance_rescale=0.7,
                )
                batch_samples = batch_samples[0, 0]
                repaired_rgb = batch_samples[:3]
                repaired_depth = batch_samples[3:]
                repaired_rgb = torch.clamp(repaired_rgb, -1.0, 1.0)
                repaired_rgb = (repaired_rgb + 1.0) / 2.0
                repaired_depth = torch.clamp(repaired_depth, -1.0, 1.0)
                repaired_depth = (repaired_depth + 1.0) / 2.0
                repaired_depth = 1 / (repaired_depth + 1e-6)
                
                repaired_rgb = F.interpolate(
                    repaired_rgb[None],
                    size=(
                        repaired_rgb.shape[1],
                        H,
                        W,
                    ),
                    mode="trilinear",
                    align_corners=False
                )[0]
                repaired_depth = F.interpolate(
                    repaired_depth[None],
                    size=(
                        repaired_depth.shape[1],
                        H,
                        W,
                    ),
                    mode="nearest"
                )[0]
                repaired_rgb = repaired_rgb[:, :self.n_generate_frames - n_border_patch]
                repaired_depth = repaired_depth[:, :self.n_generate_frames - n_border_patch]
                rgb_refined.append(repaired_rgb)
                depth_refined.append(repaired_depth)
        rgb_refined = torch.cat(rgb_refined, dim=1)
        depth_refined = torch.cat(depth_refined, dim=1)
        rgb_refined = rearrange(rgb_refined, "c t h w -> t c h w")
        depth_refined = rearrange(depth_refined, "c t h w -> t c h w")
        return rgb_refined


class ViewCrafterModel(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        dust3r_ckpt_path = "pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        if model_name == "viewcrafter_sparse":
            checkpoint = "pretrained_weights/viewcrafter_sparse.ckpt"
            # checkpoint = "pretrained_weights/viewcrafter_lowres.ckpt"
            config_path = "./baselines/diffusion/viewcrafter/viewcrafter_sparse.yaml"
        elif model_name == "viewcrafter":
            # checkpoint = "pretrained_weights/viewcrafter.ckpt"
            checkpoint = "pretrained_weights/viewcrafter_lowres.ckpt"
            config_path = "./baselines/diffusion/viewcrafter/viewcrafter_single.yaml"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # define model here
        # requires open-clip-torch==2.17.1, whereas ours are open-clip-torch==2.32.0
        from baselines.diffusion.viewcrafter.viewcrafter import ViewCrafter
        config = OmegaConf.load(config_path)
        self.n_generate_frames = config.video_length
        self.model = ViewCrafter(opts=config, ckpt_path=checkpoint, model_path=dust3r_ckpt_path, device=device)
        self.sample_view_dust3r = 5  # use external N views pair to run dust3r for scale-consistent point cloud (single view case, minimum requirement is 2)

    def forward(self, batch, device="cuda"):
        input_mask = batch["mask"]
        input_extrinsics = batch["extrinsics"][0][input_mask]
        target_extrinsics = batch["extrinsics"][0]
        target_intrinsics = batch["intrinsics"][0]
        num_input_views = input_extrinsics.shape[0]
        
        # prepare inputs for viewcrafter
        input_indices = torch.nonzero(input_mask, as_tuple=False).squeeze(-1)
        sample_view_num = min(self.sample_view_dust3r, len(input_mask))
        sampled_indices = torch.linspace(0, len(input_mask) - 1, steps=sample_view_num, device=device).long()
        sampled_indices = torch.unique(torch.cat([input_indices, sampled_indices], dim=0))
        images = self.model.preprocess_images((batch["images_all"][sampled_indices] + 1) / 2)
        self.model.run_dust3r(input_images=images, clean_pc=True)
        
        # generate scene
        if num_input_views == 1:
            output = self.model.nvs_single_view_custom(images, input_indices, sampled_indices, target_intrinsics, target_extrinsics)
        elif num_input_views > 1:
            output = self.model.nvs_sparse_view_custom(images, input_indices, sampled_indices, target_intrinsics, target_extrinsics)
        else:
            raise ValueError(f"ViewCrafter model {self.model_name} does not support {num_input_views} input views")
        
        # restore size
        H, W = batch["input_images"].shape[-2:]
        output = F.interpolate(
            output[None],
            size=(
                output.shape[1],
                H,
                W,
            ),
            mode="trilinear",
            align_corners=False
        )[0]
        output = rearrange(output, "c t h w -> t c h w")
        output = (output + 1.0) / 2.0  # to [0,1]
        
        return output
    
    
class CameraCtrlModel(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        self.n_generate_frames = 14
        checkpoint = "pretrained_weights/CameraCtrl_svd.ckpt"
        self.model_configs = OmegaConf.load("configs/svd_cameractrl.yaml")
        self.height = 320
        self.width = 576
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(1234)
        
        # define model here
        from geonvs.camctrl.modules.pose_adaptor import CameraPoseEncoder
        from geonvs.camctrl.camctrl_pipeline import StableVideoDiffusionPipelinePoseCond
        from geonvs.camctrl.modules.unet import UNetSpatioTemporalConditionModelPoseCond
        from geonvs.camctrl.camctrl_diffusers import CameraCtrlModel as CameraCtrlDiffusersModel
        from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from diffusers.utils.import_utils import is_xformers_available
        
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="vae"
        )
        unet = UNetSpatioTemporalConditionModelPoseCond.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="unet",
            down_block_types=self.model_configs['down_block_types'], up_block_types=self.model_configs['up_block_types']
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="feature_extractor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="image_encoder"
        )
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="scheduler"
        )
        pose_encoder = CameraPoseEncoder(**self.model_configs['pose_encoder_kwargs'])
        unet.set_pose_cond_attn_processor(
            enable_xformers=(self.model_configs["enable_xformers_memory_efficient_attention"] and is_xformers_available()), 
            **self.model_configs['attention_processor_kwargs']
        )
        
        main_model = CameraCtrlDiffusersModel(
            unet=unet,
            pose_encoder=pose_encoder,
            num_frames=self.n_generate_frames,
            use_gs_adapter=False
        )
        
        state_dict = torch.load(checkpoint, map_location=unet.device, weights_only=True)
        pose_encoder_state_dict = state_dict['pose_encoder_state_dict']
        pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
        assert len(pose_encoder_m) == 0 and len(pose_encoder_u) == 0
        attention_processor_state_dict = state_dict['attention_processor_state_dict']
        _, attention_processor_u = unet.load_state_dict(attention_processor_state_dict, strict=False, assign=True)
        assert len(attention_processor_u) == 0
        
        pose_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        image_encoder.to(device, dtype=weight_dtype)
        main_model.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)
        
        self.pipeline = StableVideoDiffusionPipelinePoseCond(
            vae=vae,
            image_encoder=image_encoder,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
            main_model=main_model
        )
        
    def rgb_preprocess(self, artifact_rgb):
        artifact_rgb = F.interpolate(
            artifact_rgb,
            size=(
                artifact_rgb.shape[2],
                self.height,
                self.width,
            ),
            mode="trilinear",
            align_corners=False,
        )
        return artifact_rgb
        
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
        return cur_gs_batch.unsqueeze(0), n_border_patch

    def forward(self, batch, device="cuda"):
        input_mask = batch["mask"]
        target_extrinsics = batch["extrinsics"][0]
        target_intrinsics = batch["intrinsics"][0]
        height, width = batch["input_images"].shape[-2:]
        ref_images = self.rgb_preprocess(batch['input_images'][0:1][None])
        intrinsics = target_intrinsics.clone()
        intrinsics[..., 0, :] *= (self.width / width)
        intrinsics[..., 1, :] *= (self.height / height)
        
        # prepare plucker coordinates
        w2cs = torch.linalg.inv(target_extrinsics)
        pluckers = get_plucker_coordinates(
            extrinsics_src=w2cs[input_mask][0],
            extrinsics=w2cs,
            intrinsics=intrinsics.clone(),
            target_size=(self.height, self.width),
            donwsample_factor=8,  # (sic) parameter name in seva.geometry
            convention="cameractrl"
        )
        pluckers = rearrange(pluckers, "t c h w -> 1 c t h w")
        
        n_frames = len(target_extrinsics)
        chunk_ses = self.split_into_chunks(
            n_frames, self.n_generate_frames, 0
        )
        
        pred_video_frames_all = []
        with torch.inference_mode(), torch.autocast(device):
            
            for (s_idx, e_idx) in chunk_ses:
                cur_plucker, n_border_patch = self.get_current_chunk(
                    pluckers, s_idx, e_idx, self.n_generate_frames
                )

                pred_video_frames, int_feats = self.pipeline(
                    image=ref_images[:, 0],
                    pose_embedding=cur_plucker,
                    extrinsics=None,
                    intrinsics=None,
                    near=None,
                    far=None,
                    gaussians=None,
                    scale_invariant=True,
                    e3nn=False,
                    use_gs_adapter=False,
                    height=self.height,
                    width=self.width,
                    num_frames=self.n_generate_frames,
                    num_inference_steps=self.model_configs["num_inference_steps"],
                    min_guidance_scale=self.model_configs["min_guidance_scale"],
                    max_guidance_scale=self.model_configs["max_guidance_scale"],
                    generator=self.generator,
                    output_type='pt'
                )
                pred_video_frames = rearrange(pred_video_frames[0], 't c h w -> c t h w')
                pred_video_frames = F.interpolate(
                    pred_video_frames[None],
                    size=(
                        pred_video_frames.shape[1],
                        height,
                        width,
                    ),
                    mode="trilinear",
                    align_corners=False
                )[0]
                pred_video_frames = pred_video_frames[:, :self.n_generate_frames - n_border_patch]
                pred_video_frames_all.append(pred_video_frames)
        pred_video_frames_all = torch.cat(pred_video_frames_all, dim=1)
        pred_video_frames_all = rearrange(pred_video_frames_all, "c t h w -> t c h w")
        return pred_video_frames_all
    
    
    
class CameraCtrlGeoNVSModel(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device,
                 gs_adapter_config=None, weight_path=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        self.n_generate_frames = 14
        checkpoint = "pretrained_weights/CameraCtrl_svd.ckpt"
        self.model_configs = OmegaConf.load("configs/svd_cameractrl.yaml")
        self.height = 320
        self.width = 576
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(1234)
        
        # GS-Adapter config / weights (overridable via demo_diffusion.py
        # --camctrl_gs_config / --camctrl_weight_path)
        if gs_adapter_config is None:
            gs_adapter_config = "configs/module_config/gsadapter_camctrl_gattn.yaml"
        if weight_path is None:
            weight_path = "pretrained_weights/eccv_camctrl"
        checkpoint_lora = os.path.join(weight_path, "pytorch_lora_weights.safetensors")
        checkpoint_geonvs = os.path.join(weight_path, "gs_adapter_weights.pth")
        self.gs_config = OmegaConf.load(gs_adapter_config)
        
        self.use_gs_adapter = self.gs_config['use_gs_adapter']
        add_lora = self.gs_config['add_lora']
        
        # define model here
        from geonvs.camctrl.modules.pose_adaptor import CameraPoseEncoder
        from geonvs.camctrl.camctrl_pipeline import StableVideoDiffusionPipelinePoseCond
        from geonvs.camctrl.modules.unet import UNetSpatioTemporalConditionModelPoseCond
        from geonvs.camctrl.camctrl_diffusers import CameraCtrlModel as CameraCtrlDiffusersModel
        from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from diffusers.utils.import_utils import is_xformers_available
        
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="vae"
        )
        unet = UNetSpatioTemporalConditionModelPoseCond.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="unet",
            down_block_types=self.model_configs['down_block_types'], up_block_types=self.model_configs['up_block_types']
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="feature_extractor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="image_encoder"
        )
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_configs["svd_pretrained_path"], subfolder="scheduler"
        )
        pose_encoder = CameraPoseEncoder(**self.model_configs['pose_encoder_kwargs'])
        unet.set_pose_cond_attn_processor(
            enable_xformers=(self.model_configs["enable_xformers_memory_efficient_attention"] and is_xformers_available()), 
            **self.model_configs['attention_processor_kwargs']
        )
        
        main_model = CameraCtrlDiffusersModel(
            unet=unet,
            pose_encoder=pose_encoder,
            num_frames=self.n_generate_frames,
            use_gs_adapter=self.use_gs_adapter,
            gs_adapter_config=self.gs_config,
            max_resolution=(self.height, self.width),
        )
        
        state_dict = torch.load(checkpoint, map_location=unet.device, weights_only=True)
        pose_encoder_state_dict = state_dict['pose_encoder_state_dict']
        pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
        assert len(pose_encoder_m) == 0 and len(pose_encoder_u) == 0
        attention_processor_state_dict = state_dict['attention_processor_state_dict']
        _, attention_processor_u = unet.load_state_dict(attention_processor_state_dict, strict=False, assign=True)
        assert len(attention_processor_u) == 0
        
        if add_lora:
            unet.load_lora_adapter(checkpoint_lora, adapter_name="gs_lora", prefix=None)
        if self.use_gs_adapter:
            gs_state_dict = torch.load(checkpoint_geonvs, map_location="cpu", weights_only=True)
            legacy_prefixes = ('output_proj.', 'rope.', 'feature_predictor.')
            for name in gs_state_dict.keys():
                sub_sd = gs_state_dict[name]
                # Checkpoints trained with the old adapter code contain submodules
                # (output_proj / rope / feature_predictor) and buffers
                # (fusion_layer.rope_freqs) that no longer exist; drop them before
                # the strict per-module load (same pattern as demo.py).
                dropped_keys = [
                    k for k in list(sub_sd.keys())
                    if k.startswith(legacy_prefixes) or 'fusion_layer.rope_freqs' in k
                ]
                for k in dropped_keys:
                    del sub_sd[k]
                if dropped_keys:
                    print(f"[gs-adapter] Dropped legacy keys from '{name}': {dropped_keys}")
                getattr(main_model, name).load_state_dict(sub_sd, strict=True)
            main_model.gsattn.float()

        pose_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        image_encoder.to(device, dtype=weight_dtype)
        main_model.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)
        
        self.pipeline = StableVideoDiffusionPipelinePoseCond(
            vae=vae,
            image_encoder=image_encoder,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
            main_model=main_model
        )
        
    def rgb_preprocess(self, artifact_rgb):
        artifact_rgb = F.interpolate(
            artifact_rgb,
            size=(
                artifact_rgb.shape[2],
                self.height,
                self.width,
            ),
            mode="trilinear",
            align_corners=False,
        )
        return artifact_rgb
        
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
        return cur_gs_batch.unsqueeze(0), n_border_patch
    
    def center_cameras(self, all_c2ws, c2ws):
        ref_c2ws = all_c2ws
        camera_dist_2med = torch.norm(
            ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
            dim=-1,
        )
        valid_mask = camera_dist_2med <= torch.clamp(
            torch.quantile(camera_dist_2med, 0.97) * 10,
            max=1e6,
        )
        c2ws[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
        

    def scale_cameras(self, c2ws, camera_scale=2.0):
        camera_dists = c2ws[:, :3, 3].clone()
        translation_scaling_factor = (
            camera_scale
            if torch.isclose(
                torch.norm(camera_dists[0]),
                torch.zeros(1, device=camera_dists.device),
                atol=1e-5,
            ).any()
            else (camera_scale / torch.norm(camera_dists[0]))
        )
        c2ws[:, :3, 3] *= translation_scaling_factor

    def forward(self, batch, device="cuda"):
        input_mask = batch["mask"]
        target_extrinsics = batch["extrinsics"][0]
        target_intrinsics = batch["intrinsics"][0]
        height, width = batch["input_images"].shape[-2:]
        ref_images = self.rgb_preprocess(batch['input_images'][0:1][None])
        intrinsics = target_intrinsics.clone()
        intrinsics[..., 0, :] *= (self.width / width)
        intrinsics[..., 1, :] *= (self.height / height)
        
        n_frames = len(target_extrinsics)
        target_extrinsics_norm = target_extrinsics.clone()
        self.center_cameras(target_extrinsics, target_extrinsics_norm)
        self.scale_cameras(target_extrinsics_norm)
        
        # prepare plucker coordinates
        w2cs = torch.linalg.inv(target_extrinsics_norm)
        pluckers = get_plucker_coordinates(
            extrinsics_src=w2cs[input_mask][0],
            extrinsics=w2cs,
            intrinsics=intrinsics.clone(),
            target_size=(self.height, self.width),
            donwsample_factor=8,  # (sic) parameter name in seva.geometry
            convention="cameractrl"
        )
        pluckers = rearrange(pluckers, "t c h w -> 1 t c h w")
        target_extrinsics = rearrange(target_extrinsics, "t c1 c2 -> 1 t c1 c2")
        intrinsics = rearrange(intrinsics, "t c1 c2 -> 1 t c1 c2")
        
        chunk_ses = self.split_into_chunks(
            n_frames, self.n_generate_frames, 0
        )
        
        pred_video_frames_all = []
        with torch.inference_mode(), torch.autocast(device):
            
            for (s_idx, e_idx) in chunk_ses:
                cur_plucker, n_border_patch = self.get_current_chunk(
                    pluckers, s_idx, e_idx, self.n_generate_frames
                )
                cur_extrinsics = self.get_current_chunk(
                    target_extrinsics, s_idx, e_idx, self.n_generate_frames
                )[0]
                cur_intrinsics = self.get_current_chunk(
                    intrinsics, s_idx, e_idx, self.n_generate_frames
                )[0]
                cur_nears = self.get_current_chunk(
                    batch['rendered_results']['nears'], s_idx, e_idx, self.n_generate_frames
                )[0]
                cur_fars = self.get_current_chunk(
                    batch['rendered_results']['fars'], s_idx, e_idx, self.n_generate_frames
                )[0]

                pred_video_frames, int_feats = self.pipeline(
                    image=ref_images[:, 0],
                    pose_embedding=cur_plucker,
                    extrinsics=cur_extrinsics,
                    intrinsics=cur_intrinsics,
                    near=cur_nears,
                    far=cur_fars,
                    gaussians=batch['rendered_results']['gaussians'],
                    scale_invariant=batch['rendered_results']['scale_invariant'],
                    e3nn=batch['rendered_results']['use_e3nn'],
                    use_gs_adapter=self.use_gs_adapter,
                    height=self.height,
                    width=self.width,
                    num_frames=self.n_generate_frames,
                    num_inference_steps=self.model_configs["num_inference_steps"],
                    min_guidance_scale=self.model_configs["min_guidance_scale"],
                    max_guidance_scale=self.model_configs["max_guidance_scale"],
                    generator=self.generator,
                    output_type='pt'
                )
                pred_video_frames = rearrange(pred_video_frames[0], 't c h w -> c t h w')
                pred_video_frames = F.interpolate(
                    pred_video_frames[None],
                    size=(
                        pred_video_frames.shape[1],
                        height,
                        width,
                    ),
                    mode="trilinear",
                    align_corners=False
                )[0]
                pred_video_frames = pred_video_frames[:, :self.n_generate_frames - n_border_patch]
                pred_video_frames_all.append(pred_video_frames)
        pred_video_frames_all = torch.cat(pred_video_frames_all, dim=1)
        pred_video_frames_all = rearrange(pred_video_frames_all, "c t h w -> t c h w")
        return pred_video_frames_all
    
    
    
class MotionCtrlModel(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        checkpoint = "pretrained_weights/motionctrl_svd.ckpt"
        config_path = "baselines/diffusion/motionctrl/config_motionctrl_cmcm.yaml"
        
        self.motionctrl_cfg = {
            'ddim_steps': 25,
            'n_generate_frames': 14,
            'height': 576,
            'width': 1024,
            'seed': 12345,
            'decoding_t': 2,
            'motion_bucket_id': 255,
            'fps': 10,
            'cond_aug': 0.02,
            'sample_num': 1,
            'speed': 2.0
        }

        self.ddim_steps = self.motionctrl_cfg['ddim_steps']
        self.n_generate_frames = self.motionctrl_cfg['n_generate_frames']
        self.height = self.motionctrl_cfg['height']
        self.width = self.motionctrl_cfg['width']
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.motionctrl_cfg['seed'])
        
        from baselines.diffusion.motionctrl.model_wrapper import load_model
        self.model, self.filter = load_model(config_path, checkpoint, device, self.n_generate_frames, self.ddim_steps)
        torch.manual_seed(self.motionctrl_cfg['seed'])
        self.shape = (self.n_generate_frames, 4, self.height // 8, self.width // 8)

    def _to_relative_camera_pose(
        self, camera_pose: torch.Tensor, keyframe_index: int = 0, keyframe_zero: bool = False
    ) -> torch.Tensor:
        camera_pose = camera_pose.reshape(-1, 3, 4)
        rotation_dst = camera_pose[:, :, :3]
        translation_dst = camera_pose[:, :, 3:]

        rotation_src = rotation_dst[keyframe_index : keyframe_index + 1].repeat(camera_pose.shape[0], 1, 1)
        translation_src = translation_dst[keyframe_index : keyframe_index + 1].repeat(camera_pose.shape[0], 1, 1)

        rotation_src_inv = rotation_src.permute(0, 2, 1)
        rotation_rel = rotation_dst @ rotation_src_inv
        translation_rel = translation_dst - rotation_rel @ translation_src

        rt_rel = torch.cat([rotation_rel, translation_rel], dim=-1)
        rt_rel = rt_rel.reshape(-1, 12)

        if keyframe_zero:
            rt_rel[keyframe_index] = torch.zeros_like(rt_rel[keyframe_index])

        return rt_rel
    
    def get_unique_embedder_keys_from_conditioner(self, conditioner):
        return list(set([x.input_key for x in conditioner.embedders]))
    
    def get_batch(self, keys, value_dict, N, T, device):
        batch = {}
        batch_uc = {}

        for key in keys:
            if key == "fps_id":
                batch[key] = (
                    torch.tensor([value_dict["fps_id"]])
                    .to(device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "motion_bucket_id":
                batch[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .to(device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to(device),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames":
                batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
            elif key == "cond_frames_without_noise":
                batch[key] = repeat(
                    value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
                )
            else:
                batch[key] = value_dict[key]

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc
    
    def sample(
        self,
        RT,  # [t, 3, 4], torch tensor
        cond_image,  # [1, 3, H, W], torch tensor
        device: str = "cuda",
    ):
        """
        Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
        image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
        """
        fps_id = self.motionctrl_cfg['fps']
        num_frames = self.n_generate_frames
        motion_bucket_id = self.motionctrl_cfg['motion_bucket_id']
        cond_aug = self.motionctrl_cfg['cond_aug']
        decoding_t = self.motionctrl_cfg['decoding_t']
        sample_num = self.motionctrl_cfg['sample_num']
        camera_speed = self.motionctrl_cfg['speed']

        # don't know why need [3, 1, 4] ?, from original code
        RT[..., -1] = RT[..., -1] * torch.tensor([[3., 1., 4.]], device=device) * camera_speed
        rel_RT = self._to_relative_camera_pose(RT)
        rel_RT = rel_RT.unsqueeze(0).repeat(2,1,1)  # [2, t, 12]
        
        # image : [B, 3, H, W] in range [-1, 1]
        H, W = cond_image.shape[2:]
        assert cond_image.shape[1] == 3
        assert (H, W) == (self.height, self.width), f"Conditioning frame must be {self.height}x{self.width}, but got {H}x{W}."
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )
        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")
        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = cond_image
        value_dict["cond_frames"] = cond_image + cond_aug * torch.randn_like(cond_image)

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = self.get_batch(
                    self.get_unique_embedder_keys_from_conditioner(self.model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = self.model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
                additional_model_inputs["RT"] = rel_RT

                def denoiser(input, sigma, c):
                    return self.model.denoiser(
                        self.model.model, input, sigma, c, **additional_model_inputs
                    )

                results = []
                for j in range(sample_num):
                    randn = torch.randn(self.shape, device=device)
                    samples_z = self.model.sampler(denoiser, randn, cond=c, uc=uc)
                    self.model.en_and_decode_n_samples_a_time = decoding_t
                    samples_x = self.model.decode_first_stage(samples_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0) # [1*t, c, h, w]
                    results.append(samples)

                samples = torch.stack(results, dim=0) # [sample_num, t, c, h, w]
        return samples

    def rgb_preprocess(self, artifact_rgb):
        artifact_rgb = F.interpolate(
            artifact_rgb,
            size=(
                self.height,
                self.width,
            ),
            mode="bilinear",
            align_corners=False,
        )
        return artifact_rgb
        
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

        cur_pose_batch = batch_input[start_idx:end_idx]

        if cur_pose_batch.shape[0] != chunk_size:
            n_border_patch = chunk_size - cur_pose_batch.shape[0]
            cur_pose_batch = torch.cat(
                (
                    cur_pose_batch,
                    repeat(
                        cur_pose_batch[-1:],
                        "b ... -> (b n) ...",
                        n=n_border_patch,
                    ),
                ),
                dim=0,
            )
        return cur_pose_batch, n_border_patch

    def forward(self, batch, device="cuda"):
        input_mask = batch["mask"]
        target_extrinsics = batch["extrinsics"][0]  # c2w
        target_intrinsics = batch["intrinsics"][0]
        height, width = batch["input_images"].shape[-2:]
        intrinsics = target_intrinsics.clone()
        intrinsics[..., 0, :] *= (self.width / width)
        intrinsics[..., 1, :] *= (self.height / height)
        ref_image = self.rgb_preprocess(batch['input_images'][0:1])
        
        n_frames = len(target_extrinsics)
        chunk_ses = self.split_into_chunks(
            n_frames, self.n_generate_frames, 0
        )
        
        pred_video_frames_all = []
        with torch.inference_mode(), torch.autocast(device):
            
            for (s_idx, e_idx) in chunk_ses:
                cur_pose, n_border_patch = self.get_current_chunk(
                    target_extrinsics.inverse(), s_idx, e_idx, self.n_generate_frames
                )

                pred_video_frames = self.sample(
                    RT=cur_pose[:, :3],
                    cond_image=ref_image,
                    device=device,
                )[0]  # [t, c, h, w]
                pred_video_frames = rearrange(pred_video_frames, 't c h w -> c t h w')
                pred_video_frames = F.interpolate(
                    pred_video_frames[None],
                    size=(
                        pred_video_frames.shape[1],
                        height,
                        width,
                    ),
                    mode="trilinear",
                    align_corners=False
                )[0]
                pred_video_frames = pred_video_frames[:, :self.n_generate_frames - n_border_patch]
                pred_video_frames_all.append(pred_video_frames)
        pred_video_frames_all = torch.cat(pred_video_frames_all, dim=1)
        pred_video_frames_all = rearrange(pred_video_frames_all, "c t h w -> t c h w")
        return pred_video_frames_all
    
    
class ModelWrapper(nn.Module):
    def __init__(self, dataset_path, model_name, weight_dtype, device, lrm_model=None,
                 camctrl_gs_config=None, camctrl_weight_path=None):
        super().__init__()
        self.module = get_diffusion_model(
            dataset_path, model_name, weight_dtype, device,
            camctrl_gs_config=camctrl_gs_config, camctrl_weight_path=camctrl_weight_path,
        )
        self.lrm_model = lrm_model
        self.module.to(dtype=weight_dtype, device=device)

    def forward(self, x):
        return self.module(x)
