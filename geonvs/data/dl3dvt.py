
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from jaxtyping import Float, UInt8
from torch.utils.data import Dataset
import os
import json
import random
from PIL import Image
from torch import Tensor
import torchvision.transforms as tf
import torch
import numpy as np
from einops import rearrange, repeat

from .shims.crop_shim import apply_crop_shim_to_views
from geonvs.seva.geometry import get_plucker_coordinates
from .view_sampler import get_view_sampler
from .geometry.projection import get_fov
from .view_sampler.view_sampler_bounded_v2 import ViewSamplerBoundedV2Cfg


def center_cameras(all_c2ws, c2ws):
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
    

def scale_cameras(c2ws, camera_scale=2.0):
    camera_dists = c2ws[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    c2ws[:, :3, 3] *= translation_scaling_factor



class DL3DVODataset(Dataset):
    
    def __init__(self, root_dir, stage, num_images, target_shape, step_tracker, plucker_convention='seva', square_crop=True):
        self.dataset_dir = Path(root_dir) / stage
        self.to_tensor = tf.ToTensor()
        self.stage = stage
        self.square_crop = square_crop
        self.min_views = 2
        self.max_views = 6
        self.near = 0.5
        self.far = 100.0
        self.num_context_views = 4
        self.num_total_views = num_images
        self.sort_context_index = True
        self.sort_target_index = True
        self.max_fov = 100.0
        self.chunks = self._load_scenes()
        self.chunk_keys = list(self.chunks.keys())
        if '_low' in Path(root_dir).stem:
            self.ori_image_shape = (270, 480)
            self.highres = False
        else:
            self.ori_image_shape = (540, 960)
            self.highres = True
            
        self.view_sampler_cfg = ViewSamplerBoundedV2Cfg(
            name='boundedv2',
            num_context_views=self.num_context_views,
            num_target_views=self.num_total_views - self.num_context_views,
            min_distance_between_context_views=20,
            max_distance_between_context_views=50,
            max_distance_to_context_views=0,
            context_gap_warm_up_steps=10000,
            target_gap_warm_up_steps=0,
            initial_min_distance_between_context_views=15,
            initial_max_distance_between_context_views=30,
            initial_max_distance_to_context_views=0,
            extra_views_sampling_strategy='farthest_point',
            target_views_replace_sample=False,
        )
        self.view_sampler = get_view_sampler(
            cfg=self.view_sampler_cfg, 
            stage=stage, 
            overfit=False, 
            cameras_are_circular=False, 
            step_tracker=step_tracker
        )
        self.target_shape = target_shape  # seva value
        self.plucker_convention = plucker_convention
        self.pixel_transform = tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        
        # Values fro SD 2.1 autoencoder
        self.donwsample_factor = 8
        self.scale_factor = 0.18215
        
    # collect chunks here
    def _load_scenes(self):
        
        # read dataset chunks
        with open(self.dataset_dir / "index.json", "r") as f:
            json_dict = json.load(f)
        
        return json_dict
    
    def __len__(self):
        # this is rough value
        return len(self.chunk_keys)
    
    def __getitem__(self, idx):
        
        keyname = self.chunk_keys[idx]
        chunk = torch.load(self.dataset_dir / self.chunks[keyname], weights_only=True)
        chunk_keys = [split_chunk['key'].replace("dl3dv_", "") for split_chunk in chunk]
        chunk_idx = chunk_keys.index(keyname)
        example = chunk[chunk_idx]
        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        
        extra_kwargs = {"max_num_views": 148}
        mask_valid1 = ~((extrinsics[:, :3, 3] > 1e3).any(dim=-1))
        mask_valid2 = ~torch.isnan(torch.det(extrinsics[:, :3, :3]))
        mask_valid3 = (get_fov(intrinsics).rad2deg() < self.max_fov).any(dim=1)
        mask_valid = mask_valid1 & mask_valid2 & mask_valid3
        indices = list(torch.nonzero(mask_valid).squeeze())
        extrinsics = extrinsics[mask_valid]
        intrinsics = intrinsics[mask_valid]
        
        try:
            out_data = self.view_sampler.sample(
                keyname,
                extrinsics,
                intrinsics,
                min_context_views=self.min_views,
                max_context_views=self.max_views,
                **extra_kwargs,
            )
        except:
            return {}
        
        if isinstance(out_data, tuple):
            context_indices, target_indices = out_data[:2]
            c_list = [
                (
                    context_indices.sort()[0]
                    if self.sort_context_index
                    else context_indices
                )
            ]
            t_list = [
                (
                    target_indices.sort()[0]
                    if self.sort_target_index
                    else target_indices
                )
            ]
        if isinstance(out_data, list):
            c_list = [
                (
                    a.context.sort()[0]
                    if self.sort_context_index
                    else a.context
                )
                for a in out_data
            ]
            t_list = [
                (
                    a.target.sort()[0]
                    if self.sort_target_index
                    else a.target
                )
                for a in out_data
            ]
        all_indices = torch.cat([c_list[0], t_list[0]])
        all_indices = all_indices[:self.num_total_views]
        
        if len(all_indices) < self.num_total_views:
            num_sample_views = self.num_total_views - len(all_indices)
            non_overlapped = list(set(np.arange(len(extrinsics))) - set(all_indices.tolist()))
            if len(non_overlapped) < num_sample_views:
                return {}
            indices_random = torch.tensor(
                np.random.choice(non_overlapped, size=num_sample_views, replace=False),
                dtype=torch.long
            )
            all_indices = torch.cat([all_indices, indices_random])
        
        # Load the images.
        valid_images = [img for idx, img in enumerate(example["images"]) if idx in indices]
        images = [
            valid_images[index.item()] for index in all_indices
        ]
        images = self.convert_images(images)
        extrinsics_all = extrinsics[all_indices]
        intrinsics_all = intrinsics[all_indices]
        
        # construct the indices mask
        indices_mask = torch.zeros_like(all_indices, dtype=torch.bool)
        indices_mask[:len(context_indices)] = True
        
        example_out = {
            "image": images,
            "extrinsics": extrinsics_all,  # camera to world
            "intrinsics": intrinsics_all,  # not scaled intrinsics
            "near": self.get_bound(self.near, len(all_indices)),
            "far": self.get_bound(self.far, len(all_indices)),
            "indices_mask": indices_mask,
        }
        example_out = apply_crop_shim_to_views(example_out, tuple(self.target_shape), square_crop=self.square_crop)  # output scaled intrinsics
        
        # seva preprocessing (crop will not touch the camera extrinsics)
        seva_extrinsics_all = example_out['extrinsics'].clone()
        center_cameras(extrinsics, seva_extrinsics_all)
        scale_cameras(seva_extrinsics_all)
        
        # construct the plucker coordinates
        w2cs = torch.linalg.inv(seva_extrinsics_all)
        pluckers = get_plucker_coordinates(
            extrinsics_src=w2cs[0],  # become reference camera
            extrinsics=w2cs,
            intrinsics=example_out['intrinsics'].clone(),
            target_size=(self.target_shape[0], self.target_shape[1]),
            donwsample_factor=self.donwsample_factor,
            convention=self.plucker_convention,
        )
        
        # binary mask and plcukers
        concat = torch.cat( 
            [
                repeat(
                    indices_mask,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                pluckers,
            ],
            dim=1,
        )
        
        # clean latents and binary mask (we didn't preprocess the latents)
        example_out["concat"] = concat
        example_out["plucker"] = pluckers
        
        return example_out
                    
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics
    
    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        torch_images = torch.stack(torch_images)
                
        if self.plucker_convention == 'camctrl':
            torch_images = self.pixel_transform(torch_images)
                
        return torch_images

    def get_bound(
        self,
        tensor,
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(tensor, dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)