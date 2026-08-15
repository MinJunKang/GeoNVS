
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from jaxtyping import Float, UInt8
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torch import Tensor
import torchvision.transforms as tf
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
from einops import rearrange, repeat

from .shims.crop_shim import apply_crop_shim_to_views


from geonvs.seva.geometry import get_plucker_coordinates


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



class Re10KDataset(Dataset):
    
    def __init__(self, root_dir, stage, num_images, target_shape, plucker_convention='seva', input_num=[1, 3, 6, 9, 12], square_crop=True):
        self.dataset_dir = Path(root_dir) / stage
        self.gs_dir = Path(root_dir) / f"{stage}_gs"
        self.to_tensor = tf.ToTensor()
        self.stage = stage
        self.square_crop = square_crop
        self.num_total_views = num_images
        self.chunks, self.gs_chunks = self._load_scenes(input_num=input_num)
        if '_low' in Path(root_dir).stem:
            self.ori_image_shape = (360, 640)
            self.highres = False
        else:
            self.ori_image_shape = (720, 1280)
            self.highres = True
            
        self.target_shape = target_shape  # seva value
        self.plucker_convention = plucker_convention
        self.pixel_transform = tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        
        # Values fro SD 2.1 autoencoder
        self.donwsample_factor = 8
        self.scale_factor = 0.18215
        
    # collect chunks here
    def _load_scenes(self, input_num):
        
        # read dataset chunks
        with open(self.dataset_dir / "index.json", "r") as f:
            json_dict = json.load(f)
        root_chunks = json_dict
            
        # read gs chunks
        gs_chunks = sorted(
            [path for path in self.gs_dir.iterdir() if path.suffix == ".safetensors"]
        )
        gs_chunks = [path for path in gs_chunks if int(path.stem.split('_')[1]) in input_num]
        
        return root_chunks, gs_chunks
    
    def __len__(self):
        # this is rough value
        return len(self.gs_chunks)
    
    def __getitem__(self, idx):
        keyname = self.gs_chunks[idx].stem.split('_')[0]
        chunk = torch.load(self.dataset_dir / self.chunks[keyname], weights_only=True)
        chunk_keys = [split_chunk['key'] for split_chunk in chunk]
        chunk_idx = chunk_keys.index(keyname)
        example = chunk[chunk_idx]
        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        
        tensors, metadata = load_safetensor_file(self.gs_chunks[idx])
        context_indices = tensors["input_indices"].long()
        target_indices = tensors["target_indices"].long()
        all_indices = torch.cat([context_indices, target_indices])
        all_indices = all_indices[:self.num_total_views]
        near, far = tensors["near"][0].float(), tensors["far"][0].float()
        gaussians = tensors["gaussians"].float()
        
        # Load the images.
        images = [
            example["images"][index.item()] for index in all_indices
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
            "near": self.get_bound(near, len(all_indices)),
            "far": self.get_bound(far, len(all_indices)),
            "indices_mask": indices_mask,
            "gaussians": gaussians,
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
        return repeat(tensor, "-> v", v=num_views)