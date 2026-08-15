import math
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..types import AnyExample, AnyViews
from typing import List, Literal, Optional, Tuple, Union


def get_resizing_factor(
    target_shape: Tuple[int, int],  # H, W
    current_shape: Tuple[int, int],  # H, W
    cover_target: bool = True,
    # If True, the output shape will fully cover the target shape.
    # If No, the target shape will fully cover the output shape.
) -> float:
    r_bound = target_shape[1] / target_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if cover_target:
            if aspect_r >= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r < 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r >= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r < 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    else:
        if cover_target:
            if aspect_r <= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r > 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r <= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r > 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    return factor


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: None | Float[Tensor, "*#batch h w"],
    scale: float = 1.0,
    center: Tuple[float, float] = (0.5, 0.5),
    square_crop: bool = True,
) -> (
    tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
    | tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
        Float[Tensor, "*#batch h_out w_out"],  # updated depths
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    scale_factor = get_resizing_factor(shape, (h_in, w_in))
    resize_size = rh, rw = [int(np.ceil(scale_factor * s)) for s in (h_in, w_in)]
    if scale < 1.0:
        pw = math.ceil((w_out - resize_size[1]) * 0.5)
        ph = math.ceil((h_out - resize_size[0]) * 0.5)
        image = F.pad(image, (pw, pw, ph, ph), "constant", 1.0)
    
    # reshaping first
    images = F.interpolate(
        images, (rh, rw), mode="area", antialias=False
    )
    if depths is not None:
        depths = F.interpolate(
            depths.unsqueeze(1), (rh, rw), mode="bilinear", align_corners=True
        ).squeeze(1)
        
    # cropping
    cy_center = int(center[1] * images.shape[-2])
    cx_center = int(center[0] * images.shape[-1])
    if square_crop:
        side = min(h_out, w_out)
        ct = max(0, cy_center - side // 2)
        cl = max(0, cx_center - side // 2)
        ct = min(ct, images.shape[-2] - side)
        cl = min(cl, images.shape[-1] - side)
        images = TF.crop(images, top=ct, left=cl, height=side, width=side)
    else:
        ct = max(0, cy_center - h_out // 2)
        cl = max(0, cx_center - w_out // 2)
        ct = min(ct, images.shape[-2] - h_out)
        cl = min(cl, images.shape[-1] - w_out)
        images = TF.crop(images, top=ct, left=cl, height=h_out, width=w_out)
    
    intrinsics = intrinsics.clone()
    if torch.all(intrinsics[:, :2, -1] >= 0) and torch.all(intrinsics[:, :2, -1] <= 1):
        intrinsics[:, :2] *= intrinsics.new_tensor([rw, rh])[None, :, None]  # normalized K
    else:
        intrinsics[:, :2] *= intrinsics.new_tensor([rw / w_in, rh / h_in])[None, :, None]  # unnormalized K
    intrinsics[:, :2, 2] -= intrinsics.new_tensor([cl, ct])[None]
    
    if depths is not None:
        if square_crop:
            depths = TF.crop(depths, top=ct, left=cl, height=side, width=side)
        else:
            depths = TF.crop(depths, top=ct, left=cl, height=h_out, width=w_out)
        return images, intrinsics, depths
    
    return images, intrinsics


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int], square_crop: bool) -> AnyViews:
    images, intrinsics = rescale_and_crop(
        views["image"], views["intrinsics"], shape, depths=None, square_crop=square_crop
    )
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }
