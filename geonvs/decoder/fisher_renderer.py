#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch
from math import isqrt
from rasterization_and_pup_fisher import GaussianRasterizationSettings, GaussianRasterizer

from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from .projection import get_fov


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


@torch.enable_grad()
def fisher_render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    fishers: Float[Tensor, "gaussian 6 6"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    resolution: int = 1,
    scale_invariant: bool = True,
    e3nn: bool = False,
    use_sh: bool = True,
    debug: bool = False,
    eps: float = 1e-6
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    
    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_scales = gaussian_scales * scale[:, None, None]
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape
    h, w = math.ceil(h / resolution), math.ceil(w / resolution)

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    fov_x = fov_x.clamp(min=eps, max=math.pi - eps)
    fov_y = fov_y.clamp(min=eps, max=math.pi - eps)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix
    
    for i in range(b):
        
        sym_fishers = torch.zeros(
            (fishers.shape[0], 21), dtype=fishers.dtype, device=fishers.device)
        sym_fishers.requires_grad = True
        
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True) + 0
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass
        
        raster_settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i].float(),
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,
            e3nn=e3nn,
            debug=debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = gaussian_means[i],
            means2D = mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp = None if use_sh else shs[i, :, 0, :],
            opacities = gaussian_opacities[i, ..., None],
            scales=gaussian_scales[i],
            rotations=gaussian_rotations[i],
            fishers = sym_fishers,
        )
        
        # Backward computes and stores symetric fisher values
        # in sym_fisher's grad
        image_sum = rendered_image.sum()
        image_sum.backward()
        
        fishers[:, 0, 0] += sym_fishers.grad[:, 0]
        fishers[:, 0, 1] += sym_fishers.grad[:, 1]
        fishers[:, 0, 2] += sym_fishers.grad[:, 2]
        fishers[:, 0, 3] += sym_fishers.grad[:, 3]
        fishers[:, 0, 4] += sym_fishers.grad[:, 4]
        fishers[:, 0, 5] += sym_fishers.grad[:, 5]

        fishers[:, 1, 0] += sym_fishers.grad[:, 1]
        fishers[:, 1, 1] += sym_fishers.grad[:, 6]
        fishers[:, 1, 2] += sym_fishers.grad[:, 7]
        fishers[:, 1, 3] += sym_fishers.grad[:, 8]
        fishers[:, 1, 4] += sym_fishers.grad[:, 9]
        fishers[:, 1, 5] += sym_fishers.grad[:, 10]

        fishers[:, 2, 0] += sym_fishers.grad[:, 2]
        fishers[:, 2, 1] += sym_fishers.grad[:, 7]
        fishers[:, 2, 2] += sym_fishers.grad[:, 11]
        fishers[:, 2, 3] += sym_fishers.grad[:, 12]
        fishers[:, 2, 4] += sym_fishers.grad[:, 13]
        fishers[:, 2, 5] += sym_fishers.grad[:, 14]

        fishers[:, 3, 0] += sym_fishers.grad[:, 3]
        fishers[:, 3, 1] += sym_fishers.grad[:, 8]
        fishers[:, 3, 2] += sym_fishers.grad[:, 12]
        fishers[:, 3, 3] += sym_fishers.grad[:, 15]
        fishers[:, 3, 4] += sym_fishers.grad[:, 16]
        fishers[:, 3, 5] += sym_fishers.grad[:, 17]

        fishers[:, 4, 0] += sym_fishers.grad[:, 4]
        fishers[:, 4, 1] += sym_fishers.grad[:, 9]
        fishers[:, 4, 2] += sym_fishers.grad[:, 13]
        fishers[:, 4, 3] += sym_fishers.grad[:, 16]
        fishers[:, 4, 4] += sym_fishers.grad[:, 18]
        fishers[:, 4, 5] += sym_fishers.grad[:, 19]

        fishers[:, 5, 0] += sym_fishers.grad[:, 5]
        fishers[:, 5, 1] += sym_fishers.grad[:, 10]
        fishers[:, 5, 2] += sym_fishers.grad[:, 14]
        fishers[:, 5, 3] += sym_fishers.grad[:, 17]
        fishers[:, 5, 4] += sym_fishers.grad[:, 19]
        fishers[:, 5, 5] += sym_fishers.grad[:, 20]

        del sym_fishers
        
    return fishers


@torch.enable_grad()
def pool_fisher_cuda(viewpoint_camera, pc, background: torch.Tensor, fishers: torch.Tensor, resolution: int = 4, e3nn: bool = False):
    
    sym_fishers = torch.zeros(
        (fishers.shape[0], 21), dtype=fishers.dtype, device=fishers.device)
    sym_fishers.requires_grad = True
    
    # Set up a tensor for the gradients of the screen-space means.
    mean_gradients = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True) + 0
    try:
        mean_gradients.retain_grad()
    except Exception:
        pass
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=math.ceil(viewpoint_camera.image_height / resolution),
        image_width=math.ceil(viewpoint_camera.image_width / resolution),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        e3nn=e3nn,
        debug=True
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = pc.get_xyz,
        means2D = mean_gradients,
        shs = pc.get_features,
        colors_precomp = None,
        opacities = pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        fishers = sym_fishers,
    )
    
    # Backward computes and stores symetric fisher values
    # in sym_fisher's grad
    image_sum = rendered_image.sum()
    image_sum.backward()
    
    fishers[:, 0, 0] += sym_fishers.grad[:, 0]
    fishers[:, 0, 1] += sym_fishers.grad[:, 1]
    fishers[:, 0, 2] += sym_fishers.grad[:, 2]
    fishers[:, 0, 3] += sym_fishers.grad[:, 3]
    fishers[:, 0, 4] += sym_fishers.grad[:, 4]
    fishers[:, 0, 5] += sym_fishers.grad[:, 5]

    fishers[:, 1, 0] += sym_fishers.grad[:, 1]
    fishers[:, 1, 1] += sym_fishers.grad[:, 6]
    fishers[:, 1, 2] += sym_fishers.grad[:, 7]
    fishers[:, 1, 3] += sym_fishers.grad[:, 8]
    fishers[:, 1, 4] += sym_fishers.grad[:, 9]
    fishers[:, 1, 5] += sym_fishers.grad[:, 10]

    fishers[:, 2, 0] += sym_fishers.grad[:, 2]
    fishers[:, 2, 1] += sym_fishers.grad[:, 7]
    fishers[:, 2, 2] += sym_fishers.grad[:, 11]
    fishers[:, 2, 3] += sym_fishers.grad[:, 12]
    fishers[:, 2, 4] += sym_fishers.grad[:, 13]
    fishers[:, 2, 5] += sym_fishers.grad[:, 14]

    fishers[:, 3, 0] += sym_fishers.grad[:, 3]
    fishers[:, 3, 1] += sym_fishers.grad[:, 8]
    fishers[:, 3, 2] += sym_fishers.grad[:, 12]
    fishers[:, 3, 3] += sym_fishers.grad[:, 15]
    fishers[:, 3, 4] += sym_fishers.grad[:, 16]
    fishers[:, 3, 5] += sym_fishers.grad[:, 17]

    fishers[:, 4, 0] += sym_fishers.grad[:, 4]
    fishers[:, 4, 1] += sym_fishers.grad[:, 9]
    fishers[:, 4, 2] += sym_fishers.grad[:, 13]
    fishers[:, 4, 3] += sym_fishers.grad[:, 16]
    fishers[:, 4, 4] += sym_fishers.grad[:, 18]
    fishers[:, 4, 5] += sym_fishers.grad[:, 19]

    fishers[:, 5, 0] += sym_fishers.grad[:, 5]
    fishers[:, 5, 1] += sym_fishers.grad[:, 10]
    fishers[:, 5, 2] += sym_fishers.grad[:, 14]
    fishers[:, 5, 3] += sym_fishers.grad[:, 17]
    fishers[:, 5, 4] += sym_fishers.grad[:, 19]
    fishers[:, 5, 5] += sym_fishers.grad[:, 20]

    del sym_fishers
    
    return fishers