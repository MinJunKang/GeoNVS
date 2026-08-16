import math
from typing import Literal, Optional

import torch
from diff_gaussian_rasterization_feat import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


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


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_features: Optional[Float[Tensor, "batch gaussian dim"]] = None,
    scale_invariant: bool = True,
    render_color: bool = False,
    render_depth: bool = False,
    render_feature: bool = False,
    use_sh: bool = True,
    e3nn: bool = False,
    debug: bool = False,
    eps: float = 1e-6
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        near = near * scale
        far = far * scale
    else:
        scale = None

    # One set of gaussians is shared by all the target views of a scene. The
    # caller may hand them over unexpanded (batch g_b, with b = g_b * views);
    # index them separately rather than making it materialise b copies, which
    # for a dense geometry model runs into gigabytes per rendered view.
    views_per_scene = extrinsics.shape[0] // gaussian_means.shape[0]

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = math.isqrt(n) - 1
    
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics, eps=eps).unbind(dim=-1)
    fov_x = fov_x.clamp(min=eps, max=math.pi - eps)
    fov_y = fov_y.clamp(min=eps, max=math.pi - eps)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix
    row, col = torch.triu_indices(3, 3)

    all_images = []
    all_depths = []
    all_features = []
    for i in range(b):
        
        gi = i // views_per_scene
        means_i = gaussian_means[gi]
        covariances_i = gaussian_covariances[gi]
        if scale is not None:
            means_i = means_i * scale[i]
            covariances_i = covariances_i * (scale[i] ** 2)

        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(means_i, requires_grad=True, device=means_i.device) + 0.
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass
        
        try:
            settings = GaussianRasterizationSettings(
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
                prefiltered=False,  # This matches the original usage.
                include_feature=render_feature,
                f_count=False,
                e3nn=e3nn,
                debug=debug,
            )
        except:
            print(f"tan_fov_x has NaN: {torch.isnan(tan_fov_x).any()}")
            print(f"tan_fov_y has NaN: {torch.isnan(tan_fov_y).any()}")
            print(f"view_matrix has NaN: {torch.isnan(view_matrix).any()}")
            print(f"full_projection has NaN: {torch.isnan(full_projection).any()}")
            print(f"extrinsics has NaN: {torch.isnan(extrinsics).any()}")
        rasterizer = GaussianRasterizer(settings)

        image, depth, feature, radii = rasterizer(
            means3D=means_i,
            means2D=mean_gradients,
            shs=shs[gi] if use_sh else None,
            colors_precomp=None if use_sh else shs[gi, :, 0, :],
            opacities=gaussian_opacities[gi, ..., None],
            cov3D_precomp=covariances_i[:, row, col],
            feature_precomp=gaussian_features[gi] if (gaussian_features is not None) and render_feature else None,
        )
        
        if render_color:
            all_images.append(image)
        if render_depth:
            all_depths.append(depth)
        if render_feature:
            all_features.append(feature)
            
    if render_color:
        all_images = torch.stack(all_images)
    if render_depth:
        all_depths = torch.stack(all_depths)
    if render_feature:
        all_features = torch.stack(all_features)
            
    return all_images, all_depths, all_features



def render_count(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    render_color: bool = False,
    render_depth: bool = False,
    scale_invariant: bool = True,
    max_contrib_per_pixel: int = 256,
    use_sh: bool = True,
    e3nn: bool = False,
    debug: bool = False,
    eps: float = 1e-6
):
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    
    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        near = near * scale
        far = far * scale
    else:
        scale = None

    # One set of gaussians is shared by all the target views of a scene. The
    # caller may hand them over unexpanded (batch g_b, with b = g_b * views);
    # index them separately rather than making it materialise b copies, which
    # for a dense geometry model runs into gigabytes per rendered view.
    views_per_scene = extrinsics.shape[0] // gaussian_means.shape[0]
        
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # infer SH degree from the number of coefficients (supports LRM heads with
    # degree != 3)
    _, _, _, n = gaussian_sh_coefficients.shape
    degree = math.isqrt(n) - 1

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics, eps=eps).unbind(dim=-1)
    fov_x = fov_x.clamp(min=eps, max=math.pi - eps)
    fov_y = fov_y.clamp(min=eps, max=math.pi - eps)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix
    row, col = torch.triu_indices(3, 3)

    all_images = []
    all_depths = []
    gs_indices, gs_weights = [], []
    for i in range(b):
        
        gi = i // views_per_scene
        means_i = gaussian_means[gi]
        covariances_i = gaussian_covariances[gi]
        if scale is not None:
            means_i = means_i * scale[i]
            covariances_i = covariances_i * (scale[i] ** 2)

        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(means_i, requires_grad=True, device=means_i.device) + 0.
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass
        
        settings = GaussianRasterizationSettings(
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
            prefiltered=False,  # This matches the original usage.
            include_feature=False,
            f_count=True,
            e3nn=e3nn,
            debug=debug,
        )
        rasterizer = GaussianRasterizer(settings)

        gsindex, gsweight, image, depth, radii = rasterizer(
            means3D=means_i,
            means2D=mean_gradients,
            shs=shs[gi] if use_sh else None,
            colors_precomp=None if use_sh else shs[gi, :, 0, :],
            opacities=gaussian_opacities[gi, ..., None],
            cov3D_precomp=covariances_i[:, row, col],
        )
        
        if render_color:
            all_images.append(image)
        if render_depth:
            all_depths.append(depth)
            
        gsweight, indices = torch.topk(gsweight, k=max_contrib_per_pixel, dim=-1)
        gsindex = torch.gather(gsindex, dim=-1, index=indices)
        gs_indices.append(gsindex)
        gs_weights.append(gsweight)
        
    gs_indices = torch.stack(gs_indices)
    gs_weights = torch.stack(gs_weights)
    if render_color:
        all_images = torch.stack(all_images)
    if render_depth:
        all_depths = torch.stack(all_depths)
        
    return gs_indices, gs_weights, all_images, all_depths



def render_uncertainty_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_fishers: Float[Tensor, "batch gaussian"],
    mask_invalid: Float[Tensor, "batch gaussian"],
    gaussian_features: Optional[Float[Tensor, "batch gaussian dim"]] = None,
    scale_invariant: bool = True,
    render_depth: bool = False,
    render_feature: bool = False,
    e3nn: bool = False,
    debug: bool = False,
    eps: float = 1e-6
) -> Float[Tensor, "batch 3 height width"]:

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        near = near * scale
        far = far * scale
    else:
        scale = None

    # One set of gaussians is shared by all the target views of a scene. The
    # caller may hand them over unexpanded (batch g_b, with b = g_b * views);
    # index them separately rather than making it materialise b copies, which
    # for a dense geometry model runs into gigabytes per rendered view.
    views_per_scene = extrinsics.shape[0] // gaussian_means.shape[0]

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics, eps=eps).unbind(dim=-1)
    fov_x = fov_x.clamp(min=eps, max=math.pi - eps)
    fov_y = fov_y.clamp(min=eps, max=math.pi - eps)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix
    color_precomp = repeat(gaussian_fishers, "b g -> b g c", c=3)
    row, col = torch.triu_indices(3, 3)

    all_images = []
    all_depths = []
    all_features = []
    for i in range(b):
        gi = i // views_per_scene
        mask = ~mask_invalid[gi]
        means_i = gaussian_means[gi]
        covariances_i = gaussian_covariances[gi]
        if scale is not None:
            means_i = means_i * scale[i]
            covariances_i = covariances_i * (scale[i] ** 2)
        
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(means_i[mask], requires_grad=True, device=means_i.device) + 0.
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass
        
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i].float(),
            sh_degree=3,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            include_feature=render_feature,
            f_count=False,
            e3nn=e3nn,
            debug=debug,
        )
        rasterizer = GaussianRasterizer(settings)

        image, depth, feature, radii = rasterizer(
            means3D=means_i[mask],
            means2D=mean_gradients,
            colors_precomp=color_precomp[gi][mask],
            opacities=gaussian_opacities[gi, ..., None][mask],
            cov3D_precomp=covariances_i[:, row, col][mask],
            feature_precomp=gaussian_features[gi][mask] if (gaussian_features is not None) and render_feature else None,
        )
        
        all_images.append(image)
        if render_depth:
            all_depths.append(depth)
        if render_feature:
            all_features.append(feature)
        
    if render_depth:
        all_depths = torch.stack(all_depths)
    if render_feature:
        all_features = torch.stack(all_features)
        
    return torch.stack(all_images), all_depths, all_features