from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
from einops import rearrange, repeat, reduce
from jaxtyping import Float
from torch import Tensor

from .cuda_splatting import render_cuda, render_count, render_uncertainty_cuda
from .decoder import Decoder, DecoderOutput, Gaussians

class DecoderSplattingCUDA(Decoder):

    def __init__(
        self,
        background_color: Float[Tensor, "3"],
        eps: float = 1e-6
    ) -> None:
        super().__init__(background_color)
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.eps = eps

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        scale_invariant: bool = True,
        render_color: bool = False,
        render_depth: bool = False,
        render_uncertainty: bool = False,
        render_feature: bool = False,
        render_gsindex: bool = False,
        max_contrib_per_pixel: int = 256,
        gaussian_features: Optional[Float[Tensor, "batch gaussian dim"]] = None,
        e3nn: bool = False,
        debug: bool = False,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        background_color = self.background_color.to(dtype=extrinsics.dtype)
        if gaussian_features is not None:
            gaussian_features = repeat(gaussian_features, "b g c -> (b v) g c", v=v)
            
        if render_gsindex:
            gsindex, gsweight, color, depth = render_count(
                extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
                near=rearrange(near, "b v -> (b v)"),
                far=rearrange(far, "b v -> (b v)"),
                image_shape=image_shape,
                background_color=repeat(background_color, "c -> (b v) c", b=b, v=v),
                gaussian_means=repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                gaussian_covariances=repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                gaussian_sh_coefficients=repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                gaussian_opacities=repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                scale_invariant=scale_invariant,
                render_color=render_color,
                render_depth=render_depth,
                max_contrib_per_pixel=max_contrib_per_pixel,
                e3nn=e3nn,
                debug=debug,
                eps=self.eps,
            )
            uncertainty_map, feature = None, None
            gsindex = rearrange(gsindex, "(b v) k h w -> b v k h w", b=b, v=v)
            gsweight = rearrange(gsweight, "(b v) k h w -> b v k h w", b=b, v=v)
            color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v).clamp(0.0, 1.0) if render_color else None
        else:
            if render_uncertainty:
                assert hasattr(gaussians, "fishers"), "Missing fishers in Gaussians"
                
                svdval_fishers = torch.linalg.svdvals(gaussians.fishers)
                fishers_score = torch.log(svdval_fishers + self.eps).sum(dim=-1)
                mask_invalid = torch.isnan(svdval_fishers).any(dim=-1)
                
                color, depth, feature = render_uncertainty_cuda(
                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                    intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
                    near=rearrange(near, "b v -> (b v)"),
                    far=rearrange(far, "b v -> (b v)"),
                    image_shape=image_shape,
                    background_color=repeat(background_color, "c -> (b v) c", b=b, v=v),
                    gaussian_means=repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                    gaussian_covariances=repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                    gaussian_opacities=repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                    gaussian_fishers=repeat(fishers_score, "b g -> (b v) g", v=v),
                    mask_invalid=repeat(mask_invalid, "b g -> (b v) g", v=v),
                    gaussian_features=gaussian_features,
                    scale_invariant=scale_invariant,
                    render_feature=render_feature,
                    render_depth=render_depth,
                    e3nn=e3nn,
                    debug=debug,
                    eps=self.eps,
                )
                gsindex, gsweight = None, None
                uncertainty_map = reduce(color, "(b v) c h w -> b v h w", "mean", v=v)
                color = (uncertainty_map - uncertainty_map.amin(dim=(0, 1), keepdim=True)) / (
                    uncertainty_map.amax(dim=(0, 1), keepdim=True) - uncertainty_map.amin(dim=(0, 1), keepdim=True)
                )
                color = repeat(color.clamp(0.0, 1.0), "b v h w -> b v c h w", c=3)
            else:
                color, depth, feature = render_cuda(
                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                    intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
                    near=rearrange(near, "b v -> (b v)"),
                    far=rearrange(far, "b v -> (b v)"),
                    image_shape=image_shape,
                    background_color=repeat(background_color, "c -> (b v) c", b=b, v=v),
                    gaussian_means=repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                    gaussian_covariances=repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                    gaussian_sh_coefficients=repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                    gaussian_opacities=repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                    gaussian_features=gaussian_features,
                    scale_invariant=scale_invariant,
                    render_color=render_color,
                    render_depth=render_depth,
                    render_feature=render_feature,
                    e3nn=e3nn,
                    debug=debug,
                    eps=self.eps,
                )
                uncertainty_map = None
                gsindex, gsweight = None, None
                color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v).clamp(0.0, 1.0) if render_color else None
            
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v) if render_feature else None
        depth = rearrange(depth, "(b v) 1 h w -> b v h w", b=b, v=v) if render_depth else None
        
        return DecoderOutput(
            color=color,
            depth=depth,
            uncertainty=uncertainty_map,
            feature=feature,
            gsindex=gsindex,
            gsweight=gsweight
        )
