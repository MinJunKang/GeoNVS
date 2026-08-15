

import torch
from jaxtyping import Float
from torch import Tensor
from einops import repeat, rearrange
from .fisher_renderer import fisher_render_cuda
from .decoder import Gaussians


# https://github.com/j-alex-hanson/gaussian-splatting-pup
def run_fisher(
    gaussians: Gaussians,
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    near: Float[Tensor, "batch view"],
    far: Float[Tensor, "batch view"],
    image_shape: tuple[int, int],
    resolution: int = 4,
    scale_invariant: bool = True,
    e3nn: bool = False,
    device: str = 'cuda',
    debug: bool = False,
    eps: float = 1e-8,
    return_score: bool = False,
):
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    num_gaussians = gaussians.means.shape[1]
    fishers = torch.zeros(num_gaussians,6,6,device=device).float()
    
    b, v, _, _ = extrinsics.shape
    fisher_render_cuda(
        rearrange(extrinsics, "b v i j -> (b v) i j"),
        rearrange(intrinsics, "b v i j -> (b v) i j"),
        rearrange(near, "b v -> (b v)"),
        rearrange(far, "b v -> (b v)"),
        image_shape,
        repeat(background, "c -> (b v) c", b=b, v=v),
        fishers,
        repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
        repeat(gaussians.scales, "b g xyz -> (b v) g xyz", v=v),
        repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v),
        repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
        repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        resolution=resolution,
        scale_invariant=scale_invariant,
        e3nn=e3nn,
        debug=debug,
        eps=eps,
    )

    if return_score:
        # per-Gaussian sensitivity score (used e.g. for PUP pruning in datagen)
        fishers_score = torch.linalg.slogdet(fishers + eps * torch.eye(6, device=device))[1]
        return fishers_score.abs(), fishers

    return fishers