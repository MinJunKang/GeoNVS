from typing import Dict, Union

import torch
import torch.nn as nn
from geonvs.seva.sampling import append_dims
from .denoiser_scaling import EDMScaling, EpsScaling, VScaling, VScalingWithEDMcNoise
from .discretizer import DDPMDiscretization



class Denoiser(nn.Module):
    def __init__(self, scaling_method: str, kwargs: Dict):
        super().__init__()
        if scaling_method == 'edm':
            self.scaling = EDMScaling(kwargs['sigma_data'])
        elif scaling_method == 'eps':
            self.scaling = EpsScaling()
        elif scaling_method == 'v':
            self.scaling = VScaling()
        elif scaling_method == 'vedm':
            self.scaling = VScalingWithEDMcNoise()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        input: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return c_skip, c_out, c_in, c_noise


class DiscreteDenoiserM(Denoiser):
    def __init__(
        self,
        scaling_method: str,
        num_idx: int,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
        kwargs: Dict = {},
    ):
        super().__init__(scaling_method, kwargs)
        self.discretization = DDPMDiscretization()
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
