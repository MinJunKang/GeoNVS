import torch
from typing import Optional, Union
from inspect import isfunction
from .discretizer import DDPMDiscretization


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        # For SVD, P_mean=0.7 P_std=1.6
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, num_idx, do_append_zero=False, flip=True):
        self.num_idx = num_idx
        self.sigmas = DDPMDiscretization()(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)


class ZeroSampler:
    def __call__(
        self, n_samples: int, rand: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.zeros_like(default(rand, torch.randn((n_samples,)))) + 1.0e-5
