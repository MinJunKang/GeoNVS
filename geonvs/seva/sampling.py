import threading
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from diffusers.utils.torch_utils import randn_tensor
from geonvs.seva.geometry import get_camera_dist


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (x - denoised) / append_dims(sigma, x.ndim)


def make_betas(
    num_timesteps: int, linear_start: float = 1e-4, linear_end: float = 2e-2
) -> np.ndarray:
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64
        )
        ** 2
    )
    return betas.numpy()


def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


#######################################################
# Discretization
#######################################################


class Discretization(object):
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps

    def __call__(
        self,
        n: int,
        do_append_zero: bool = True,
        flip: bool = False,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))


class DDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start: float = 5e-06,
        linear_end: float = 0.012,
        log_snr_shift: float | None = 2.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        betas = make_betas(
            self.num_timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        self.log_snr_shift = log_snr_shift

        alphas = 1.0 - betas  # first alpha here is on data side
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError(f"Expected n <= {self.num_timesteps}, but got n = {n}.")

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        if self.log_snr_shift is not None:
            sigmas = sigmas * np.exp(self.log_snr_shift)
        return torch.flip(
            torch.tensor(sigmas, dtype=torch.float32, device=device), (0,)
        )


#######################################################
# Denoiser
#######################################################


class DiscreteDenoiser(object):
    sigmas: torch.Tensor

    def __init__(
        self,
        discretization: Discretization | None = None,
        num_idx: int = 1000,
        device: str | torch.device = "cpu",
    ):
        self.discretization = discretization or DDPMDiscretization()
        self.num_idx = num_idx
        self.device = device
        self.register_sigmas()

    def scaling(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise

    def register_sigmas(self):
        self.sigmas = self.discretization(
            self.num_idx, do_append_zero=False, flip=True, device=self.device
        )

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: torch.Tensor | int) -> torch.Tensor:
        return self.sigmas[idx]

    def __call__(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: dict,
        log_int_features: bool,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.idx_to_sigma(self.sigma_to_idx(sigma))
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.sigma_to_idx(c_noise.reshape(sigma_shape))

        if "replace" in cond:
            x, mask = cond.pop("replace").split((input.shape[1], 1), dim=1)
            input = input * (1 - mask) + x * mask

        c_skip, c_out, c_in = c_skip.to(dtype=input.dtype), c_out.to(dtype=input.dtype), c_in.to(dtype=input.dtype)
        x = torch.cat([input * c_in, cond['concat']], dim=1)
        net_out, net_feats = network(
            sample=x,
            input_mask=cond['mask'],
            dense_sample=cond["dense_vector"],
            timestep=c_noise,
            encoder_hidden_states=cond["crossattn"],
            extrinsics=cond.get("extrinsics", None),
            intrinsics=cond.get("intrinsics", None),
            near=cond.get("near", None),
            far=cond.get("far", None),
            gaussians=cond.get("gaussians", None),
            log_int_features=log_int_features,
            training_mode=False,
            **additional_model_inputs
        )

        # MIX INPUT : INPUT <- MODEL(INPUT, COND) * C_OUT + INPUT * C_SKIP
        net_out = net_out * c_out + input * c_skip

        return net_out, net_feats


#######################################################
# Scale rules and schedules
#######################################################


class MultiviewScaleRule(object):
    def __init__(self, min_scale: float = 1.0):
        self.min_scale = min_scale

    def __call__(
        self,
        scale: float | torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        c2w_input = c2w[input_frame_mask]
        rotation_diff = get_camera_dist(c2w, c2w_input, mode="rotation").min(-1).values
        translation_diff = (
            get_camera_dist(c2w, c2w_input, mode="translation").min(-1).values
        )
        K_diff = (
            ((K[:, None] - K[input_frame_mask][None]).flatten(-2) == 0).all(-1).any(-1)
        )
        close_frame = (rotation_diff < 10.0) & (translation_diff < 1e-5) & K_diff
        if isinstance(scale, torch.Tensor):
            scale = scale.clone()
            scale[close_frame] = self.min_scale
        elif isinstance(scale, float):
            scale = torch.where(close_frame, self.min_scale, scale)
        else:
            raise ValueError(f"Invalid scale type {type(scale)}.")
        return scale


class VanillaCFG(object):
    def __init__(self):
        self.scale_rule = lambda scale: scale

    def _expand_scale(
        self, sigma: float | torch.Tensor, scale: float | torch.Tensor
    ) -> float | torch.Tensor:
        if isinstance(sigma, float):
            return scale
        elif isinstance(sigma, torch.Tensor):
            if len(sigma.shape) == 1 and isinstance(scale, torch.Tensor):
                sigma = append_dims(sigma, scale.ndim)
            return scale * torch.ones_like(sigma)
        else:
            raise ValueError(f"Invalid sigma type {type(sigma)}.")

    def guidance(
        self,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        scale: float | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(scale, torch.Tensor) and len(scale.shape) == 1:
            scale = append_dims(scale, cond.ndim)
        return uncond + scale * (cond - uncond)

    def __call__(
        self, x: torch.Tensor, sigma: float | torch.Tensor, scale: float | torch.Tensor
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        scale = self.scale_rule(scale)
        x_pred = self.guidance(x_u, x_c, self._expand_scale(sigma, scale))
        return x_pred

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict, log_int_features: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # c / uc are constant across the sampling loop, so their (uncond, cond)
        # concatenation is computed once per run and reused for every step.
        cache = getattr(self, "_cond_cache", None)
        if cache is None or cache[0] is not c or cache[1] is not uc:
            c_out = dict()
            for k in c:
                if c[k] is not None:
                    c_out[k] = torch.cat((uc[k], c[k]), 0)
            self._cond_cache = (c, uc, c_out)
        # return a shallow copy: the denoiser pops "replace" from this dict
        c_out = dict(self._cond_cache[2])
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out, log_int_features


class MultiviewCFG(VanillaCFG):
    def __init__(self, cfg_min: float = 1.0):
        self.scale_min = cfg_min
        self.scale_rule = MultiviewScaleRule(min_scale=cfg_min)

    def __call__(  # type: ignore
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        scale: float | torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        scale = self.scale_rule(scale, c2w, K, input_frame_mask)
        x_pred = self.guidance(x_u, x_c, self._expand_scale(sigma, scale))
        return x_pred


class MultiviewTemporalCFG(MultiviewCFG):
    def __init__(self, num_frames: int, cfg_min: float = 1.0):
        super().__init__(cfg_min=cfg_min)
        self.num_frames = num_frames
        distance_matrix = (
            torch.arange(num_frames)[None] - torch.arange(num_frames)[:, None]
        ).abs()
        self.distance_matrix = distance_matrix

    def __call__(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        scale: float | torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_frame_mask = rearrange(
            input_frame_mask, "(b t) ... -> b t ...", t=self.num_frames
        )
        min_distance = (
            self.distance_matrix[None].to(x.device)
            + (~input_frame_mask[:, None]) * self.num_frames
        ).min(-1)[0]
        min_distance = min_distance / min_distance.max(-1, keepdim=True)[0].clamp(min=1)
        scale = min_distance * (scale - self.scale_min) + self.scale_min
        scale = rearrange(scale, "b t ... -> (b t) ...")
        scale = append_dims(scale, x.ndim)
        return super().__call__(x, sigma, scale, c2w, K, input_frame_mask.flatten(0, 1))


#######################################################
# Samplers
#######################################################


class TrackedSampler(object):
    """Mixin that adds progress tracking (any object exposing `.update()`) and
    abort support (via a `threading.Event`) to a sampler, e.g. for demos."""

    def __init__(self, *args, abort_event: threading.Event | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.abort_event = abort_event

    def possibly_update_pbar(self, global_pbar=None):
        if global_pbar is not None:
            global_pbar.update()
        if self.abort_event is not None and self.abort_event.is_set():
            return False
        return True


class EulerEDMSampler(TrackedSampler):
    def __init__(
        self,
        discretization: Discretization,
        guider: VanillaCFG | MultiviewCFG | MultiviewTemporalCFG,
        num_steps: int | None = None,
        verbose: bool = False,
        device: str | torch.device = "cuda",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.discretization = discretization
        self.guider = guider
        self.verbose = verbose
        self.device = device

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def prepare_sampling_loop(
        self, x: torch.Tensor, cond: dict, uc: dict, num_steps: int | None = None,
        start_strength: float = 1.0, initial_latents: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict, dict]:
        num_steps = num_steps or self.num_steps
        assert num_steps is not None, "num_steps must be specified"
        sigmas = self.discretization(num_steps, device=self.device)

        # set start index (start_strength < 1.0 resumes denoising mid-schedule,
        # e.g. for SDEdit-style inpainting)
        start_step_idx = int(num_steps * (1 - start_strength))
        start_step_idx = min(max(start_step_idx, 0), num_steps - 1)

        if initial_latents is not None:
            # SDEdit based inpainting
            mask = cond['mask']
            assert initial_latents.shape[0] == (~mask).count_nonzero(), "only support novel-view inpainting"
            x *= sigmas[start_step_idx]
            x[~mask] += initial_latents.to(x.dtype)
        else:
            x *= torch.sqrt(1.0 + sigmas[start_step_idx] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, start_step_idx, num_sigmas, cond, uc

    def get_sigma_gen(self, start_idx: int, num_sigmas: int, verbose: bool = True) -> range | tqdm:
        sigma_generator = range(start_idx, num_sigmas - 1)
        if self.verbose and verbose:
            sigma_generator = tqdm(
                sigma_generator,
                total=len(sigma_generator),
                desc="Sampling",
                leave=False,
            )
        return sigma_generator

    def sampler_step(
        self,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict,
        gamma: float = 0.0,
        log_int_features: bool = False,
        generator: torch.Generator | None = None,
        **guider_kwargs,
    ) -> torch.Tensor:
        sigma_hat = sigma * (gamma + 1.0) + 1e-6

        eps = randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype) * self.s_noise
        x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised, feats = denoiser(*self.guider.prepare_inputs(x, sigma_hat, cond, uc, log_int_features))
        denoised = self.guider(denoised, sigma_hat, scale, **guider_kwargs).to(dtype=x.dtype)

        feats_dict = {}
        for i, feat in enumerate(feats):
            if i == len(feats) - 1:  # the last two features are reserved for color and uncertainty logging
                feats_dict[f"uncertainty"] = feat
            elif i == len(feats) - 2:
                feats_dict[f"color"] = feat
            else:
                feats_dict[f"feat_{i}"] = self.guider(feat, sigma_hat, scale, **guider_kwargs)

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        return x + dt * d, feats_dict

    def __call__(
        self,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict | None = None,
        num_steps: int | None = None,
        step_int_features: list[int] = [],
        verbose: bool = True,
        global_pbar=None,
        generator: torch.Generator | None = None,
        initial_latents: torch.Tensor | None = None,
        start_strength: float = 1.0,
        **guider_kwargs,
    ) -> torch.Tensor:
        uc = cond if uc is None else uc
        x, s_in, sigmas, start_idx, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x,
            cond,
            uc,
            num_steps,
            start_strength=start_strength,
            initial_latents=initial_latents
        )

        int_features = {}
        for i in self.get_sigma_gen(start_idx, num_sigmas, verbose=verbose):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )

            log_int_features = i in step_int_features
            x, int_feature = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                scale,
                cond,
                uc,
                gamma,
                log_int_features=log_int_features,
                generator=generator,
                **guider_kwargs,
            )

            for key, value in int_feature.items():
                if key == 'color':
                    int_features[f"color"] = value
                elif key == 'uncertainty':
                    int_features[f"uncertainty"] = value
                else:
                    int_features[f"{key}_{i}/feature"] = value

            if not self.possibly_update_pbar(global_pbar):
                return None

        return x, int_features
