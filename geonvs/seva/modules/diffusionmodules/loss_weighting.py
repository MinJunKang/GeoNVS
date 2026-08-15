from abc import ABC, abstractmethod
import torch
from geonvs.seva.sampling import append_dims
from .util import pose_cdistance


class DiffusionLossWeighting(ABC):
    def __init__(self, name):
        self.name = name
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        pass


class UnitWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting(DiffusionLossWeighting):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma**-2.0
    
class SevaWeighting(DiffusionLossWeighting):
    def __call__(
        self, 
        sigma: torch.Tensor, 
        mask: torch.Tensor, 
        c2w: torch.Tensor,
        min_weight=0.0,
        max_weight=5.0,
        eps: float = 1e-6
    ) -> torch.Tensor:
        
        # camera center distance method
        # camera_centers = c2w[..., :3, 3]
        # dist_matrix = torch.cdist(camera_centers, camera_centers, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        
        # camera viewpoint distance method
        dist_matrix = pose_cdistance(c2w, c2w)
        mask_matrix = (~mask[..., None]) & (mask[:, None, :])
        
        # min distance / mean distance
        masked_dist_matrix = dist_matrix.masked_fill(~mask_matrix, float('inf'))
        dist_to_src = torch.where(mask, 0.0, masked_dist_matrix.min(dim=-1).values)
        # dist_to_src = torch.where(mask, 0.0, (dist_matrix * mask_matrix).sum(dim=-1) / mask_matrix.sum(dim=-1))
        
        weights = dist_to_src / (dist_to_src.max(dim=-1, keepdim=True).values + eps) * (max_weight - min_weight) + min_weight
        weights = append_dims(weights, sigma.ndim)
        weights = weights.to(dtype=sigma.dtype)
        
        if self.name == 'seva_eps':
            weights = weights / (sigma**2)

        return weights
    
    
def get_loss_weighting(loss_weighting: str):
    if loss_weighting == "unit":
        return UnitWeighting(loss_weighting)
    elif loss_weighting == "edm":
        return EDMWeighting(loss_weighting)
    elif loss_weighting == "v":
        return VWeighting(loss_weighting)
    elif loss_weighting == "eps":
        return EpsWeighting(loss_weighting)
    elif loss_weighting in ["seva", "seva_eps"]:
        return SevaWeighting(loss_weighting)
    else:
        raise ValueError(f"Unknown loss weighting: {loss_weighting}")

