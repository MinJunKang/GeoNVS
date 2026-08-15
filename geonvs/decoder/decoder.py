from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, Optional

from jaxtyping import Float
from torch import Tensor, nn

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    covariances: Optional[Float[Tensor, "batch gaussian dim dim"]] = None
    rotations: Optional[Float[Tensor, "batch gaussian 4"]] = None
    scales: Optional[Float[Tensor, "batch gaussian 3"]] = None
    fishers: Optional[Float[Tensor, "batch gaussian 6 6"]] = None
    fisher_scores: Optional[Float[Tensor, "batch gaussian"]] = None
    feature_harmonics: Optional[Float[Tensor, "batch gaussian 4 d_sh"]] = None


@dataclass
class DecoderOutput:
    color: Optional[Float[Tensor, "batch view 3 height width"]] = None
    depth: Optional[Float[Tensor, "batch view height width"]] = None
    uncertainty: Optional[Float[Tensor, "batch view height width"]] = None
    feature: Optional[Float[Tensor, "batch view dim height width"]] = None
    gsindex: Optional[Float[Tensor, "batch view dim height width"]] = None
    gsweight: Optional[Float[Tensor, "batch view dim height width"]] = None


T = TypeVar("T")


class Decoder(nn.Module, ABC):

    def __init__(self, background_color) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
    ) -> DecoderOutput:
        pass
