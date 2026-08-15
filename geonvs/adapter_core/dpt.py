
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from typing import List, Dict, Tuple, Union


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners) 
    
    
# custom implementation to avoid checkerboard artifacts
# https://distill.pub/2016/deconv-checkerboard/
class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleLayer, self).__init__()
        assert scale_factor % 2 == 0, "scale_factor should be a multiple of 2"
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=scale_factor + 1, stride=1, padding=scale_factor//2)

    def forward(self, x):
        x = custom_interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        x = self.conv(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups=1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.GELU(),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )
    
    
    
class MultiScaleFusionBlock(nn.Module):
    def __init__(self, gs_channel_dim, feature_dims, use_bn, size=None):
        super(MultiScaleFusionBlock, self).__init__()
        self.feature_dims = feature_dims
        self.num_features = len(feature_dims)

        # upsampling layer
        base_feature_dim = feature_dims[0]
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=base_feature_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for feature_dim in feature_dims
        ])
        self.resize_layers = nn.ModuleList()
        for i in range(len(self.projects)):
            if i == 0:
                self.resize_layers.append(
                    nn.Conv2d(
                        in_channels=base_feature_dim,
                        out_channels=base_feature_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                )
            else:
                self.resize_layers.append(
                    UpsampleLayer(
                        in_channels=base_feature_dim,
                        out_channels=base_feature_dim,
                        scale_factor=2**i
                    )
                )
        
        # channel modulation
        self.scratch = nn.ModuleList([
            nn.Conv2d(
                in_channels=base_feature_dim,
                out_channels=gs_channel_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ) for i in range(self.num_features)
        ])

        # dpt style fusion layer
        self.fusion_blocks = nn.ModuleList(
            [_make_fusion_block(gs_channel_dim, use_bn, size) for _ in range(self.num_features)]
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        
        out = []
        for i, x in enumerate(input_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            x = self.scratch[i](x)
            out.append(x)
            
        for i in range(1, len(out)+1):
            if i == 1:
                out[-i] = self.fusion_blocks[-i](out[-i], size=out[-i].shape[2:])
            else:
                out[-i] = self.fusion_blocks[-i](out[-i], out[-i+1], size=out[-i].shape[2:])

        return out