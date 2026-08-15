#!/bin/bash
# Install script for the GeoNVS data-generation pipeline.
# The CUDA rasterizers are shared with the training/evaluation code
# (../third_party), so datapreprocess can run in the SAME environment as training —
# this script only adds the datapreprocess-specific packages on top.
#
# Uses uv (https://docs.astral.sh/uv/) when available, falls back to pip.
set -e

if command -v uv >/dev/null 2>&1; then
    PIP="uv pip install --python $(command -v python)"
    echo "==> using uv ($(uv --version))"
else
    PIP="python -m pip install"
    echo "==> uv not found, using pip"
fi

# VGGT (vendored)
$PIP -e vggt

# PyTorch3D (matrix_to_quaternion)
$PIP --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# Shared CUDA rasterizers (skip any that are already installed for training;
# setup.py imports torch, so build isolation must stay disabled)
$PIP --no-build-isolation ../third_party/simple-knn
$PIP --no-build-isolation ../third_party/diff-gaussian-rasterization
$PIP --no-build-isolation ../third_party/langsplat-rasterization
$PIP --no-build-isolation ../third_party/rasterization_and_pup_fisher

# Optional: fused SSIM for faster per-scene Gaussian optimization
# $PIP --no-build-isolation ../third_party/fused-ssim

# Optional: xformers speeds up the Pi3 backbone (benchmark generation)
# $PIP -U xformers --index-url https://download.pytorch.org/whl/cu121

$PIP -r requirements.txt
