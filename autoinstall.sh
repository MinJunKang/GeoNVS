#!/bin/bash
# Install script for GeoNVS training + evaluation
# (assumes torch 2.5.x + CUDA 12.1 already installed in the active environment).
#
# Uses uv (https://docs.astral.sh/uv/) when available — identical installs,
# much faster resolution — and falls back to pip otherwise.
set -e

if command -v uv >/dev/null 2>&1; then
    PIP="uv pip install --python $(command -v python)"
    echo "==> using uv ($(uv --version))"
else
    PIP="python -m pip install"
    echo "==> uv not found, using pip"
fi

# NOTE: the CUDA extensions below import torch in setup.py, so build isolation
# must be disabled (uv isolates builds by default; pip honors the flag too).

# ---------------------------------------------------------------------------
# Required: feature-capable Gaussian rasterizer used by the GS-Adapter,
# and the Fisher renderer used by the LRM / evaluation stack
# ---------------------------------------------------------------------------
$PIP --no-build-isolation third_party/langsplat-rasterization
$PIP --no-build-isolation third_party/rasterization_and_pup_fisher

# xformers (memory-efficient attention; version must match your torch build)
$PIP -U xformers --index-url https://download.pytorch.org/whl/cu121

# torch-scatter (LRM voxel fusion; wheel index must match your torch/CUDA build)
$PIP torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# ---------------------------------------------------------------------------
# Required for the `rattn` / `gattn` fusion configs (RoPE attention kernels)
# ---------------------------------------------------------------------------
$PIP --no-build-isolation transformer_engine[pytorch]

# ---------------------------------------------------------------------------
# Optional: extra LRM backbones for training (dl3dvo/re10ko) and evaluation
# ---------------------------------------------------------------------------
# $PIP --no-build-isolation third_party/diff-gaussian-rasterization-depth
# $PIP --no-build-isolation third_party/diff-gaussian-rasterization
# $PIP --no-build-isolation third_party/latent-gaussian-rasterization   # mvsplat360
# $PIP -e baselines/lrm/Depth-Anything-3                                        # da3 (falls back to sys.path if skipped)
# $PIP --no-build-isolation third_party/simple-knn

$PIP -r requirements.txt
