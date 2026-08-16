#!/bin/bash
# Weights needed to train and evaluate GeoNVS itself.
# Run from the repository root: bash tools/scripts/download_weights_ours.sh
#
# For the comparison baselines as well, use download_weights_all.sh instead.
set -e
mkdir -p pretrained_weights

# ---------------------------------------------------------------------------
# GeoNVS trained weights (GS-Adapter + LoRA)
#   eccv_gattn   : SEVA backbone, gated-attention fusion (main results)
#   eccv_base    : SEVA backbone, concat-FeedForward fusion (ablation)
#   eccv_camctrl : CameraCtrl backbone
# ---------------------------------------------------------------------------
mkdir -p pretrained_weights/eccv_gattn
wget -c -O pretrained_weights/eccv_gattn/gs_adapter_weights.pth \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_gattn/gs_adapter_weights.pth"
wget -c -O pretrained_weights/eccv_gattn/pytorch_lora_weights.safetensors \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_gattn/pytorch_lora_weights.safetensors"

mkdir -p pretrained_weights/eccv_base
wget -c -O pretrained_weights/eccv_base/gs_adapter_weights.pth \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_base/gs_adapter_weights.pth"
wget -c -O pretrained_weights/eccv_base/pytorch_lora_weights.safetensors \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_base/pytorch_lora_weights.safetensors"

mkdir -p pretrained_weights/eccv_camctrl
wget -c -O pretrained_weights/eccv_camctrl/gs_adapter_weights.pth \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_camctrl/gs_adapter_weights.pth"
wget -c -O pretrained_weights/eccv_camctrl/pytorch_lora_weights.safetensors \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_camctrl/pytorch_lora_weights.safetensors"

# ---------------------------------------------------------------------------
# CameraCtrl (SVD) backbone — required for train_camctrl.py and the
# CameraCtrl-based GeoNVS variant
# (see https://github.com/hehao13/CameraCtrl)
# ---------------------------------------------------------------------------
wget -c -O pretrained_weights/CameraCtrl_svd.ckpt \
    "https://huggingface.co/hehao13/CameraCtrl_SVD_ckpts/resolve/main/CameraCtrl_svd.ckpt"

# ---------------------------------------------------------------------------
# DepthSplat — geometry prior for --lrm_model_name depthsplat and for the
# on-the-fly training datasets (dl3dvo / re10ko).
# NOTE: upstream replaced the checkpoint used in the paper (f40abc4f) with a
# retrained one (f8ddd845). They are NOT equivalent - measured on our benchmark
# the retrained file loses ~5.5 dB PSNR on RE10K and ~2.7 dB on DL3DV - so the
# original is mirrored (MIT, (c) 2024 Haofei Xu) and used by default.
# Set DEPTHSPLAT_CKPT to point at another file.
# ---------------------------------------------------------------------------
wget -c -O pretrained_weights/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f40abc4f.pth \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/third_party/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f40abc4f.pth"
# current upstream release (retrained):
# wget -c -O pretrained_weights/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
#     "https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"

# ---------------------------------------------------------------------------
# Downloaded automatically from the HuggingFace Hub / torch.hub on first use
# (no action needed):
#   - stabilityai/stable-virtual-camera        (SEVA backbone; accept license)
#   - stabilityai/stable-video-diffusion-img2vid (VAE)
#   - OpenCLIP ViT-H-14 laion2b_s32b_b79k      (conditioner)
#   - facebookresearch/dinov2 via torch.hub    (DepthSplat backbone)
#   - facebook/VGGT-1B, yyfz233/Pi3, facebook/map-anything (data preprocessing)
# vggt_iv / vggt_av / pi3_iv / pi3_av need no weights: they read the
# precomputed Gaussians produced by datapreprocess/process_benchmark.py.
