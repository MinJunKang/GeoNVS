#!/bin/bash
# Download pretrained weights for GeoNVS training and evaluation.
# Run from the repository root: bash tools/scripts/download_weights.sh
# Google Drive downloads require gdown: pip install gdown
set -e
mkdir -p pretrained_weights

# ---------------------------------------------------------------------------
# GeoNVS trained weights (GS-Adapter + LoRA) — required for evaluation
# ---------------------------------------------------------------------------
mkdir -p pretrained_weights/eccv_gattn
wget -c -O pretrained_weights/eccv_gattn/gs_adapter_weights.pth \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_gattn/gs_adapter_weights.pth"
wget -c -O pretrained_weights/eccv_gattn/pytorch_lora_weights.safetensors \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_gattn/pytorch_lora_weights.safetensors"

# ---------------------------------------------------------------------------
# CameraCtrl (SVD) checkpoint — required for train_camctrl.py
# (see https://github.com/hehao13/CameraCtrl)
# ---------------------------------------------------------------------------
wget -c -O pretrained_weights/CameraCtrl_svd.ckpt \
    "https://huggingface.co/hehao13/CameraCtrl_SVD_ckpts/resolve/main/CameraCtrl_svd.ckpt"

# ---------------------------------------------------------------------------
# DepthSplat — used by --lrm_model_name depthsplat and the on-the-fly
# training datasets (dl3dvo / re10ko).
# NOTE: upstream replaced the checkpoint used in the paper (f40abc4f) with a
# retrained one (f8ddd845). They are NOT equivalent - measured on our benchmark
# the retrained file loses ~5.5 dB PSNR on RE10K and ~2.7 dB on DL3DV - so the
# original is mirrored (MIT, (c) 2024 Haofei Xu) for reproducibility and used
# by default. Set DEPTHSPLAT_CKPT to point at another file.
# ---------------------------------------------------------------------------
wget -c -O pretrained_weights/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f40abc4f.pth \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/third_party/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f40abc4f.pth"
# current upstream release (retrained):
# wget -c -O pretrained_weights/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
#     "https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"

# ---------------------------------------------------------------------------
# MVSplat — used by --lrm_model_name mvsplat
# (official release folder: https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU)
# ---------------------------------------------------------------------------
gdown 1Y5OAJN9AX5yMjkbyxag5OnwacdjPnl-N -O pretrained_weights/mvsplat_re10k.ckpt

# ---------------------------------------------------------------------------
# HiSplat — used by --lrm_model_name hisplat
# (official release folder: https://drive.google.com/drive/folders/1U6GGbvk-oCMq-HTXxuIJf1q7PaRKiWdB)
# ---------------------------------------------------------------------------
gdown 1p6v4gyN8nvhSRiu7aBPJ2QWe7Mx2Qyqn -O pretrained_weights/hisplat_re10k.ckpt
# DINOv2 backbone init used by HiSplat's pyramid encoder. Only needed to silence
# the init warning (at eval the weights come from hisplat_re10k.ckpt).
# wget -c -O pretrained_weights/dinov2_vitb14_pretrain.pth \
#     "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

# ---------------------------------------------------------------------------
# MVSplat360 — used by --lrm_model_name mvsplat360
# ---------------------------------------------------------------------------
wget -c -O pretrained_weights/dl3dv_480p.ckpt \
    "https://huggingface.co/donydchen/mvsplat360/resolve/main/dl3dv_480p.ckpt"

# ---------------------------------------------------------------------------
# Baseline diffusion checkpoints (demo_diffusion.py)
# ---------------------------------------------------------------------------
# MotionCtrl (SVD / CMCM) — used by --dm_model_name motionctrl
wget -c -O pretrained_weights/motionctrl_svd.ckpt \
    "https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_svd.ckpt"

# ViewCrafter — used by --dm_model_name viewcrafter (/ viewcrafter_sparse).
# The 320x512 25-frame models matching baselines/diffusion/viewcrafter/viewcrafter_{single,sparse}.yaml.
wget -c -O pretrained_weights/viewcrafter_lowres.ckpt \
    "https://huggingface.co/Drexubery/ViewCrafter_25_512/resolve/main/model.ckpt"
wget -c -O pretrained_weights/viewcrafter_sparse.ckpt \
    "https://huggingface.co/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt"

# DUSt3R (ViewCrafter's geometry backbone). The official
# download.europe.naverlabs.com link is no longer served and
# naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt only ships model.safetensors
# (the vendored dust3r loader needs the original .pth with ckpt['args']),
# so this uses a byte-identical community mirror of the official file.
wget -c -O pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    "https://huggingface.co/camenduru/dust3r/resolve/main/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# GenFusion — used by --dm_model_name genfusion_*. Upstream names the file
# "epoch=59-step=34000.ckpt" (DL3DV generation model); renamed here to match
# baselines/diffusion_models.py.
# (the CDN rejects plain wget for this file, so use the HF client)
python -c "
from huggingface_hub import hf_hub_download; import shutil
p = hf_hub_download('Sibo2rr/GenFusion-GenerationModel', 'epoch=59-step=34000.ckpt')
shutil.copy(p, 'pretrained_weights/genfusion_dl3dv.ckpt')"

# Difix3D (--dm_model_name difix3d_*) needs no manual download: it pulls
# nvidia/difix_ref from the HuggingFace Hub on first use. (That remote code
# targets diffusers 0.25.1; baselines/diffusion_models.py shims the renamed
# symbols so it runs in the main environment.)
# MVSplat360 (--dm_model_name mvsplat360) reuses pretrained_weights/dl3dv_480p.ckpt
# from the MVSplat360 section above.
# CameraCtrl variants (cameractrl / geonvs_cameractrl_*)
# reuse pretrained_weights/CameraCtrl_svd.ckpt from the CameraCtrl section above.

# GeoNVS camctrl weights (GS-Adapter + LoRA on CameraCtrl-SVD)
mkdir -p pretrained_weights/eccv_camctrl
wget -c -O pretrained_weights/eccv_camctrl/gs_adapter_weights.pth \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_camctrl/gs_adapter_weights.pth"
wget -c -O pretrained_weights/eccv_camctrl/pytorch_lora_weights.safetensors \
    "https://huggingface.co/HugMinjun/GeoNVS/resolve/main/eccv_camctrl/pytorch_lora_weights.safetensors"

# ---------------------------------------------------------------------------
# Downloaded automatically from the HuggingFace Hub / torch.hub on first use
# (no action needed):
#   - stabilityai/stable-virtual-camera        (SEVA backbone; accept license)
#   - stabilityai/stable-video-diffusion-img2vid (VAE)
#   - OpenCLIP ViT-H-14 laion2b_s32b_b79k      (conditioner)
#   - depth-anything/DA3-GIANT-1.1             (--lrm_model_name da3)
#   - facebookresearch/dinov2 via torch.hub    (depthsplat backbone)
#   - facebook/VGGT-1B, yyfz233/Pi3, facebook/map-anything (datagen)
# vggt_iv / vggt_av / pi3_iv / pi3_av need no weights: they read the
# precomputed Gaussians produced by datagen/process_benchmark.py.
