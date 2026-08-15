#!/bin/bash
# Everything: GeoNVS weights plus every comparison baseline evaluated in the
# paper (~55 GB). Run from the repository root:
#     bash tools/scripts/download_weights_all.sh
#
# Google Drive downloads require gdown: pip install gdown
set -e

# GeoNVS + backbones + DepthSplat prior
bash tools/scripts/download_weights_ours.sh

# ---------------------------------------------------------------------------
# Feed-forward geometry baselines
# ---------------------------------------------------------------------------
# MVSplat — --lrm_model_name mvsplat
# (release folder: https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU)
gdown 1Y5OAJN9AX5yMjkbyxag5OnwacdjPnl-N -O pretrained_weights/mvsplat_re10k.ckpt

# HiSplat — --lrm_model_name hisplat
# (release folder: https://drive.google.com/drive/folders/1U6GGbvk-oCMq-HTXxuIJf1q7PaRKiWdB)
gdown 1p6v4gyN8nvhSRiu7aBPJ2QWe7Mx2Qyqn -O pretrained_weights/hisplat_re10k.ckpt
# DINOv2 backbone init used by HiSplat's pyramid encoder. Only needed to silence
# the init warning (at eval the weights come from hisplat_re10k.ckpt).
# wget -c -O pretrained_weights/dinov2_vitb14_pretrain.pth \
#     "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

# MVSplat360 — --lrm_model_name mvsplat360 and --dm_model_name mvsplat360
wget -c -O pretrained_weights/dl3dv_480p.ckpt \
    "https://huggingface.co/donydchen/mvsplat360/resolve/main/dl3dv_480p.ckpt"

# ---------------------------------------------------------------------------
# Video-diffusion baselines (demo_diffusion.py)
# ---------------------------------------------------------------------------
# MotionCtrl (SVD / CMCM) — --dm_model_name motionctrl
wget -c -O pretrained_weights/motionctrl_svd.ckpt \
    "https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_svd.ckpt"

# ViewCrafter — --dm_model_name viewcrafter (/ viewcrafter_sparse).
# The 320x512 25-frame models matching
# baselines/diffusion/viewcrafter/viewcrafter_{single,sparse}.yaml.
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

# GenFusion — --dm_model_name genfusion_*. Upstream names the file
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
# Depth-Anything-3 (--lrm_model_name da3) likewise auto-downloads
# depth-anything/DA3-GIANT-1.1.
