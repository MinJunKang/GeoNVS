#!/bin/bash
# GeoNVS training on top of CameraCtrl (SVD backbone).
# DL3DV, 14 frames, 576x320, 8 GPUs.

CONFIG=${1:-configs/module_config/gsadapter_camctrl_base.yaml}
OUTPUT_DIR=${2:-outputs_camctrl_$(basename ${CONFIG%.yaml})}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --main_process_port 23881 \
    --config_file configs/accelerate_config/deepspeed_zero_2.yaml \
    train_camctrl.py \
    --report_to wandb \
    --do_validation \
    --base_folder datasets \
    --gs_adapter_config ${CONFIG} \
    --output_dir ${OUTPUT_DIR}
