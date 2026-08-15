#!/bin/bash
# GeoNVS training on top of SEVA (Stable Virtual Camera).
# Reproduces the main paper results (DL3DV, 21 frames, 384x384, 8 GPUs).
#
# Variants:
#   configs/module_config/gsadapter_eccv_base.yaml   : concat fusion (base)
#   configs/module_config/gsadapter_eccv_gattn.yaml  : gated-attention fusion

CONFIG=${1:-configs/module_config/gsadapter_eccv_gattn.yaml}
OUTPUT_DIR=${2:-outputs_seva_$(basename ${CONFIG%.yaml})}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --main_process_port 23880 \
    --config_file configs/accelerate_config/deepspeed_zero_2.yaml \
    train_seva.py \
    --report_to wandb \
    --do_validation \
    --base_folder datasets \
    --gs_adapter_config ${CONFIG} \
    --output_dir ${OUTPUT_DIR}
