#!/usr/bin/env bash


# base model
# first train on re10k, 2 views, 256x448
# train on 8x (8 nodes) 4x GPUs (>=80GB VRAM) for 150K steps, batch size 8 on each gpu 
python -m src.main +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
dataset.image_shape=[256,448] \
trainer.max_steps=150000 \
trainer.num_nodes=8 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vitb.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/re10k-256x448-depthsplat-base


# finetune on dl3dv, random view 2-6
# train on 4x GPUs (>=80GB VRAM) for 100K steps, batch size 1 on each gpu
# resume from the previously pretrained model on re10k
python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.roots=[datasets/dl3dv] \
dataset.view_sampler.num_target_views=8 \
dataset.view_sampler.num_context_views=6 \
dataset.min_views=2 \
dataset.max_views=6 \
trainer.max_steps=100000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x448-view2-76a0605a.pth \
wandb.project=depthsplat \
output_dir=checkpoints/dl3dv-256x448-depthsplat-base-randview2-6


