#!/usr/bin/env bash


# base model: depth prediction on re10k: 2 input views 352x640
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=10 \
mode=test \
dataset/view_sampler=evaluation \
dataset.image_shape=[352,640] \
test.compute_scores=false \
dataset.view_sampler.num_context_views=2 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
train.forward_depth_only=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth \
test.compute_scores=false \
test.save_depth=true \
test.save_depth_concat_img=true \
output_dir=outputs/depthsplat-depth-base-re10k


# base model: depth prediction on re10k: 6 input views 352x640
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=10 \
mode=test \
dataset.roots=[datasets/re10k] \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/re10k_ctx_6v_video.json \
dataset.image_shape=[352,640] \
dataset.ori_image_shape=[360,640] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
train.forward_depth_only=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth \
test.compute_scores=false \
test.save_depth=true \
test.save_depth_concat_img=true \
output_dir=outputs/depthsplat-depth-base-re10k-view6


# base model: depth prediction on dl3dv: 12 input views 512x960
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
mode=test \
dataset.roots=[datasets/dl3dv_960p] \
dataset/view_sampler=evaluation \
dataset.image_shape=[512,960] \
dataset.ori_image_shape=[540,960] \
dataset.view_sampler.num_context_views=12 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_100_ctx_12v_video.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
train.forward_depth_only=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth \
test.compute_scores=false \
test.save_depth=true \
test.save_depth_concat_img=true \
output_dir=outputs/depthsplat-depth-base-dl3dv-view12

