
import os, gc
import cv2
import glob
import argparse
import os.path as osp
import warnings
from tqdm import tqdm
from einops import repeat, rearrange
from pathlib import Path
from safetensors.torch import save_file
from data_io import get_parser

warnings.filterwarnings('ignore')


import torch
import numpy as np
import time
import torchvision.transforms as tf
from load_fn import load_img_and_K, load_and_preprocess_vggt, load_and_preprocess, convert_poses, load_safetensor_file
from view_selector import select_views, check_scene, write_log, CLIPDistance
from load_model import VGGtModel, PI3Model, MapAnythingModel


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose


def get_value_dict(
    curr_c2ws,
    all_c2ws,
    camera_scale,
):
    c2w = to_hom_pose(curr_c2ws.float())

    # camera centering
    ref_c2ws = all_c2ws
    camera_dist_2med = torch.norm(
        ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
        dim=-1,
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )
    c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    c2w[:, :3, 3] *= translation_scaling_factor
    w2c = torch.linalg.inv(c2w)

    return w2c


# dataloader configuration (defaults; overridden by CLI args in __main__)
dataset_path = 'datasets/benchmarkset'
dataset_name = 'reflection_real'  # 're10k-pixelsplat', 'omniobject3d', 'wildrgbd/wildrgbd_easy', 'wildrgbd/wildrgbd_hard', 'dl3dv140'
model_names = ['vggt', 'pi3']  # vggt / pi3 / mapanything
include_target = False  # include target views in the point cloud prediction
camera_scale_default = 2.0
resolution = 518
measure_confidence = True
device = "cuda" if torch.cuda.is_available() else "cpu"


# scale mapping (single view)
camera_scales_single_view = {
    'co3d': 0.8,
    'co3d-viewcrafter': 0.8,
    'dtu': 0.2,
    'llff': 1.0,
    'mipnerf360': 2.0,
    'omniobject3d': 2.0,
    're10k': 1.2,
    're10k-4dim': 0.8,
    're10k-viewcrafter': 0.4,
    'tnt-viewcrafter': 0.6,
    'glossy_real': 2.0,
    'reflection_real': 2.0,
}
view_usage_scenes = {
    'co3d': [1, 2, 3, 6, 9],
    'co3d-viewcrafter': [1],
    'dl3dv10': [9], # [3, 6, 9, 16, 32],
    'dl3dv10_vipe': [3, 6, 9],
    'dl3dv140': [3, 6, 9, 16, 32],
    'dtu': [1, 3, 6, 9],
    'llff': [1, 2, 3, 6, 9],
    'mipnerf360': [1, 2, 3, 6, 9],
    'omniobject3d': [1, 3, 6, 9],
    're10k': [1, 2, 3, 6, 9],
    're10k-4dim': [1],
    're10k-pixelsplat': [2],
    're10k-viewcrafter': [1],
    'wildrgbd/wildrgbd_easy': [1, 3, 6, 9],
    'wildrgbd/wildrgbd_hard': [1, 3, 6, 9],
    'tnt-viewcrafter': [1],
    'tnt-cf3dgs': [9, 16], # [2, 3, 6],
    'tnt-cf3dgs_vipe': [2, 3, 6],
    'veo3_demo': [2],
    'glossy_real': [3, 6],
    'reflection_real': [3, 6],
}
# these scenes must be normalized
scene_normalize = ["co3d", "co3d-viewcrafter", "wildrgbd/wildrgbd_easy", "wildrgbd/wildrgbd_hard"]


def run_function(model_lrm):
    
    if osp.isfile(osp.join(dataset_path, "selected_scenes.txt")):
        with open(osp.join(dataset_path, "selected_scenes.txt"), 'r') as f:
            scenes = f.read().splitlines()
        print(f"Loaded {len(scenes)} selected scenes from selected_scenes.txt")
        scenes = [osp.join(dataset_path, s) for s in scenes]
    else:
        scenes = [
            item for item in glob.glob(osp.join(dataset_path, "*")) if os.path.isdir(item)
        ]
    
    progress_bar = tqdm(scenes)
    for scene in progress_bar:
        tag = "av" if include_target else "iv"
        output_path = Path(scene) / f"{model_lrm.model_name}_{tag}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        parser = get_parser(
            parser_type="reconfusion",
            data_dir=scene,
            normalize=False,
            file_name="transforms.json"
        )
        splits_per_num_input_frames = {k: v for k, v in parser.splits_per_num_input_frames.items() if k in view_usage_scenes.get(dataset_name, [])}
        
        # check already processed scenes
        input_frame_keys = []
        for num_inputs in splits_per_num_input_frames.keys():
            split_dict = splits_per_num_input_frames[num_inputs]
            output_name = f"{len(split_dict['train_ids'])}_{len(split_dict['test_ids'])}.safetensors"
            # if (output_path / output_name).exists() or len(split_dict['test_ids']) + len(split_dict['train_ids']) > 64:
            #     continue
            input_frame_keys.append(num_inputs)
        
        all_imgs_path = parser.image_paths
        c2ws = parser.camtoworlds
        camera_ids = parser.camera_ids
        Ks = np.concatenate([parser.Ks_dict[cam_id][None] for cam_id in camera_ids], 0)
        
        Ks = torch.tensor(Ks).float()
        c2ws = torch.tensor(c2ws).float()
        images_all, intrinsics = [], []
        for id, path in enumerate(all_imgs_path):
            images_, Ks_ = load_img_and_K(path, resolution, K=Ks[id], image_as_tensor=False, device='cpu')
            Ks_[0, :] /= images_.size[0]
            Ks_[1, :] /= images_.size[1]
            images_all.append(images_)
            intrinsics.append(Ks_)
        images_vgg = load_and_preprocess_vggt(images_all, target_size=resolution).to(device)
        intrinsics = torch.stack(intrinsics, 0).to(device)

        for num_inputs in input_frame_keys:
            
            split_dict = splits_per_num_input_frames[num_inputs]
            output_name = f"{len(split_dict['train_ids'])}_{len(split_dict['test_ids'])}.safetensors"
            input_images_vgg = images_vgg[split_dict["train_ids"]]
            target_images_vgg = images_vgg[split_dict["test_ids"]]
            input_intrinsics = intrinsics[split_dict["train_ids"]]
            target_intrinsics = intrinsics[split_dict["test_ids"]]
            
            # scale adjustment
            if dataset_name in scene_normalize:
                camera_scale = camera_scales_single_view.get(dataset_name, camera_scale_default) if len(split_dict["train_ids"]) == 1 else camera_scale_default
                w2cs = get_value_dict(
                    c2ws.clone(),
                    c2ws,
                    camera_scale=camera_scale,
                )
            else:
                w2cs = torch.linalg.inv(c2ws)
        
            input_poses = w2cs[split_dict["train_ids"]]
            target_poses = w2cs[split_dict["test_ids"]]
            
            H, W = input_images_vgg.shape[-2:]
            
            batch = {
                'input_images': input_images_vgg.to(device),
                'input_extrinsics': input_poses.to(device),
                'input_intrinsics': input_intrinsics.to(device),
                'target_images': target_images_vgg.to(device),
                'target_extrinsics': target_poses.to(device),
                'target_intrinsics': target_intrinsics.to(device),
                'original_size': (H, W),
            }
            
            # get gaussian
            gaussians, metrics = model_lrm.prediction(batch, include_target=include_target)
            if gaussians == None:
                write_log(msg=output_name, path=f'{dataset_name}_{tag}_failure.txt')
                continue
            
            # save data
            if measure_confidence:
                gaussians_tensor = torch.cat([
                        gaussians.means,  # 3
                        gaussians.rotations,  # 4
                        gaussians.scales,  # 3
                        gaussians.opacities[..., None],  # 1
                        rearrange(gaussians.fishers, 'b g p q -> b g (p q)'),  # 36
                        rearrange(gaussians.harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 3 * (n ** 2)
                    ], dim=-1
                )
            else:
                gaussians_tensor = torch.cat([
                        gaussians.means,  # 3
                        rearrange(gaussians.covariances, 'b g p q -> b g (p q)'),  # 9
                        gaussians.opacities[..., None],  # 1
                        rearrange(gaussians.harmonics, 'b g c d_sh -> b g (c d_sh)'),  # 3 * (n ** 2)
                    ], dim=-1
                )
                
            input_indices_pt = torch.tensor(split_dict["train_ids"], dtype=torch.short)
            target_indices_pt = torch.tensor(split_dict["test_ids"], dtype=torch.short)
                
            outputs = {
                'input_indices': input_indices_pt,
                'target_indices': target_indices_pt,
                'sh_degree': torch.tensor([model_lrm.sh_degree], dtype=torch.short),
                'near': torch.tensor([model_lrm.near], dtype=torch.bfloat16),
                'far': torch.tensor([model_lrm.far], dtype=torch.bfloat16),
                'gaussians': gaussians_tensor
            }
            progress_bar.set_description(f"psnr_min {metrics['psnr'].min().item():.1f} psnr_max {metrics['psnr'].max().item():.1f}")
            
            # save outputs
            save_file(outputs, output_path / output_name, metadata={"model_name": model_lrm.model_name, "dataset_name": dataset_name})
    
    
    
if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Generate benchmark Gaussians for GeoNVS evaluation.")
    cli.add_argument("--data_root", type=str, default="datasets/benchmarkset",
                     help="Root folder that contains the benchmark scene sets.")
    cli.add_argument("--dataset", type=str, default=dataset_name,
                     help="Benchmark set name (e.g. dl3dv140, re10k-pixelsplat, reflection_real).")
    cli.add_argument("--models", type=str, nargs="+", default=model_names,
                     choices=["vggt", "pi3", "mapanything"])
    cli.add_argument("--include_target", action="store_true",
                     help="Include target views in the point cloud prediction.")
    cli.add_argument("--no_confidence", action="store_true", help="Skip Fisher information.")
    args = cli.parse_args()

    dataset_name = args.dataset
    dataset_path = osp.join(args.data_root, dataset_name)
    model_names = args.models
    include_target = args.include_target
    measure_confidence = not args.no_confidence

    for model_name in model_names:
        if model_name == 'vggt':
            model_lrm = VGGtModel(dataset_name, measure_confidence, device)
        elif model_name == 'pi3':
            model_lrm = PI3Model(dataset_name, measure_confidence, device)
        elif model_name == 'mapanything':
            model_lrm = MapAnythingModel(dataset_name, measure_confidence, device)
        # model_lrm.prune_percent = 0.0
        run_function(model_lrm)
        
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
