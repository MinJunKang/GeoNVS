"""Generate precomputed 3D Gaussian chunks (.safetensors) for GeoNVS training.

For every scene chunk of a pixelSplat-style dataset (RE10K / DL3DV .torch
chunks), this script
  1. selects input / target view combinations with a frustum-overlap and
     CLIP-distance based view selector,
  2. runs VGGT to get dense geometry, aligns it to the GT camera poses, and
     optimizes a per-scene 3D Gaussian representation (with PUP pruning),
  3. computes per-Gaussian Fisher information (36 channels), and
  4. saves the packed Gaussians to `<data_root>/<dataset>_<res>/<stage>_gs/`.

The resulting files are consumed by the `dl3dv` / `re10k` datasets of the
training code (see `data/` in the repository root).

Example:
    python process_dataset.py --dataset dl3dv --stage test \
        --data_root /path/to/datasets --start_index 0 --end_index 100
"""

import os
import sys
import argparse
import warnings
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from einops import rearrange
from pathlib import Path
from safetensors.torch import save_file

warnings.filterwarnings('ignore')

import torch

from load_fn import load_and_preprocess_vggt, convert_poses, load_safetensor_file
from view_selector import select_views, check_scene, write_log, CLIPDistance
from load_model import VGGtModel


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Gaussian chunks for GeoNVS training.")
    parser.add_argument("--dataset", type=str, default="dl3dv", choices=["dl3dv", "re10k"])
    parser.add_argument("--stage", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--data_root", type=str, default="datasets",
                        help="Root folder that contains <dataset>_low / <dataset>_high.")
    parser.add_argument("--highres", action="store_true")
    parser.add_argument("--start_index", type=int, required=True, help="First chunk index (inclusive).")
    parser.add_argument("--end_index", type=int, required=True, help="Last chunk index (inclusive).")
    parser.add_argument("--num_group_images", type=int, default=21,
                        help="Number of views per training sample (matches --num_frames of train_seva.py).")
    parser.add_argument("--num_input_images", type=int, nargs="+", default=[1, 3, 6, 9, 12],
                        help="Numbers of input (context) views to precompute per scene.")
    parser.add_argument("--num_input_variations", type=int, default=None,
                        help="View-combination variations per input count (default: 3 for dl3dv, 2 for re10k).")
    parser.add_argument("--adjacent_frame_sampling_rate", type=float, default=0.6,
                        help="Fraction of easy (adjacent-frame) samples.")
    parser.add_argument("--min_frustum_iou_threshold", type=float, default=0.4,
                        help="Minimum frustum overlap between selected views.")
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--pretrained_model", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--no_confidence", action="store_true",
                        help="Skip Fisher information (saves covariance parameterization instead).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process scenes even if their outputs already exist.")
    return parser.parse_args()


def run_function(args, model_lrm, clip_metric, root_path, output_path, target_shape, device):
    dataset_name = args.dataset
    stage = args.stage
    num_group_images = args.num_group_images
    num_input_images = args.num_input_images
    num_input_variations = args.num_input_variations
    measure_confidence = not args.no_confidence

    # re10k chunks need one spare frame for the view selector
    min_num_images = num_group_images + (1 if dataset_name == 're10k' else 0)

    progress_bar = tqdm(range(args.start_index, args.end_index + 1))
    for chunk_index in progress_bar:

        chunk_path = root_path / f"{chunk_index:0>6}.torch"
        if not chunk_path.exists():
            progress_bar.write(f"Chunk {chunk_index} does not exist. Skipping...")
            continue

        scenes = torch.load(chunk_path)
        for scene in scenes:

            # state outputs
            keyname = scene['key'].replace("dl3dv_", "") if dataset_name == 'dl3dv' else scene['key']

            # skip scene with not enough images
            if len(scene['images']) <= min_num_images:
                continue

            # check scene if already processed
            if check_scene(output_path, keyname, num_input_images, num_input_variations) and not args.overwrite:
                continue

            # get input / target pairs : num_input_variations x len(num_input_images)
            images = [Image.open(BytesIO(image.numpy().tobytes())).resize(target_shape, Image.Resampling.BICUBIC) for image in scene['images']]
            W, H = images[0].size
            poses_w2c, intrinsics = convert_poses(scene['cameras'])
            view_select_inputs = {
                "images": images,
                "clip_metric": clip_metric,
                "poses_w2c": poses_w2c,
                "intrinsics": intrinsics,
                "num_group_images": num_group_images,
                "num_input_images": num_input_images,
                "num_input_variations": num_input_variations,
                "near": args.near,
                "far": args.far,
                "H": H,
                "W": W,
                "adjacent_frame_sampling_rate": args.adjacent_frame_sampling_rate,
                "min_frustum_iou_threshold": args.min_frustum_iou_threshold
            }
            view_combinations = select_views(**view_select_inputs)

            for idx, view_combination in enumerate(view_combinations):
                input_indices = view_combination[0]
                target_indices = view_combination[1]
                input_indices_pt = torch.tensor(input_indices, dtype=torch.short)
                target_indices_pt = torch.tensor(target_indices, dtype=torch.short)

                # read images, poses, intrinsics
                input_images = [images[i] for i in input_indices]
                input_poses, input_intrinsics = poses_w2c[input_indices], intrinsics[input_indices]
                target_images = [images[i] for i in target_indices]
                target_poses, target_intrinsics = poses_w2c[target_indices], intrinsics[target_indices]

                # assign to device
                input_poses = input_poses.to(device)
                input_intrinsics = input_intrinsics.to(device)
                target_poses = target_poses.to(device)
                target_intrinsics = target_intrinsics.to(device)

                # run vggt
                output_name = f"{keyname}_{num_group_images - len(target_indices)}_{len(target_indices)}_{idx}.safetensors"
                name_wo_key = f"{num_group_images - len(target_indices)}_{len(target_indices)}_{idx}"

                # preprocessing
                input_images_vgg = load_and_preprocess_vggt(input_images).to(device)
                target_images_vgg = load_and_preprocess_vggt(target_images).to(device)

                batch = {
                    'input_images': input_images_vgg,
                    'input_extrinsics': input_poses,
                    'input_intrinsics': input_intrinsics,
                    'input_indices': input_indices,
                    'target_images': target_images_vgg,
                    'target_extrinsics': target_poses,
                    'target_intrinsics': target_intrinsics,
                    'target_indices': target_indices,
                    'original_size': (H, W),
                }

                # get gaussian
                gaussians, metrics = model_lrm.prediction(batch)
                if gaussians == None:
                    write_log(msg=output_name, path=f'{dataset_name}_{stage}_failure.txt')
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

                gaussians_tensor = gaussians_tensor.squeeze().detach().cpu().to(dtype=torch.bfloat16)

                outputs = {
                    'input_indices': input_indices_pt,
                    'target_indices': target_indices_pt,
                    'sh_degree': torch.tensor([model_lrm.sh_degree], dtype=torch.short),
                    'near': torch.tensor([model_lrm.near], dtype=torch.bfloat16),
                    'far': torch.tensor([model_lrm.far], dtype=torch.bfloat16),
                    'gaussians': gaussians_tensor
                }
                progress_bar.set_description(f"{name_wo_key}: psnr_min {metrics['psnr'].min().item():.1f} psnr_max {metrics['psnr'].max().item():.1f}")

                if (~torch.isfinite(gaussians_tensor)).any():
                    # do not save this scene and log this scene
                    write_log(msg=output_name, path=f'{dataset_name}_{stage}_log.txt')
                    continue

                # save outputs
                save_file(outputs, output_path / output_name, metadata={"model_name": "vggt", "dataset_name": dataset_name})
                # read back with: loaded, metadata = load_safetensor_file(output_path / output_name)


if __name__ == "__main__":
    args = parse_args()
    assert args.start_index < args.end_index

    if args.num_input_variations is None:
        args.num_input_variations = 3 if args.dataset == 'dl3dv' else 2

    res_tag = 'high' if args.highres else 'low'
    root_path = Path(args.data_root) / f'{args.dataset}_{res_tag}' / args.stage
    output_path = Path(args.data_root) / f'{args.dataset}_{res_tag}' / f'{args.stage}_gs'
    output_path.mkdir(parents=True, exist_ok=True)

    # PIL-style (W, H) resize targets
    if args.highres:
        target_shape = (1280, 720) if args.dataset == 're10k' else (960, 540)
    else:
        target_shape = (640, 360) if args.dataset == 're10k' else (480, 270)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_metric = CLIPDistance(device=device).to(device)

    print(f"Processing chunks from {args.start_index} to {args.end_index}... in {args.dataset} dataset")
    model_lrm = VGGtModel(
        args.dataset,
        measure_confidence=not args.no_confidence,
        device=device,
        pretrained_model=args.pretrained_model,
    )

    run_function(args, model_lrm, clip_metric, root_path, output_path, target_shape, device)
