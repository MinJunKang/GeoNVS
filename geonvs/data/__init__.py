
import os
import random
import torch
import numpy as np
from .dl3dv import DL3DVDataset
from .dl3dvt import DL3DVODataset
from .re10k import Re10KDataset
from .re10kt import Re10KODataset


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


def build_dataset(args, dataset='dl3dv', stage='train', step_tracker=None, plucker_convention='seva', input_num=[1, 3, 6, 9, 12]):  # 9 was missing
    tag = 'high' if args.highres else 'low'
    square_crop = (args.height == args.width)
    if dataset == 'dl3dv':  # include gaussian splat
        return DL3DVDataset(
            root_dir=os.path.join(args.base_folder, f'dl3dv_{tag}'), 
            stage=stage, 
            num_images=args.num_frames,
            target_shape=(args.height, args.width),
            plucker_convention=plucker_convention,
            input_num=input_num,
            square_crop=square_crop,
        )
    elif dataset == 'dl3dvo':
        return DL3DVODataset(
            root_dir=os.path.join(args.base_folder, f'dl3dv_{tag}'), 
            stage=stage, 
            num_images=args.num_frames,
            target_shape=(args.height, args.width),
            step_tracker=step_tracker,
            plucker_convention=plucker_convention,
            square_crop=square_crop,
        )
    elif dataset == 're10k':  # include gaussian splat
        return Re10KDataset(
            root_dir=os.path.join(args.base_folder, f're10k_{tag}'), 
            stage=stage, 
            num_images=args.num_frames,
            target_shape=(args.height, args.width),
            plucker_convention=plucker_convention,
            input_num=input_num,
            square_crop=square_crop,
        )
    elif dataset == 're10ko':
        return Re10KODataset(
            root_dir=os.path.join(args.base_folder, f're10k_{tag}'), 
            stage=stage, 
            num_images=args.num_frames,
            target_shape=(args.height, args.width),
            step_tracker=step_tracker,
            plucker_convention=plucker_convention,
            square_crop=square_crop,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")