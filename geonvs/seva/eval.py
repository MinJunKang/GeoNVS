import collections
import json
import math
import os
import re
import gc
import time
import pyiqa
import pandas as pd
import threading
from typing import List, Literal, Optional, Tuple, Union
from colorama import Fore, Style, init

init(autoreset=True)

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import repeat
from PIL import Image
from tqdm.auto import tqdm
from einops import rearrange, repeat

from geonvs.seva.geometry import get_camera_dist, get_plucker_coordinates, to_hom_pose
from geonvs.seva.sampling import (
    Discretization,
    EulerEDMSampler,
    MultiviewCFG,
    MultiviewTemporalCFG,
    VanillaCFG,
)
from geonvs.seva.utils import seed_everything
from diffusers.utils.torch_utils import randn_tensor

try:
    # Check if version string contains 'dev' or 'nightly'
    version = torch.__version__
    IS_TORCH_NIGHTLY = "dev" in version
    if IS_TORCH_NIGHTLY:
        torch._dynamo.config.cache_size_limit = 128  # type: ignore[assignment]
        torch._dynamo.config.accumulated_cache_size_limit = 1024  # type: ignore[assignment]
        torch._dynamo.config.force_parameter_static_shapes = False  # type: ignore[assignment]
except Exception:
    IS_TORCH_NIGHTLY = False


# Lazily-created pyiqa metric singletons: created once per process and reused
# across scenes/calls, keyed by (metric_name, device).
_PYIQA_METRICS: dict = {}


def get_pyiqa_metric(name: str, device: str = "cuda:0"):
    key = (name, str(device))
    if key not in _PYIQA_METRICS:
        _PYIQA_METRICS[key] = pyiqa.create_metric(name).to(device)
    return _PYIQA_METRICS[key]


def pad_indices(
    input_indices: List[int],
    test_indices: List[int],
    T: int,
    padding_mode: Literal["first", "last", "none"] = "last",
):
    assert padding_mode in ["last", "none"], "`first` padding is not supported yet."
    if padding_mode == "last":
        padded_indices = [
            i for i in range(T) if i not in (input_indices + test_indices)
        ]
    else:
        padded_indices = []
    input_selects = list(range(len(input_indices)))
    test_selects = list(range(len(test_indices)))
    if max(input_indices) > max(test_indices):
        # last elem from input
        input_selects += [input_selects[-1]] * len(padded_indices)
        input_indices = input_indices + padded_indices
        sorted_inds = np.argsort(input_indices)
        input_indices = [input_indices[ind] for ind in sorted_inds]
        input_selects = [input_selects[ind] for ind in sorted_inds]
    else:
        # last elem from test
        test_selects += [test_selects[-1]] * len(padded_indices)
        test_indices = test_indices + padded_indices
        sorted_inds = np.argsort(test_indices)
        test_indices = [test_indices[ind] for ind in sorted_inds]
        test_selects = [test_selects[ind] for ind in sorted_inds]

    if padding_mode == "last":
        input_maps = np.array([-1] * T)
        test_maps = np.array([-1] * T)
    else:
        input_maps = np.array([-1] * (len(input_indices) + len(test_indices)))
        test_maps = np.array([-1] * (len(input_indices) + len(test_indices)))
    input_maps[input_indices] = input_selects
    test_maps[test_indices] = test_selects
    return input_indices, test_indices, input_maps, test_maps


def assemble(
    input,
    test,
    input_maps,
    test_maps,
):
    T = len(input_maps)
    assembled = torch.zeros_like(test[-1:]).repeat_interleave(T, dim=0)
    assembled[input_maps != -1] = input[input_maps[input_maps != -1]]
    assembled[test_maps != -1] = test[test_maps[test_maps != -1]]
    assert np.logical_xor(input_maps != -1, test_maps != -1).all()
    return assembled


def get_resizing_factor(
    target_shape: Tuple[int, int],  # H, W
    current_shape: Tuple[int, int],  # H, W
    cover_target: bool = True,
    # If True, the output shape will fully cover the target shape.
    # If No, the target shape will fully cover the output shape.
) -> float:
    r_bound = target_shape[1] / target_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if cover_target:
            if aspect_r >= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r < 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r >= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r < 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    else:
        if cover_target:
            if aspect_r <= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r > 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r <= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r > 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    return factor


def get_unique_embedder_keys_from_conditioner(conditioner):
    keys = [x.input_key for x in conditioner.embedders if x.input_key is not None]
    keys = [item for sublist in keys for item in sublist]  # Flatten list
    return set(keys)


def get_wh_with_fixed_shortest_side(w, h, size):
    # size is smaller or equal to zero, we return original w h
    if size is None or size <= 0:
        return w, h
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return new_w, new_h


def load_img_and_K(
    image_path_or_size: Union[str, torch.Size],
    size: Optional[Union[int, Tuple[int, int]]],
    scale: float = 1.0,
    center: Tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | None = None,
    size_stride: int = 1,
    center_crop: bool = False,
    image_as_tensor: bool = True,
    context_rgb: np.ndarray | None = None,
    device: str = "cuda",
):
    if isinstance(image_path_or_size, torch.Size):
        image = Image.new("RGBA", image_path_or_size[::-1])
    else:
        image = Image.open(image_path_or_size).convert("RGBA")

    w, h = image.size
    if size is None:
        size = (w, h)

    image = np.array(image).astype(np.float32) / 255
    if image.shape[-1] == 4:
        rgb, alpha = image[:, :, :3], image[:, :, 3:]
        if context_rgb is not None:
            image = rgb * alpha + context_rgb * (1 - alpha)
        else:
            image = rgb * alpha  # + (1 - alpha)  # default is black background
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32)
    image = image.unsqueeze(0)

    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W, H = (
        math.floor(W / size_stride + 0.5) * size_stride,
        math.floor(H / size_stride + 0.5) * size_stride,
    )

    rfs = get_resizing_factor((math.floor(H * scale), math.floor(W * scale)), (h, w))
    resize_size = rh, rw = [int(np.ceil(rfs * s)) for s in (h, w)]
    image = torch.nn.functional.interpolate(
        image, resize_size, mode="area", antialias=False
    )
    if scale < 1.0:
        pw = math.ceil((W - resize_size[1]) * 0.5)
        ph = math.ceil((H - resize_size[0]) * 0.5)
        image = F.pad(image, (pw, pw, ph, ph), "constant", 1.0)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if center_crop:
        side = min(H, W)
        ct = max(0, cy_center - side // 2)
        cl = max(0, cx_center - side // 2)
        ct = min(ct, image.shape[-2] - side)
        cl = min(cl, image.shape[-1] - side)
        image = TF.crop(image, top=ct, left=cl, height=side, width=side)
    else:
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = TF.crop(image, top=ct, left=cl, height=H, width=W)

    if K is not None:
        K = K.clone()
        if torch.all(K[:2, -1] >= 0) and torch.all(K[:2, -1] <= 1):
            K[:2] *= K.new_tensor([rw, rh])[:, None]  # normalized K
        else:
            K[:2] *= K.new_tensor([rw / w, rh / h])[:, None]  # unnormalized K
        K[:2, 2] -= K.new_tensor([cl, ct])

    if image_as_tensor:
        # tensor of shape (1, 3, H, W) with values ranging from (-1, 1)
        image = image.to(device) * 2.0 - 1.0
    else:
        # PIL Image with values ranging from (0, 255)
        image = image.permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype(np.uint8))
    return image, K


def transform_img_and_K(
    image: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    scale: float = 1.0,
    center: Tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | None = None,
    size_stride: int = 1,
    mode: str = "crop",
):
    assert mode in [
        "crop",
        "pad",
        "stretch",
    ], f"mode should be one of ['crop', 'pad', 'stretch'], got {mode}"

    h, w = image.shape[-2:]
    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W, H = (
        math.floor(W / size_stride + 0.5) * size_stride,
        math.floor(H / size_stride + 0.5) * size_stride,
    )

    if mode == "stretch":
        rh, rw = H, W
    else:
        rfs = get_resizing_factor(
            (H, W),
            (h, w),
            cover_target=mode != "pad",
        )
        (rh, rw) = [int(np.ceil(rfs * s)) for s in (h, w)]

    rh, rw = int(rh / scale), int(rw / scale)
    image = torch.nn.functional.interpolate(
        image, (rh, rw), mode="area", antialias=False
    )

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if mode != "pad":
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = TF.crop(image, top=ct, left=cl, height=H, width=W)
        pl, pt = 0, 0
    else:
        pt = max(0, H // 2 - cy_center)
        pl = max(0, W // 2 - cx_center)
        pb = max(0, H - pt - image.shape[-2])
        pr = max(0, W - pl - image.shape[-1])
        image = TF.pad(
            image,
            [pl, pt, pr, pb],
        )
        cl, ct = 0, 0

    if K is not None:
        K = K.clone()
        # K[:, :2, 2] += K.new_tensor([pl, pt])
        if torch.all(K[:, :2, -1] >= 0) and torch.all(K[:, :2, -1] <= 1):
            K[:, :2] *= K.new_tensor([rw, rh])[None, :, None]  # normalized K
        else:
            K[:, :2] *= K.new_tensor([rw / w, rh / h])[None, :, None]  # unnormalized K
        K[:, :2, 2] += K.new_tensor([pl - cl, pt - ct])

    return image, K


lowvram_mode = False


def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def load_model(model, device: str = "cuda"):
    model.to(device)


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()


def infer_prior_stats(
    T,
    num_input_frames,
    num_total_frames,
    version_dict,
):
    options = version_dict["options"]
    chunk_strategy = options.get("chunk_strategy", "nearest")
    T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
    T_second_pass = T[1] if isinstance(T, (list, tuple)) else T
    # get traj_prior_c2ws for 2-pass sampling
    if chunk_strategy.startswith("interp"):
        # Start and end have alreay taken up two slots
        # +1 means we need X + 1 prior frames to bound X times forwards for all test frames

        # Tuning up `num_prior_frames_ratio` is helpful when you observe sudden jump in the
        # generated frames due to insufficient prior frames. This option is effective for
        # complicated trajectory and when `interp` strategy is used (usually semi-dense-view
        # regime). Recommended range is [1.0 (default), 1.5].
        if num_input_frames >= options.get("num_input_semi_dense", 9):
            num_prior_frames = (
                math.ceil(
                    num_total_frames
                    / (T_second_pass - 2)
                    * options.get("num_prior_frames_ratio", 1.0)
                )
                + 1
            )

            if num_prior_frames + num_input_frames < T_first_pass:
                num_prior_frames = T_first_pass - num_input_frames

            num_prior_frames = max(
                num_prior_frames,
                options.get("num_prior_frames", 0),
            )

            T_first_pass = num_prior_frames + num_input_frames

            if "gt" in chunk_strategy:
                T_second_pass = T_second_pass + num_input_frames

            # Dynamically update context window length.
            version_dict["T"] = [T_first_pass, T_second_pass]

        else:
            num_prior_frames = (
                math.ceil(
                    num_total_frames
                    / (
                        T_second_pass
                        - 2
                        - (num_input_frames if "gt" in chunk_strategy else 0)
                    )
                    * options.get("num_prior_frames_ratio", 1.0)
                )
                + 1
            )

            if num_prior_frames + num_input_frames < T_first_pass:
                num_prior_frames = T_first_pass - num_input_frames

            num_prior_frames = max(
                num_prior_frames,
                options.get("num_prior_frames", 0),
            )
    else:
        num_prior_frames = max(
            T_first_pass - num_input_frames,
            options.get("num_prior_frames", 0),
        )

        if num_input_frames >= options.get("num_input_semi_dense", 9):
            T_first_pass = num_prior_frames + num_input_frames

            # Dynamically update context window length.
            version_dict["T"] = [T_first_pass, T_second_pass]

    return num_prior_frames


def infer_prior_inds(
    c2ws,
    num_prior_frames,
    input_frame_indices,
    options,
):
    chunk_strategy = options.get("chunk_strategy", "nearest")
    if chunk_strategy.startswith("interp"):
        prior_frame_indices = np.array(
            [i for i in range(c2ws.shape[0]) if i not in input_frame_indices]
        )
        prior_frame_indices = prior_frame_indices[
            np.ceil(
                np.linspace(
                    0, prior_frame_indices.shape[0] - 1, num_prior_frames, endpoint=True
                )
            ).astype(int)
        ]  # having a ceil here is actually safer for corner case
    else:
        prior_frame_indices = []
        while len(prior_frame_indices) < num_prior_frames:
            closest_distance = np.abs(
                np.arange(c2ws.shape[0])[None]
                - np.concatenate(
                    [np.array(input_frame_indices), np.array(prior_frame_indices)]
                )[:, None]
            ).min(0)
            prior_frame_indices.append(np.argsort(closest_distance)[-1])
    return np.sort(prior_frame_indices)


def compute_relative_inds(
    source_inds,
    target_inds,
):
    assert len(source_inds) > 2
    # compute relative indices of target_inds within source_inds
    relative_inds = []
    for ind in target_inds:
        if ind in source_inds:
            relative_ind = int(np.where(source_inds == ind)[0][0])
        elif ind < source_inds[0]:
            # extrapolate
            relative_ind = -((source_inds[0] - ind) / (source_inds[1] - source_inds[0]))
        elif ind > source_inds[-1]:
            # extrapolate
            relative_ind = len(source_inds) + (
                (ind - source_inds[-1]) / (source_inds[-1] - source_inds[-2])
            )
        else:
            # interpolate
            lower_inds = source_inds[source_inds < ind]
            upper_inds = source_inds[source_inds > ind]
            if len(lower_inds) > 0 and len(upper_inds) > 0:
                lower_ind = lower_inds[-1]
                upper_ind = upper_inds[0]
                relative_lower_ind = int(np.where(source_inds == lower_ind)[0][0])
                relative_upper_ind = int(np.where(source_inds == upper_ind)[0][0])
                relative_ind = relative_lower_ind + (ind - lower_ind) / (
                    upper_ind - lower_ind
                ) * (relative_upper_ind - relative_lower_ind)
            else:
                # Out of range
                relative_inds.append(float("nan"))  # Or some other placeholder
        relative_inds.append(relative_ind)
    return relative_inds


def find_nearest_source_inds(
    source_c2ws,
    target_c2ws,
    nearest_num=1,
    mode="translation",
):
    dists = get_camera_dist(source_c2ws, target_c2ws, mode=mode).cpu().numpy()
    sorted_inds = np.argsort(dists, axis=0).T
    return sorted_inds[:, :nearest_num]


def chunk_input_and_test(
    T,
    input_c2ws,
    test_c2ws,
    input_ords,  # orders
    test_ords,  # orders
    options,
    task: str = "img2img",
    chunk_strategy: str = "gt",
    gt_input_inds: list = [],
):
    M, N = input_c2ws.shape[0], test_c2ws.shape[0]

    chunks = []
    if chunk_strategy.startswith("gt"):
        assert len(gt_input_inds) < T, (
            f"Number of gt input frames {len(gt_input_inds)} should be "
            f"less than {T} when `gt` chunking strategy is used."
        )
        assert (
            list(range(M)) == gt_input_inds
        ), "All input_c2ws should be gt when `gt` chunking strategy is used."

        num_test_seen = 0
        while num_test_seen < N:
            chunk = [f"!{i:03d}" for i in gt_input_inds]
            if chunk_strategy != "gt" and num_test_seen > 0:
                pseudo_num_ratio = options.get("pseudo_num_ratio", 0.33)
                if (N - num_test_seen) >= math.floor(
                    (T - len(gt_input_inds)) * pseudo_num_ratio
                ):
                    pseudo_num = math.ceil((T - len(gt_input_inds)) * pseudo_num_ratio)
                else:
                    pseudo_num = (T - len(gt_input_inds)) - (N - num_test_seen)
                pseudo_num = min(pseudo_num, options.get("pseudo_num_max", 10000))

                if "ltr" in chunk_strategy:
                    chunk.extend(
                        [
                            f"!{i + len(gt_input_inds):03d}"
                            for i in range(num_test_seen - pseudo_num, num_test_seen)
                        ]
                    )
                elif "nearest" in chunk_strategy:
                    source_inds = np.concatenate(
                        [
                            find_nearest_source_inds(
                                test_c2ws[:num_test_seen],
                                test_c2ws[num_test_seen:],
                                nearest_num=1,  # pseudo_num,
                                mode="rotation",
                            ),
                            find_nearest_source_inds(
                                test_c2ws[:num_test_seen],
                                test_c2ws[num_test_seen:],
                                nearest_num=1,  # pseudo_num,
                                mode="translation",
                            ),
                        ],
                        axis=1,
                    )
                    ####### [HACK ALERT] keep running until pseudo num is stablized ########
                    temp_pseudo_num = pseudo_num
                    while True:
                        nearest_source_inds = np.concatenate(
                            [
                                np.sort(
                                    [
                                        ind
                                        for (ind, _) in collections.Counter(
                                            [
                                                item
                                                for item in source_inds[
                                                    : T
                                                    - len(gt_input_inds)
                                                    - temp_pseudo_num
                                                ]
                                                .flatten()
                                                .tolist()
                                                if item
                                                != (
                                                    num_test_seen - 1
                                                )  # exclude the last one here
                                            ]
                                        ).most_common(pseudo_num - 1)
                                    ],
                                ).astype(int),
                                [num_test_seen - 1],  # always keep the last one
                            ]
                        )
                        if len(nearest_source_inds) >= temp_pseudo_num:
                            break  # stablized
                        else:
                            temp_pseudo_num = len(nearest_source_inds)
                    pseudo_num = len(nearest_source_inds)
                    ########################################################################
                    chunk.extend(
                        [f"!{i + len(gt_input_inds):03d}" for i in nearest_source_inds]
                    )
                else:
                    raise NotImplementedError(
                        f"Chunking strategy {chunk_strategy} for the first pass is not implemented."
                    )

                chunk.extend(
                    [
                        f">{i:03d}"
                        for i in range(
                            num_test_seen,
                            min(num_test_seen + T - len(gt_input_inds) - pseudo_num, N),
                        )
                    ]
                )
            else:
                chunk.extend(
                    [
                        f">{i:03d}"
                        for i in range(
                            num_test_seen,
                            min(num_test_seen + T - len(gt_input_inds), N),
                        )
                    ]
                )

            num_test_seen += sum([1 for c in chunk if c.startswith(">")])
            if len(chunk) < T:
                chunk.extend(["NULL"] * (T - len(chunk)))
            chunks.append(chunk)

    elif chunk_strategy.startswith("nearest"):
        input_imgs = np.array([f"!{i:03d}" for i in range(M)])
        test_imgs = np.array([f">{i:03d}" for i in range(N)])

        match = re.match(r"^nearest-(\d+)$", chunk_strategy)
        if match:
            nearest_num = int(match.group(1))
            assert (
                nearest_num < T
            ), f"Nearest number of {nearest_num} should be less than {T}."
            source_inds = find_nearest_source_inds(
                input_c2ws,
                test_c2ws,
                nearest_num=nearest_num,
                mode="translation",  # during the second pass, consider translation only is enough
            )

            for i in range(0, N, T - nearest_num):
                nearest_source_inds = np.sort(
                    [
                        ind
                        for (ind, _) in collections.Counter(
                            source_inds[i : i + T - nearest_num].flatten().tolist()
                        ).most_common(nearest_num)
                    ]
                )
                chunk = (
                    input_imgs[nearest_source_inds].tolist()
                    + test_imgs[i : i + T - nearest_num].tolist()
                )
                chunks.append(chunk + ["NULL"] * (T - len(chunk)))

        else:
            # do not always condition on gt cond frames
            if "gt" not in chunk_strategy:
                gt_input_inds = []

            source_inds = find_nearest_source_inds(
                input_c2ws,
                test_c2ws,
                nearest_num=1,
                mode="translation",  # during the second pass, consider translation only is enough
            )[:, 0]

            test_inds_per_input = {}
            for test_idx, input_idx in enumerate(source_inds):
                if input_idx not in test_inds_per_input:
                    test_inds_per_input[input_idx] = []
                test_inds_per_input[input_idx].append(test_idx)

            num_test_seen = 0
            chunk = input_imgs[gt_input_inds].tolist()
            candidate_input_inds = sorted(list(test_inds_per_input.keys()))

            while num_test_seen < N:
                input_idx = candidate_input_inds[0]
                test_inds = test_inds_per_input[input_idx]
                input_is_cond = input_idx in gt_input_inds
                prefix_inds = [] if input_is_cond else [input_idx]
                
                if len(chunk) == T - len(prefix_inds) or not candidate_input_inds:
                    if chunk:
                        chunk += ["NULL"] * (T - len(chunk))
                        chunks.append(chunk)
                        chunk = input_imgs[gt_input_inds].tolist()
                    if num_test_seen >= N:
                        break
                    continue

                candidate_chunk = (
                    input_imgs[prefix_inds].tolist() + test_imgs[test_inds].tolist()
                )

                space_left = T - len(chunk)
                if len(candidate_chunk) <= space_left:
                    chunk.extend(candidate_chunk)
                    num_test_seen += len(test_inds)
                    candidate_input_inds.pop(0)
                else:
                    chunk.extend(candidate_chunk[:space_left])
                    num_input_idx = 0 if input_is_cond else 1
                    num_test_seen += space_left - num_input_idx
                    test_inds_per_input[input_idx] = test_inds[
                        space_left - num_input_idx :
                    ]

                if len(chunk) == T:
                    chunks.append(chunk)
                    chunk = input_imgs[gt_input_inds].tolist()

            if chunk and chunk != input_imgs[gt_input_inds].tolist():
                chunks.append(chunk + ["NULL"] * (T - len(chunk)))

    elif chunk_strategy.startswith("interp"):
        # `interp` chunk requires ordering info
        assert input_ords is not None and test_ords is not None, (
            "When using `interp` chunking strategy, ordering of input "
            "and test frames should be provided."
        )

        # if chunk_strategy is `interp*`` and task is `img2trajvid*`, we will not
        # use input views since their order info within target views is unknown
        if "img2trajvid" in task:
            assert (
                list(range(len(gt_input_inds))) == gt_input_inds
            ), "`img2trajvid` task should put `gt_input_inds` in start."
            input_c2ws = input_c2ws[
                [ind for ind in range(M) if ind not in gt_input_inds]
            ]
            input_ords = [
                input_ords[ind] for ind in range(M) if ind not in gt_input_inds
            ]
            M = input_c2ws.shape[0]

        input_ords = [0] + input_ords  # this is a  hack accounting for test views
        # before the first input view
        input_ords[-1] += 0.01  # this is a hack ensuring last test stop is included
        # in the last forward when input_ords[-1] == test_ords[-1]
        input_ords = np.array(input_ords)[:, None]
        input_ords_ = np.concatenate([input_ords[1:], np.full((1, 1), np.inf)])
        test_ords = np.array(test_ords)[None]

        in_stop_ranges = np.logical_and(
            np.repeat(input_ords, N, axis=1) <= np.repeat(test_ords, M + 1, axis=0),
            np.repeat(input_ords_, N, axis=1) > np.repeat(test_ords, M + 1, axis=0),
        )  # (M, N)
        assert (in_stop_ranges.sum(1) <= T - 2).all(), (
            "More anchor frames need to be sampled during the first pass to ensure "
            f"#target frames during each forward in the second pass will not exceed {T - 2}."
        )
        if input_ords[1, 0] <= test_ords[0, 0]:
            assert not in_stop_ranges[0].any()
        if input_ords[-1, 0] >= test_ords[0, -1]:
            assert not in_stop_ranges[-1].any()

        gt_chunk = (
            [f"!{i:03d}" for i in gt_input_inds] if "gt" in chunk_strategy else []
        )
        chunk = gt_chunk + []
        # any test views before the first input views
        if in_stop_ranges[0].any():
            for j, in_range in enumerate(in_stop_ranges[0]):
                if in_range:
                    chunk.append(f">{j:03d}")
        in_stop_ranges = in_stop_ranges[1:]

        i = 0
        base_i = len(gt_input_inds) if "img2trajvid" in task else 0
        chunk.append(f"!{i + base_i:03d}")
        while i < len(in_stop_ranges):
            in_stop_range = in_stop_ranges[i]
            if not in_stop_range.any():
                i += 1
                continue

            input_left = i + 1 < M
            space_left = T - len(chunk)
            if sum(in_stop_range) + input_left <= space_left:
                for j, in_range in enumerate(in_stop_range):
                    if in_range:
                        chunk.append(f">{j:03d}")
                i += 1
                if input_left:
                    chunk.append(f"!{i + base_i:03d}")

            else:
                chunk += ["NULL"] * space_left
                chunks.append(chunk)
                chunk = gt_chunk + [f"!{i + base_i:03d}"]

        if len(chunk) > 1:
            chunk += ["NULL"] * (T - len(chunk))
            chunks.append(chunk)

    else:
        raise NotImplementedError

    (
        input_inds_per_chunk,
        input_sels_per_chunk,
        test_inds_per_chunk,
        test_sels_per_chunk,
    ) = (
        [],
        [],
        [],
        [],
    )
    for chunk in chunks:
        input_inds = [
            int(img.removeprefix("!")) for img in chunk if img.startswith("!")
        ]
        input_sels = [chunk.index(img) for img in chunk if img.startswith("!")]
        test_inds = [int(img.removeprefix(">")) for img in chunk if img.startswith(">")]
        test_sels = [chunk.index(img) for img in chunk if img.startswith(">")]
        input_inds_per_chunk.append(input_inds)
        input_sels_per_chunk.append(input_sels)
        test_inds_per_chunk.append(test_inds)
        test_sels_per_chunk.append(test_sels)

    if options.get("sampler_verbose", True):

        def colorize(item):
            if item.startswith("!"):
                return f"{Fore.RED}{item}{Style.RESET_ALL}"  # Red for items starting with '!'
            elif item.startswith(">"):
                return f"{Fore.GREEN}{item}{Style.RESET_ALL}"  # Green for items starting with '>'
            return item  # Default color if neither '!' nor '>'

        print("\nchunks:")
        for chunk in chunks:
            print(", ".join(colorize(item) for item in chunk))

    return (
        chunks,
        input_inds_per_chunk,  # ordering of input in raw sequence
        input_sels_per_chunk,  # ordering of input in one-forward sequence of length T
        test_inds_per_chunk,  # ordering of test in raw sequence
        test_sels_per_chunk,  # oredering of test in one-forward sequence of length T
    )


def is_k_in_dict(d, k):
    return any(map(lambda x: x.startswith(k), d.keys()))


def get_k_from_dict(d, k):
    media_d = {}
    for key, value in d.items():
        if key == k:
            return value
        if key.startswith(k):
            media = key.split("/")[-1]
            if media == "raw":
                return value
            media_d[media] = value
    if len(media_d) == 0:
        return torch.tensor([])
    assert (
        len(media_d) == 1
    ), f"multiple media found in {d} for key {k}: {media_d.keys()}"
    return media_d[media]


def update_kv_for_dict(d, k, v):
    for key in d.keys():
        if key.startswith(k):
            d[key] = v
    return d


def extend_dict(ds, d):
    for key in d.keys():
        if key in ds:
            ds[key] = torch.cat([ds[key], d[key]], 0)
        else:
            ds[key] = d[key]
    return ds


def replace_or_include_input_for_dict(
    samples,
    test_indices,
    imgs,
    c2w,
    K,
):
    samples_new = {}
    for sample, value in samples.items():
        imgs_ = imgs.clone()
        c2w_ = c2w.clone()
        K_ = K.clone()
        if "rgb" in sample:
            imgs_[test_indices] = (
                value[test_indices] if value.shape[0] == imgs_.shape[0] else value
            ).to(device=imgs_.device, dtype=imgs_.dtype)
            samples_new[sample] = imgs_
        elif "c2w" in sample:
            c2w_[test_indices] = (
                value[test_indices] if value.shape[0] == c2w_.shape[0] else value
            ).to(device=c2w_.device, dtype=c2w_.dtype)
            samples_new[sample] = c2w_
        elif "intrinsics" in sample:
            K_[test_indices] = (
                value[test_indices] if value.shape[0] == K_.shape[0] else value
            ).to(device=K_.device, dtype=K_.dtype)
            samples_new[sample] = K_
        else:
            samples_new[sample] = value
    return samples_new


def decode_output(
    samples,
    T,
    indices=None,
):
    # decode model output into dict if it is not
    if isinstance(samples, dict):
        # model with postprocessor and outputs dict
        for sample, value in samples.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            elif isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            else:
                value = torch.tensor(value)

            if indices is not None and value.shape[0] == T:
                value = value[indices]
            samples[sample] = value
    else:
        # model without postprocessor and outputs tensor (rgb)
        samples = samples.detach().cpu()

        if indices is not None and samples.shape[0] == T:
            samples = samples[indices]
        samples = {"samples-rgb/image": samples}

    return samples


def save_output(
    samples,
    save_path,
    video_save_fps=2,
):
    os.makedirs(save_path, exist_ok=True)
    for sample in samples:
        media_type = "video"
        if "/" in sample:
            sample_, media_type = sample.split("/")
        else:
            sample_ = sample

        value = samples[sample]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        else:
            value = torch.tensor(value)

        if media_type == "image":
            value = (value.permute(0, 2, 3, 1) + 1) / 2.0
            value = (value * 255).clamp(0, 255).to(torch.uint8)
            iio.imwrite(
                os.path.join(save_path, f"{sample_}.mp4")
                if sample_
                else f"{save_path}.mp4",
                value,
                fps=video_save_fps,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
            os.makedirs(os.path.join(save_path, sample_), exist_ok=True)
            for i, s in enumerate(value):
                iio.imwrite(
                    os.path.join(save_path, sample_, f"{i:03d}.png"),
                    s,
                )
        elif media_type == "video":
            value = (value.permute(0, 2, 3, 1) + 1) / 2.0
            value = (value * 255).clamp(0, 255).to(torch.uint8)
            iio.imwrite(
                os.path.join(save_path, f"{sample_}.mp4"),
                value,
                fps=video_save_fps,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
        elif media_type == "raw":
            torch.save(
                value,
                os.path.join(save_path, f"{sample_}.pt"),
            )
        else:
            pass


def create_transforms_simple(save_path, img_paths, img_whs, c2ws, Ks):
    import os.path as osp

    out_frames = []
    for img_path, img_wh, c2w, K in zip(img_paths, img_whs, c2ws, Ks):
        out_frame = {
            "fl_x": K[0][0].item(),
            "fl_y": K[1][1].item(),
            "cx": K[0][2].item(),
            "cy": K[1][2].item(),
            "w": img_wh[0].item(),
            "h": img_wh[1].item(),
            "file_path": f"./{osp.relpath(img_path, start=save_path)}"
            if img_path is not None
            else None,
            "transform_matrix": c2w.tolist(),
        }
        out_frames.append(out_frame)
    out = {
        # "camera_model": "PINHOLE",
        "orientation_override": "none",
        "frames": out_frames,
    }
    with open(osp.join(save_path, "transforms.json"), "w") as of:
        json.dump(out, of, indent=5)


def create_samplers(
    guider_types: int | list[int],
    discretization: Discretization,
    num_frames: list[int] | None,
    num_steps: int,
    cfg_min: float = 1.0,
    device: str | torch.device = "cuda",
    abort_event: threading.Event | None = None,
):
    guider_mapping = {
        0: VanillaCFG,
        1: MultiviewCFG,
        2: MultiviewTemporalCFG,
    }
    samplers = []
    if not isinstance(guider_types, (list, tuple)):
        guider_types = [guider_types]
    for i, guider_type in enumerate(guider_types):
        if guider_type not in guider_mapping:
            raise ValueError(
                f"Invalid guider type {guider_type}. Must be one of {list(guider_mapping.keys())}"
            )
        guider_args = ()
        if guider_type > 0:
            guider_args += (cfg_min,)
            if guider_type == 2:
                assert num_frames is not None
                guider_args = (num_frames[i], cfg_min)
        guider = guider_mapping[guider_type](*guider_args)
        sampler = EulerEDMSampler(
            abort_event=abort_event,
            discretization=discretization,
            guider=guider,
            num_steps=num_steps,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=999.0,
            s_noise=1.0,
            verbose=True,
            device=device,
        )
        samplers.append(sampler)
    return samplers


def get_value_dict(
    curr_imgs,
    curr_input_frame_indices,
    curr_input_frame_indices_nopad,
    curr_c2ws,
    curr_Ks,
    curr_input_camera_indices,
    all_c2ws,
    camera_scale,
):
    assert sorted(curr_input_camera_indices) == sorted(
        range(len(curr_input_camera_indices))
    )
    H, W, T, F = curr_imgs.shape[-2], curr_imgs.shape[-1], len(curr_imgs), 8

    value_dict = {}
    value_dict["cond_frames"] = curr_imgs + 0.0 * torch.randn_like(curr_imgs)
    value_dict["cond_frames_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["cond_frames_mask"][curr_input_frame_indices] = True
    value_dict["input_frames_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["input_frames_mask"][curr_input_frame_indices_nopad] = True
    value_dict["cond_aug"] = 0.0

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
    w2c = torch.linalg.inv(c2w)

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
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor
    value_dict["plucker_coordinate"] = get_plucker_coordinates(
        extrinsics_src=w2c[0],
        extrinsics=w2c,
        intrinsics=curr_Ks.float().clone(),
        # GeoNVS's get_plucker_coordinates (convention='seva') divides
        # target_size by donwsample_factor=8 internally, matching SVC's
        # original target_size=(H // F, W // F) with F=8.
        target_size=(H, W),
    )

    value_dict["c2w"] = c2w
    value_dict["c2w_raw"] = to_hom_pose(curr_c2ws.float())  # un-normalized
    value_dict["K"] = curr_Ks
    value_dict["camera_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["camera_mask"][curr_input_camera_indices] = True

    return value_dict


def normalize_cam(
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
    
    return c2w
    


def run_lrm_model(
    model,
    H,
    W,
    input_imgs,
    input_Ks,
    input_c2ws,
    all_c2ws,
    camera_scale,
    scene_config_path,
    downsample_factor,
    device="cuda",
    **options,
):
    lrm_outputs = None
    if model is not None:
        
        # use normalized c2w for lrm model
        if options.get("traj_prior", None) == "orbit" or options.get("normalize_camera", False):
            c2ws = normalize_cam(
                curr_c2ws=input_c2ws,
                all_c2ws=all_c2ws,
                camera_scale=camera_scale
            )
        else:
            c2ws = to_hom_pose(input_c2ws.float())
        intrinsics = input_Ks.to(device)
        intrinsics[:, 0] *= W
        intrinsics[:, 1] *= H
            
        load_model(model)
        batch_input = {
            'original_size': (H, W),
            'voxel_size': options.get('lrm_voxel_size', -1.0),
            'downsample_factor': downsample_factor,
            'pixel_values': input_imgs[None].to(device),
            'intrinsics': intrinsics[None],
            'extrinsics': c2ws[None].to(device),
            'scene_config_path': scene_config_path,
        }
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        gaussians, near, far = model.forward_geometry(batch_input, device)
        unload_model(model)
        
        lrm_outputs = {
            "gaussians": gaussians,
            "near": near,
            "far": far,
        }
        
    return lrm_outputs



def do_sample(
    model,
    ae,
    conditioner,
    denoiser,
    sampler,
    value_dict,
    H,
    W,
    C,
    F,
    T,
    cfg,
    scene_config_path,
    downsample_factor,
    encoding_t=1,
    decoding_t=1,
    verbose=True,
    device="cuda",
    global_pbar=None,
    generator=None,
    **options,
):
    # memory overhead measurement before sampling
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        # Reset the peak memory stats before measurement starts.
        torch.cuda.reset_peak_memory_stats()
        # Empty the cache to report a stricter pure-allocation peak.
        torch.cuda.empty_cache()
    
    
    imgs = value_dict["cond_frames"].to(device=device)
    input_masks = value_dict["cond_frames_mask"].to(device=device)
    pluckers = value_dict["plucker_coordinate"].to(device=device)
    if options.get("traj_prior", None) == "orbit":
        c2w = value_dict["c2w"].to(device=device)
    else:
        c2w = value_dict["c2w_raw"].to(device=device)
    intrinsics = value_dict["K"].to(device=device)
    intrinsics[:, 0] *= W
    intrinsics[:, 1] *= H
    extrinsics, intrinsics = c2w[None], intrinsics[None]
    num_viewpoints = imgs.shape[0]

    # if need, get lrm model computation to get gaussian splats
    if value_dict.get("lrm_outputs", None) is not None:
        gaussians = value_dict["lrm_outputs"]["gaussians"]
        near = repeat(value_dict["lrm_outputs"]["near"], "b -> b v", v=num_viewpoints)
        far = repeat(value_dict["lrm_outputs"]["far"], "b -> b v", v=num_viewpoints)
        scale_invariant = model.lrm_model.scale_invariant
        use_e3nn = model.lrm_model.use_e3nn
    else:
        if model.lrm_model is not None:
            inputv_masks = value_dict["input_frames_mask"].to(device)
            load_model(model.lrm_model)
            batch_input = {
                'original_size': (H, W),
                'voxel_size': options.get('lrm_voxel_size', -1.0),
                'downsample_factor': downsample_factor,
                'pixel_values': imgs[inputv_masks][None],
                'intrinsics': intrinsics[:, inputv_masks],
                'extrinsics': extrinsics[:, inputv_masks],
                'scene_config_path': scene_config_path,
            }
            gaussians, near, far = model.lrm_model.forward_geometry(batch_input, device)
            unload_model(model.lrm_model)
            near = repeat(near, "b -> b v", v=num_viewpoints)
            far = repeat(far, "b -> b v", v=num_viewpoints)
            scale_invariant = model.lrm_model.scale_invariant
            use_e3nn = model.lrm_model.use_e3nn
        else:
            scale_invariant, use_e3nn = False, False
            gaussians, near, far = None, None, None

    num_samples = [1, T]
    with torch.inference_mode(), torch.autocast(device):
        
        load_model(ae)
        load_model(conditioner)
        latents = torch.nn.functional.pad(
            ae.encode(imgs[input_masks], encoding_t), (0, 0, 0, 0, 0, 1), value=1.0
        )
        c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
        uc_crossattn = torch.zeros_like(c_crossattn)
        c_replace = latents.new_zeros(T, *latents.shape[1:])
        c_replace[input_masks] = latents
        uc_replace = torch.zeros_like(c_replace)
        c_concat = torch.cat(
            [
                repeat(
                    input_masks,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                pluckers,
            ],
            1,
        )
        uc_concat = torch.cat(
            [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
        )
        c_dense_vector = pluckers
        uc_dense_vector = c_dense_vector
            
        c = {
            "mask": input_masks,
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": c_dense_vector,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "near": near,
            "far": far,
            "gaussians": gaussians,
        }
        
        uc = {
            "mask": input_masks,
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": uc_dense_vector,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "near": near,
            "far": far,
            "gaussians": gaussians,
        }

        additional_model_inputs = {
            "num_frames": T, "scale_invariant": scale_invariant, "e3nn": use_e3nn
        }
        additional_sampler_inputs = {
            "c2w": value_dict["c2w"].to(device),
            "K": value_dict["K"].to(device),
            "input_frame_mask": input_masks,
        }
        if global_pbar is not None:
            additional_sampler_inputs["global_pbar"] = global_pbar
        additional_model_inputs["use_gs_adapter"] = model.module.use_gs_adapter

        # initial noise here
        shape = (math.prod(num_samples), C, H // F, W // F)
        randn = randn_tensor(shape, generator=generator, device=imgs.device, dtype=imgs.dtype)
        
        # noisy image inpainting (ablation study)
        if options.get('inpainting_mode', False):
            
            # rendering noisy novel-views
            batch_input = {
                "original_size": (H, W),
                "gaussians": gaussians,
                "near": near[:, ~input_masks],
                "far": far[:, ~input_masks],
                "extrinsics": extrinsics[:, ~input_masks],
                "intrinsics": intrinsics[:, ~input_masks],
            }
            rendered_outputs = model.lrm_model.forward_render(batch_input, device=device)
            rendered_latents = ae.encode(rendered_outputs['rendered_images'] * 2.0 - 1.0, encoding_t)
            start_strength = options.get('inpainting_strength', 0.6)
        else:
            rendered_latents = None
            start_strength = 1.0
            
        unload_model(ae)
        unload_model(conditioner)
        load_model(model)
        
        start_time = time.perf_counter()
        samples_z, samples_feat = sampler(
            lambda input, sigma, c, log_int_features: denoiser(
                model,
                input,
                sigma,
                c,
                log_int_features,
                **additional_model_inputs,
            ),
            randn,
            scale=cfg,
            cond=c,
            uc=uc,
            step_int_features=[],
            verbose=verbose,
            generator=generator,
            initial_latents=rendered_latents,
            start_strength=start_strength,
            **additional_sampler_inputs,
        )
        end_time = time.perf_counter()
        runtime = (end_time - start_time) / len(samples_z)
        
        # Measure peak memory at the end of sampling.
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated(device)

            # Convert bytes to megabytes.
            peak_allocated_mb = peak_allocated / (1024 ** 2)

        # log runtime in seconds
        if 'runtime_log_path' in options:
            with open(options['runtime_log_path'], 'a') as f:
                f.write(f"{runtime}\n")

        if 'memory_log_path' in options:
            with open(options['memory_log_path'], 'a') as f:
                f.write(f"{peak_allocated_mb}\n")

        if samples_z is None:
            return
        unload_model(model)

        load_model(ae)
        samples = ae.decode(samples_z, decoding_t)  # (N, 3, H, W)
        unload_model(ae)

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    return samples


def do_sample_regression(
    model,
    ae,
    conditioner,
    denoiser,
    sampler,
    value_dict,
    H,
    W,
    C,
    F,
    T,
    cfg,
    scene_config_path,
    downsample_factor,
    encoding_t=1,
    decoding_t=1,
    verbose=True,
    device="cuda",
    global_pbar=None,
    generator=None,
    **options,
):
    imgs = value_dict["cond_frames"].to(device)
    input_masks = value_dict["input_frames_mask"].to(device)
    if options.get("traj_prior", None) == "orbit":
        c2w = value_dict["c2w"].to(device)
    else:
        c2w = value_dict["c2w_raw"].to(device)
    intrinsics = value_dict["K"].to(device)
    intrinsics[:, 0] *= W
    intrinsics[:, 1] *= H
    extrinsics, intrinsics = c2w[None], intrinsics[None]
    num_viewpoints = imgs.shape[0]
    
    # if need, get lrm model computation to get gaussian splats
    if value_dict.get("lrm_outputs", None) is not None:
        gaussians = value_dict["lrm_outputs"]["gaussians"]
        near = repeat(value_dict["lrm_outputs"]["near"], "b -> b v", v=num_viewpoints)
        far = repeat(value_dict["lrm_outputs"]["far"], "b -> b v", v=num_viewpoints)
    else:
        load_model(model)
        batch_input = {
            'original_size': (H, W),
            'voxel_size': options.get('lrm_voxel_size', -1.0),
            'downsample_factor': downsample_factor,
            'pixel_values': imgs[input_masks][None],
            'intrinsics': intrinsics[:, input_masks],
            'extrinsics': extrinsics[:, input_masks],
            'scene_config_path': scene_config_path,
        }
        gaussians, near, far = model.forward_geometry(batch_input, device)
        unload_model(model)
        near = repeat(near, "b -> b v", v=num_viewpoints)
        far = repeat(far, "b -> b v", v=num_viewpoints)
    
    # rendering
    batch = {
        "original_size": (H, W),
        "gaussians": gaussians,
        "near": near,
        "far": far,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }
    samples = model.forward_render(batch, device=device)["rendered_images"]  # should be range in [0, 1]
    samples = samples * 2.0 - 1.0  # to [-1, 1]

    return samples


def do_sample_diffusion(
    model,
    ae,
    conditioner,
    denoiser,
    sampler,
    value_dict,
    H,
    W,
    C,
    F,
    T,
    cfg,
    scene_config_path,
    downsample_factor,
    encoding_t=1,
    decoding_t=1,
    verbose=True,
    device="cuda",
    global_pbar=None,
    generator=None,
    **options,
):
    imgs = value_dict["cond_frames"].to(device=device)
    input_masks = value_dict["cond_frames_mask"].to(device=device)
    pluckers = value_dict["plucker_coordinate"].to(device=device)
    if options.get("traj_prior", None) == "orbit":
        c2w = value_dict["c2w"].to(device=device)
    else:
        c2w = value_dict["c2w_raw"].to(device=device)
    intrinsics = value_dict["K"].to(device=device)
    intrinsics[:, 0] *= W
    intrinsics[:, 1] *= H
    extrinsics, intrinsics = c2w[None], intrinsics[None]
    num_viewpoints = imgs.shape[0]

    # if need, get lrm model computation to get gaussian splats
    if value_dict.get("lrm_outputs", None) is not None:
        gaussians = value_dict["lrm_outputs"]["gaussians"]
        near = repeat(value_dict["lrm_outputs"]["near"], "b -> b v", v=num_viewpoints)
        far = repeat(value_dict["lrm_outputs"]["far"], "b -> b v", v=num_viewpoints)
    else:
        if model.lrm_model is not None:
            inputv_masks = value_dict["input_frames_mask"].to(device)
            load_model(model.lrm_model)
            batch_input = {
                'original_size': (H, W),
                'voxel_size': options.get('lrm_voxel_size', -1.0),
                'downsample_factor': downsample_factor,
                'pixel_values': imgs[inputv_masks][None],
                'intrinsics': intrinsics[:, inputv_masks],
                'extrinsics': extrinsics[:, inputv_masks],
                'scene_config_path': scene_config_path,
            }
            gaussians, near, far = model.lrm_model.forward_geometry(batch_input, device)
            unload_model(model.lrm_model)
            near = repeat(near, "b -> b v", v=num_viewpoints)
            far = repeat(far, "b -> b v", v=num_viewpoints)
        else:
            gaussians, near, far = None, None, None
            
    batch_input = {
        "original_size": (H, W),
        "mask": input_masks,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "input_images": imgs[input_masks],
        "images_all": imgs,
    }
    if gaussians is not None:
        # rendering
        batch = {
            "original_size": (H, W),
            "gaussians": gaussians,
            "near": near,
            "far": far,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }
        batch_input["rendered_results"] = model.lrm_model.forward_render(batch, device=device)
            
    load_model(model.module)
    samples = model.module(batch_input, device=device)  # should be range in [0, 1]
    samples = samples * 2.0 - 1.0  # to [-1, 1]
    unload_model(model.module)
        
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    return samples


def run_one_scene(
    task,
    scene_path,
    version_dict,
    model,
    ae,
    conditioner,
    denoiser,
    image_cond,
    camera_cond,
    save_path,
    use_traj_prior,
    traj_prior_Ks,
    traj_prior_c2ws,
    seed=23,
    abort_event=None,
    first_pass_pbar=None,
    second_pass_pbar=None,
    model_type='seva',
):
    H, W, T, C, F, options = (
        version_dict["H"],
        version_dict["W"],
        version_dict["T"],
        version_dict["C"],
        version_dict["f"],
        version_dict["options"],
    )
    
    if model_type == 'seva':
        sampler = do_sample
    elif model_type == 'regression':
        sampler = do_sample_regression
    elif model_type == 'diffusion':
        sampler = do_sample_diffusion
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    vconfig_name = f"train_test_split_{len(image_cond['input_indices'])}.json"
    scene_config_path = os.path.join(scene_path, vconfig_name)

    if isinstance(image_cond, str):
        image_cond = {"img": [image_cond]}
    imgs, img_size = [], None
    for i, (img, K) in enumerate(zip(image_cond["img"], camera_cond["K"])):
        if isinstance(img, str) or img is None:
            img, K = load_img_and_K(img or img_size, None, K=K, device="cpu")  # type: ignore
            img_size = img.shape[-2:]
            if options.get("L_short", -1) == -1:
                img, K = transform_img_and_K(
                    img,
                    (W, H),
                    K=K[None],
                    mode=(
                        options.get("transform_input", "crop")
                        if i in image_cond["input_indices"]
                        else options.get("transform_target", "crop")
                    ),
                    scale=(
                        1.0
                        if i in image_cond["input_indices"]
                        else options.get("transform_scale", 1.0)
                    ),
                )
            else:
                downsample = 3
                assert options["L_short"] % F * 2**downsample == 0, (
                    "Short side of the image should be divisible by "
                    f"F*2**{downsample}={F * 2**downsample}."
                )
                img, K = transform_img_and_K(
                    img,
                    options["L_short"],
                    K=K[None],
                    size_stride=F * 2**downsample,
                    mode=(
                        options.get("transform_input", "crop")
                        if i in image_cond["input_indices"]
                        else options.get("transform_target", "crop")
                    ),
                    scale=(
                        1.0
                        if i in image_cond["input_indices"]
                        else options.get("transform_scale", 1.0)
                    ),
                )
                version_dict["W"] = W = img.shape[-1]
                version_dict["H"] = H = img.shape[-2]
            K = K[0]
            K[0] /= W
            K[1] /= H
            camera_cond["K"][i] = K
        elif isinstance(img, np.ndarray):
            img_size = torch.Size(img.shape[:2])
            img = torch.as_tensor(img).permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img / 255.0 * 2.0 - 1.0
            img, K = transform_img_and_K(img, (W, H), K=K[None])
            assert K is not None
            K = K[0]
            K[0] /= W
            K[1] /= H
            camera_cond["K"][i] = K
        else:
            assert (
                False
            ), f"Variable `img` got {type(img)} type which is not supported!!!"
        imgs.append(img)
    imgs = torch.cat(imgs, dim=0)

    if traj_prior_Ks is not None:
        assert img_size is not None
        for i, prior_k in enumerate(traj_prior_Ks):
            img, prior_k = load_img_and_K(img_size, None, K=prior_k, device="cpu")  # type: ignore
            img, prior_k = transform_img_and_K(
                img,
                (W, H),
                K=prior_k[None],
                mode=options.get(
                    "transform_target", "crop"
                ),  # mode for prior is always same as target
                scale=options.get(
                    "transform_scale", 1.0
                ),  # scale for prior is always same as target
            )
            prior_k = prior_k[0]
            prior_k[0] /= W
            prior_k[1] /= H
            traj_prior_Ks[i] = prior_k

    options["num_frames"] = T
    torch.cuda.empty_cache()

    seed_everything(seed)
    generator = torch.Generator(device='cuda:0')
    generator.manual_seed(seed)

    # Get Data
    input_indices = image_cond["input_indices"]
    input_imgs = imgs[input_indices]
    input_c2ws = camera_cond["c2w"][input_indices]
    input_Ks = camera_cond["K"][input_indices]

    test_indices = [i for i in range(len(imgs)) if i not in input_indices]
    test_imgs = imgs[test_indices]
    test_c2ws = camera_cond["c2w"][test_indices]
    test_Ks = camera_cond["K"][test_indices]

    if options.get("save_input", True):
        save_output(
            {"/image": input_imgs},
            save_path=os.path.join(save_path, "input"),
            video_save_fps=2,
        )
        
    # get lrm model prior if needed
    if not options.get("continuous_rendering", False):
        # lrm outputs: dict with keys "gaussians", "near", "far"
        lrm_model = model if model_type == "regression" else model.lrm_model
        if lrm_model is not None:
            lrm_outputs = run_lrm_model(
                lrm_model,
                H,
                W,
                input_imgs,
                input_Ks,
                input_c2ws,
                camera_cond["c2w"],
                camera_scale=options.get("camera_scale", 2.0),
                scene_config_path=scene_config_path,
                **{k: options[k] for k in options if k not in ["cfg", "T", "camera_scale"]},
            )
        else:
            lrm_outputs = None
    else:
        lrm_outputs = None

    if not use_traj_prior:
        chunk_strategy = options.get("chunk_strategy", "gt")

        (
            _,
            input_inds_per_chunk,
            input_sels_per_chunk,
            test_inds_per_chunk,
            test_sels_per_chunk,
        ) = chunk_input_and_test(
            T,
            input_c2ws,
            test_c2ws,
            input_indices,
            test_indices,
            options=options,
            task=task,
            chunk_strategy=chunk_strategy,
            gt_input_inds=list(range(input_c2ws.shape[0])),
        )
        print(
            f"One pass - chunking with `{chunk_strategy}` strategy: total "
            f"{len(input_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_test_inds = []
        for i, (
            chunk_input_inds,
            chunk_input_sels,
            chunk_test_inds,
            chunk_test_sels,
        ) in tqdm(
            enumerate(
                zip(
                    input_inds_per_chunk,
                    input_sels_per_chunk,
                    test_inds_per_chunk,
                    test_sels_per_chunk,
                )
            ),
            total=len(input_inds_per_chunk),
            leave=False,
        ):
            (
                curr_input_sels,
                curr_test_sels,
                curr_input_maps,
                curr_test_maps,
            ) = pad_indices(
                chunk_input_sels,
                chunk_test_sels,
                T=T,
                padding_mode=options.get("t_padding_mode", "last"),
            )
            curr_imgs, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_input_inds],
                    test=y[chunk_test_inds],
                    input_maps=curr_input_maps,
                    test_maps=curr_test_maps,
                )
                for x, y in zip(
                    [
                        torch.cat(
                            [
                                input_imgs,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat([input_c2ws, test_c2ws[all_test_inds]], dim=0),
                        torch.cat([input_Ks, test_Ks[all_test_inds]], dim=0),
                    ],  # procedually append generated prior views to the input views
                    [test_imgs, test_c2ws, test_Ks],
                )
            ]
            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_input_sels
                + [
                    sel
                    for (ind, sel) in zip(
                        np.array(chunk_test_inds)[curr_test_maps[curr_test_maps != -1]],
                        curr_test_sels,
                    )
                    if test_indices[ind] in image_cond["input_indices"]
                ],
                chunk_input_sels
                + [
                    sel
                    for (ind, sel) in zip(
                        chunk_test_inds,
                        chunk_test_sels,
                    )
                    if test_indices[ind] in image_cond["input_indices"]
                ],
                curr_c2ws,
                curr_Ks,
                curr_input_sels
                + [
                    sel
                    for (ind, sel) in zip(
                        np.array(chunk_test_inds)[curr_test_maps[curr_test_maps != -1]],
                        curr_test_sels,
                    )
                    if test_indices[ind] in camera_cond["input_indices"]
                ],
                all_c2ws=camera_cond["c2w"],
                camera_scale=options.get("camera_scale", 2.0),
            )
            value_dict.update({"lrm_outputs": lrm_outputs})
            
            if model_type == "seva":
                samplers = create_samplers(
                    options["guider_types"],
                    denoiser.discretization,
                    [len(curr_imgs)],
                    options["num_steps"],
                    options["cfg_min"],
                    abort_event=abort_event,
                )
                assert len(samplers) == 1
            else: samplers = None
            samples = sampler(
                model,
                ae,
                conditioner,
                denoiser,
                samplers[0] if samplers is not None else None,
                value_dict,
                H,
                W,
                C,
                F,
                T=len(curr_imgs),
                cfg=(
                    options["cfg"][0]
                    if isinstance(options["cfg"], (list, tuple))
                    else options["cfg"]
                ),
                scene_config_path=scene_config_path,
                generator=generator,
                **{k: options[k] for k in options if k not in ["cfg", "T"]},
            )
            samples = decode_output(
                samples, len(curr_imgs), chunk_test_sels
            )  # decode into dict
            if options.get("save_first_pass", False):
                save_output(
                    replace_or_include_input_for_dict(
                        samples,
                        chunk_test_sels,
                        curr_imgs,
                        curr_c2ws,
                        curr_Ks,
                    ),
                    save_path=os.path.join(save_path, "first-pass", f"forward_{i}"),
                    video_save_fps=2,
                )
            extend_dict(all_samples, samples)
            all_test_inds.extend(chunk_test_inds)
    else:
        assert traj_prior_c2ws is not None, (
            "`traj_prior_c2ws` should be set when using 2-pass sampling. One "
            "potential reason is that the amount of input frames is larger than "
            "T. Set `num_prior_frames` manually to overwrite the infered stats."
        )
        traj_prior_c2ws = torch.as_tensor(
            traj_prior_c2ws,
            device=input_c2ws.device,
            dtype=input_c2ws.dtype,
        )

        if traj_prior_Ks is None:
            traj_prior_Ks = test_Ks[:1].repeat_interleave(
                traj_prior_c2ws.shape[0], dim=0
            )
        traj_prior_imgs = imgs.new_zeros(traj_prior_c2ws.shape[0], *imgs.shape[1:])

        # ---------------------------------- first pass ----------------------------------
        T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
        T_second_pass = T[1] if isinstance(T, (list, tuple)) else T
        chunk_strategy_first_pass = options.get(
            "chunk_strategy_first_pass", "gt-nearest"
        )
        (
            _,
            input_inds_per_chunk,
            input_sels_per_chunk,
            prior_inds_per_chunk,
            prior_sels_per_chunk,
        ) = chunk_input_and_test(
            T_first_pass,
            input_c2ws,
            traj_prior_c2ws,
            input_indices,
            image_cond["prior_indices"],
            options=options,
            task=task,
            chunk_strategy=chunk_strategy_first_pass,
            gt_input_inds=list(range(input_c2ws.shape[0])),
        )
        print(
            f"Two passes (first) - chunking with `{chunk_strategy_first_pass}` strategy: total "
            f"{len(input_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_prior_inds = []
        for i, (
            chunk_input_inds,
            chunk_input_sels,
            chunk_prior_inds,
            chunk_prior_sels,
        ) in tqdm(
            enumerate(
                zip(
                    input_inds_per_chunk,
                    input_sels_per_chunk,
                    prior_inds_per_chunk,
                    prior_sels_per_chunk,
                )
            ),
            total=len(input_inds_per_chunk),
            leave=False,
        ):
            (
                curr_input_sels,
                curr_prior_sels,
                curr_input_maps,
                curr_prior_maps,
            ) = pad_indices(
                chunk_input_sels,
                chunk_prior_sels,
                T=T_first_pass,
                padding_mode=options.get("t_padding_mode", "last"),
            )
            curr_imgs, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_input_inds],
                    test=y[chunk_prior_inds],
                    input_maps=curr_input_maps,
                    test_maps=curr_prior_maps,
                )
                for x, y in zip(
                    [
                        torch.cat(
                            [
                                input_imgs,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat([input_c2ws, traj_prior_c2ws[all_prior_inds]], dim=0),
                        torch.cat([input_Ks, traj_prior_Ks[all_prior_inds]], dim=0),
                    ],  # procedually append generated prior views to the input views
                    [
                        traj_prior_imgs,
                        traj_prior_c2ws,
                        traj_prior_Ks,
                    ],
                )
            ]
            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_input_sels,
                chunk_input_sels,
                curr_c2ws,
                curr_Ks,
                list(range(T_first_pass)),
                all_c2ws=camera_cond["c2w"],
                camera_scale=options.get("camera_scale", 2.0),
            )
            value_dict.update({"lrm_outputs": lrm_outputs})
            
            if model_type == "seva":
                samplers = create_samplers(
                    options["guider_types"],
                    denoiser.discretization,
                    [T_first_pass, T_second_pass],
                    options["num_steps"],
                    options["cfg_min"],
                    abort_event=abort_event,
                )
            else: samplers = None
            samples = sampler(
                model,
                ae,
                conditioner,
                denoiser,
                (
                    samplers[1]
                    if len(samplers) > 1
                    and options.get("ltr_first_pass", False)
                    and chunk_strategy_first_pass != "gt"
                    and i > 0
                    else samplers[0]
                ) if samplers is not None else None,
                value_dict,
                H,
                W,
                C,
                F,
                cfg=(
                    options["cfg"][0]
                    if isinstance(options["cfg"], (list, tuple))
                    else options["cfg"]
                ),
                scene_config_path=scene_config_path,
                T=T_first_pass,
                global_pbar=first_pass_pbar,
                generator=generator,
                **{k: options[k] for k in options if k not in ["cfg", "T", "sampler"]},
            )
            if samples is None:
                return
            samples = decode_output(
                samples, T_first_pass, chunk_prior_sels
            )  # decode into dict
            extend_dict(all_samples, samples)
            all_prior_inds.extend(chunk_prior_inds)

        if options.get("save_first_pass", True):
            save_output(
                all_samples,
                save_path=os.path.join(save_path, "first-pass"),
                video_save_fps=5,
            )
            video_path_0 = os.path.join(save_path, "first-pass", "samples-rgb.mp4")
            yield video_path_0

        # ---------------------------------- second pass ----------------------------------
        prior_indices = image_cond["prior_indices"]
        assert (
            prior_indices is not None
        ), "`prior_frame_indices` needs to be set if using 2-pass sampling."
        prior_argsort = np.argsort(input_indices + prior_indices).tolist()
        prior_indices = np.array(input_indices + prior_indices)[prior_argsort].tolist()
        gt_input_inds = [prior_argsort.index(i) for i in range(input_c2ws.shape[0])]

        traj_prior_imgs = torch.cat(
            [input_imgs, get_k_from_dict(all_samples, "samples-rgb")], dim=0
        )[prior_argsort]
        traj_prior_c2ws = torch.cat([input_c2ws, traj_prior_c2ws], dim=0)[prior_argsort]
        traj_prior_Ks = torch.cat([input_Ks, traj_prior_Ks], dim=0)[prior_argsort]

        update_kv_for_dict(all_samples, "samples-rgb", traj_prior_imgs)
        update_kv_for_dict(all_samples, "samples-c2ws", traj_prior_c2ws)
        update_kv_for_dict(all_samples, "samples-intrinsics", traj_prior_Ks)

        chunk_strategy = options.get("chunk_strategy", "nearest")
        (
            _,
            prior_inds_per_chunk,
            prior_sels_per_chunk,
            test_inds_per_chunk,
            test_sels_per_chunk,
        ) = chunk_input_and_test(
            T_second_pass,
            traj_prior_c2ws,
            test_c2ws,
            prior_indices,
            test_indices,
            options=options,
            task=task,
            chunk_strategy=chunk_strategy,
            gt_input_inds=gt_input_inds,
        )
        print(
            f"Two passes (second) - chunking with `{chunk_strategy}` strategy: total "
            f"{len(prior_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_test_inds = []
        for i, (
            chunk_prior_inds,
            chunk_prior_sels,
            chunk_test_inds,
            chunk_test_sels,
        ) in tqdm(
            enumerate(
                zip(
                    prior_inds_per_chunk,
                    prior_sels_per_chunk,
                    test_inds_per_chunk,
                    test_sels_per_chunk,
                )
            ),
            total=len(prior_inds_per_chunk),
            leave=False,
        ):
            (
                curr_prior_sels,
                curr_test_sels,
                curr_prior_maps,
                curr_test_maps,
            ) = pad_indices(
                chunk_prior_sels,
                chunk_test_sels,
                T=T_second_pass,
                padding_mode="last",
            )
            curr_imgs, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_prior_inds],
                    test=y[chunk_test_inds],
                    input_maps=curr_prior_maps,
                    test_maps=curr_test_maps,
                )
                for x, y in zip(
                    [
                        traj_prior_imgs,
                        traj_prior_c2ws,
                        traj_prior_Ks,
                    ],
                    [test_imgs, test_c2ws, test_Ks],
                )
            ]
            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_prior_sels,
                chunk_prior_sels,
                curr_c2ws,
                curr_Ks,
                list(range(T_second_pass)),
                all_c2ws=camera_cond["c2w"],
                camera_scale=options.get("camera_scale", 2.0),
            )
            value_dict.update({"lrm_outputs": lrm_outputs})
            
            samples = sampler(
                model,
                ae,
                conditioner,
                denoiser,
                (
                    samplers[1] 
                    if len(samplers) > 1 
                    else samplers[0]
                ) if samplers is not None else None,
                value_dict,
                H,
                W,
                C,
                F,
                T=T_second_pass,
                cfg=(
                    options["cfg"][1]
                    if isinstance(options["cfg"], (list, tuple))
                    and len(options["cfg"]) > 1
                    else options["cfg"]
                ),
                scene_config_path=scene_config_path,
                global_pbar=second_pass_pbar,
                generator=generator,
                **{k: options[k] for k in options if k not in ["cfg", "T", "sampler"]},
            )
            if samples is None:
                return
            samples = decode_output(
                samples, T_second_pass, chunk_test_sels
            )  # decode into dict
            if options.get("save_second_pass", False):
                save_output(
                    replace_or_include_input_for_dict(
                        samples,
                        chunk_test_sels,
                        curr_imgs,
                        curr_c2ws,
                        curr_Ks,
                    ),
                    save_path=os.path.join(save_path, "second-pass", f"forward_{i}"),
                    video_save_fps=2,
                )
            extend_dict(all_samples, samples)
            all_test_inds.extend(chunk_test_inds)
    all_samples = {
        key: value[np.argsort(all_test_inds)] for key, value in all_samples.items()
    }
    
    # Metric measurement
    metric_outputs = {}
    qual_metrics = {}
    if task in ['img2img', 'img2vid']:
        
        render_metrics = {
            "psnr": get_pyiqa_metric('psnr'),
            "ssim": get_pyiqa_metric('ssim'),
            "lpips": get_pyiqa_metric('lpips'),
        }
        for key, render_metric in render_metrics.items():
            pred_imgs = ((all_samples['samples-rgb/image'] + 1) / 2.0).clamp(0, 1).float().to('cuda:0')
            gt_imgs = ((imgs[test_indices] + 1) / 2.0).clamp(0, 1).float().to('cuda:0')
            metric_outputs[key] = render_metric(pred_imgs, gt_imgs).cpu().numpy()
            
        qual_metrics = {
            # "fid": get_pyiqa_metric("fid"),
            "piqe": get_pyiqa_metric("piqe"),
            "niqe": get_pyiqa_metric("niqe"),
            "brisque": get_pyiqa_metric("brisque"),
        }
        all_samples['gt-rgb/image'] = imgs[test_indices].clone()
    
    save_output(
        replace_or_include_input_for_dict(
            all_samples,
            test_indices,
            imgs,
            camera_cond["c2w"],
            camera_cond["K"],
        )
        if options.get("replace_or_include_input", False)
        else all_samples,
        save_path=save_path,
        video_save_fps=options.get("video_save_fps", 2),
    )
    
    if task in ['img2img', 'img2vid']:
        for key, qual_metric in qual_metrics.items():
            try:
                if 'fid' in key and task == 'img2img':
                    metric_outputs[key] = qual_metric(os.path.join(save_path, "samples-rgb"), os.path.join(save_path, "gt-rgb"))
                else:
                    pred_imgs = ((all_samples['samples-rgb/image'] + 1) / 2.0).float().clamp(0, 1).to('cuda:0')
                    metric_outputs[key] = qual_metric(pred_imgs).cpu().numpy()
            except:
                metric_outputs[key] = np.array([float('nan')])

        save_metrics_to_excel(metric_outputs, os.path.join(save_path, "performance_table.xlsx"))
    
    video_path_1 = os.path.join(save_path, "samples-rgb.mp4")
    yield video_path_1


def save_metrics_to_excel(metrics: dict, filename: str):
    data = {}
    mean_only_keys = {}

    for key, value in metrics.items():
        flat = value.flatten()
        if len(flat) == 1:
            mean_only_keys[key] = flat[0]
        else:
            data[key] = flat

    df = pd.DataFrame(data)

    # Compute mean row for normal columns
    mean_row = df.mean() if not df.empty else pd.Series()
    
    # Append metrics that had only one value directly into mean row
    for k, v in mean_only_keys.items():
        mean_row[k] = v
    mean_row.name = 'mean'

    if not df.empty:
        df_with_mean = pd.concat([df, mean_row.to_frame().T])
    else:
        df_with_mean = pd.DataFrame([mean_row])

    df_with_mean.to_excel(filename, index=True)
