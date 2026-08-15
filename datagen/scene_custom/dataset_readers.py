#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from scene_custom.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    train_poses: list
    test_poses: list


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
    
    
def prepareScene(dataset):
    # images, overlapping_masks, extrinsics, intrinsics, pts3d
    cam_infos = []
    poses=[]
    num_samples = len(dataset['images'])
    for idx in range(num_samples):
        image = dataset['images'][idx]
        depth = dataset['depths'][idx][..., None]
        height, width = image.shape[:2]
        focal_length_x, focal_length_y = dataset['intrinsics'][idx][0, 0], dataset['intrinsics'][idx][1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        R, T = dataset['extrinsics'][idx][:3, :3], dataset['extrinsics'][idx][:3, 3]  # w2c
        R = np.transpose(R)
        poses.append(dataset['extrinsics'][idx])
        
        image = np.uint8(image * 255.0)
        image_pil = Image.fromarray(image)
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_pil, depth=depth, width=width, height=height)
        cam_infos.append(cam_info)
        
    train_cam_infos = cam_infos
    test_cam_infos = []
    train_poses = poses
    test_poses = []
    
    masks = dataset['overlapping_masks']
    points = dataset['pts3d'][masks > 0].reshape(-1, 3)
    colors = dataset['color3d'][masks > 0].reshape(-1, 3)
    if 'normal3d' in dataset:
        normals = dataset['normal3d'][masks > 0].reshape(-1, 3)
    else:
        normals = None
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           train_poses=train_poses,
                           test_poses=test_poses)
    return scene_info