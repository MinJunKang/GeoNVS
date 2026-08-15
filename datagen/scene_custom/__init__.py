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

import os
import random
from scene_custom.dataset_readers import prepareScene
from scene_custom.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos

class Scene:

    gaussians : GaussianModel

    def __init__(self, resolution, gaussians : GaussianModel, inputs, shuffle=True, resolution_scales=[1.0], data_device="cuda"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.gaussians = gaussians
        self.train_cameras = {}
        self.test_cameras = {}
        scene_info = prepareScene(inputs)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, resolution, data_device)
            # print('train_camera_num: ', len(self.train_cameras[resolution_scale]))
            # print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, resolution, data_device)
            # print('test_camera_num: ', len(self.test_cameras[resolution_scale]))

        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]