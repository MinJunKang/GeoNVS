import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import json
import ffmpeg
from ffmpeg import probe

class VideoAdapter:
    def __init__(self, scene_dir, num_frames = 16, crop_height = 512, crop_width = 960, width = 1920, height = 540):
        self.num_frames = num_frames
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.width = width
        self.height = height
        self.scene_dir = scene_dir

    def get_rgb_file_names(self):
        rgb_files = sorted([f for f in os.listdir(self.scene_dir) if f.endswith(('.mp4'))])
        return rgb_files
    
    def get_depth_file_names(self):
        depth_files = sorted([f for f in os.listdir(self.scene_dir) if f.endswith('.npz') and 'depth_part' in f])
        return depth_files

    def get_video_depth_file_names(self):
        depth_files = sorted([f for f in os.listdir(self.scene_dir) if f.endswith('.npz') and 'video_depth' in f])
        return depth_files

    # output: (C, T, H, W)
    def read_rgbs(self, rgb_file_name):
        rgb_path = os.path.join(self.scene_dir, rgb_file_name)
        out, _ = (
            ffmpeg
            .input(rgb_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args('-loglevel', 'quiet')
            .run(capture_stdout=True)
        )
        rgbs = np.frombuffer(out, np.uint8)
        rgbs = rgbs.reshape([-1, self.height, self.width, 3])
        rgbs = np.transpose(rgbs, (3, 0, 1, 2))
        return rgbs
    
    # output: (1, T, H, W)
    def read_depths(self, depth_file_name):
        npz_path = os.path.join(self.scene_dir, depth_file_name)
        depths = np.load(npz_path)
        array_name = depths.files
        assert len(array_name) == 1, "depth_merge should contain only one array"
        depths = depths[array_name[0]]
        depths = np.expand_dims(depths, axis=0).astype(np.float32)
        return depths
    
    # output: (1, T, H, W)
    def split_depths(self, depths):
        half_width = depths.shape[3] // 2
        gt_depths = depths[:, :, :, :half_width]
        renders_depths = depths[:, :, :, half_width:]
        return gt_depths, renders_depths
    
    # output: (C, T, H, W)
    def split_rgbs(self, rgbs):
        half_width = self.width // 2
        gt_rgbs = rgbs[:, :, :, :half_width]
        renders_rgbs = rgbs[:, :, :, half_width:]
        return gt_rgbs, renders_rgbs
    
    # input: (C, T, H, W)
    # output: (C, T, H, W)
    def crop_frames(self, frames, start_x, start_y):
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)
        cropped = transforms.functional.crop(frames, start_y, start_x, self.crop_height, self.crop_width)
        return cropped
    
    def generate_sequence(self, gt_depths, renders_depths, gt_rgbs, renders_rgbs):
        # Randomly select a sequence of frames from gt_depths and gt_rgbs
        seq_idx = np.random.randint(0, gt_depths.shape[1] - self.num_frames + 1)
        gt_depths = gt_depths[:, seq_idx:seq_idx+self.num_frames]
        renders_depths = renders_depths[:, seq_idx:seq_idx+self.num_frames]
        gt_rgbs = gt_rgbs[:, seq_idx:seq_idx+self.num_frames]
        renders_rgbs = renders_rgbs[:, seq_idx:seq_idx+self.num_frames]

        return gt_depths, renders_depths, gt_rgbs, renders_rgbs
    
    
    # output: (C, T, H, W)
    def get_inputs(self, start_x, start_y):
        # Randomly select one depth and one rgb file
        depth_file_names = self.get_depth_file_names()
        rgb_file_names = self.get_rgb_file_names()
        video_depth_file_names = self.get_depth_file_names()
        idx = np.random.randint(0, len(depth_file_names))
        depth_file_name = depth_file_names[idx]
        rgb_file_name = rgb_file_names[idx]
        video_depth_file_name = video_depth_file_names[idx]
        

        # Read depths and rgbs
        depths = self.read_depths(depth_file_name)
        rgbs = self.read_rgbs(rgb_file_name)
        _, renders_depths = self.split_depths(depths)
        gt_depths = self.read_depths(video_depth_file_name)
        gt_rgbs, renders_rgbs = self.split_rgbs(rgbs)

        gt_depths, renders_depths, gt_rgbs, renders_rgbs = self.generate_sequence(gt_depths, renders_depths, gt_rgbs, renders_rgbs)

        # Randomly select one frame from gt_rgbs as reference frame
        ref_idx = np.random.randint(0, gt_rgbs.shape[0])
        ref_frame = gt_rgbs[:, ref_idx:ref_idx+1]

        # Crop frames
        gt_depths_cropped = self.crop_frames(gt_depths, start_x, start_y)
        renders_depths_cropped = self.crop_frames(renders_depths, start_x, start_y)
        gt_rgbs_cropped = self.crop_frames(gt_rgbs, start_x, start_y)
        renders_rgbs_cropped = self.crop_frames(renders_rgbs, start_x, start_y)

        return gt_depths_cropped, renders_depths_cropped, gt_rgbs_cropped, renders_rgbs_cropped, torch.from_numpy(ref_frame)


class ImageSequenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.video_list = []
        
        json_path = "/home/sibo/Archive_Code/GenFusion/filelist.json"
        with open(json_path, "r") as f:
            self.video_list = json.load(f)
        
        # Ensure video_list is a list
        if not isinstance(self.video_list, list):
            raise ValueError("The JSON file should contain a list of video paths.")
        
        self.transform = transform or transforms.ToTensor()
        print('len(self.video_list): ', len(self.video_list))
        self.crop_height = 512
        self.crop_width = 960
        self.input_height = 512
        self.input_width = 960
        self.num_frames = 16

    def __len__(self):
        return len(self.video_list)
    
    def rgb_normalize(self, rgbs):
        return (rgbs / 255. - 0.5) * 2
    
    def depth_normalize(self, depths):
        # Normalize depths using min-max normalization
        epsilon = 1e-10
        valid_mask = depths > 0
        disparity = torch.zeros_like(depths)
        disparity[valid_mask] = 1.0 / (depths[valid_mask] + epsilon)
        valid_disparities = torch.masked_select(disparity, valid_mask)
        if valid_disparities.numel() > 0:
            disp_min = valid_disparities.min()
            disp_max = valid_disparities.max()
            normalized_disparity = torch.zeros_like(disparity)
            normalized_disparity[valid_mask] = (disparity[valid_mask] - disp_min) / (disp_max - disp_min)
            print("normalized_disparity:", normalized_disparity.max(), normalized_disparity.min())
        else:
            print("Warning: No valid depth values found")
            normalized_disparity = torch.zeros_like(disparity)

        return normalized_disparity


    def __getitem__(self, index):
        scene_dir = self.video_list[index]
        video_adapter = VideoAdapter(scene_dir, self.num_frames, self.crop_height, self.crop_width)
        width, height = 960, 540#video_adapter.width, video_adapter.height
        start_x = 0#torch.randint(0, max(1, width - self.crop_width), (1,)).item()
        start_y = 0#torch.randint(0, max(1, height - self.crop_height), (1,)).item()

        gt_depth_tensor, artifact_depth_tensor, gt_image_tensor, artifact_image_tensor, ref_image_tensor = video_adapter.get_inputs(start_x, start_y)
        ref_image_tensor = self.rgb_normalize(ref_image_tensor)
        gt_image_tensor = self.rgb_normalize(gt_image_tensor)
        artifact_image_tensor = self.rgb_normalize(artifact_image_tensor)
        
        
        ######### Depth #########
        gt_depth_tensor = (gt_depth_tensor - 0.5) * 2
        artifact_depth_tensor = (artifact_depth_tensor - 0.5) * 2

        # Define resize transform
        resize_transform = transforms.Resize((self.input_height, self.input_width))
        gt_image_tensor = torch.stack([resize_transform(img) for img in gt_image_tensor.unbind(1)], dim=1)

        artifact_image_tensor = torch.stack([resize_transform(img) for img in artifact_image_tensor.unbind(1)], dim=1)
        
        # Resize depth tensors
        gt_depth_tensor = torch.stack([resize_transform(depth) for depth in gt_depth_tensor.unbind(1)], dim=1)
        
        test_tensor = artifact_depth_tensor.detach().clone()
        artifact_depth_tensor = torch.nn.functional.interpolate(test_tensor.unsqueeze(0), size=(self.num_frames, self.input_height, self.input_width)).squeeze(0)
        
        # Resize reference image tensor
        ref_image_tenscurrent_channelor = torch.nn.functional.interpolate(ref_image_tensor.unsqueeze(0), size=(1, self.input_height, self.input_width), mode='trilinear', align_corners=False).squeeze(0)


        # Append scene_dir to json file
        ####### only for debug #######
        json_file = 'processed_scenes.json'
        
        # Load existing data if file exists
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                try:
                    processed_scenes = json.load(f)
                except json.JSONDecodeError:
                    processed_scenes = []
        else:
            processed_scenes = []
            
        # Append new scene if not already present
        if scene_dir not in processed_scenes:
            processed_scenes.append(scene_dir)
            
            # Write back to file
            with open(json_file, 'w') as f:
                json.dump(processed_scenes, f, indent=2)
        

        data = {'caption': scene_dir, 
                'ref_rgb': ref_image_tensor,
                'gt_rgb': gt_image_tensor, 
                'gt_depth': gt_depth_tensor, 
                'artifact_rgb': artifact_image_tensor,
                'artifact_depth': artifact_depth_tensor,
                'path': './data', 
                'fps': 30, 
                'frame_stride': 1}
        return data
