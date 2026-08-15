

import open_clip
import torch
import torch.nn as nn
import trimesh
import random
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class CLIPDistance(nn.Module):
    
    def __init__(self, device='cuda', backbone='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        super().__init__()
        
        self.device = device
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        
    @torch.no_grad()
    def forward(self, images, src_indices, tgt_indices):
        
        src_images = [self.preprocess(images[idx]) for idx in src_indices]
        tgt_images = [self.preprocess(images[idx]) for idx in tgt_indices]
        src_images = torch.stack(src_images, dim=0).half().to(self.device)
        tgt_images = torch.stack(tgt_images, dim=0).half().to(self.device)
        
        with torch.autocast(self.device):
            src_image_features = self.clip_model.encode_image(src_images)
            tgt_image_features = self.clip_model.encode_image(tgt_images)
            src_image_features /= src_image_features.norm(dim=-1, keepdim=True)
            tgt_image_features /= tgt_image_features.norm(dim=-1, keepdim=True)
            
            cosine_sim = src_image_features @ tgt_image_features.T
            cosine_dist = 1.0 - cosine_sim
        
        return cosine_dist.cpu().numpy()


def get_frustum_mesh(K, img_size, near, far):
    """
    Constructs a frustum mesh from intrinsics, pose, and depth range.
    K: (3, 3), intrinsics
    img_size: (H, W)
    near, far: depth bounds
    """
    H, W = img_size
    fx, fy = K[0, 0] * W, K[1, 1] * H
    cx, cy = K[0, 2] * W, K[1, 2] * H
    
    def get_corners(depth):
        xs = torch.tensor([0, W, W, 0])
        ys = torch.tensor([0, 0, H, H])
        z = torch.full((4,), depth)
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        return torch.stack([x, y, z], dim=1)

    # 3D frustum points in camera frame
    near_corners = get_corners(near)
    far_corners = get_corners(far)
    origin = torch.zeros(1, 3)

    # Combine into full frustum vertices
    frustum_cam = torch.cat([origin, near_corners, far_corners], dim=0)  # (9, 3)
    
    return frustum_cam


def kmeans_clustering(poses_c2w, n):
    # Prepare the data for K-means
    camera_centers = poses_c2w[..., :3, 3]
    camera_directions = poses_c2w[..., :3, 2] / np.linalg.norm(poses_c2w[..., :3, 2], axis=-1, keepdims=True)
    positions = np.concatenate([camera_centers, camera_directions], axis=1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(positions)
    centers = kmeans.cluster_centers_
    
    # Find the index closest to each cluster center
    distmatrix = cdist(positions, centers)
    selected_ids = np.argmin(distmatrix, axis=0)
    selected_ids = np.unique(selected_ids)
    
    return selected_ids


def get_camera_frustum_mesh(frustum_cam, pose):
    # pose: (4, 4), world-to-camera (or camera-to-world if inverted)

    # Transform to world coordinates
    frustum_world = (pose[:3, :3] @ frustum_cam.T + pose[:3, 3:4]).T.numpy()

    # Build mesh
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # Sides

        [1, 2, 3], [1, 3, 4],                       # Near plane (correct winding)
        [5, 6, 7], [5, 7, 8],                       # Far plane

        [1, 2, 6], [1, 6, 5],                       # Edge 1
        [2, 3, 7], [2, 7, 6],                       # Edge 2
        [3, 4, 8], [3, 8, 7],                       # Edge 3
        [4, 1, 5], [4, 5, 8],                       # Edge 4
    ])
    return trimesh.Trimesh(vertices=frustum_world, faces=faces, process=False)


def compute_frustum_iou(frustum_cam, pose1, pose2):
    mesh1 = get_camera_frustum_mesh(frustum_cam, pose1)
    mesh2 = get_camera_frustum_mesh(frustum_cam, pose2)
    mesh1 = mesh1.convex_hull
    mesh2 = mesh2.convex_hull
    inter = mesh1.intersection(mesh2)
    
    if inter.is_empty:
        return 0.0

    vol_inter = inter.volume
    vol_union = mesh1.volume + mesh2.volume - vol_inter
    return float(vol_inter / vol_union)

# pose distance from https://github.com/ardaduz/deep-video-mvs/blob/fa14288f149c5af7b2a49092f729f5c4f44517ba/dvmvs/utils.py#L17
def pose_cdistance(reference_pose, measurement_pose):
    """
    :param reference_pose: Nx4x4 torch array, reference frame camera-to-world pose (not extrinsic matrix!)
    :param measurement_pose: Mx4x4 torch array, measurement frame camera-to-world pose (not extrinsic matrix!)
    :return combined_measure: float, combined pose distance measure
    :return R_measure: float, rotation distance measure
    :return t_measure: float, translation distance measure
    """
    rel_pose = torch.einsum('nij,mjk->nmik', torch.linalg.inv(reference_pose), measurement_pose)
    R = rel_pose[..., :3, :3]
    t = rel_pose[..., :3, 3]
    batch_R_trace = torch.einsum('...ii', R)
    R_measure = torch.sqrt(2 * (1 - batch_R_trace.clip(max=3.0) / 3))
    t_measure = torch.norm(t, dim=-1)
    combined_measure = torch.sqrt(t_measure ** 2 + R_measure ** 2)
    return combined_measure, R_measure, t_measure


def select_views(images, clip_metric, poses_w2c, intrinsics, num_group_images, num_input_images, num_input_variations, near, far, H, W, adjacent_frame_sampling_rate=0.7, min_frustum_iou_threshold=0.2):
    # For the same scene, intrinsics will be the same
    
    # camera centers
    poses_c2w = poses_w2c.inverse()
    poses_c2w_np = np.array(poses_c2w)
    det_poses_c2w = torch.det(poses_c2w[:, :3, :3])
    valid_mask1 = ~((poses_c2w[:, :3, 3] > 1e3).any(dim=-1))
    valid_mask2 = ~torch.isnan(det_poses_c2w)
    valid_mask3 = torch.isclose(det_poses_c2w, torch.ones_like(det_poses_c2w))
    valid_mask = valid_mask1 & valid_mask2 & valid_mask3
    valid_mask_2d = valid_mask[:, None] & valid_mask[None, :]
    
    # determine the distance threshold
    min_number_existance = max(num_group_images, int(len(poses_c2w) * 0.1))
    combined_dist, R_dist, t_dist = pose_cdistance(poses_c2w, poses_c2w)
    sorted_combined_dist, indices_dist = torch.sort(combined_dist, dim=-1)
    dist_threshold = (sorted_combined_dist * valid_mask_2d)[:, min_number_existance + 1].max()  # this ensure that there will be at least num_group_images near neighbors
    
    # build graph
    mask = (combined_dist < dist_threshold) & (~torch.eye(len(poses_c2w), dtype=torch.bool, device=combined_dist.device)) & valid_mask_2d
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_list = list(zip(src.tolist(), dst.tolist()))
    G = nx.Graph()
    G.add_edges_from(edge_list)
    degree_array = np.array(list(G.degree()))  # [N, 2] : (idx, number of connection)
    
    # weighted random selection by connectivity
    prob = degree_array[:, 1] / np.sum(degree_array[:, 1])
    selected_indices = np.random.choice(len(degree_array), size=num_input_variations, replace=False, p=prob)
    selected_node_indices = degree_array[selected_indices, 0].tolist()
    
    # make frustum volume
    frustum_cam = get_frustum_mesh(intrinsics[0], (H, W), near, far)
    
    # input-view, novel-view selection
    input_target_combinations = []
    for node in selected_node_indices:
        
        # gather indices belonging to node
        neighbors = list(G.neighbors(node))
        neighbors = np.array([node] + neighbors)
        
        for num_input_image in num_input_images:
            
            # novel-view number
            num_target_image = num_group_images - num_input_image
            easy_sample_num = int(num_target_image * adjacent_frame_sampling_rate)
        
            # k-means clustering based input-view selection
            ids_selected_iv = kmeans_clustering(poses_c2w_np[neighbors], num_input_image)
            input_nodes = neighbors[ids_selected_iv]
            input_nodes_list = input_nodes.tolist()
            assert len(input_nodes_list) == num_input_image
            
            # novel-view candidates
            mask_nv = ~np.isin(neighbors, input_nodes)
            target_nodes = neighbors[mask_nv]
            
            # measure clip distance between input_nodes and target_nodes
            clip_dists = clip_metric(images, input_nodes_list, target_nodes.tolist())  # [n, m]
            clip_min_dists = clip_dists.min(axis=0)
            
            # select easy samples
            minimum_percent = max(easy_sample_num / len(clip_min_dists) * 100, 20)  # low 20% clip score
            clip_distance_threshold = np.percentile(clip_min_dists, minimum_percent)
            small_viewpoint_idx = target_nodes[clip_min_dists <= clip_distance_threshold]
            large_viewpoint_idx = target_nodes[clip_min_dists > clip_distance_threshold]
            ids_small_nv = kmeans_clustering(poses_c2w_np[small_viewpoint_idx], easy_sample_num)
            easy_target_nodes = small_viewpoint_idx[ids_small_nv].tolist()
            
            # frustum IOU check
            final_target_nodes = []
            for tgt_node in easy_target_nodes:
                min_iou_value = 1.0
                for input_node in input_nodes_list:
                    min_iou_value = min(compute_frustum_iou(frustum_cam, poses_c2w[input_node], poses_c2w[tgt_node]), min_iou_value)
                if min_iou_value >= min_frustum_iou_threshold:
                    final_target_nodes.append(tgt_node)
            
            # select hard samples + frustum IOU check
            hard_target_nodes = np.random.permutation(large_viewpoint_idx).tolist()
            for tgt_node in hard_target_nodes:
                min_iou_value = 1.0
                for input_node in input_nodes_list:
                    min_iou_value = min(compute_frustum_iou(frustum_cam, poses_c2w[input_node], poses_c2w[tgt_node]), min_iou_value)
                if min_iou_value >= min_frustum_iou_threshold:
                    final_target_nodes.append(tgt_node)
                if len(final_target_nodes) == num_target_image:
                    break
            
            # if not done, select any remained samples
            if len(final_target_nodes) < num_target_image:
                num_remain_target = num_target_image - len(final_target_nodes)
                valid_indices = list(set(neighbors) - set(input_nodes_list) - set(final_target_nodes))
                final_target_nodes += random.sample(valid_indices, num_remain_target)
            assert len(final_target_nodes) == num_target_image
            
            input_nodes_list = sorted(input_nodes_list)
            target_nodes_list = sorted(final_target_nodes)
            input_target_combinations.append((input_nodes_list, target_nodes_list))
            
    return input_target_combinations


def check_scene(output_path, keyname, num_input_images, num_input_variations):
    
    num_file_all = len(num_input_images) * num_input_variations
    
    safetensors_files = list(output_path.rglob('*.safetensors'))
    valid_files = [file for file in safetensors_files if keyname in file.stem]
    
    return len(valid_files) == num_file_all


def write_log(msg: str, path="log.txt"):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg.strip() + "\n")
    except Exception as e:
        print(f"Log write failed: {e}")