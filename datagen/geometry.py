
import torch
import torch.nn.functional as F
import open3d as o3d
import numpy as np
from einops import rearrange, repeat

from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
# from evo.tools import plot
import matplotlib.pyplot as plt
from utils.sh_rotation import rotate_sh
from roma import rotmat_to_unitquat, rigid_points_registration


def rigid_points_registration_(pts1, pts2, conf=None):
    R, T, s = rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf, compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)

def apply_transform_to_gaussian(gaussian_mean, gaussian_cov, gaussian_sh, s, R, T):
    mean = torch.einsum('pq,bnq->bnp', R, gaussian_mean) * s + T[None, None, :]
    covariances = R[None, None] @ gaussian_cov @ R[None, None].transpose(-1, -2)
    covariances = s**2 * covariances
    sh = rotate_sh(gaussian_sh, R)
    return mean, covariances, sh


def convert_to_trajectory(poses):
    """
    Convert a tensor of poses to a PoseTrajectory3D object.

    Args:
        poses (torch.Tensor): Tensor of poses.

    Returns:
        PoseTrajectory3D: Converted trajectory.
    """
    quat = rotmat_to_unitquat(poses[:, :3, :3])[..., [3, 0, 1, 2]]  # xyzw to wxyz
    timestamps = np.arange(poses.shape[0], dtype=np.float32)
    trajectory = PoseTrajectory3D(positions_xyz=poses[:, :3, 3].cpu().numpy(), orientations_quat_wxyz=quat.cpu().numpy(), timestamps=timestamps)
    
    return trajectory


# def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True):
#     assert isinstance(pred_traj, PoseTrajectory3D)

#     if gt_traj is not None:
#         assert isinstance(gt_traj, PoseTrajectory3D)
#         gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

#         if align:
#             pred_traj.align(gt_traj, correct_scale=correct_scale)

#     plot_collection = plot.PlotCollection("PlotCol")
#     fig = plt.figure(figsize=(8, 8))
#     plot_mode = plot.PlotMode.xz # ideal for planar movement
#     ax = plot.prepare_axis(fig, plot_mode)
#     ax.set_title(title)
#     if gt_traj is not None:
#         plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
#     plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
#     plot_collection.add_figure("traj (error)", fig)
#     plot_collection.export(filename, confirm_overwrite=False)
#     plt.close(fig=fig)
    
    
def visualize_pcd(points, colors, filename="point_cloud.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    return pcd


def align_to_ground_truth(poses_gt, poses_pred):
    
    # compute camera centers
    camera_center_gt = -poses_gt[..., :3, :3].transpose(-1, -2) @ poses_gt[..., :3, 3:]
    camera_center_pred = -poses_pred[..., :3, :3].transpose(-1, -2) @ poses_pred[..., :3, 3:]
    
    # compute scale difference
    scale = rigid_points_registration(camera_center_pred[..., 0], camera_center_gt[..., 0], compute_scaling=True)[-1]
    
    if torch.isnan(scale).any() or scale < 1e-6:
        norm_gt = torch.norm(camera_center_gt[..., 0], dim=-1)
        norm_pred = torch.norm(camera_center_pred[..., 0], dim=-1)
        
        valid_mask = norm_pred > 1e-6
        if valid_mask.any():
            scale = (norm_gt[valid_mask] / norm_pred[valid_mask]).mean()
        else:
            scale = torch.tensor(1.0, device=poses_gt.device)
    
    return scale


def project_pc(w2c, intrinsic, pc, image_size, device_):
    batch_num = len(w2c)
    height, width = image_size
    assert 'pts3d' in pc
    num_points = len(pc['pts3d'])
    Kmat = intrinsic.clone()
    Kmat[..., 0, :] *= width
    Kmat[..., 1, :] *= height
    
    # projection
    prev_point_maps_homo = torch.cat([pc['pts3d'], torch.ones_like(pc['pts3d'][..., :1])], dim=-1)
    points2cam = torch.einsum('bpq,nq->bnp', w2c, prev_point_maps_homo)
    points_prj = torch.einsum('bpq,bnq->bnp', Kmat, points2cam[..., :-1])
    points_2d = points_prj[..., :2] / points_prj[..., -1:]
    valid_mask = (points_2d[..., 0] >= 0) & (points_2d[..., 1] >= 0) & \
                 (points_2d[..., 0] < width) & (points_2d[..., 1] < height) & (points_prj[..., -1] > 0)
    
    valid_batch_idx = repeat(torch.arange(batch_num, device=device_), 'b -> b n', n=num_points)[valid_mask]
    valid_pc_index = repeat(torch.arange(num_points, device=device_), 'n -> b n', b=batch_num)[valid_mask]
    valid_pixel_u = points_2d[..., 0][valid_mask].long()
    valid_pixel_v = points_2d[..., 1][valid_mask].long()
    valid_z_buffer = points_prj[..., -1][valid_mask]
    flat_indices = (valid_batch_idx, valid_pixel_v, valid_pixel_u)
                 
    depth_buffer = torch.full((batch_num, height, width), float('inf'), device=device_)
    depth_buffer[flat_indices] = torch.minimum(depth_buffer[flat_indices], valid_z_buffer)
    depth_buffer[depth_buffer == float('inf')] = 0
    
    index_buffer = torch.full((batch_num, height, width), -1, dtype=torch.long, device=device_)
    index_buffer[flat_indices] = torch.where(depth_buffer[flat_indices] == valid_z_buffer, valid_pc_index, index_buffer[flat_indices])
    index_buffer[index_buffer == -1] = num_points
    
    color_buffer, normal_buffer = None, None
    if 'color3d' in pc:
        colors = repeat(pc['color3d'], 'n c -> b n c', b=batch_num)
        bg_color = torch.zeros((batch_num, 1, 3), device=device_)
        color_buffer = torch.gather(torch.cat([colors, bg_color], dim=1), 1, repeat(index_buffer, 'b h w -> b (h w) c', c=3))
        color_buffer = rearrange(color_buffer, 'b (h w) c -> b h w c', h=height, w=width)
    if 'normal3d' in pc:
        normals = repeat(pc['normal3d'], 'n c -> b n c', b=batch_num)
        bg_normal = torch.zeros((batch_num, 1, 3), device=device_)
        normal_buffer = torch.gather(torch.cat([normals, bg_normal], dim=1), 1, repeat(index_buffer, 'b h w -> b (h w) c', c=3))
        normal_buffer = rearrange(normal_buffer, 'b (h w) c -> b h w c', h=height, w=width)
    index_buffer[index_buffer == num_points] = -1
        
    return depth_buffer, index_buffer, color_buffer, normal_buffer


def normalize_depth(depth_maps):
    return (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min() + 1e-6)


def compute_co_vis_masks(sorted_conf_indices, depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1, device_='cuda'):
    
    H, W = image_sizes
    overlapping_masks = torch.zeros((len(sorted_conf_indices), H, W), dtype=torch.bool, device=device_)
    
    if len(overlapping_masks) == 1:
        return overlapping_masks
    
    # Get the current and previous depth maps and point maps
    curr_depth_maps = depthmaps[sorted_conf_indices[1:]]
    curr_intrinsics = camera_intrinsics[sorted_conf_indices[1:]]
    curr_extrinsics = extrinsics_w2c[sorted_conf_indices[1:]]
    prev_depth_maps = depthmaps[sorted_conf_indices[:-1]]
    prev_point_maps = pointmaps[sorted_conf_indices[:-1]]
    
    # Normalize depth maps
    prev_depth_maps = normalize_depth(prev_depth_maps)
    curr_depth_maps = normalize_depth(curr_depth_maps)
    
    # projection of points
    prev_point_maps_homo = torch.cat([prev_point_maps, torch.ones_like(prev_point_maps[..., :1])], dim=-1)
    points2cam = torch.einsum('bpq,bhwq->bhwp', curr_extrinsics, prev_point_maps_homo)
    points_prj = torch.einsum('bpq,bhwq->bhwp', curr_intrinsics, points2cam[..., :-1])
    points_2d = points_prj[..., :2] / points_prj[..., -1:]
    valid_mask = (points_2d[..., 0] >= 0) & (points_2d[..., 1] >= 0) & \
                 (points_2d[..., 0] < W) & (points_2d[..., 1] < H) & (points_prj[..., -1] > 0)
    
    # sample curr depths corresponding to previous depths
    grid_2d = points_2d.clone()
    grid_2d[..., 0] = (grid_2d[..., 0] + 0.5) / W * 2.0 - 1.0
    grid_2d[..., 1] = (grid_2d[..., 1] + 0.5) / H * 2.0 - 1.0
    curr_sampled_depths = F.grid_sample(curr_depth_maps[:, None], grid=grid_2d, mode='bilinear', align_corners=False)[:, 0]
    
    # depth difference
    depth_differences = torch.abs(prev_depth_maps - curr_sampled_depths)
    consistency_mask = (depth_differences < depth_threshold) & valid_mask
    
    # update overlapping masks
    points2d_valid = points_2d[consistency_mask].long()
    batchidx_valid = repeat(sorted_conf_indices[1:], 'b -> b h w', h=H, w=W)[consistency_mask]
    overlapping_masks[batchidx_valid, points2d_valid[..., 1], points2d_valid[..., 0]] = True

    return overlapping_masks


def compute_input_to_gt_co_vis_masks(
    depthmaps_all,    # [N_all, H, W]  scale-aligned depth maps for all frames
    pointmaps_input,  # [N_in, H, W, 3]  3D point maps for input frames (fallback)
    intrinsics_all,   # [N_all, 3, 3]  pixel-space intrinsics for all frames
    extrinsics_all,   # [N_all, 4, 4]  w2c for all frames
    input_indices,    # list/tensor: indices of input frames into all_frames
    image_sizes,      # (H, W)
    depth_threshold=0.05,
    device_='cuda',
    has_depth=None,   # [N_all] bool or None
    dilation_radius=4,  # morphological dilation to fill gaps; 0 = disabled
):
    """
    For every GT frame (all N_all frames), compute which pixels are visible
    from at least one input view.

    Uses **backward projection** (target pixel → 3D world → input frame) which
    produces dense masks by design, avoiding the point-splatting holes that the
    old forward-projection approach suffered from.

    For novel frames that have no depth map (has_depth=False), falls back to
    forward projection from input point maps.

    Returns
    -------
    overlapping_masks : [N_all, H, W] bool
        True  = pixel is co-visible with at least one input frame
        False = pixel is occluded / not seen by any input frame
    """
    H, W = image_sizes
    N_all = depthmaps_all.shape[0]

    overlapping_masks = torch.zeros((N_all, H, W), dtype=torch.bool, device=device_)

    if isinstance(input_indices, list):
        input_indices_t = torch.tensor(input_indices, device=device_, dtype=torch.long)
    else:
        input_indices_t = input_indices.long()
    N_in = input_indices_t.shape[0]
    input_set = set(input_indices_t.tolist())

    # ── 1. Input frames are directly observed → all pixels visible ──────────
    for i in input_set:
        overlapping_masks[i] = True

    # ── 2. Pixel grid for unprojection ──────────────────────────────────────
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device_, dtype=torch.float32),
        torch.arange(W, device=device_, dtype=torch.float32),
        indexing='ij',
    )  # each [H, W]

    # ── 3. Input cameras & depths (used in backward projection check) ────────
    in_depthmaps  = depthmaps_all[input_indices_t]    # [N_in, H, W]
    in_extrinsics = extrinsics_all[input_indices_t]   # [N_in, 4, 4]
    in_intrinsics = intrinsics_all[input_indices_t]   # [N_in, 3, 3]

    # Per-input max depth for absolute occlusion threshold (robust to invalid/zero pixels)
    in_d_max = in_depthmaps.flatten(1).max(dim=1).values.clamp(min=1e-6)  # [N_in]

    # ── 4. Classify novel frames ─────────────────────────────────────────────
    novel_with_depth = [
        i for i in range(N_all)
        if i not in input_set and (has_depth is None or bool(has_depth[i]))
    ]
    novel_without_depth = [
        i for i in range(N_all)
        if i not in input_set and has_depth is not None and not bool(has_depth[i])
    ]

    # ── 5. Backward projection for novel frames that have a depth map ────────
    #
    # For each target pixel:
    #   1. unproject with target depth → 3D world point
    #   2. project into each input frame
    #   3. check geometric bounds + depth consistency in INPUT frame
    #   4. mark target pixel visible if consistent with any input
    #
    for tgt_idx in novel_with_depth:
        depth_t = depthmaps_all[tgt_idx]     # [H, W]
        K_t     = intrinsics_all[tgt_idx]    # [3, 3]
        w2c_t   = extrinsics_all[tgt_idx]    # [4, 4]
        c2w_t   = torch.linalg.inv(w2c_t)   # [4, 4]

        valid_depth = depth_t > 0            # [H, W]

        # Unproject: image → camera → world
        fx, fy = K_t[0, 0], K_t[1, 1]
        cx, cy = K_t[0, 2], K_t[1, 2]
        z_t = depth_t                        # [H, W]
        pts_cam_homo = torch.stack([
            ((xs - cx) * z_t / fx).reshape(-1),
            ((ys - cy) * z_t / fy).reshape(-1),
            z_t.reshape(-1),
            torch.ones(H * W, device=device_),
        ], dim=-1)  # [H*W, 4]

        pts_world = (c2w_t @ pts_cam_homo.T).T  # [H*W, 4]  (homogeneous world)

        # Project world points into every input frame
        pts2cam_in = torch.einsum('bpq,nq->bnp', in_extrinsics, pts_world)            # [N_in, H*W, 4]
        pts_prj_in = torch.einsum('bpq,bnq->bnp', in_intrinsics, pts2cam_in[..., :3]) # [N_in, H*W, 3]
        proj_z_in  = pts2cam_in[..., 2]                                                # [N_in, H*W]
        pts_2d_in  = pts_prj_in[..., :2] / proj_z_in.unsqueeze(-1).clamp(min=1e-6)   # [N_in, H*W, 2]

        geom_valid = (
            (pts_2d_in[..., 0] >= 0) & (pts_2d_in[..., 1] >= 0) &
            (pts_2d_in[..., 0] < W)  & (pts_2d_in[..., 1] < H)  &
            (proj_z_in > 0)
        )  # [N_in, H*W]

        # Sample input-frame depth at projected locations
        grid_in = pts_2d_in.reshape(N_in, H, W, 2).clone()
        grid_in[..., 0] = (grid_in[..., 0] + 0.5) / W * 2.0 - 1.0
        grid_in[..., 1] = (grid_in[..., 1] + 0.5) / H * 2.0 - 1.0
        sampled_in = F.grid_sample(
            in_depthmaps[:, None], grid=grid_in, mode='bilinear', align_corners=False
        )[:, 0]  # [N_in, H, W]

        # One-sided occlusion test:
        #   - A point is OCCLUDED only if its projected depth significantly EXCEEDS
        #     the closest surface visible in the input frame (proj_z > sampled + tol).
        #   - Two-sided abs() was too strict and caused sparse masks for noisy depth.
        #   - When input has no depth at projected location (sampled_in == 0),
        #     skip the depth check and rely on geometric validity alone.
        proj_z_2d     = proj_z_in.reshape(N_in, H, W)          # [N_in, H, W]
        sampled_valid = sampled_in > 0                          # [N_in, H, W]
        abs_tol       = depth_threshold * in_d_max[:, None, None]  # [N_in, 1, 1]
        depth_ok = ((proj_z_2d - sampled_in) < abs_tol) | ~sampled_valid  # [N_in, H, W]

        visible = (
            valid_depth[None].expand(N_in, -1, -1) &
            geom_valid.reshape(N_in, H, W)          &
            depth_ok
        )  # [N_in, H, W]

        mask = visible.any(dim=0)  # [H, W]

        # Small dilation to close gaps at object boundaries / depth=0 pixels
        if dilation_radius > 0:
            k = 2 * dilation_radius + 1
            mask = F.max_pool2d(
                mask.float()[None, None], kernel_size=k, stride=1, padding=dilation_radius
            )[0, 0].bool()

        overlapping_masks[tgt_idx] = mask

    # ── 6. Fallback: forward projection for no-depth novel frames ────────────
    #
    # Project input 3D points into the target frame and mark landing pixels.
    # This is inherently sparse (point splatting); dilation fills the gaps.
    #
    if novel_without_depth:
        nd_t          = torch.tensor(novel_without_depth, device=device_, dtype=torch.long)
        nd_extrinsics = extrinsics_all[nd_t]   # [N_nd, 4, 4]
        nd_intrinsics = intrinsics_all[nd_t]   # [N_nd, 3, 3]
        N_nd = len(novel_without_depth)

        for in_idx in range(N_in):
            pts = pointmaps_input[in_idx]  # [H, W, 3]
            pts_homo = torch.cat(
                [pts.reshape(-1, 3), torch.ones(H * W, 1, device=device_)], dim=-1
            )  # [H*W, 4]

            pts2cam = torch.einsum('bpq,nq->bnp', nd_extrinsics, pts_homo)            # [N_nd, H*W, 4]
            pts_prj = torch.einsum('bpq,bnq->bnp', nd_intrinsics, pts2cam[..., :3])   # [N_nd, H*W, 3]
            proj_z  = pts2cam[..., 2]                                                   # [N_nd, H*W]
            pts_2d  = pts_prj[..., :2] / proj_z.unsqueeze(-1).clamp(min=1e-6)         # [N_nd, H*W, 2]

            geom_valid = (
                (pts_2d[..., 0] >= 0) & (pts_2d[..., 1] >= 0) &
                (pts_2d[..., 0] < W)  & (pts_2d[..., 1] < H)  &
                (proj_z > 0)
            )  # [N_nd, H*W]

            # Scatter to pixel grid
            sparse = torch.zeros(N_nd, H, W, dtype=torch.bool, device=device_)
            pts_2d_hw  = pts_2d.reshape(N_nd, H, W, 2)
            valid_flat = geom_valid.reshape(N_nd, H, W)
            b_idx = repeat(torch.arange(N_nd, device=device_), 'b -> b h w', h=H, w=W)[valid_flat]
            pix   = pts_2d_hw[valid_flat].long()
            pix[:, 0].clamp_(0, W - 1)
            pix[:, 1].clamp_(0, H - 1)
            if pix.shape[0] > 0:
                sparse[b_idx, pix[:, 1], pix[:, 0]] = True

            # Dilation to fill point-splatting holes.
            # Use 2× the radius compared to backward projection because forward
            # splatting leaves larger gaps (especially for perspective distortion).
            if dilation_radius > 0:
                fwd_r = dilation_radius * 2
                k = 2 * fwd_r + 1
                sparse = F.max_pool2d(
                    sparse.float(), kernel_size=k, stride=1, padding=fwd_r
                ).bool()

            overlapping_masks[nd_t] |= sparse

    return overlapping_masks