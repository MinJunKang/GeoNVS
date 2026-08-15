#!/usr/bin/env python3
"""
Combined script: convert ViPE results to COLMAP format, then visualize with viser.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Tuple

import cv2
import imageio
import numpy as np
import torch
import viser
import viser.transforms as tf
import PIL.Image as Image

from scipy.spatial.transform import Rotation

from vipe.slam.interface import SLAMMap
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion functions taken from vipe_to_colmap.py
# ---------------------------------------------------------------------------

def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to [w, x, y, z]


def matrix_to_colmap_pose(c2w_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world matrix to COLMAP format (world-to-camera)."""
    w2c = np.linalg.inv(c2w_matrix)
    quaternion = quaternion_from_matrix(w2c)
    translation = w2c[:3, 3]
    return quaternion, translation


def write_cameras_txt(output_dir: Path, artifact: ArtifactPath, frame_width: int, frame_height: int):
    """Write COLMAP cameras.txt file."""
    cameras_file = output_dir / "cameras.txt"

    _, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)

    assert camera_types[0] == CameraType.PINHOLE, "Only PINHOLE camera type is supported"
    fx, fy, cx, cy = intrinsics[0].cpu().numpy()

    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        fx, fy, cx, cy = intrinsics[0]
        f.write(f"1 PINHOLE {frame_width} {frame_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    logger.info(f"Written cameras.txt with intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


def write_images_txt(output_dir: Path, artifact: ArtifactPath):
    """Write COLMAP images.txt file."""
    images_file = output_dir / "images.txt"

    pose_data = np.load(artifact.pose_path)
    poses = pose_data["data"]   # (N, 4, 4)
    indices = pose_data["inds"]  # frame indices

    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")

        for i, (pose_matrix, frame_idx) in enumerate(zip(poses, indices)):
            quaternion, translation = matrix_to_colmap_pose(pose_matrix)
            qw, qx, qy, qz = quaternion
            tx, ty, tz = translation
            image_name = f"images/frame_{frame_idx:06d}.jpg"
            f.write(f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {image_name}\n")
            f.write("\n")

    logger.info(f"Written images.txt with {len(poses)} images")


def write_points3d_txt_from_slam_map(output_dir: Path, artifact: ArtifactPath):
    """Write points3D.txt from SLAM map."""
    assert artifact.slam_map_path.exists(), "SLAM map not found, please refer to README.md for more details."

    slam_map = SLAMMap.load(artifact.slam_map_path, device=torch.device("cpu"))

    points3d_file = output_dir / "points3D.txt"
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {slam_map.dense_disp_xyz.shape[0]}\n")

        point_id = 1
        for keyframe_idx, frame_idx in enumerate(slam_map.dense_disp_frame_inds):
            xyz, rgb = slam_map.get_dense_disp_pcd(keyframe_idx)
            xyz = xyz.cpu().numpy()
            rgb = rgb.cpu().numpy()

            for xyz, rgb in zip(xyz, rgb):
                x, y, z = xyz
                r, g, b = (rgb * 255).astype(np.uint8)
                f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0 {frame_idx} {point_id} 0 0 0 0\n")
                point_id += 1


def write_points3d_txt_from_depth(
    output_dir: Path, artifact: ArtifactPath, depth_step: int, spatial_subsample: int = 4
):
    """Write COLMAP points3D.txt from depth maps."""
    _, pose_data = read_pose_artifacts(artifact.pose_path)
    _, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)
    camera_type = camera_types[0]

    image_dir = output_dir / "images"
    images = sorted(list(image_dir.glob("*.jpg")))

    all_points = []
    point_id = 1
    rays: np.ndarray | None = None

    for idx, (_, depth) in enumerate(read_depth_artifacts(artifact.depth_path)):
        if idx % 30 == 0:
            logger.info(f"Processed {idx} depth maps")

        if idx % depth_step != 0:
            continue

        rgb = cv2.cvtColor(cv2.imread(str(images[idx]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frame_height, frame_width = rgb.shape[:2]
        rgb = rgb[::spatial_subsample, ::spatial_subsample]

        if rays is None:
            camera_model = camera_type.build_camera_model(intrinsics[idx])
            disp_v, disp_u = torch.meshgrid(
                torch.arange(frame_height).float()[::spatial_subsample],
                torch.arange(frame_width).float()[::spatial_subsample],
                indexing="ij",
            )
            if camera_type == CameraType.PANORAMA:
                disp_v = disp_v / (frame_height - 1)
                disp_u = disp_u / (frame_width - 1)
            disp = torch.ones_like(disp_v)
            pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
            rays = pts[..., :3].numpy()
            if camera_type != CameraType.PANORAMA:
                rays /= rays[..., 2:3]

        if depth is not None:
            pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
            depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
            rgb, pcd = rgb[depth_mask], pcd[depth_mask]
            c2w_matrix = pose_data[idx].matrix().numpy()
            pcd = pcd @ c2w_matrix[:3, :3].T + c2w_matrix[:3, 3][None]

            for pts_rgb, pts_xyz in zip(rgb, pcd):
                all_points.append(
                    (
                        point_id,
                        pts_xyz[0],
                        pts_xyz[1],
                        pts_xyz[2],
                        int(pts_rgb[0]),
                        int(pts_rgb[1]),
                        int(pts_rgb[2]),
                        0.0,
                        idx + 1,
                    )
                )
                point_id += 1

    points3d_file = output_dir / "points3D.txt"
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(all_points)}\n")

        for point_data in all_points:
            pid, x, y, z, r, g, b, error, image_id = point_data
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f} {image_id} {pid} 0 0 0 0\n")

    logger.info(f"Written points3D.txt with {len(all_points)} points")


def extract_frames(artifact: ArtifactPath, output_dir: Path) -> Tuple[int, int]:
    """Extract frames from video to individual image files."""
    video_path = artifact.rgb_path
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info(f"Extracting frames from {video_path}")

    frame_idx = 0
    for frame_idx, rgb in read_rgb_artifacts(video_path):
        frame_path = images_dir / f"frame_{frame_idx:06d}.jpg"
        frame_height, frame_width = rgb.shape[:2]
        imageio.imwrite(str(frame_path), (rgb.cpu().numpy() * 255).astype(np.uint8))
        if frame_idx % 30 == 0:
            logger.info(f"Extracted {frame_idx} frames")

    logger.info(f"Extracted {frame_idx} frames to {images_dir}")
    return frame_width, frame_height


def convert_vipe_to_colmap(artifact: ArtifactPath, output_path: Path, depth_step: int, use_slam_map: bool):
    """Convert ViPE reconstruction results to COLMAP format."""
    logger.info(
        f"Converting ViPE results from {artifact.base_path} ({artifact.artifact_name}) to COLMAP format at {output_path}"
    )

    required_files = [artifact.rgb_path, artifact.pose_path, artifact.intrinsics_path, artifact.depth_path]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    frame_width, frame_height = extract_frames(artifact, output_path)
    write_cameras_txt(output_path, artifact, frame_width, frame_height)
    write_images_txt(output_path, artifact)

    if use_slam_map:
        write_points3d_txt_from_slam_map(output_path, artifact)
    else:
        write_points3d_txt_from_depth(output_path, artifact, depth_step)

    logger.info("COLMAP conversion completed successfully!")
    logger.info(f"Output directory: {output_path}")


# ---------------------------------------------------------------------------
# Visualization functions taken from colmap_viser_plot.py
# ---------------------------------------------------------------------------

def read_colmap_text_data(path: Path):
    """Read data directly from the .txt files without pycolmap."""
    points3d = {}
    with open(path / "points3D.txt", "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            elems = line.split()
            pid = int(elems[0])
            xyz = np.array([float(elems[1]), float(elems[2]), float(elems[3])])
            rgb = np.array([int(elems[4]), int(elems[5]), int(elems[6])])
            points3d[pid] = (xyz, rgb)

    images = []
    with open(path / "images.txt", "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                continue
            elems = line.split()
            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            name = elems[9]
            images.append({
                "name": name,
                "quat": [qw, qx, qy, qz],  # wxyz
                "trans": [tx, ty, tz],
            })

    return points3d, images


def visualize_colmap(dataset_path: Path):
    """Visualize COLMAP results using viser."""
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    screenshot_dir = dataset_path / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    print("Reading text files directly...")
    points3d, images = read_colmap_text_data(dataset_path)

    # Point cloud
    xyz_list = np.array([v[0] for v in points3d.values()])
    rgb_list = np.array([v[1] for v in points3d.values()])
    server.scene.add_point_cloud("/points3D", points=xyz_list, colors=rgb_list, point_size=0.02)

    # Camera frustums and trajectory
    camera_centers = []
    camera_scale = 0.05
    for img in images:
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(np.array(img["quat"])),
            np.array(img["trans"]),
        ).inverse()

        center = T_world_camera.translation()
        camera_centers.append(center)

        server.scene.add_camera_frustum(
            f"/cameras/{img['name']}",
            fov=0.8,
            aspect=1.0,
            scale=camera_scale,
            wxyz=T_world_camera.rotation().wxyz,
            position=center,
            color=(0, 255, 255),
        )

    if len(camera_centers) > 1:
        camera_centers = np.array(camera_centers)
        points_start = camera_centers[:-1]
        points_end = camera_centers[1:]
        line_vertices = np.stack([points_start, points_end], axis=1)  # (N-1, 2, 3)
        server.scene.add_line_segments(
            "/trajectory",
            points=line_vertices,
            colors=np.array([255, 0, 0]),
            line_width=4.0,
        )

    # Viewpoint save/load GUI
    json_path = dataset_path / "viser_viewpoint.json"
    with server.gui.add_folder("Viewpoint Control"):
        save_btn = server.gui.add_button("Save Current View", icon=viser.Icon.CAMERA)
        load_btn = server.gui.add_button("Load Last View", icon=viser.Icon.DOWNLOAD)
        status_text = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    @server.on_client_connect
    def setup_viewpoint_handlers(client: viser.ClientHandle) -> None:
        @save_btn.on_click
        def _(_) -> None:
            view_data = {
                "position": client.camera.position.tolist(),
                "wxyz": client.camera.wxyz.tolist(),
                "look_at": client.camera.look_at.tolist(),
                "fov": client.camera.fov,
                "aspect": client.camera.aspect,
            }
            with open(json_path, "w") as f:
                json.dump(view_data, f, indent=4)
            status_text.value = f"Saved to {json_path.name}"
            print(f"Viewpoint saved: {json_path}")

        @load_btn.on_click
        def _(_) -> None:
            if not json_path.exists():
                status_text.value = "No saved view found!"
                return
            with open(json_path, "r") as f:
                data = json.load(f)
            client.camera.position = np.array(data["position"])
            client.camera.wxyz = np.array(data["wxyz"])
            client.camera.look_at = np.array(data["look_at"])
            client.camera.fov = data["fov"]
            status_text.value = "View loaded!"
            print("Viewpoint loaded and applied.")

    print(f"Visualization ready! Connect at: {server.get_host()}")
    while True:
        time.sleep(1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ViPE results to COLMAP format, then visualize with viser"
    )

    # Conversion arguments
    conv_group = parser.add_argument_group("conversion options")
    conv_group.add_argument(
        "--base_path", type=Path, required=False,
        help="ViPE artifact base path (e.g. /data/vipe_output)"
    )
    conv_group.add_argument(
        "--artifact_name", type=str, required=False,
        help="ViPE artifact name (e.g. scene_001)"
    )
    conv_group.add_argument(
        "--output_path", type=Path, required=False,
        help="COLMAP output directory path"
    )
    conv_group.add_argument(
        "--depth_step", type=int, default=1,
        help="Depth processing interval (default: 1, i.e. every frame)"
    )
    conv_group.add_argument(
        "--use_slam_map", action="store_true",
        help="Build points3D from the SLAM map instead of depth"
    )
    conv_group.add_argument(
        "--skip_conversion", action="store_true",
        help="Skip the conversion and directly visualize existing COLMAP data"
    )

    # Visualization arguments
    viz_group = parser.add_argument_group("visualization options")
    viz_group.add_argument(
        "--colmap_path", type=Path, required=False,
        help="Path to an existing COLMAP directory when using --skip_conversion"
    )
    viz_group.add_argument(
        "--no_visualize", action="store_true",
        help="Do not run the visualization after conversion"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip_conversion:
        # Visualize directly without conversion
        colmap_path = args.colmap_path
        if colmap_path is None:
            raise ValueError("--colmap_path must be specified when using --skip_conversion.")
        logger.info(f"Conversion skipped. COLMAP path: {colmap_path}")
    else:
        # Run the conversion
        if args.base_path is None or args.artifact_name is None or args.output_path is None:
            raise ValueError(
                "--base_path, --artifact_name and --output_path are all required for conversion."
            )
        artifact = ArtifactPath(base_path=args.base_path, artifact_name=args.artifact_name)
        convert_vipe_to_colmap(artifact, args.output_path, args.depth_step, args.use_slam_map)
        colmap_path = args.output_path

    if not args.no_visualize:
        logger.info(f"Starting viser visualization: {colmap_path}")
        visualize_colmap(colmap_path)


if __name__ == "__main__":
    main()
