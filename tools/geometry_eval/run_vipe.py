
import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import torch
import shutil
import logging
import pycolmap
import numpy as np
from scipy.spatial import cKDTree

# Import the ViPE library and the conversion function (the script must be in the same
# directory or the package must be installed)
from vipe.utils.io import ArtifactPath
from vipe_to_colmap import convert_vipe_to_colmap

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Argument parsing
parser = argparse.ArgumentParser(description="ViPE pose estimation and evaluation script")
parser.add_argument("scene", type=str, help="Scene name to process (e.g. scene0001)")
parser.add_argument("--eval_gt", action="store_true", help="Evaluate using the GT videos")
parser.add_argument("--root_path", type=str, default="eccv_ours",
                    help="Root results directory containing {scene}/img2vid (default: eccv_ours)")
args = parser.parse_args()

eval_gt = args.eval_gt
root_path = Path(args.root_path)
source_path = root_path / args.scene / "img2vid"

def load_colmap_poses(colmap_path: Path):
    """Load poses from images.txt and return them as a dict of T_c2w matrices."""

    reconstruction = pycolmap.Reconstruction(colmap_path)
    poses = {}

    # RDF (COLMAP) -> RUB (OpenCV/NeRF) conversion
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1

    for img_id, img in reconstruction.images.items():
        # 1. Get the W2C matrix
        try:
            cfw = img.cam_from_world
            raw_w2c = cfw().matrix() if callable(cfw) else cfw.matrix()
        except AttributeError:
            # Fallback for older API versions
            raw_w2c = np.eye(4)
            # Recent versions may require composing attributes directly instead of rotation_matrix()
            try:
                raw_w2c[:3, :3] = img.rotation_matrix()
            except AttributeError:
                # The most reliable way to turn qvec into a matrix in pycolmap 3.x
                from scipy.spatial.transform import Rotation as R_scipy
                # If img.qvec is unavailable, use img.cam_from_world().rotation.quat etc.
                # Here we use the most common qvec/tvec fallback
                q = img.qvec if hasattr(img, 'qvec') else img.cam_from_world().rotation.quat
                raw_w2c[:3, :3] = R_scipy.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            raw_w2c[:3, 3] = img.tvec if hasattr(img, 'tvec') else img.cam_from_world().translation

        # 2. Expand 3x4 to 4x4 if needed (key to avoiding LinAlgError)
        w2c_4x4 = np.eye(4)
        w2c_4x4[:raw_w2c.shape[0], :raw_w2c.shape[1]] = raw_w2c

        # 3. Compute C2W via matrix inverse
        c2w_mat = np.linalg.inv(w2c_4x4)

        # 4. Coordinate-system correction (RDF -> RUB)
        res_c2w = c2w_mat @ flip_yz

        # Filename key
        name = os.path.basename(img.name)
        idx = int(re.search(r"\d+", name).group())
        key = f"{idx:03d}.png"

        poses[key] = res_c2w

    return poses, reconstruction

def load_transform_json(transform_json_path: str, only_subdir: str = None, stride: int = 1):
    with open(transform_json_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])

    # Apply stride: subsample frames at the given interval
    if stride > 1:
        frames = frames[::stride]

    gt_poses = {}
    for fr in frames:
        fp = fr["file_path"].replace("\\", "/")  # Handle Windows paths
        if only_subdir is not None:
            # Only use frames whose path contains samples-rgb/
            if f"/{only_subdir}/" not in fp and not fp.startswith(f"{only_subdir}/") and not fp.startswith(f"./{only_subdir}/"):
                continue

        key = os.path.basename(fp)

        T = np.array(fr["transform_matrix"], dtype=np.float64)
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64)
            T4[:3, :4] = T
            T = T4

        gt_poses[key] = T

    return gt_poses

def compute_chamfer_distance(pcd1, pcd2):
    """Compute the Chamfer Distance between two point clouds."""
    # pcd1, pcd2: (N, 3) numpy array
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)

    dist1, _ = tree1.query(pcd2)
    dist2, _ = tree2.query(pcd1)

    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer_dist

def umeyama_alignment(x, y):
    n = x.shape[0]
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    sigma_x = np.mean(np.sum((x - mu_x)**2, axis=1))
    X, Y = (x - mu_x).T, (y - mu_y).T
    K = Y @ X.T / n
    U, D, VT = np.linalg.svd(K)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VT) < 0: S[2, 2] = -1
    R_found = U @ S @ VT
    s_found = (1.0 / sigma_x) * np.trace(np.diag(D) @ S)
    t_found = mu_y - s_found * R_found @ mu_x
    return s_found, R_found, t_found

def evaluate_vipe_scene(vipe_colmap_path, gt_transforms_json, only_subdir=None):
    # 1. Load poses and reconstruction info
    est_poses, reconstruction = load_colmap_poses(Path(vipe_colmap_path))
    gt_poses = load_transform_json(str(gt_transforms_json), only_subdir=only_subdir)

    # 2. Find common frames and align (Umeyama)
    common_names = sorted(set(est_poses.keys()) & set(gt_poses.keys()))
    if len(common_names) < 3:
        return None

    p_est = np.array([est_poses[n][:3, 3] for n in common_names])
    p_gt = np.array([gt_poses[n][:3, 3] for n in common_names])

    s, R_align, t_align = umeyama_alignment(p_est, p_gt)

    # 3. Compute pose error (with alignment applied)
    t_errs, r_errs = [], []
    for n in common_names:
        T_est = est_poses[n]
        # Align Est to GT coordinate
        T_est_aligned = np.eye(4)
        T_est_aligned[:3, :3] = R_align @ T_est[:3, :3]
        T_est_aligned[:3, 3] = s * (R_align @ T_est[:3, 3]) + t_align

        # Error computation
        err_t = np.linalg.norm(gt_poses[n][:3, 3] - T_est_aligned[:3, 3])
        R_rel = T_est_aligned[:3, :3] @ gt_poses[n][:3, :3].T
        err_r = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)))

        t_errs.append(err_t)
        r_errs.append(err_r)

    # 4. Compute Chamfer Distance
    # Est PCD: transform sparse points from ViPE into the GT coordinate frame
    est_pcd_raw = np.array([p.xyz for p in reconstruction.points3D.values()])
    chamfer = -1

    return {
        "t_mean": np.mean(t_errs),
        "r_mean": np.mean(r_errs),
        "chamfer": chamfer,
        "reg_rate": len(common_names) / len(gt_poses)
    }

####################

save_path = Path(source_path.parent.parent.stem + "_vipe") / source_path.parent.stem if not eval_gt else Path("gt_vipe") / source_path.parent.stem
# if save_path.exists():
#     shutil.rmtree(save_path)
save_path.mkdir(parents=True, exist_ok=True)


all_scene_results = {}
scene_lists = [d for d in source_path.iterdir() if d.is_dir()]
print(f"Number of folders found: {len(scene_lists)}")
for scene in tqdm(scene_lists):

    target_video_path = scene / 'samples-rgb.mp4' if not eval_gt else scene / 'gt-rgb.mp4'
    save_path_target = save_path / scene.stem
    save_path_target.mkdir(parents=True, exist_ok=True)
    # Final destination in COLMAP format
    colmap_output_path = save_path / f"{scene.stem}_colmap"

    # # 2. Check that the video file actually exists
    if not target_video_path.exists():
        print(f"Warning: {target_video_path} not found, skipping.")
        continue

    # 2. Run ViPE infer (video -> pose/depth estimation)
    infer_command = [
        "vipe", "infer", str(target_video_path),
        "--output", str(save_path_target),
        "--pipeline", "dav3"  # should install depth-anything3
    ]

    try:
        if not colmap_output_path.exists():
            logger.info(f"Running ViPE infer for {scene.name}...")
            subprocess.run(infer_command, check=True, capture_output=True)

        # 3. Convert ViPE results to COLMAP format
        # Find the generated files via ArtifactPath.
        artifacts = list(ArtifactPath.glob_artifacts(save_path_target, use_video=True))

        if not artifacts:
            logger.warning(f"No artifacts found in {save_path_target}")
            continue

        for artifact in artifacts:
            logger.info(f"Converting {artifact.artifact_name} to COLMAP format...")
            # Call the conversion function
            # depth_step: 16 (default), use_slam_map: False (default setting)
            convert_vipe_to_colmap(
                artifact=artifact,
                output_path=colmap_output_path,
                depth_step=16,
                use_slam_map=False
            )

        logger.info(f"Successfully processed and converted: {scene.name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"ViPE infer failed for {scene.name}: {e}")
    except Exception as e:
        logger.error(f"Conversion failed for {scene.name}: {e}")

    result = evaluate_vipe_scene(str(colmap_output_path), str(scene / 'transforms.json'), only_subdir="samples-rgb")
    if result:
        all_scene_results[scene.name] = result
        print(f"  Success: T_err={result['t_mean']:.6f}, R_err={result['r_mean']:.6f}, Reg={result['reg_rate']*100:.1f}%")
    else:
        print(f"  Failed: Could not reconstruct scene {scene.name}")


# 4. Overall statistics report
if all_scene_results:
    final_t = [res['t_mean'] for res in all_scene_results.values()]
    final_r = [res['r_mean'] for res in all_scene_results.values()]

    print("\n" + "="*50)
    print(f"TOTAL EVALUATION SUMMARY ({len(all_scene_results)} scenes)")
    print(f"Avg Translation Error: {np.mean(final_t):.6f} ± {np.std(final_t):.6f}")
    print(f"Avg Rotation Error:    {np.mean(final_r):.6f} ± {np.std(final_r):.6f} deg")
    print("="*50)

    # Save results to a separate JSON file
    with open(save_path / "final_metrics.json", "w") as f:
        json.dump(all_scene_results, f, indent=4)
