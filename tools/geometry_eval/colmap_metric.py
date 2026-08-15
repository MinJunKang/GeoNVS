import os
import json
import random
import shutil
import argparse
import numpy as np
import pycolmap
from pathlib import Path

# Set environment variables to make results as deterministic as possible (top of the file)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
random.seed(0)
np.random.seed(0)

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

def compute_errors(gt_poses, est_poses):
    t_errors, r_errors = [], []
    common_keys = sorted(set(gt_poses.keys()) & set(est_poses.keys()))
    for k in common_keys:
        R_gt, T_gt = gt_poses[k][:3, :3], gt_poses[k][:3, 3]
        R_est, T_est = est_poses[k][:3, :3], est_poses[k][:3, 3]
        err_t = np.linalg.norm(T_gt - T_est)
        t_errors.append(err_t)
        R_rel = np.dot(R_est, R_gt.T)
        cos_theta = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
        r_errors.append(np.degrees(np.arccos(cos_theta)))
    return np.array(t_errors), np.array(r_errors)

def run_pycolmap_reconstruction(image_dir, output_dir, image_names=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    # If image_names is given, use only that list; otherwise use all images
    if image_names is None:
        image_names = sorted([p.name for p in Path(image_dir).glob("*.png")])

    # 1. Feature Extraction & Matching
    pycolmap.extract_features(database_path=str(db_path), image_path=str(image_dir), image_names=image_names)
    pycolmap.match_exhaustive(str(db_path))

    # 2. Configure Incremental Pipeline options (key to fixing the error)
    # IncrementalPipelineOptions must be used instead of IncrementalMapperOptions.
    options = pycolmap.IncrementalPipelineOptions()

    # Limit the thread count to 1 for deterministic results
    options.num_threads = 1

    # In recent versions, detailed options are accessed via the mapper sub-object.
    # Tune BA iteration counts etc. to improve convergence precision.
    try:
        options.mapper.random_seed = 42  # Use a fixed value instead of -1 (random)
        options.mapper.ba_global_options.max_num_iterations = 100
        options.mapper.ba_local_options.max_num_iterations = 50
    except AttributeError:
        # The mapper attribute may not exist depending on the version, so handle the exception
        pass

    # 3. Run Incremental Mapping
    # Pass the arguments explicitly converted to str
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_dir),
        output_path=str(output_dir),
        options=options
    )
    if not maps:
        return None

    recon = maps[0]



    return recon

def align_and_evaluate(reconstruction, gt_poses):
    est_poses_raw = {}
    flip_mat = np.diag([1, -1, -1, 1]) # OpenGL <-> OpenCV correction

    for img_id, img in reconstruction.images.items():
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        try:
            R_w2c, t_w2c = cfw.rotation.matrix(), cfw.translation
        except AttributeError:
            R_w2c, t_w2c = cfw.matrix()[:3, :3], cfw.matrix()[:3, 3]
        T_w2c = np.eye(4)
        T_w2c[:3, :3], T_w2c[:3, 3] = R_w2c, t_w2c
        est_poses_raw[os.path.basename(img.name)] = np.linalg.inv(T_w2c) @ flip_mat

    common_names = sorted(set(est_poses_raw.keys()) & set(gt_poses.keys()))
    if not common_names: return

    p_est = np.array([est_poses_raw[n][:3, 3] for n in common_names])
    p_gt = np.array([gt_poses[n][:3, 3] for n in common_names])

    s, R_align, t = umeyama_alignment(p_est, p_gt)

    final_est_poses = {}
    for name, T_c2w in est_poses_raw.items():
        T_aligned = np.eye(4)
        T_aligned[:3, 3] = s * (R_align @ T_c2w[:3, 3]) + t
        T_aligned[:3, :3] = R_align @ T_c2w[:3, :3]
        final_est_poses[name] = T_aligned

    # print("GT example keys:", list(sorted(gt_poses.keys())))
    # print("EST example keys:", list(sorted(est_poses_raw.keys())))
    # print("Common:", len(common_names), "first:", common_names)

    t_err, r_err = compute_errors(gt_poses, final_est_poses)

    # print(f"\n--- Aligned Evaluation Results ---")
    # print(f"Mean Translation Error: {np.mean(t_err):.6f}")
    # print(f"Mean Rotation Error: {np.mean(r_err):.6f} degrees")
    # print(f"Scale Factor (COLMAP -> GT): {s:.6f}")

    # Changed to also return the aligned estimated poses (final_est_poses) and the matched GT poses
    matched_gt_poses = {k: gt_poses[k] for k in common_names}

    return t_err, r_err, final_est_poses, matched_gt_poses, (s, R_align, t, flip_mat)


def save_camera_wireframes_ply(poses_dict, output_path, color=[255, 0, 0], scale=0.3):
    """
    poses_dict: {image_name: 4x4 T_c2w matrix}
    color: [R, G, B] - recommended: [255, 0, 0] for predictions (PRED), [0, 255, 0] for GT
    scale: size of the camera pyramid (set to 0.5 or larger to see it bigger)
    """
    vertices = []
    edges = []

    # Camera pyramid shape definition
    pyramid_local = np.array([
        [0, 0, 0],                         # 0: camera center
        [-0.5, -0.5, 1], [0.5, -0.5, 1],   # 1, 2: top corners
        [0.5, 0.5, 1], [-0.5, 0.5, 1]      # 3, 4: bottom corners
    ]) * scale

    keys = sorted(poses_dict.keys())
    v_offset = 0

    for i, key in enumerate(keys):
        T_c2w = poses_dict[key]
        R = T_c2w[:3, :3]
        t = T_c2w[:3, 3]

        # 1. Compute vertices and assign colors
        world_pts = (R @ pyramid_local.T).T + t
        for p in world_pts:
            # Apply the color argument to every point
            vertices.append(f"{p[0]} {p[1]} {p[2]} {color[0]} {color[1]} {color[2]}")

        # 2. Add the per-camera wireframe edges
        # (4 center-to-corner, 4 corner-to-corner)
        cam_edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
        for start, end in cam_edges:
            edges.append(f"{v_offset + start} {v_offset + end}")

        # 3. Add trajectory lines connecting cameras (previous center - current center)
        if i > 0:
            prev_center_idx = (i - 1) * 5  # Index 0 of the previous camera
            curr_center_idx = i * 5        # Index 0 of the current camera
            edges.append(f"{prev_center_idx} {curr_center_idx}")

        v_offset += 5

    # Write the PLY file
    header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element edge {len(edges)}
property int vertex1
property int vertex2
end_header
"""
    with open(output_path, "w") as f:
        f.write(header)
        f.write("\n".join(vertices) + "\n")
        f.write("\n".join(edges))

    print(f"Saved: {output_path} (Color: {color}, Scale: {scale})")


def evaluate_single_scene(scene_path, workspace_root, num_runs=5, stride=1, eval_gt=False):
    """Run a single scene n times and return the average errors."""
    image_path = scene_path / "samples-rgb" if not eval_gt else scene_path / "gt-rgb"
    json_path = scene_path / "transforms.json"

    if not json_path.exists():
        print(f"Skipping {scene_path.name}: transforms.json not found.")
        return None

    gt_data = load_transform_json(str(json_path), only_subdir="samples-rgb", stride=stride)
    selected_image_names = sorted(gt_data.keys())

    scene_t_means = []
    scene_r_means = []
    registered_counts = []

    for i in range(num_runs):
        print(f"  > Run [{i+1}/{num_runs}]...")
        run_workspace = workspace_root / scene_path.name / f"colmap_run_{i}"

        recon = run_pycolmap_reconstruction(image_path, run_workspace, image_names=selected_image_names)

        if recon:
            # 1. Get the alignment parameters (params)
            t_errs, r_errs, aligned_est, aligned_gt, params = align_and_evaluate(recon, gt_data)
            s, R_align, t_align, flip_mat = params # Renamed to t_align (to avoid duplication)

            if t_errs is not None and len(t_errs) > 0:
                scene_t_means.append(np.mean(t_errs))
                scene_r_means.append(np.mean(r_errs))
                registered_counts.append(len(t_errs))

                # 2. Save trajectories
                save_camera_wireframes_ply(aligned_est, run_workspace / "traj_est.ply", color=[255, 0, 0], scale=0.5)
                save_camera_wireframes_ply(aligned_gt, run_workspace / "traj_gt.ply", color=[0, 255, 0], scale=0.5)

                # 3. Transform and save the point cloud (coordinate-frame unification)
                points3D = recon.points3D
                transformed_points = []
                # Extract only the rotation/flip component from flip_mat (3x3)
                R_flip = flip_mat[:3, :3]

                T_align = np.eye(4)
                T_align[:3, :3] = s * R_align
                T_align[:3, 3] = t_align

                for p_id, p in points3D.items():
                    xyz_raw = p.xyz

                    # Point transform order for orientation correction
                    # 1. First rotate the raw point by the rotation component of flip_mat (orientation match)
                    xyz_flipped = R_flip @ xyz_raw

                    # 2. Umeyama alignment (position and global rotation match)
                    # s * (R_align @ xyz_flipped) + t_align
                    xyz_aligned = (T_align[:3, :3] @ xyz_flipped) + T_align[:3, 3]

                    rgb = p.color
                    transformed_points.append(
                        f"{xyz_aligned[0]} {xyz_aligned[1]} {xyz_aligned[2]} {rgb[0]} {rgb[1]} {rgb[2]}"
                    )

                # Save the transformed point cloud as PLY
                ply_path = run_workspace / "sparse_points_aligned.ply"
                header = f"ply\nformat ascii 1.0\nelement vertex {len(transformed_points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"

                with open(ply_path, "w") as f:
                    f.write(header + "\n".join(transformed_points))

                print(f"  [COLMAP] Aligned Point cloud saved: {ply_path}")

    if not scene_t_means:
        return None

    return {
        "t_mean": np.mean(scene_t_means),
        "r_mean": np.mean(scene_r_means),
        "reg_rate": np.mean(registered_counts) / len(gt_data)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COLMAP-based pose evaluation (multi-run pycolmap reconstruction vs. GT transforms.json)"
    )
    parser.add_argument("--dataset_root", type=str, default="./ours/tnt-cf3dgs_3_1.0/img2vid",
                        help="Root folder containing per-scene subdirectories "
                             "(default: ./ours/tnt-cf3dgs_3_1.0/img2vid; "
                             "other examples: ./seva/dl3dv140_9_1.0/img2vid, "
                             "./longvidseq_rebuttal/seva/mipnerf360_6_1.0/img2vid, "
                             "./ours_vggt_iv/dl3dv10_6_1.0/img2img)")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of pycolmap runs per scene (default: 3)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride applied to transforms.json frames (default: 1)")
    parser.add_argument("--eval_gt", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate the GT videos (gt-rgb) instead of predictions (samples-rgb). "
                             "Default: True; use --no-eval_gt to evaluate predictions.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    NUM_RUNS = args.num_runs
    STRIDE = args.stride
    EVAL_GT = args.eval_gt
    tag = 'GT' if EVAL_GT else 'PRED'

    GLOBAL_WORKSPACE = Path(f"./colmap_results_{dataset_root.parent.stem}_{dataset_root.parent.parent.stem}_{tag}")

    if GLOBAL_WORKSPACE.exists():
        shutil.rmtree(GLOBAL_WORKSPACE)

    # 2. Get the scene list (directories only)
    scenes = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    print(f"Found {len(scenes)} scenes to evaluate.")

    all_scene_results = {}

    # 3. Iterate over all scenes
    for idx, scene_path in enumerate(scenes):
        print(f"\n[{idx+1}/{len(scenes)}] Evaluating Scene: {scene_path.name}")
        result = evaluate_single_scene(scene_path, GLOBAL_WORKSPACE, num_runs=NUM_RUNS, stride=STRIDE, eval_gt=EVAL_GT)

        if result:
            all_scene_results[scene_path.name] = result
            print(f"  Success: T_err={result['t_mean']:.6f}, R_err={result['r_mean']:.6f}, Reg={result['reg_rate']*100:.1f}%")
        else:
            print(f"  Failed: Could not reconstruct scene {scene_path.name}")

    # 4. Overall statistics report
    if all_scene_results:
        final_t = [res['t_mean'] for res in all_scene_results.values()]
        final_r = [res['r_mean'] for res in all_scene_results.values()]
        final_reg = [res['reg_rate'] for res in all_scene_results.values()]

        print("\n" + "="*50)
        print(f"TOTAL EVALUATION SUMMARY ({len(all_scene_results)} scenes)")
        print(f"Avg Translation Error: {np.mean(final_t):.6f} ± {np.std(final_t):.6f}")
        print(f"Avg Rotation Error:    {np.mean(final_r):.6f} ± {np.std(final_r):.6f} deg")
        print(f"Avg Registration Rate: {np.mean(final_reg)*100:.2f}%")
        print("="*50)

        # Save results to a separate JSON file
        with open(GLOBAL_WORKSPACE / "final_metrics.json", "w") as f:
            json.dump(all_scene_results, f, indent=4)
