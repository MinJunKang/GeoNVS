import json
import argparse
import numpy as np
from pathlib import Path
import torch
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance


def read_colmap_text_data(path: Path):
    """Read data directly from the .txt files without pycolmap."""
    points3d = {}
    # Read points3D.txt
    with open(path / "points3D.txt", "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            elems = line.split()
            id = int(elems[0])
            xyz = np.array([float(elems[1]), float(elems[2]), float(elems[3])])
            rgb = np.array([int(elems[4]), int(elems[5]), int(elems[6])])
            points3d[id] = (xyz, rgb)

    images = []
    with open(path / "images.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            elems = line.split()
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            name = elems[9]
            images.append({"name": name, "quat": [qw, qx, qy, qz], "trans": [tx, ty, tz]})

            _ = f.readline()  # Consume the next line (2D points)

    return points3d, images




def calculate_metrics(p1_np, p2_np, threshold=0.01, device="cuda"):
    """
    Compute F-score and Chamfer Distance at the same time.
    p1: Model (Source), p2: GT (Target)
    """
    if len(p1_np) == 0 or len(p2_np) == 0:
        return -1, -1, -1, -1

    p1 = torch.from_numpy(p1_np).float().to(device).unsqueeze(0)
    p2 = torch.from_numpy(p2_np).float().to(device).unsqueeze(0)

    # --- 1. Chamfer Distance (PyTorch3D built-in function) ---
    # Computes the mean point-to-point distance.
    # norm=2 (L2^2), norm=1 (L1)
    cd_l2, _ = chamfer_distance(p1, p2, norm=2)

    # --- 2. F-score computation ---
    # For precision (p1 -> p2)
    dist_p1, _, _ = knn_points(p1, p2, K=1)
    # For recall (p2 -> p1)
    dist_p2, _, _ = knn_points(p2, p1, K=1)

    # knn_points returns squared distances, so take the square root
    dist_p1 = torch.sqrt(dist_p1)
    dist_p2 = torch.sqrt(dist_p2)

    precision = (dist_p1 < threshold).float().mean()
    recall = (dist_p2 < threshold).float().mean()

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = torch.tensor(0.0).to(device)

    return f1.item(), precision.item(), recall.item(), cd_l2.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Chamfer Distance / F-score between method and GT ViPE point clouds"
    )
    parser.add_argument("--model", type=str, default="eccv_seva",
                        help="Method results prefix, e.g. eccv_ours or eccv_seva (default: eccv_seva)")
    parser.add_argument("--dataset", type=str, default="dl3dv140_9_1",
                        help="Dataset name, e.g. dl3dv140_9_1 or mipnerf360_3_1 (default: dl3dv140_9_1)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="F-score distance threshold (default: 0.01)")
    args = parser.parse_args()

    MODEL = args.model  # ours or seva
    DATASET = args.dataset
    THRESHOLD = args.threshold

    path1 = f"{MODEL}_vipe/{DATASET}/final_metrics.json"
    path2 = f"gt_vipe/{DATASET}/final_metrics.json"

    # 1. Read the JSON files
    with open(path1, 'r') as f:
        data1 = json.load(f)
    with open(path2, 'r') as f:
        data2 = json.load(f)

    # 2. Extract the intersection of scenes (using set intersection)
    scene_names1 = set(data1.keys())
    scene_names2 = set(data2.keys())

    # Keep only scenes present in both datasets, sorted
    scene_names = sorted(list(scene_names1.intersection(scene_names2)))

    # 3. Collect Chamfer Distance and other metrics
    final_output = {}
    for name in scene_names:
        model_dir = Path(f"{MODEL}_vipe/{DATASET}/{name}_colmap")
        gt_dir = Path(f"gt_vipe/{DATASET}/{name}_colmap")

        # 1. Take the existing metrics (from data1)
        scene_metrics = {
            "t_mean": data1[name].get("t_mean", 0),
            "r_mean": data1[name].get("r_mean", 0),
            "f_score": -1,
            "precision": -1,
            "recall": -1,
            "chamfer": -1,
            "reg_rate": data1[name].get("reg_rate", 1.0)
        }

        # 2. Chamfer Distance computation
        if (model_dir / "points3D.txt").exists() and (gt_dir / "points3D.txt").exists():
            try:
                p3d_model, images_model = read_colmap_text_data(model_dir)
                p3d_gt, images_gt = read_colmap_text_data(gt_dir)

                xyz_model = np.array([v[0] for v in p3d_model.values()])
                xyz_gt = np.array([v[0] for v in p3d_gt.values()])

                f1, prec, rec, cd = calculate_metrics(xyz_model, xyz_gt, threshold=THRESHOLD)
                scene_metrics.update({
                    "f_score": f1,
                    "precision": prec,
                    "recall": rec,
                    "chamfer": cd
                })
                print(f"Processed: {name[:8]} | F1: {f1:.4f} | CD: {cd:.6f}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

        # 3. Add to the dictionary
        final_output[name] = scene_metrics

    # 4. Save to a JSON file
    output_filename = f"{MODEL}_{DATASET}_final_results.json"
    with open(output_filename, "w") as f:
        json.dump(final_output, f, indent=4)

    # Extended statistics output
    valid_f1s = [m["f_score"] for m in final_output.values() if m["f_score"] != -1]
    valid_cds = [m["chamfer"] for m in final_output.values() if m["chamfer"] != -1]

    if valid_f1s:
        print("\n" + "="*50)
        print(f"Results Summary (Threshold: {THRESHOLD})")
        print(f"Mean F1-score   : {np.mean(valid_f1s):.6f}")
        print(f"Mean Chamfer    : {np.mean(valid_cds):.6f}")
        print(f"Median Chamfer  : {np.median(valid_cds):.6f}") # To check outlier influence
        print(f"")
        print("="*50)
