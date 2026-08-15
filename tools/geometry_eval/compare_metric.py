import json
import argparse
import numpy as np

def pose_score(t_mean, r_mean_deg):
    """translation error + rotation cosine distance (1 - cos(r))"""
    r_rad = np.deg2rad(r_mean_deg)
    return t_mean + (1 - np.cos(r_rad))

def calculate_filtered_means(path1, path2, output_path="sorted_scenes_with_metrics2.json"):
    # 1. Read the JSON files
    with open(path1, 'r') as f:
        data1 = json.load(f)
    with open(path2, 'r') as f:
        data2 = json.load(f)

    detailed_diff_list = []

    # Compare each scene in data1 against data2
    for scene_name, metrics1 in data1.items():
        if scene_name in data2:
            t1 = metrics1.get("t_mean", 0)
            r1 = metrics1.get("r_mean", 0)
            t2 = data2[scene_name].get("t_mean", 0)
            r2 = data2[scene_name].get("r_mean", 0)

            pose1 = pose_score(t1, r1)
            pose2 = pose_score(t2, r2)
            pose_diff = pose2 - pose1

            c1 = metrics1.get("chamfer", 0)
            c2 = data2[scene_name].get("chamfer", 0)
            chamfer_diff = c2 - c1

            # Store the details as a dictionary
            detailed_diff_list.append({
                "scene_name": scene_name,
                "chamfer_1": c1,
                "chamfer_2": c2,
                "chamfer_diff": chamfer_diff,
                "pose_score_1": pose1,
                "pose_score_2": pose2,
                "pose_diff": pose_diff,
                "t_mean_1": t1,
                "r_mean_1": r1,
                "t_mean_2": t2,
                "r_mean_2": r2,
            })

    # 2. Sort by chamfer difference (descending)
    sorted_details = sorted(detailed_diff_list, key=lambda x: x["chamfer_diff"], reverse=True)

    # 3. Print results and save to file
    print(f"{'No.':<4} {'Scene Name':<20} {'chamfer_1':<12} {'chamfer_2':<12} {'chamfer_diff':<14} {'pose_diff':<12}")
    print("-" * 80)
    for i, item in enumerate(sorted_details[:10]): # Print top 10
        print(f"{i+1:<4} {item['scene_name']:<20} {item['chamfer_1']:<12.6f} {item['chamfer_2']:<12.6f} {item['chamfer_diff']:<14.6f} {item['pose_diff']:<12.6f}")

    # Save JSON including detailed information
    with open(output_path, "w") as f:
        json.dump(sorted_details, f, indent=4)

    print(f"\nDetailed list saved to '{output_path}'.")

    return sorted_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare per-scene metrics between two final_results JSON files"
    )
    parser.add_argument("--path1", type=str, default="eccv_ours_dtu_3_1_final_results.json",
                        help="First results JSON (default: eccv_ours_dtu_3_1_final_results.json)")
    parser.add_argument("--path2", type=str, default="eccv_seva_dtu_3_1_final_results.json",
                        help="Second results JSON (default: eccv_seva_dtu_3_1_final_results.json)")
    parser.add_argument("--output", type=str, default="sorted_scenes_with_metrics2.json",
                        help="Output JSON path (default: sorted_scenes_with_metrics2.json)")
    args = parser.parse_args()

    calculate_filtered_means(args.path1, args.path2, output_path=args.output)
