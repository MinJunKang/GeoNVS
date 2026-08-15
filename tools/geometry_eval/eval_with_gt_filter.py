"""
Script that excludes scenes whose t_mean/r_mean is excessively high according to the
GT final_metrics.json, then computes the mean metrics for each method.

Two file patterns are supported:
  1. final_results mode (default): {base}/{method}_{dataset}_final_results.json
     - Files produced by chamfer_distance.py, containing pose error + chamfer distance
  2. final_metrics mode (--use_metrics_folder):  {base}/{method}/{dataset}/final_metrics.json
     - Files containing only pose error after vipe alignment
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_valid_scenes(gt_metrics, t_threshold=None, r_threshold=None):
    """Return the list of valid scenes based on the GT metrics."""
    excluded = []
    valid = []
    for scene, m in gt_metrics.items():
        t = m.get("t_mean", 0)
        r = m.get("r_mean", 0)
        reasons = []
        if t_threshold is not None and t > t_threshold:
            reasons.append(f"t_mean={t:.4f} > {t_threshold}")
        if r_threshold is not None and r > r_threshold:
            reasons.append(f"r_mean={r:.4f} > {r_threshold}")
        if reasons:
            excluded.append((scene, ", ".join(reasons)))
        else:
            valid.append(scene)
    return valid, excluded


def compute_mean_metrics(metrics, valid_scenes, metric_keys):
    """Compute mean metrics over the valid scenes. -1 is treated as an invalid value."""
    results = {k: [] for k in metric_keys}
    missing = []
    for scene in valid_scenes:
        if scene not in metrics:
            missing.append(scene)
            continue
        for k in metric_keys:
            v = metrics[scene].get(k, None)
            if v is not None and v >= 0:
                results[k].append(v)
    means = {k: np.mean(v) if v else float("nan") for k, v in results.items()}
    counts = {k: len(v) for k, v in results.items()}
    return means, counts, missing


def evaluate(gt_path, method_paths, t_threshold=None, r_threshold=None,
             metric_keys=("t_mean", "r_mean", "chamfer", "f_score")):
    """
    gt_path      : Path to the GT final_metrics.json (filtering reference)
    method_paths : {method_name: json_path} – result file for each method to evaluate
    """
    gt_metrics = load_json(gt_path)

    # Filter valid scenes based on GT
    valid_scenes, excluded = get_valid_scenes(gt_metrics, t_threshold, r_threshold)

    col_w = 16
    print("=" * 70)
    print(f"GT path : {gt_path}")
    print(f"Thresholds: t_mean <= {t_threshold}, r_mean <= {r_threshold}")
    print(f"Total scenes  : {len(gt_metrics)}")
    print(f"Valid scenes ({len(valid_scenes)}): {valid_scenes}")
    if excluded:
        print(f"Excluded scenes ({len(excluded)}):")
        for scene, reason in excluded:
            print(f"  - {scene}: {reason}")
    print("=" * 70)

    # Header
    print(f"\n{'Method':<30}", end="")
    for k in metric_keys:
        print(f"  {k:<{col_w}}", end="")
    print()
    print("-" * (30 + (col_w + 2) * len(metric_keys)))

    # GT's own pose averages (chamfer/f_score may be absent from GT, display omitted)
    gt_means, gt_counts, _ = compute_mean_metrics(gt_metrics, valid_scenes, metric_keys)
    print(f"{'GT (pose ref)':<30}", end="")
    for k in metric_keys:
        v = gt_means.get(k, float("nan"))
        print(f"  {v:<{col_w}.6f}", end="")
    print()

    # Per-method averages
    all_results = {}
    for method_name, method_path in method_paths.items():
        if not os.path.exists(method_path):
            print(f"{'[MISSING] ' + method_name:<30}  (not found: {method_path})")
            continue
        method_metrics = load_json(method_path)
        means, counts, missing = compute_mean_metrics(method_metrics, valid_scenes, metric_keys)
        all_results[method_name] = means

        print(f"{method_name:<30}", end="")
        for k in metric_keys:
            v = means.get(k, float("nan"))
            n = counts.get(k, 0)
            suffix = f"(n={n})" if not np.isnan(v) else "  N/A  "
            print(f"  {v:<{col_w}.6f}", end="")
        if missing:
            print(f"  [scene missing: {missing}]", end="")
        print()

    print()
    return valid_scenes, excluded, all_results


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-method mean metrics after GT-based filtering"
    )
    parser.add_argument(
        "--base_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to the folder containing the metric result files"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dl3dv140_9_1"],
        help="List of datasets to evaluate"
    )
    parser.add_argument(
        "--gt_method",
        default="gt_vipe",
        help="GT metrics folder name (gt_vipe/{dataset}/final_metrics.json)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["eccv_ours", "eccv_seva"],  # , "eccv_seva_input", "eccv_difix3d"
        help=(
            "List of method names to compare.\n"
            "  final_results mode (default): {base}/{method}_{dataset}_final_results.json\n"
            "  final_metrics mode (--use_metrics_folder): {base}/{method}/{dataset}/final_metrics.json"
        )
    )
    parser.add_argument(
        "--use_metrics_folder",
        action="store_true",
        help="Change the file pattern to {method}/{dataset}/final_metrics.json"
    )
    parser.add_argument(
        "--metric_keys",
        nargs="+",
        default=["t_mean", "r_mean", "chamfer", "f_score"],
        help="List of metric columns to aggregate"
    )
    parser.add_argument(
        "--t_threshold",
        type=float,
        default=0.5,
        help="GT t_mean threshold (scenes above it are excluded)"
    )
    parser.add_argument(
        "--r_threshold",
        type=float,
        default=50.0,
        help="GT r_mean threshold (scenes above it are excluded)"
    )
    parser.add_argument(
        "--no_t_filter",
        action="store_true",
        help="Disable the t_mean filter"
    )
    parser.add_argument(
        "--no_r_filter",
        action="store_true",
        help="Disable the r_mean filter"
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    t_thr = None if args.no_t_filter else args.t_threshold
    r_thr = None if args.no_r_filter else args.r_threshold

    for dataset in args.datasets:
        gt_path = base / args.gt_method / dataset / "final_metrics.json"
        if not gt_path.exists():
            print(f"\n[SKIP] GT not found for dataset '{dataset}': {gt_path}")
            continue

        if args.use_metrics_folder:
            # {method}/{dataset}/final_metrics.json
            method_paths = {
                m: str(base / m / dataset / "final_metrics.json")
                for m in args.methods
            }
        else:
            # {method}_{dataset}_final_results.json  (output files of chamfer_distance.py)
            method_paths = {
                m: str(base / f"{m}_{dataset}_final_results.json")
                for m in args.methods
            }

        print(f"\n{'#' * 70}")
        print(f"# Dataset: {dataset}")
        print(f"{'#' * 70}")
        evaluate(
            gt_path=str(gt_path),
            method_paths=method_paths,
            t_threshold=t_thr,
            r_threshold=r_thr,
            metric_keys=tuple(args.metric_keys),
        )
