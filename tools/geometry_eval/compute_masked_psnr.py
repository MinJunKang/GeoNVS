"""
Compute PSNR separately for occluded and non-occluded regions.

Examples:
    # Single scene
    python compute_masked_psnr.py path/to/bonsai

    # Parent folder -> automatically process all scenes inside
    python compute_masked_psnr.py path/to/img2vid/

    # Mixed paths (scene + parent folder)
    python compute_masked_psnr.py path/to/img2vid/ path/to/bonsai

    # When folder names differ
    python compute_masked_psnr.py path/to/scene --pred-dir rendered --mask-dir masks

    # Skip per-frame output (summary only)
    python compute_masked_psnr.py path/to/img2vid/ -q

    # Save as CSV
    python compute_masked_psnr.py path/to/img2vid/ -o results.csv
"""

import os
import math
import glob
import argparse
import csv
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR between float32 [0,1] images. mask=bool array (pixels to include)."""
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        diff = (img1[mask] - img2[mask]).astype(np.float64)
    else:
        diff = (img1 - img2).astype(np.float64)
    mse = (diff ** 2).mean()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8) > 0


def nanmean(lst: list) -> float:
    valid = [v for v in lst if not math.isnan(v) and not math.isinf(v)]
    return sum(valid) / len(valid) if valid else float("nan")


# ---------------------------------------------------------------------------
# Per-scene computation
# ---------------------------------------------------------------------------

def compute_scene_psnr(scene_dir: str,
                       gt_dir_name: str = "gt-rgb",
                       pred_dir_name: str = "samples-rgb",
                       mask_dir_name: str = "occlusion-masks",
                       verbose: bool = True) -> dict:

    gt_dir   = os.path.join(scene_dir, gt_dir_name)
    pred_dir = os.path.join(scene_dir, pred_dir_name)
    mask_dir = os.path.join(scene_dir, mask_dir_name)

    for d, name in [(gt_dir, gt_dir_name), (pred_dir, pred_dir_name), (mask_dir, mask_dir_name)]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}  (check the --{name.replace('-','_')} option)")

    gt_files   = sorted(glob.glob(os.path.join(gt_dir,   "*.png")))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if not gt_files:
        raise FileNotFoundError(f"No PNG files: {gt_dir}")
    if len(gt_files) != len(pred_files) or len(gt_files) != len(mask_files):
        raise ValueError(
            f"Frame count mismatch — gt:{len(gt_files)}, "
            f"pred:{len(pred_files)}, mask:{len(mask_files)}"
        )

    psnr_occ_list, psnr_nonocc_list, psnr_full_list = [], [], []

    if verbose:
        print(f"\n{'Frame':<8} {'PSNR_occ':>12} {'PSNR_non-occ':>14} {'PSNR_full':>11}"
              f"  {'occ_px':>10} {'non-occ_px':>12}")
        print("-" * 72)

    for gt_p, pred_p, mask_p in zip(gt_files, pred_files, mask_files):
        frame = os.path.splitext(os.path.basename(gt_p))[0]
        gt    = load_rgb(gt_p)
        pred  = load_rgb(pred_p)
        mask  = load_mask(mask_p)

        mask3 = mask[:, :, None].repeat(3, axis=2)

        p_occ    = psnr(gt, pred, mask3)
        p_nonocc = psnr(gt, pred, ~mask3)
        p_full   = psnr(gt, pred)

        psnr_occ_list.append(p_occ)
        psnr_nonocc_list.append(p_nonocc)
        psnr_full_list.append(p_full)

        if verbose:
            fmt = lambda v, w: f"{v:{w}.4f}" if math.isfinite(v) else f"{'N/A':>{w}}"
            print(f"{frame:<8} {fmt(p_occ,12)} {fmt(p_nonocc,14)} {fmt(p_full,11)}"
                  f"  {mask.sum():>10d} {(~mask).sum():>12d}")

    avg_occ    = nanmean(psnr_occ_list)
    avg_nonocc = nanmean(psnr_nonocc_list)
    avg_full   = nanmean(psnr_full_list)

    if verbose:
        print("-" * 72)
        print(f"{'AVERAGE':<8} {avg_occ:12.4f} {avg_nonocc:14.4f} {avg_full:11.4f}\n")

    return {
        "scene":        os.path.basename(scene_dir.rstrip("/")),
        "psnr_occ":     avg_occ,
        "psnr_non_occ": avg_nonocc,
        "psnr_full":    avg_full,
    }


# ---------------------------------------------------------------------------
# Path resolution: auto-detect scene directory vs parent directory
# ---------------------------------------------------------------------------

def is_scene_dir(path: str, gt: str, pred: str, mask: str) -> bool:
    """Check whether the path itself is a scene directory (has gt/pred/mask subfolders)."""
    return all(os.path.isdir(os.path.join(path, d)) for d in (gt, pred, mask))


def resolve_scene_dirs(paths: list[str], gt: str, pred: str, mask: str) -> list[str]:
    """
    If each path is a scene, use it as-is; otherwise automatically collect
    the scene subdirectories inside it.
    """
    scenes = []
    for p in paths:
        p = p.rstrip("/")
        if is_scene_dir(p, gt, pred, mask):
            scenes.append(p)
        else:
            # Collect only subdirectories that have a scene structure (sorted)
            children = sorted(
                d for d in (os.path.join(p, name) for name in os.listdir(p))
                if os.path.isdir(d) and is_scene_dir(d, gt, pred, mask)
            )
            if children:
                print(f"[auto] Found {len(children)} scenes inside '{os.path.basename(p)}'")
                scenes.extend(children)
            else:
                print(f"[warn] No scenes found: {p}")
    return scenes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Occlusion-mask based PSNR computation (occluded / non-occluded / full)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths", nargs="+", metavar="PATH",
        help="Scene directory or a parent directory containing scenes (multiple allowed)",
    )
    parser.add_argument(
        "--gt-dir", default="gt-rgb", metavar="NAME",
        help="GT image folder name (default: gt-rgb)",
    )
    parser.add_argument(
        "--pred-dir", default="samples-rgb", metavar="NAME",
        help="Predicted image folder name (default: samples-rgb)",
    )
    parser.add_argument(
        "--mask-dir", default="occlusion-masks", metavar="NAME",
        help="Occlusion mask folder name (default: occlusion-masks)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Skip per-frame output (show summary only)",
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE",
        help="Save results to a CSV file (e.g. results.csv)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    scenes = resolve_scene_dirs(args.paths, args.gt_dir, args.pred_dir, args.mask_dir)
    if not scenes:
        print("No scenes to process.")
        return

    all_results = []
    for scene in scenes:
        print(f"{'='*60}")
        print(f"Scene: {scene}")
        try:
            result = compute_scene_psnr(
                scene,
                gt_dir_name=args.gt_dir,
                pred_dir_name=args.pred_dir,
                mask_dir_name=args.mask_dir,
                verbose=not args.quiet,
            )
            all_results.append(result)
            if args.quiet:
                print(f"  PSNR occ     : {result['psnr_occ']:.4f} dB")
                print(f"  PSNR non-occ : {result['psnr_non_occ']:.4f} dB")
                print(f"  PSNR full    : {result['psnr_full']:.4f} dB")
        except (FileNotFoundError, ValueError) as e:
            print(f"  [ERROR] {e}")

    # Combined summary for multiple scenes
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"{'Scene':<30} {'PSNR_occ':>10} {'PSNR_non-occ':>14} {'PSNR_full':>11}")
        print("-" * 68)
        for r in all_results:
            print(f"{r['scene']:<30} {r['psnr_occ']:10.4f} {r['psnr_non_occ']:14.4f} {r['psnr_full']:11.4f}")
        avgs = {k: nanmean([r[k] for r in all_results])
                for k in ("psnr_occ", "psnr_non_occ", "psnr_full")}
        print("-" * 68)
        print(f"{'MEAN':<30} {avgs['psnr_occ']:10.4f} {avgs['psnr_non_occ']:14.4f} {avgs['psnr_full']:11.4f}")

    # CSV export
    if args.output and all_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene", "psnr_occ", "psnr_non_occ", "psnr_full"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCSV saved: {args.output}")


if __name__ == "__main__":
    main()
