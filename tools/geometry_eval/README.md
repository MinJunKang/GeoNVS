# Geometry-Fidelity Evaluation

Scripts for the paper's geometry-fidelity metrics, computed on videos generated
by `demo.py` (the `img2vid` outputs under `work_dirs_*/`):

- **Pose error (T_err / R_err)** — [ViPE](https://github.com/nv-tlabs/vipe) estimates
  camera poses from each generated video; after Umeyama (Sim(3)) alignment they are
  compared against the GT `transforms.json` poses (mean translation error and mean
  rotation error in degrees).
- **Chamfer distance / F-score** — the ViPE point clouds of the generated video and
  of the GT video are compared (PyTorch3D `chamfer_distance` + kNN F-score).
- **Masked PSNR** — PSNR split into occluded / non-occluded regions using the
  occlusion masks produced during rendering.

## Dependencies

### ViPE (external, not vendored)

Pose estimation relies on NVIDIA **ViPE**: <https://github.com/nv-tlabs/vipe>.
It is referenced as a git submodule at `third_party/vipe` (see the repo-level
`.gitmodules`):

```bash
git submodule update --init third_party/vipe
cd third_party/vipe && # install per ViPE's own instructions (requires CUDA)
```

The scripts call the `vipe` CLI (`vipe infer ... --pipeline dav3`, which requires
Depth-Anything-3 support) and import `vipe.utils.io` / `vipe.slam.interface`, so
the package must be installed in the evaluation environment.

### Other Python packages

`pycolmap`, `viser`, `scipy`, `opencv-python` (cv2), `imageio`, `pytorch3d`
(for `chamfer_distance.py`), `torch`, `numpy`, `tqdm`, `Pillow`.

### Which scripts need ViPE installed

| Script | Needs `vipe` | Other main deps |
|---|---|---|
| `run_vipe.py` | yes (CLI + import) | pycolmap, scipy |
| `vipe_to_colmap.py` | yes (import) | opencv, imageio, scipy, torch |
| `vipe_to_colmap_and_viz.py` | yes (import) | viser, opencv, imageio |
| `chamfer_distance.py` | no | pytorch3d, torch |
| `colmap_metric.py` | no | pycolmap |
| `colmap_viser_plot.py` | no | viser, Pillow |
| `compare_metric.py` | no | numpy |
| `compute_masked_psnr.py` | no | Pillow, numpy |
| `eval_with_gt_filter.py` | no | numpy |

## Pipeline

The expected layout is a results folder (produced by `demo.py`) of the form
`{method}/{dataset}/img2vid/{scene}/` containing `samples-rgb.mp4`, `gt-rgb.mp4`,
`samples-rgb/`, `gt-rgb/`, `occlusion-masks/` and `transforms.json`.

### 1. ViPE pose estimation + COLMAP conversion + pose error (`run_vipe.py`)

Run per dataset (once for the generated videos, once with `--eval_gt` for the GT
videos). It runs `vipe infer` on every scene's video, converts the ViPE artifacts
to COLMAP text format via `vipe_to_colmap.py`, and evaluates aligned T_err / R_err
against `transforms.json`, writing `{method}_vipe/{dataset}/final_metrics.json`:

```bash
cd geometry_eval
python run_vipe.py dl3dv140_9_1 --root_path /path/to/work_dirs/eccv_ours              # predictions
python run_vipe.py dl3dv140_9_1 --root_path /path/to/work_dirs/eccv_ours --eval_gt    # GT videos -> gt_vipe/
```

### 2. Chamfer distance / F-score (`chamfer_distance.py`)

Compares the method's ViPE point clouds (`{model}_vipe/{dataset}/{scene}_colmap`)
against the GT ones (`gt_vipe/...`) and merges pose + geometry metrics into
`{model}_{dataset}_final_results.json`:

```bash
python chamfer_distance.py --model eccv_ours --dataset dl3dv140_9_1 --threshold 0.01
```

Alternative pose metric without ViPE: `colmap_metric.py` runs a plain pycolmap
incremental reconstruction (multiple deterministic runs per scene) on the rendered
frames and reports aligned T_err / R_err / registration rate:

```bash
python colmap_metric.py --dataset_root /path/to/{method}/{dataset}/img2vid --num_runs 3 --no-eval_gt
```

### 3. Aggregation / comparison

- `eval_with_gt_filter.py` — filters out scenes whose **GT** ViPE metrics are
  unreliable (`t_mean > 0.5` or `r_mean > 50` by default) and prints per-method
  mean metrics over the remaining scenes:

  ```bash
  python eval_with_gt_filter.py --base_dir /path/to/results --datasets dl3dv140_9_1 \
      --methods eccv_ours eccv_seva
  ```

- `compare_metric.py` — per-scene diff between two `*_final_results.json` files
  (chamfer / pose-score deltas, sorted by chamfer difference):

  ```bash
  python compare_metric.py --path1 eccv_ours_dtu_3_1_final_results.json \
      --path2 eccv_seva_dtu_3_1_final_results.json
  ```

### Auxiliary scripts

- `compute_masked_psnr.py` — occlusion-mask based PSNR (occluded / non-occluded /
  full) per scene or over a folder of scenes:

  ```bash
  python compute_masked_psnr.py /path/to/{method}/{dataset}/img2vid -o results.csv
  ```

- `colmap_viser_plot.py` — interactive viser visualization of a `{scene}_colmap`
  reconstruction (input views highlighted in red, viewpoint sync + screenshots).
- `vipe_to_colmap_and_viz.py` — standalone ViPE→COLMAP conversion followed by
  viser visualization for a single artifact.
- `vipe_to_colmap.py` — the ViPE→COLMAP conversion library used by `run_vipe.py`
  (imported from the same directory, so run the pipeline from `geometry_eval/`).
