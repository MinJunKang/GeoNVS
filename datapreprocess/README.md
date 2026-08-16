# Data Preprocess Guideline for GeoNVS

Two pipelines share this directory. Both lift posed images into 3D Gaussians
with a geometry foundation model, refine them with a short per-scene 3DGS
optimization, and attach per-Gaussian Fisher information, which together form
the geometry prior the GS-Adapter consumes.

| Script | Produces | Consumed by |
|---|---|---|
| `process_dataset.py` | **training data**: `<stage>_gs/*.safetensors` for RE10K / DL3DV | `geonvs/data/{dl3dv,re10k}.py` during `train_seva.py` / `train_camctrl.py` |
| `process_benchmark.py` | **benchmark priors**: `<scene>/{vggt,pi3,mapanything}_{iv,av}/*.safetensors` | `--lrm_model_name vggt_iv / pi3_iv / …` in `demo.py`, `demo_regression.py`, `demo_diffusion.py` |

The CUDA rasterizers and the Gaussian decoder are shared with the training /
evaluation stack (`../third_party`, `../geonvs/decoder`), so everything runs in
the same environment. `autoinstall.sh` only adds the datapreprocess-specific extras
(vendored VGGT, PyTorch3D, per-scene optimizer dependencies).

```bash
cd datapreprocess && bash autoinstall.sh
```

Model weights (VGGT `facebook/VGGT-1B`, Pi3 `yyfz233/Pi3`, MapAnything
`facebook/map-anything`, OpenCLIP for the view selector) download automatically
from the HuggingFace Hub on first use. A CUDA GPU is required.

---

## 1. Training data (`process_dataset.py`)

### Input

[pixelSplat](https://github.com/dcharatan/pixelsplat)-style chunked datasets,
as released by pixelSplat (RE10K) and DepthSplat (DL3DV). This repository does
not re-implement that conversion, so download the chunks from those projects and
lay them out as:

```
<data_root>/
└── dl3dv_low/                 # or dl3dv_high / re10k_low / re10k_high
    ├── train/  000000.torch, 000001.torch, ..., index.json
    └── test/   000000.torch, ...
```

Each `.torch` chunk is a list of scenes with `key`, `cameras` (N×18: fx, fy,
cx, cy normalized + 12 row-major w2c values) and `images` (list of encoded JPEG
byte tensors).

### Run

```bash
# train split (shard the chunk range across GPUs / processes)
CUDA_VISIBLE_DEVICES=0 python process_dataset.py \
    --dataset dl3dv --stage train --data_root /path/to/datasets \
    --start_index 0 --end_index 3000

# test split (used for validation during training)
CUDA_VISIBLE_DEVICES=0 python process_dataset.py \
    --dataset dl3dv --stage test --data_root /path/to/datasets \
    --start_index 0 --end_index 100
```

Chunks are independent, so N GPUs can process N disjoint `--start_index /
--end_index` ranges concurrently. Re-running skips scenes whose outputs already
exist (`--overwrite` forces regeneration).

Useful options: `--highres` (use `*_high`), `--num_group_images` (views per
sample, default 21, must match `--num_frames` in training),
`--num_input_images` (reference-view counts, default `1 3 6 9 12`),
`--num_input_variations`, `--min_frustum_iou_threshold`,
`--adjacent_frame_sampling_rate`, `--no_confidence` (skip Fisher; **not**
supported by the training code, kept for ablations).

### Output

```
<data_root>/dl3dv_low/train_gs/<scene>_<P>_<Q>_<variation>.safetensors
```
where `P` = number of reference views, `Q` = number of target views
(`P + Q = --num_group_images`).

| key | shape | note |
|---|---|---|
| `gaussians` | `(N, 95)` | `means(3) ⊕ rot(4, wxyz) ⊕ scale(3) ⊕ opacity(1) ⊕ fisher(36) ⊕ SH(48)`, bf16 |
| `input_indices` / `target_indices` | `(P,)` / `(Q,)` | indices into the scene's frame list |
| `near` / `far` / `sh_degree` | `(1,)` | scene bounds and SH degree (3) |

Point `--base_folder` of the training scripts at `<data_root>`.

---

## 2. Benchmark geometry priors (`process_benchmark.py`)

### Input

Reconfusion-style benchmark scenes (see [`../tools/benchmark/`](../tools/benchmark/)):

```
<benchmarkset>/<dataset>/<scene>/
├── images/
├── transforms.json
└── train_test_split_{1,3,6,9,...}.json
```

### Run

```bash
CUDA_VISIBLE_DEVICES=0 python process_benchmark.py \
    --data_root /path/to/benchmarkset --dataset dl3dv10 --models vggt pi3
```

- `--models vggt pi3 mapanything`: geometry backbone(s); one output folder each.
- `--include_target`: reconstruct from reference **and** target views
  (`*_av` folders). Without it only reference views are used (`*_iv`), which is
  the setting used for the paper's evaluation.
- `--no_confidence`: skip Fisher information. **Do not use for evaluation**:
  the renderer expects the 95-channel layout with Fisher.

Nested datasets take the sub-path, e.g. `--dataset wildrgbd/wildrgbd_hard`.

### Output

```
<benchmarkset>/<dataset>/<scene>/vggt_iv/<P>_<Q>.safetensors
```
Same keys as above, but `gaussians` keeps a leading batch dimension
`(1, N, 95)`, which is what `GaussianModelWrapper` in
`baselines/lrm_models.py` expects. One file per `train_test_split_<P>.json`.

Evaluate with the matching name, e.g.
`--lrm_model_name vggt_iv` (or `vggt_av`, `pi3_iv`, `pi3_av`).

---

## How it works

1. **View selection** (`view_selector.py`, training only) builds a pose-distance
   graph, samples anchors proportionally to degree, picks reference views by
   K-means, splits easy/hard combinations with a CLIP distance, and rejects
   combinations whose frustum IoU is below `--min_frustum_iou_threshold`.
   Benchmark scenes skip this: their splits are fixed by
   `train_test_split_<P>.json`.
2. **Geometry** (`load_model.py`): VGGT / Pi3 / MapAnything predict cameras and
   depth; depth is aligned to the ground-truth poses (`geometry.py`) and
   unprojected into a point map, with co-visibility masks from the confidence
   maps.
3. **Per-scene 3DGS**: Gaussians initialized from the point map are optimized
   for 3000 iterations (InstantSplat-style, `scene_custom/`,
   `gaussian_renderer/`), with PUP Fisher-based pruning of 70% of the Gaussians
   at iteration 1000.
4. **Fisher information**: 6×6 per-Gaussian Fisher matrices
   (`../geonvs/decoder/fisher_metric.py`, `rasterization_and_pup_fisher`) are
   packed into the saved tensor and used by the GS-Adapter as a confidence
   signal.
5. **Quality gate**: scenes whose reconstruction fails or produces non-finite
   values are skipped and logged to `<dataset>_<stage>_failure.txt` /
   `<dataset>_<stage>_log.txt`.

Runtime is dominated by step 3: roughly 4–8 minutes per (scene, view
combination) on an A6000, so full-dataset generation is a multi-GPU,
multi-day job. Shard by chunk range (training) or by dataset (benchmark).
