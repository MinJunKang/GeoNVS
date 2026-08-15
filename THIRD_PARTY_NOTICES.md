# Third-party notices

This repository vendors modified copies of several third-party research
codebases, kept in-tree so that the paper's comparisons and data pipeline are
reproducible as-released. Each vendored directory retains its upstream license,
which governs that directory and takes precedence over the repository-level
license. All copies contain GeoNVS-specific modifications (wrappers, API
adaptations, bug fixes) unless noted otherwise.

| Directory | Upstream | License |
|---|---|---|
| `baselines/lrm/depthsplat/` | [cvg/depthsplat](https://github.com/cvg/depthsplat) | MIT |
| `baselines/lrm/mvsplat/` | [donydchen/mvsplat](https://github.com/donydchen/mvsplat) | MIT |
| `baselines/lrm/mvsplat360/` | [donydchen/mvsplat360](https://github.com/donydchen/mvsplat360) | MIT |
| `baselines/lrm/hisplat/` | [Open3DVLab/HiSplat](https://github.com/Open3DVLab/HiSplat) | no upstream license file (see note) |
| `baselines/lrm/Depth-Anything-3/` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Apache-2.0 |
| `datapreprocess/vggt/` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | CC BY-NC 4.0 (**non-commercial**) |
| `datapreprocess/pi3/` | [yyfz/Pi3](https://github.com/yyfz/Pi3) | BSD-3-Clause |
| `datapreprocess/mapanything/` | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Apache-2.0 |
| `baselines/diffusion/viewcrafter/` (incl. vendored DUSt3R under `extern/dust3r/`) | [Drexubery/ViewCrafter](https://github.com/Drexubery/ViewCrafter) | Apache-2.0 (in-tree LICENSE); vendored DUSt3R is CC BY-NC-SA 4.0 (**non-commercial**) |
| `baselines/diffusion/genfusion/` | [Inception3D/GenFusion](https://github.com/Inception3D/GenFusion) | MIT (upstream; no license file in vendored dir, see note) |
| `baselines/diffusion/difix3d/` | [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | NVIDIA License (**non-commercial**) + Stability AI Community License for the underlying SD-Turbo model (in-tree LICENSE.txt) |
| `baselines/diffusion/motionctrl/` | [TencentARC/MotionCtrl](https://github.com/TencentARC/MotionCtrl) | Apache-2.0 (upstream; no license file in vendored dir, see note) |
| `baselines/diffusion/mvsplat360/` (refiner-only copy) | [donydchen/mvsplat360](https://github.com/donydchen/mvsplat360) | MIT |
| `third_party/langsplat-rasterization/` | [minghanqin/LangSplat](https://github.com/minghanqin/LangSplat) (3DGS derivative) | Gaussian-Splatting License (research use) |
| `third_party/diff-gaussian-rasterization/` | [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) | Gaussian-Splatting License (research use) |
| `third_party/diff-gaussian-rasterization-depth/` | 3DGS derivative | Gaussian-Splatting License (research use) |
| `third_party/rasterization_and_pup_fisher/` | [j-alex-hanson/gaussian-splatting-pup](https://github.com/j-alex-hanson/gaussian-splatting-pup) (3DGS derivative) | Gaussian-Splatting License (research use) |
| `third_party/latent-gaussian-rasterization/` | 3DGS derivative | Gaussian-Splatting License (research use) |
| `third_party/simple-knn/` | [graphdeco-inria (simple-knn)](https://gitlab.inria.fr/bkerbl/simple-knn) | Gaussian-Splatting License (research use) |
| `third_party/fused-ssim/` | [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) | MIT |
| `third_party/vipe` (git submodule, not vendored) | [nv-tlabs/vipe](https://github.com/nv-tlabs/vipe) | see upstream |

Notes:

- **`datapreprocess/vggt/` is CC BY-NC 4.0**: the data-generation pipeline that uses
  it is for non-commercial research use only.
- The Gaussian-Splatting License (Inria / MPII) restricts the rasterizer
  submodules to non-commercial research use.
- The `baselines/diffusion/{viewcrafter,genfusion,difix3d,motionctrl,mvsplat360}/`
  directories are the baseline models compared against by
  `demo_diffusion.py`. **`baselines/diffusion/difix3d/`** (NVIDIA License) and the
  DUSt3R copy under **`baselines/diffusion/viewcrafter/extern/dust3r/`**
  (CC BY-NC-SA 4.0) are non-commercial research use only.
- **`baselines/diffusion/genfusion/`** and **`baselines/diffusion/motionctrl/`** ship without a
  license file in the vendored subtree (upstream keeps it at the repository
  root); the licenses recorded above were read from the upstream GitHub
  repositories (MIT and Apache-2.0 respectively).
- The upstream GenFusion `lvdm/data/` loaders (and
  `baselines/diffusion/viewcrafter/lvdm/data/`) contain dead hard-coded dataset paths
  from the original authors' machines; they are unused by `demo_diffusion.py`
  and kept as-is for fidelity to the upstream code.
- The GeoNVS weight repository additionally mirrors one third-party
  checkpoint: `third_party/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f40abc4f.pth`
  from [DepthSplat](https://github.com/cvg/depthsplat) (MIT License,
  © 2024 Haofei Xu). Upstream replaced this file with a retrained checkpoint,
  which measurably changes results, so the original is mirrored for
  reproducibility under the terms of its MIT license.
- **`baselines/lrm/hisplat/`** has no license file upstream. It is included in good
  faith for research reproducibility with full attribution; we will remove it
  promptly upon request by the authors.
- `geonvs/seva/` derives from [Stability-AI/stable-virtual-camera](https://github.com/Stability-AI/stable-virtual-camera)
  and `geonvs/camctrl/` from [hehao13/CameraCtrl](https://github.com/hehao13/CameraCtrl);
  both are heavily modified for the GS-Adapter and are subject to their
  upstream licenses.
