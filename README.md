<div align="center">

# GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis

[![Project page](https://img.shields.io/badge/Project%20page-GeoNVS-1a73e8.svg)](https://sites.google.com/view/minjun-kang/geonvs-eccv26)
[![Paper](https://img.shields.io/badge/arXiv-2603.14965-b31b1b.svg)](https://arxiv.org/abs/2603.14965)
[![Weights](https://img.shields.io/badge/🤗%20HuggingFace-Weights-yellow)](https://huggingface.co/HugMinjun/GeoNVS)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Minjun Kang<sup>1</sup>, Inkyu Shin<sup>2</sup>, Taeyeop Lee<sup>1</sup>, Myungchul Kim<sup>1</sup>, In So Kweon<sup>1</sup>, Kuk-Jin Yoon<sup>1</sup>

<sup>1</sup>KAIST, South Korea &nbsp;&nbsp; <sup>2</sup>Luma AI, USA

<img src="assets/geonvs_method.png" width="88%">

</div>

The GS-Adapter reconstructs 3D Gaussians from the input views, uplifts the
diffusion features onto them, and renders them into every target view, so every
denoising step is guided by the 3D-GS enhanced feature. Only the adapter and the
LoRA layers are trained; the diffusion backbone stays frozen.

---

## 📌 Release status

- [x] Training code (SEVA and CameraCtrl backbones)
- [x] Evaluation code and benchmark protocol runners
- [x] Pretrained GS-Adapter + LoRA weights ([HuggingFace](https://huggingface.co/HugMinjun/GeoNVS))
- [x] Baseline comparison methods (geometry models and video-diffusion models)
- [x] Geometry-fidelity metrics (ViPE camera pose error, Chamfer distance)
- [x] Dataset preprocessing code ([`datapreprocess/`](datapreprocess/))

## ✨ Highlights

- **Geometry as features, not pixels.** The GS-Adapter lifts input-view
  diffusion features into 3D Gaussians, renders them into every target view, and
  fuses them back into the diffusion features. Injecting geometry in *feature*
  space avoids the view-dependent color noise that degrades structural
  consistency when rendered images are injected at the input level.

- **Adaptive Fusion that knows when to trust geometry.** A gating MLP predicts a
  per-pixel confidence weight from the diffusion and geometry features, so the
  prior is downweighted exactly where it is unreliable, such as reflective or
  occluded regions.

- **Plug-and-play with any geometry model.** The same trained adapter works with
  VGGT, Pi3, DepthSplat, MVSplat and others **without retraining**.

- **Backbone agnostic.** Demonstrated on both SEVA and CameraCtrl with 11.3% and
  14.9% improvements, up to 2× lower translation error and 7× lower Chamfer
  distance.

## 🔧 Installation

Tested with Python ≥ 3.10, PyTorch 2.5.1 + CUDA 12.1.

```bash
git clone --recursive https://github.com/MinJunKang/GeoNVS.git && cd GeoNVS
# install torch 2.5.x for your CUDA version first, then:
bash autoinstall.sh
```

The script uses [uv](https://docs.astral.sh/uv/) when available and falls back
to pip. It builds the feature-capable Gaussian rasterizer
(`third_party/langsplat-rasterization`) and the Fisher renderer, and installs
xformers, torch-scatter and transformer_engine. Optional blocks cover the extra
baseline backbones.

```bash
bash tools/scripts/download_weights_ours.sh   # GeoNVS weights + backbones (~2 GB)
bash tools/scripts/download_weights_all.sh    # + every comparison baseline (~55 GB)
```

`stabilityai/stable-virtual-camera` (SEVA) and the SVD VAE are pulled from the
HuggingFace Hub on first run, so accept their licenses first.

## 🚀 Quick start

`demo.py` runs GeoNVS over a benchmark split and scores it. It is the
evaluation entry point, not a single-scene demo:

```bash
python demo.py --data_path <benchmarkset>/dl3dv10 --num_inputs 3 \
    --gs_adapter_config configs/module_config/gsadapter_eccv_gattn.yaml \
    --gs_adapter_weight_path pretrained_weights/eccv_gattn \
    --lrm_model_name vggt_iv --H=384 --W=384 --version=1.0 --cfg=1.4
```

Outputs land in `runs/<tag>/` with rendered frames, videos, per-scene
`performance_table.xlsx` and an aggregated `metric_result.xlsx`.

## 📊 Evaluation

Evaluation runs through the `demo_*.py` entry points: `demo.py` for GeoNVS and
the SEVA backbone, `demo_regression.py` for a geometry prior on its own, and
`demo_diffusion.py` for the video-diffusion baselines. Each writes rendered
frames and metrics for a whole benchmark split, so they are the evaluation
scripts rather than single-scene demos.

The runners that drive them over the paper's tables (384×384 protocol) live in
[`tools/scripts/eval/`](tools/scripts/eval/):

```bash
bash tools/scripts/eval/eval_geonvs.sh --gpu 0,1,2,3 vggt_iv   # main table
bash tools/scripts/eval/eval_baseline.sh --gpu 0 seva          # baselines
bash tools/scripts/eval/eval_long.sh                           # long trajectory
bash tools/scripts/eval/eval_ablation.sh cfg                   # ablations
```

Multiple GPUs shard the benchmark table and run in parallel.

`--lrm_model_name` selects the geometry prior: `vggt_iv` / `vggt_av` /
`pi3_iv` / `pi3_av` (precomputed) or on-the-fly `depthsplat`, `mvsplat`,
`hisplat`, `mvsplat360`, `da3`. The baselines are chosen with
`--dm_model_name cameractrl | motionctrl | viewcrafter | mvsplat360 |
genfusion_<lrm> | difix3d_<lrm> | geonvs_cameractrl_<lrm>`.

Geometry-fidelity metrics (camera pose error, Chamfer distance) use the ViPE
pipeline in [`tools/geometry_eval/`](tools/geometry_eval/). The benchmark data
format and a scene converter are documented in
[`tools/benchmark/`](tools/benchmark/).

## 🏋️ Training

```bash
# SEVA backbone, main results (8 GPUs, 21 frames, 384x384)
bash tools/scripts/train_seva.sh configs/module_config/gsadapter_eccv_gattn.yaml

# CameraCtrl backbone (14 frames, 576x320)
bash tools/scripts/train_camctrl.sh configs/module_config/gsadapter_camctrl_gattn.yaml
```

| Config | Fusion |
|---|---|
| `gsadapter_eccv_gattn.yaml` | Adaptive Fusion (`fusion_method: gattn`), **final model** |
| `gsadapter_eccv_base.yaml` | Naive Fusion (`fusion_method: concat`) |

Defaults: DL3DV-10K, LoRA rank 16, lr 1e-5 (LoRA) / 5e-5 (adapter), 100k steps,
bf16, per-GPU batch 1. LoRA and GS-Adapter weights are exported next to every
checkpoint. See `python train_seva.py --help`.

## 🗂️ Data Preprocess

GeoNVS never reads raw images alone: every scene comes with a precomputed
Gaussian prior. Two pipelines in [`datapreprocess/`](datapreprocess/) produce
them, and both follow the same recipe (a geometry foundation model lifts the
posed images into 3D Gaussians, a short per-scene 3DGS optimization refines
them, and Fisher information is attached per Gaussian):

| Pipeline | Produces | Used by |
|---|---|---|
| `process_dataset.py` | training data, `<stage>_gs/*.safetensors` | `train_seva.py`, `train_camctrl.py` |
| `process_benchmark.py` | benchmark priors, `<scene>/{vggt,pi3}_{iv,av}/*.safetensors` | `--lrm_model_name vggt_iv / pi3_iv / ...` at evaluation |

Training expects [pixelSplat](https://github.com/dcharatan/pixelsplat)-style
chunks next to the priors:

```
<base_folder>/dl3dv_low/
├── train/     000000.torch ...  index.json
├── train_gs/  <scene>_<P>_<Q>_<v>.safetensors     # geometry prior
├── test/
└── test_gs/
```

Each prior stores `means(3) ⊕ rot(4) ⊕ scale(3) ⊕ opacity(1) ⊕ fisher(36) ⊕
SH(48)`.

**➡️ Read [`datapreprocess/README.md`](datapreprocess/README.md)** for the full
guideline: environment setup, the input layout and exact commands for each
pipeline, what they write out, and a step-by-step walkthrough of view
selection, geometry prediction, per-scene 3DGS with PUP pruning, and the Fisher
information the GS-Adapter consumes (including runtime and sharding advice).

## 📁 Repository structure

```
GeoNVS/
│                                       # training entry points:
├── train_seva.py                       #   train GeoNVS with SEVA baseline
├── train_camctrl.py                    #   train GeoNVS with CameraCtrl baseline
│                                       # evaluation entry points (demo_*.py):
├── demo.py                             #   GeoNVS, and the SEVA backbone alone
├── demo_regression.py                  #   a geometry prior alone, no diffusion
├── demo_diffusion.py                   #   video-diffusion baselines
├── configs/                            # adapter / backbone / accelerate configs
├── geonvs/                             # our model
│   ├── adapter_core/                   # GS-Adapter (lifting, refine, fusion)
│   ├── seva/  camctrl/                 # diffusion backbones (diffusers ports)
│   ├── decoder/                        # feature-capable Gaussian rasterization
│   └── data/  utils/                   # datasets and helpers
├── baselines/                          # third-party geometry / diffusion models
│   ├── lrm_models.py                   # geometry-model dispatch (also used by GeoNVS)
│   ├── diffusion_models.py             # video-diffusion baseline dispatch
│   ├── lrm/                            # geometry prior models (vendored)
│   └── diffusion/                      # diffusion methods for comparison (vendored)
├── datapreprocess/                     # training data + benchmark geometry priors
├── third_party/                        # CUDA extensions, ViPE submodule
└── tools/                              # eval runners, geometry metrics, benchmark
```

## 🙏 Acknowledgements

Built on [Stable Virtual Camera](https://github.com/Stability-AI/stable-virtual-camera),
[CameraCtrl](https://github.com/hehao13/CameraCtrl),
[Stable Video Diffusion](https://github.com/Stability-AI/generative-models),
[VGGT](https://github.com/facebookresearch/vggt),
[Pi3](https://github.com/yyfz/Pi3),
[DepthSplat](https://github.com/cvg/depthsplat),
[MVSplat](https://github.com/donydchen/mvsplat),
[pixelSplat](https://github.com/dcharatan/pixelsplat),
[InstantSplat](https://github.com/NVlabs/InstantSplat),
[LangSplat](https://github.com/minghanqin/LangSplat),
[PUP-3DGS](https://github.com/j-alex-hanson/gaussian-splatting-pup) and
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
We thank the authors for releasing their code.

## 📄 License

The code is released under Apache 2.0. Vendored third-party components
keep their own licenses, which take precedence for those directories. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular `datapreprocess/vggt`
(CC BY-NC 4.0) and the Gaussian rasterizers (Gaussian-Splatting License) are
restricted to non-commercial research use, and the released weights inherit the
[Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/stable-virtual-camera/blob/main/LICENSE).

## 📚 Citation

```bibtex
@article{kang2026geonvs,
  title   = {GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis},
  author  = {Kang, Minjun and Shin, Inkyu and Lee, Taeyeop and Kim, Myungchul
             and Kweon, In So and Yoon, Kuk-Jin},
  journal = {arXiv preprint arXiv:2603.14965},
  year    = {2026}
}
```
