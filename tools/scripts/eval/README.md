# Evaluation scripts

Minimal set of runners that reproduce the experiments in the GeoNVS paper.
All reported numbers use the **384×384** protocol (`--H=384 --W=384`,
`--cfg=1.4`, SEVA `--version=1.0`); the benchmark table and these defaults live
in [`benchmark.sh`](benchmark.sh) and can be overridden from the environment.

Run everything from the repository root.

| Script | Purpose |
|---|---|
| `benchmark.sh` | Dataset × reference-view table (Tables 1–2) + shared defaults. Sourced by the runners, not executed directly. |
| `eval_geonvs.sh` | GeoNVS (SEVA + GS-Adapter + LoRA) — main results |
| `eval_baseline.sh` | SEVA, input-level injection, feed-forward geometry, diffusion baselines |
| `eval_long.sh` | Long-trajectory protocol (trajectory prior + `nearest-gt` chunking + voxel pruning) |
| `eval_ablation.sh` | top-k / voxel pruning / CFG / fusion / geometry-model ablations |

## Examples

```bash
# main table, all dataset × P entries, VGGT geometry prior
bash tools/scripts/eval/eval_geonvs.sh --gpu 0 vggt_iv

# several GPUs shard the table and run in parallel
bash tools/scripts/eval/eval_geonvs.sh --gpu 0,1,2,3 vggt_iv

# a single entry (dataset:num_reference_views)
bash tools/scripts/eval/eval_geonvs.sh depthsplat dl3dv10:3

# baselines
bash tools/scripts/eval/eval_baseline.sh seva
bash tools/scripts/eval/eval_baseline.sh lrm depthsplat dl3dv10:3
bash tools/scripts/eval/eval_baseline.sh diffusion geonvs_cameractrl_depthsplat
bash tools/scripts/eval/eval_baseline.sh input_level 0.2 dl3dv10:3   # paper: s=0.2

# long-trajectory setting
bash tools/scripts/eval/eval_long.sh geonvs vggt_iv

# ablations (default target dl3dv10:3)
bash tools/scripts/eval/eval_ablation.sh cfg
bash tools/scripts/eval/eval_ablation.sh pruning
```

Pick GPUs with `--gpu 0` or `--gpu 0,1,2,3` (entries are sharded
round-robin, one process per GPU). Other useful environment variables:
`BENCHMARK_ROOT`, `CONFIG`, `WEIGHTS`,
`LRM`, `CFG`, `RES`, `DIR_TAG`.

Results are written to `work_dirs_<dir_tag>_*/`, with per-scene
`performance_table.xlsx` and an aggregated `metric_result.xlsx`. Geometry
fidelity metrics (camera pose error, Chamfer distance) are computed separately
by the ViPE pipeline in [`../../geometry_eval/`](../../geometry_eval/).

## Baseline environments

Everything in the table above runs in the single environment installed by
`autoinstall.sh`. Two vendored video-diffusion baselines pin older
dependencies than the shared environment, and are patched here only so far as
is needed to make them run:

| Baseline | Upstream pins | Shared env | Effect |
|---|---|---|---|
| GenFusion | `open-clip-torch==2.12.0`, `transformers==4.46.2` | 2.32.0 / 4.49.0 | reproduces below the reported numbers |
| ViewCrafter | `open-clip-torch==2.17.1` | 2.32.0 | see note |

`open_clip >= 2.26` removed the `input_patchnorm` attribute and changed how the
text-transformer attention mask is shaped, so both models' CLIP conditioners
take a slightly different path in the shared environment. Measured for
GenFusion:

| | this env | paper |
|---|---|---|
| DL3DV-10, P=6 | 10.94 / 0.329 / 0.607 | 13.11 / 0.394 / 0.537 |
| RE10K, P=3 | 20.58 / 0.833 / 0.181 | 22.69 / 0.864 / 0.159 |

(PSNR / SSIM / LPIPS.) To reproduce the reported numbers, run these two
baselines in their upstream environments:

```bash
pip install open-clip-torch==2.12.0 transformers==4.46.2   # GenFusion
pip install open-clip-torch==2.17.1                        # ViewCrafter
```
