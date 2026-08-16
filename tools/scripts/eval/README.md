# Evaluation scripts

Minimal set of runners that reproduce the experiments in the GeoNVS paper.
All reported numbers use the **384×384** protocol (`--H=384 --W=384`,
`--cfg=1.4`, SEVA `--version=1.0`); the benchmark table and these defaults live
in [`benchmark.sh`](benchmark.sh) and can be overridden from the environment.

Run everything from the repository root.

| Script | Purpose |
|---|---|
| `benchmark.sh` | Dataset × reference-view table (Tables 1–2) + shared defaults. Sourced by the runners, not executed directly. |
| `eval_geonvs.sh` | GeoNVS (SEVA + GS-Adapter + LoRA), main results |
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

Results are written to `runs/<dir_tag>_*/`, with per-scene
`performance_table.xlsx` and an aggregated `metric_result.xlsx`. Geometry
fidelity metrics (camera pose error, Chamfer distance) are computed separately
by the ViPE pipeline in [`../../geometry_eval/`](../../geometry_eval/).

## Baseline environments

Everything above runs in the single environment installed by `autoinstall.sh`.
Two vendored video-diffusion baselines were written against older CLIP
releases. GenFusion pins `open-clip-torch==2.12.0` / `transformers==4.46.2`,
ViewCrafter pins `open-clip-torch==2.17.1`, and this repository installs
2.32.0 / 4.49.0. Their conditioners
([GenFusion](../../../baselines/diffusion/genfusion/lvdm/modules/encoders/condition.py),
[ViewCrafter](../../../baselines/diffusion/viewcrafter/lvdm/modules/encoders/condition.py))
branch on the two changes that matter, so no separate environment is needed:

- `input_patchnorm` was removed in open_clip 2.26; its absence means the
  standard (non dual-patchnorm) path.
- The transformer became batch-first in open_clip 2.24. The original
  `NLD -> LND` permute hands it a length-1 sequence: the text encoder raises on
  the 77x77 causal mask, and, silently, the image encoder runs with
  self-attention disabled, which costs about 2 dB.

Both baselines reproduce their published numbers here (10 scenes per entry,
PSNR / SSIM / LPIPS):

| | measured | paper |
|---|---|---|
| GenFusion, DL3DV-10 P=6 | 13.97 / 0.432 / 0.519 | 13.11 / 0.394 / 0.537 |
| GenFusion, RE10K P=3 | 22.79 / 0.869 / 0.153 | 22.69 / 0.864 / 0.159 |
| ViewCrafter, DL3DV-10 P=6 | 10.49 / 0.244 / 0.641 | 10.51 / 0.241 / 0.649 |
| ViewCrafter, RE10K P=3 | 16.37 / 0.658 / 0.368 | 15.83 / 0.647 / 0.385 |
