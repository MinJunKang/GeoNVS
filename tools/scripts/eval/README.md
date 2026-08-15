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
| `eval_baseline.sh` | SEVA, LoRA-only, input-level injection, feed-forward geometry, diffusion baselines |
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
