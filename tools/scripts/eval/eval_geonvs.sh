#!/bin/bash
# GeoNVS (SEVA + GS-Adapter + LoRA) on the small-viewpoint benchmark.
#
#   bash tools/scripts/eval/eval_geonvs.sh [--gpu 0[,1,...]] [lrm_model_name] [dataset:P ...]
#
# With no dataset arguments the full paper table is run (see benchmark.sh).
# Several GPUs shard the table and run in parallel.
# Examples:
#   bash tools/scripts/eval/eval_geonvs.sh --gpu 3 vggt_iv
#   bash tools/scripts/eval/eval_geonvs.sh --gpu 0,1,2,3 vggt_iv
#   bash tools/scripts/eval/eval_geonvs.sh --gpu 0 depthsplat dl3dv10:3
#
# Environment overrides: CONFIG, WEIGHTS, CFG, RES, BENCHMARK_ROOT, DIR_TAG
set -e
cd "$(dirname "$0")/../../.."
source tools/scripts/eval/benchmark.sh

parse_common_flags "$@"
set -- "${REMAINING_ARGS[@]}"

LRM=${1:-vggt_iv}; shift || true
CONFIG=${CONFIG:-configs/module_config/gsadapter_eccv_gattn.yaml}
WEIGHTS=${WEIGHTS:-pretrained_weights/eccv_gattn}
DIR_TAG=${DIR_TAG:-low}

run_entry() {  # <dataset> <num_inputs>
    echo "==> [gpu ${CUDA_VISIBLE_DEVICES}] GeoNVS | $1 | P=$2 | lrm=${LRM}"
    python demo.py \
        --data_path "${BENCHMARK_ROOT}/$1" \
        --num_inputs "$2" \
        --video_save_fps ${VIDEO_FPS} \
        --gs_adapter_config "${CONFIG}" \
        --gs_adapter_weight_path "${WEIGHTS}" \
        --lrm_model_name "${LRM}" \
        --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} --version=${VERSION} --cfg=${CFG}
}

ENTRIES=("$@")
[ ${#ENTRIES[@]} -eq 0 ] && ENTRIES=("${SMALL_VIEWPOINT[@]}")
dispatch "${ENTRIES[@]}"
