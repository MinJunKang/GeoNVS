#!/bin/bash
# Ablations reported in Sec. 4.6 / the supplementary material.
#
#   bash tools/scripts/eval/eval_ablation.sh [--gpu 0[,1,...]] <study> [dataset:P ...]
#
# Studies:
#   topk        top-k Gaussians per ray (k = 1/16/32/64/128; needs the matching
#               configs/module_config/gsadapter_eccv_gattn_k<k>.yaml)
#   pruning     voxel-based Gaussian pruning (voxel size sweep)
#   cfg         classifier-free guidance scale sweep
#   fusion      Naive Fusion vs Adaptive Fusion
#   geometry    plug-and-play geometry models with the same adapter
#
# Default target is dl3dv10:3 (the ablation set used in the paper).
set -e
cd "$(dirname "$0")/../../.."
source tools/scripts/eval/benchmark.sh

parse_common_flags "$@"
set -- "${REMAINING_ARGS[@]}"

STUDY=$1; shift || { echo "usage: $0 [--gpu N] <topk|pruning|cfg|fusion|geometry> [dataset:P ...]"; exit 1; }
WEIGHTS=${WEIGHTS:-pretrained_weights/eccv_gattn}
CONFIG=${CONFIG:-configs/module_config/gsadapter_eccv_gattn.yaml}
LRM=${LRM:-vggt_iv}

# Sweep values (override to run a single setting, e.g. VOXELS=0.05)
KS=${KS:-"1 16 32 64 128"}
VOXELS=${VOXELS:-"0.05 0.1 0.2 0.25 0.5 0.75"}
CFGS=${CFGS:-"1.0 1.25 1.4 1.5 1.75 2.0"}

_run() {  # _run <dir_tag> <lrm> <extra args...>
    local tag=$1 lrm=$2; shift 2
    python demo.py \
        --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
        --video_save_fps ${VIDEO_FPS} \
        --gs_adapter_weight_path "${WEIGHTS}" \
        --lrm_model_name "${lrm}" \
        --dir_tag=${tag} --H=${RES} --W=${RES} --version=${VERSION} "$@"
}

run_entry() {  # <dataset> <num_inputs>
    DATA="${BENCHMARK_ROOT}/$1"; NUM_INPUTS="$2"
    echo "==> [gpu ${CUDA_VISIBLE_DEVICES}] ablation:${STUDY} | $1 | P=$2"
    case "${STUDY}" in
        topk)
            for k in ${KS}; do
                _run "low_k${k}" "${LRM}" \
                    --gs_adapter_config "configs/module_config/gsadapter_eccv_gattn_k${k}.yaml" --cfg=${CFG}
            done ;;
        pruning)
            for v in ${VOXELS}; do
                _run "low_prune_${v}" "${LRM}" --gs_adapter_config "${CONFIG}" --lrm_voxel_size ${v} --cfg=${CFG}
            done ;;
        cfg)
            for c in ${CFGS}; do
                _run "low_cfg${c}" "${LRM}" --gs_adapter_config "${CONFIG}" --cfg=${c}
            done ;;
        fusion)
            for cfg_name in gsadapter_eccv_base gsadapter_eccv_gattn; do
                _run "low_${cfg_name}" "${LRM}" \
                    --gs_adapter_config "configs/module_config/${cfg_name}.yaml" --cfg=${CFG}
            done ;;
        geometry)
            for lrm in vggt_iv vggt_av pi3_iv depthsplat; do
                _run "low_geo_${lrm}" "${lrm}" --gs_adapter_config "${CONFIG}" --cfg=${CFG}
            done ;;
        *) echo "unknown study: ${STUDY}"; exit 1 ;;
    esac
}

ENTRIES=("$@")
[ ${#ENTRIES[@]} -eq 0 ] && ENTRIES=("dl3dv10:3")
dispatch "${ENTRIES[@]}"
