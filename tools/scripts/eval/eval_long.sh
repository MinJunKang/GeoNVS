#!/bin/bash
# Long-trajectory protocol (Sec. 4.3): img2vid with a trajectory prior and
# nearest-gt chunking, plus voxel-pruned Gaussians (Sec. 4.6).
#
#   bash tools/scripts/eval/eval_long.sh [--gpu 0[,1,...]] [method] [lrm_model_name] [dataset:P ...]
#
# method: geonvs (default) | seva | lrm
# Examples:
#   bash tools/scripts/eval/eval_long.sh --gpu 0,1              # GeoNVS, full set
#   bash tools/scripts/eval/eval_long.sh --gpu 2 seva
#   bash tools/scripts/eval/eval_long.sh lrm depthsplat dl3dv10:6
set -e
cd "$(dirname "$0")/../../.."
source tools/scripts/eval/benchmark.sh

parse_common_flags "$@"
set -- "${REMAINING_ARGS[@]}"

METHOD=${1:-geonvs}; shift || true
LRM=${1:-vggt_iv}; shift || true
CONFIG=${CONFIG:-configs/module_config/gsadapter_eccv_gattn.yaml}
WEIGHTS=${WEIGHTS:-pretrained_weights/eccv_gattn}
VOXEL=${VOXEL:-0.2}
DIR_TAG=${DIR_TAG:-low_long}

TRAJ_ARGS=(--task img2vid --replace_or_include_input True
           --use_traj_prior True --chunking_strategy nearest-gt)

run_entry() {  # <dataset> <num_inputs>
    local DATA="${BENCHMARK_ROOT}/$1" NUM_INPUTS="$2"
    echo "==> [gpu ${CUDA_VISIBLE_DEVICES}] long ${METHOD} | $1 | P=$2 | lrm=${LRM}"
    case "${METHOD}" in
        geonvs)
            python demo.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} "${TRAJ_ARGS[@]}" \
                --gs_adapter_config "${CONFIG}" \
                --gs_adapter_weight_path "${WEIGHTS}" \
                --lrm_model_name "${LRM}" --lrm_voxel_size ${VOXEL} \
                --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} --version=${VERSION} --cfg=${CFG} ;;
        seva)
            python demo.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} "${TRAJ_ARGS[@]}" \
                --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} --version=${VERSION} ;;
        lrm)
            python demo_regression.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} "${TRAJ_ARGS[@]}" \
                --lrm_model_name "${LRM}" \
                --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} ;;
        *) echo "unknown method: ${METHOD}"; exit 1 ;;
    esac
}

ENTRIES=("$@")
[ ${#ENTRIES[@]} -eq 0 ] && ENTRIES=("${LONG_TRAJECTORY[@]}")
dispatch "${ENTRIES[@]}"
