#!/bin/bash
# Baselines on the same benchmark table as eval_geonvs.sh.
#
#   bash tools/scripts/eval/eval_baseline.sh [--gpu 0[,1,...]] <method> [extra_arg] [dataset:P ...]
#
# Methods:
#   seva                      SEVA backbone, no adapter          (demo.py)
#   lora                      LoRA-only ablation                 (demo.py)
#   input_level <strength>    input-level geometry injection,    (demo.py)
#                             i.e. denoise from rendered views (paper: s=0.2)
#   lrm <lrm_name>            feed-forward geometry only         (demo_regression.py)
#   diffusion <dm_name>       video-diffusion baselines          (demo_diffusion.py)
#
# Examples:
#   bash tools/scripts/eval/eval_baseline.sh --gpu 0,1 seva
#   bash tools/scripts/eval/eval_baseline.sh --gpu 2 lrm depthsplat dl3dv10:3
#   bash tools/scripts/eval/eval_baseline.sh diffusion geonvs_cameractrl_depthsplat
#   bash tools/scripts/eval/eval_baseline.sh input_level 0.2 dl3dv10:3
set -e
cd "$(dirname "$0")/../../.."
source tools/scripts/eval/benchmark.sh

parse_common_flags "$@"
set -- "${REMAINING_ARGS[@]}"

METHOD=$1; shift || { echo "usage: $0 [--gpu N] <seva|lora|input_level|lrm|diffusion> ..."; exit 1; }
DIR_TAG=${DIR_TAG:-low}

case "${METHOD}" in
    lrm|diffusion|input_level) ARG=$1; shift || { echo "'${METHOD}' needs an argument"; exit 1; } ;;
esac

run_entry() {  # <dataset> <num_inputs>
    local DATA="${BENCHMARK_ROOT}/$1" NUM_INPUTS="$2"
    echo "==> [gpu ${CUDA_VISIBLE_DEVICES}] ${METHOD} ${ARG} | $1 | P=$2"
    case "${METHOD}" in
        seva)
            python demo.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} \
                --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} --version=${VERSION} ;;
        lora)
            python demo.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} \
                --gs_adapter_config configs/module_config/lora_only.yaml \
                --gs_adapter_weight_path "${WEIGHTS:-pretrained_weights/eccv_gattn}" \
                --lrm_model_name "${LRM:-vggt_iv}" \
                --dir_tag=${DIR_TAG}_lora --H=${RES} --W=${RES} --version=${VERSION} --cfg=${CFG} ;;
        input_level)
            python demo.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} \
                --inpainting_mode 1 --inpainting_strength "${ARG}" \
                --lrm_model_name "${LRM:-vggt_iv}" \
                --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} --version=${VERSION} ;;
        lrm)
            python demo_regression.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} \
                --lrm_model_name "${ARG}" \
                --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} ;;
        diffusion)
            python demo_diffusion.py \
                --data_path "${DATA}" --num_inputs "${NUM_INPUTS}" \
                --video_save_fps ${VIDEO_FPS} \
                --dm_model_name "${ARG}" \
                --dir_tag=${DIR_TAG} --H=${RES} --W=${RES} ;;
        *) echo "unknown method: ${METHOD}"; exit 1 ;;
    esac
}

ENTRIES=("$@")
[ ${#ENTRIES[@]} -eq 0 ] && ENTRIES=("${SMALL_VIEWPOINT[@]}")
dispatch "${ENTRIES[@]}"
