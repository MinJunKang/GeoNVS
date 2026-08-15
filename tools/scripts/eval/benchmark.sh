#!/bin/bash
# Benchmark table used by the GeoNVS paper (Sec. 4.3, Tables 1-2).
#
# Each entry is "<dataset-path-under-benchmarkset>:<number-of-reference-views>".
# The small-viewpoint set covers 9 scene collections; the long-trajectory set
# (Sec. 4.3) re-runs a subset with the trajectory-prior protocol.
#
# Sourced by the eval_*.sh runners; override any of these from the environment.

# Root of the benchmark data (see tools/benchmark/README.md).
BENCHMARK_ROOT=${BENCHMARK_ROOT:-datasets/benchmarkset}

# --- small-viewpoint set (main results) ------------------------------------
SMALL_VIEWPOINT=(
    "omniobject3d:3"
    "re10k:1"       "re10k:2"       "re10k:3"
    "llff:1"        "llff:3"
    "dtu:1"         "dtu:3"
    "co3d:1"        "co3d:3"
    "wildrgbd/wildrgbd_easy:3"
    "wildrgbd/wildrgbd_hard:1"  "wildrgbd/wildrgbd_hard:3"  "wildrgbd/wildrgbd_hard:6"
    "mipnerf360:1"  "mipnerf360:3"  "mipnerf360:6"
    "dl3dv10:1"     "dl3dv10:3"     "dl3dv10:6"
    "tnt-cf3dgs:2"  "tnt-cf3dgs:3"
)

# --- long-trajectory set (Sec. 4.3) ----------------------------------------
LONG_TRAJECTORY=(
    "dl3dv10:6"
    "dtu:3"
    "mipnerf360:6"
    "re10k:3"
    "tnt-cf3dgs:3"
)

# --- defaults shared by every runner (Sec. 4.2 inference setup) ------------
RES=${RES:-384}              # 384x384 protocol used for all reported numbers
CFG=${CFG:-1.4}
VERSION=${VERSION:-1.0}      # SEVA model version
VIDEO_FPS=${VIDEO_FPS:-10}
GPUS=${GPUS:-${GPU:-0}}      # --gpu 0  |  --gpu 0,1,2,3  (also honours $GPU)

# Strip common flags (currently --gpu/--gpus) from "$@"; the rest is returned
# in REMAINING_ARGS.
parse_common_flags() {
    REMAINING_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu|--gpus)   GPUS="$2"; shift 2 ;;
            --gpu=*|--gpus=*) GPUS="${1#*=}"; shift ;;
            *) REMAINING_ARGS+=("$1"); shift ;;
        esac
    done
}

# Run every "<dataset>:<P>" entry through the caller-provided `run_entry`
# function. With several GPUs the entries are sharded round-robin and the
# shards run in parallel, one process per GPU.
dispatch() {
    local entries=("$@")
    IFS=',' read -ra _gpu_list <<< "${GPUS}"

    if [ ${#_gpu_list[@]} -le 1 ]; then
        export CUDA_VISIBLE_DEVICES=${_gpu_list[0]}
        for entry in "${entries[@]}"; do
            run_entry "${entry%:*}" "${entry##*:}"
        done
        return
    fi

    local n=${#_gpu_list[@]} i j shard
    for i in "${!_gpu_list[@]}"; do
        shard=()
        for j in "${!entries[@]}"; do
            [ $(( j % n )) -eq "$i" ] && shard+=("${entries[$j]}")
        done
        [ ${#shard[@]} -eq 0 ] && continue
        (
            export CUDA_VISIBLE_DEVICES=${_gpu_list[$i]}
            for entry in "${shard[@]}"; do
                run_entry "${entry%:*}" "${entry##*:}"
            done
        ) &
    done
    wait
}
