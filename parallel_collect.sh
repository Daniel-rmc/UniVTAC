#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash parallel_collect.sh <task_name> [config_name] [gpu_list] [num_processes]" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/workspace/UniVTAC/.runtime/bin/python}"

TASK_NAME="$1"
CONFIG_NAME="${2:-demo}"
GPU="${3:-0}"
NUM_PROCESSES="${4:-3}"

export CUDA_VISIBLE_DEVICES="${GPU}"
"${PYTHON_BIN}" scripts/parallel_collect_data.py "${TASK_NAME}" "${CONFIG_NAME}" \
    --workers="${NUM_PROCESSES}"
