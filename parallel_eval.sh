#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash parallel_eval.sh <task_name> <task_config> <policy_config> [gpu_list] [num_processes] [total_num]" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/workspace/UniVTAC/.runtime/bin/python}"

TASK_NAME="$1"
TASK_CONFIG="$2"
POLICY_CONFIG="$3"
GPU="${4:-0}"
NUM_PROCESSES="${5:-2}"
TOTAL_NUM="${6:-100}"

export CUDA_VISIBLE_DEVICES="${GPU}"
"${PYTHON_BIN}" scripts/parallel_eval_policy.py "${TASK_NAME}" "${TASK_CONFIG}" "${POLICY_CONFIG}" \
    --total_num "${TOTAL_NUM}" \
    --workers "${NUM_PROCESSES}"
