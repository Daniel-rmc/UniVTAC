#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: bash eval_policy.sh <task_name> <task_config> <policy_config> <gpu_list>" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/workspace/UniVTAC/.runtime/bin/python}"

TASK_NAME="$1"
TASK_CONFIG="$2"
POLICY_CONFIG="$3"
GPU="$4"

export CUDA_VISIBLE_DEVICES="${GPU}"
"${PYTHON_BIN}" scripts/eval_policy.py "${TASK_NAME}" "${TASK_CONFIG}" "${POLICY_CONFIG}"

# bash eval_policy.sh ACT/deploy_policy_insert_lean 0
