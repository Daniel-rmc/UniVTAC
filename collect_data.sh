#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash collect_data.sh <task_name> <config_name> [gpu] [start_seed] [max_seed] [episode]" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/workspace/UniVTAC/.runtime/bin/python}"

TASK_NAME="$1"
CONFIG_NAME="$2"
GPU="${3:-0}"
START_SEED="${4:--1}"
MAX_SEED="${5:--1}"
EPISODE="${6:--1}"

"${PYTHON_BIN}" scripts/collect_data.py \
    "${TASK_NAME}" "${CONFIG_NAME}" \
    --start_seed "${START_SEED}" \
    --max_seed "${MAX_SEED}" \
    --episode_num "${EPISODE}" \
    --gpu "${GPU}"
