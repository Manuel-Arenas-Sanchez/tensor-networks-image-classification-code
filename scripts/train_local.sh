#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_INPUT="${1:-configs/experiments/mnist_mps.yaml}"
RUNTIME_INPUT="${2:-configs/runtime/local.yaml}"
DEVICE="${3:-auto}"

if [[ "${CONFIG_INPUT}" = /* ]]; then
    CONFIG="${CONFIG_INPUT}"
else
    CONFIG="${PROJECT_ROOT}/${CONFIG_INPUT}"
fi

if [[ "${RUNTIME_INPUT}" = /* ]]; then
    RUNTIME="${RUNTIME_INPUT}"
else
    RUNTIME="${PROJECT_ROOT}/${RUNTIME_INPUT}"
fi

"${PROJECT_ROOT}/.venv/bin/python" -m tn_dl.cli.train --config "${CONFIG}" --runtime "${RUNTIME}" --device "${DEVICE}"
