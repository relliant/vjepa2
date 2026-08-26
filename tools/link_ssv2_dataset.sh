#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 DATASET_ROOT" >&2
    echo "Example: $0 /data-ai/lsy_ws/dataset/ssv2/something_something_2" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$1"
LINK_PATH="${REPO_ROOT}/dataset/something_something_2"

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset directory does not exist: ${DATASET_ROOT}" >&2
    exit 1
fi

if [[ -e "${LINK_PATH}" || -L "${LINK_PATH}" ]]; then
    if [[ -L "${LINK_PATH}" ]]; then
        rm -- "${LINK_PATH}"
    else
        echo "Refusing to replace existing directory: ${LINK_PATH}" >&2
        echo "Move it outside the repository first, then rerun this script." >&2
        exit 1
    fi
fi

ln -s "$(cd "${DATASET_ROOT}" && pwd)" "${LINK_PATH}"
echo "Created: ${LINK_PATH} -> $(readlink "${LINK_PATH}")"
