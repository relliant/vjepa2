#!/usr/bin/env bash
# Web/portal Slurm entrypoint. The portal supplies resources; this script
# follows the cluster's working srun-based multi-GPU pattern.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/eval/vitg-384/ssv2.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

# Match the known-good cluster launcher for a single node.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + (${SLURM_JOB_ID:-0} % 50000)))}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1

# The cluster's working scripts activate a prepared Conda environment. The
# repository-local .venv may only be a uv-created Python shim without torch,
# so prefer the known environment and allow an explicit override.
if [[ -n "${VJEPA_PYTHON:-}" ]]; then
    PYTHON_BIN="${VJEPA_PYTHON}"
elif [[ -f "/share/app/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/share/app/anaconda3/etc/profile.d/conda.sh"
    conda activate "${VJEPA_CONDA_ENV:-vjepa2}"
    PYTHON_BIN="$(command -v python)"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

"${PYTHON_BIN}" -c 'import torch, yaml' || {
    echo "The selected Python environment is missing torch or yaml." >&2
    echo "Set VJEPA_PYTHON to the prepared interpreter or VJEPA_CONDA_ENV to its Conda name." >&2
    exit 1
}

echo "=== V-JEPA 2 SSv2 probe ==="
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "config=${CONFIG}"
echo "python=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
echo "SLURM_PROCID=${SLURM_PROCID:-<unset>}"
echo "SLURM_LOCALID=${SLURM_LOCALID:-<unset>}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name --format=csv,noheader
fi

# srun launches one task per GPU. Each task receives its own rank from Slurm.
# evals.main's debug path consumes these variables and does not submit a job.
export VJEPA_CONFIG="${CONFIG}"
export VJEPA_PYTHON="${PYTHON_BIN}"
srun bash -c '
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export LOCAL_RANK="${SLURM_LOCALID}"
    exec "${VJEPA_PYTHON}" -u -m evals.main \
        --fname "${VJEPA_CONFIG}" \
        --debugmode True \
        --use_fsdp
' 2>&1 | tee "${LOG_DIR}/ssv2_probe_${SLURM_JOB_ID:-local}.log"
