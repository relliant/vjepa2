#!/usr/bin/env bash
# Web/portal Slurm entrypoint. Submit this file as one job with 8 GPUs.
# Account, partition, QoS, and time can be set in the web form.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=220G
# Keep these files in the submission working directory. Slurm does not create
# missing parent directories before opening --output/--error.
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/eval/vitg-384/ssv2.yaml}"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7}"

if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run --active python)
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
else
    PYTHON_CMD=(python)
fi

echo "=== V-JEPA 2 SSv2 probe ==="
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "config=${CONFIG}"
echo "python=$("${PYTHON_CMD[@]}" -c 'import sys; print(sys.executable)')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "devices=${DEVICES}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name --format=csv,noheader
fi

# evals.main launches one process per device on this node. Do not call
# evals.main_distributed here: that entrypoint submits a second Slurm job.
read -r -a DEVICE_ARGS <<< "${DEVICES}"
"${PYTHON_CMD[@]}" -u -m evals.main \
    --fname "${CONFIG}" \
    --devices "${DEVICE_ARGS[@]}"
