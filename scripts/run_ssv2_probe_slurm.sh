#!/usr/bin/env bash
# Web/portal Slurm entrypoint. Submit this file as one job with 8 GPUs.
# Account, partition, QoS, and time can be set in the web form.
#SBATCH --nodes=1
# Match the cluster's working multi-GPU scripts.
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=220G

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/eval/vitg-384/ssv2.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    # Use the already-synced uv environment; do not let uv resolve or create
    # another environment on the compute node.
    source "${REPO_ROOT}/.venv/bin/activate"
    PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
else
    PYTHON_CMD=(python)
fi

echo "=== V-JEPA 2 SSv2 probe ==="
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "config=${CONFIG}"
echo "python=$("${PYTHON_CMD[@]}" -c 'import sys; print(sys.executable)')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"
echo "nproc_per_node=${NPROC_PER_NODE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name --format=csv,noheader
fi

# torchrun launches one process per GPU on this node. Do not call
# evals.main_distributed here: that entrypoint submits a second Slurm job.
"${PYTHON_CMD[@]}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m evals.main \
    --fname "${CONFIG}" \
    --debugmode True \
    --use_fsdp
