#!/usr/bin/env bash
# V-JEPA 2 SSv2 attentive-probe job for the CHESS Slurm cluster.
#
# The allocation deliberately contains one Slurm task and eight GPUs.  The
# evaluation itself is launched with torchrun so that it owns one process per
# GPU, matching evals/main.py's FSDP debug entrypoint.
#
# The portal may provide account, qos, partition, and time limits.  The
# directives below are defaults for direct `sbatch` submission and are ignored
# when the portal supplies its own allocation.
#SBATCH --job-name=train_vjepa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
# Slurm opens these files before executing the script.  Use absolute paths
# because CHESS starts jobs from the user's home directory by default.
#SBATCH --chdir=/data-ai/lsy_ws/project/vjepa2
#SBATCH --output=/data-ai/lsy_ws/project/vjepa2/scripts/slurm-%x-%j.out
#SBATCH --error=/data-ai/lsy_ws/project/vjepa2/scripts/slurm-%x-%j.err

set -euo pipefail

# The CHESS launcher starts batch jobs from the user's home directory.  Set and
# enter the repository explicitly before resolving any relative paths.
BASE="${BASE:-/data-ai/lsy_ws}"
REPO_ROOT="${REPO_ROOT:-${BASE}/project/vjepa2}"
cd "${REPO_ROOT}" || {
    echo "ERROR: repository directory not found: ${REPO_ROOT}" >&2
    exit 1
}

# ------------------------------------------------------------------------------
# Paths and run configuration
# ------------------------------------------------------------------------------

VENV="${VENV:-${REPO_ROOT}/.venv}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/eval/vitg-384/ssv2.yaml}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
JOB_ID="${SLURM_JOB_ID:-local-$$}"

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/pretrained_model/vitg-384.pt}"
TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/dataset/ssv2_train_paths.csv}"
VAL_CSV="${VAL_CSV:-${REPO_ROOT}/dataset/ssv2_val_paths.csv}"

# A single node uses local rendezvous.  evals.main's distributed helper uses
# the same local process group for all eight ranks.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ -z "${MASTER_PORT:-}" ]]; then
    if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
        export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"
    else
        export MASTER_PORT=37129
    fi
fi
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# Do not override CUDA_VISIBLE_DEVICES: Slurm sets it to the GPUs assigned to
# this job.  GPUS_PER_NODE can be overridden for a smaller smoke test.
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# ------------------------------------------------------------------------------
# Helpers and preflight checks
# ------------------------------------------------------------------------------

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local description="$2"
    [[ -f "${path}" ]] || die "${description} not found: ${path}"
}

require_dir() {
    local path="$1"
    local description="$2"
    [[ -d "${path}" ]] || die "${description} not found: ${path}"
}

[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GPUS_PER_NODE must be a positive integer"
require_dir "${REPO_ROOT}" "V-JEPA repository"
require_file "${CONFIG}" "SSv2 config"
require_file "${CHECKPOINT}" "V-JEPA checkpoint"
require_file "${TRAIN_CSV}" "SSv2 training CSV"
require_file "${VAL_CSV}" "SSv2 validation CSV"
require_file "${VENV}/bin/python" "uv environment Python"
[[ -x "${VENV}/bin/python" ]] || die "Python is not executable: ${VENV}/bin/python"

mkdir -p "${LOG_DIR}"

# Emit diagnostics before importing PyTorch.  This makes the batch entrypoint
# observable immediately in CHESS's web log viewer.
PYTHON_BIN="${PYTHON_BIN:-${VENV}/bin/python}"
LOG_FILE="${LOG_DIR}/ssv2_probe_${JOB_ID}.log"
{
    echo "=== V-JEPA 2 SSv2 attentive probe ==="
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "host=$(hostname)"
    echo "job_id=${JOB_ID}"
    echo "repo=${REPO_ROOT}"
    echo "base=${BASE}"
    echo "venv=${VENV}"
    echo "python=${PYTHON_BIN}"
    echo "config=${CONFIG}"
    echo "checkpoint=${CHECKPOINT}"
    echo "train_csv=${TRAIN_CSV}"
    echo "val_csv=${VAL_CSV}"
    echo "gpus_per_node=${GPUS_PER_NODE}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "slurm_nnodes=${SLURM_NNODES:-unset}"
    echo "slurm_ntasks=${SLURM_NTASKS:-unset}"
    echo "master_addr=${MASTER_ADDR}"
    echo "master_port=${MASTER_PORT}"
    echo "==============================="
} | tee -a "${LOG_FILE}"

# Activate the already-created uv environment for auxiliary executables and
# make the selected interpreter explicit for the actual job.
source "${VENV}/bin/activate"
[[ -x "${PYTHON_BIN}" ]] || die "Python executable is not executable: ${PYTHON_BIN}"

if ! "${PYTHON_BIN}" - <<'PY'
import torch
import yaml
import decord

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"decord={getattr(decord, '__version__', 'installed')}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA device")
PY
then
    die "Python preflight failed; verify torch, pyyaml, decord, and CUDA in ${VENV}"
fi

CUDA_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
[[ "${CUDA_COUNT}" =~ ^[0-9]+$ ]] || die "Could not determine CUDA device count"
(( CUDA_COUNT >= GPUS_PER_NODE )) || die "Requested ${GPUS_PER_NODE} GPUs but PyTorch sees ${CUDA_COUNT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "${LOG_FILE}"
fi

# ------------------------------------------------------------------------------
# Launch one local process per GPU through a single Slurm task
# ------------------------------------------------------------------------------

echo "Starting torchrun; live output is appended to ${LOG_FILE}" | tee -a "${LOG_FILE}"
set +e
srun --ntasks=1 --kill-on-bad-exit=1 \
"${PYTHON_BIN}" -u -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m evals.main \
    --fname "${CONFIG}" \
    --debugmode True \
    --use_fsdp 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

if (( STATUS != 0 )); then
    echo "SSv2 probe failed with exit code ${STATUS}" | tee -a "${LOG_FILE}" >&2
    exit "${STATUS}"
fi

echo "SSv2 probe completed successfully" | tee -a "${LOG_FILE}"
