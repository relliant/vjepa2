#!/usr/bin/env bash
set -euo pipefail

# Launches the repository's Submitit-based SLURM evaluator from a login node.
# Usage: ./scripts/submit_ssv2_probe.sh ACCOUNT PARTITION QOS [TIME_MINUTES]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ACCOUNT="${1:-${SLURM_ACCOUNT:-}}"
PARTITION="${2:-${SLURM_PARTITION:-learn}}"
QOS="${3:-${SLURM_QOS:-}}"
TIME_MINUTES="${4:-${SLURM_TIME_MINUTES:-8600}}"
CONFIG="${CONFIG:-configs/eval/vitg-384/ssv2.yaml}"
SUBMIT_FOLDER="${SUBMIT_FOLDER:-experiments/slurm/ssv2-vitg-384}"

if [[ -z "${ACCOUNT}" || -z "${QOS}" ]]; then
    cat >&2 <<'EOF'
Usage: ./scripts/submit_ssv2_probe.sh ACCOUNT PARTITION QOS [TIME_MINUTES]

Example:
  ./scripts/submit_ssv2_probe.sh my_account learn my_qos 8600
EOF
    exit 2
fi

if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run --active python)
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
else
    PYTHON_CMD=(python)
fi

echo "Repository: ${REPO_ROOT}"
echo "Python: $("${PYTHON_CMD[@]}" -c 'import sys; print(sys.executable)')"
echo "Config: ${CONFIG}"
echo "SLURM account=${ACCOUNT} partition=${PARTITION} qos=${QOS} time_min=${TIME_MINUTES}"

"${PYTHON_CMD[@]}" -m evals.main_distributed \
    --fname "${CONFIG}" \
    --folder "${SUBMIT_FOLDER}" \
    --account "${ACCOUNT}" \
    --partition "${PARTITION}" \
    --qos "${QOS}" \
    --time "${TIME_MINUTES}"
