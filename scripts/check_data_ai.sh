#!/bin/bash
#SBATCH --job-name=check_data_ai
#SBATCH --nodes=1
#SBATCH --nodelist=fg02
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/tmp
#SBATCH --output=/data-sl/lsy_ws/diag-data-ai-%x-%j.out
#SBATCH --error=/data-sl/lsy_ws/diag-data-ai-%x-%j.err

set -euxo pipefail

echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "===== mount visibility ====="
for path in \
    /data-ai \
    /data-ai/lsy_ws \
    /data-ai/lsy_ws/project/vjepa2 \
    /data-ai/lsy_ws/project/vjepa2/scripts \
    /data-sl; do
    if [[ -e "${path}" ]]; then
        ls -ld "${path}"
    else
        echo "MISSING: ${path}"
    fi
done

echo "===== data-ai write test ====="
write_test="/data-ai/lsy_ws/project/vjepa2/scripts/.compute-write-test-${SLURM_JOB_ID:-$$}"
if touch "${write_test}"; then
    echo "WRITE_OK: /data-ai"
    rm -f "${write_test}"
else
    echo "WRITE_FAILED: /data-ai" >&2
fi

echo "===== gpu ====="
nvidia-smi
