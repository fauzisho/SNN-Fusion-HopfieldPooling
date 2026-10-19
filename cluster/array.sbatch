#!/bin/bash
# Generic Slurm wrapper: runs one pipeline step for array task t (seed=(t-1)/18+1, subject=(t-1)%18).
#
#   sbatch --array=1-54%6 cluster/array.sbatch pipeline/02_train_event_snn.sh
#   CONTAINER=/path/to/wearsnn_v2e.sif sbatch --array=1-54 cluster/array.sbatch pipeline/06_simb_heads.sh
#
# Adjust the resources to your cluster. The paper runs used one RTX 2080 Ti per task.
#SBATCH --job-name=nwear
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
STEP="${1:?usage: sbatch --array=1-54 cluster/array.sbatch pipeline/<step>.sh}"
mkdir -p logs
if [[ -n "${CONTAINER:-}" ]]; then
    # --cleanenv + PYTHONNOUSERSITE keep host packages (e.g. ~/.local torch) out of the container
    # EXTRA_BINDS: further "-B dir" arguments, e.g. when outputs are symlinks to other disks
    # shellcheck disable=SC2086
    singularity exec --nv --cleanenv -B "${NWEAR_DATA:-$PWD}" -B "${NWEAR_OUTPUT:-$PWD}" -B "$PWD" ${EXTRA_BINDS:-} "${CONTAINER}" \
        env PYTHONNOUSERSITE=1 SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}" \
            NWEAR_DATA="${NWEAR_DATA:-}" NWEAR_OUTPUT="${NWEAR_OUTPUT:-}" WEAR_REPO="${WEAR_REPO:-}" \
            NEUROBENCH_ROOT="${NEUROBENCH_ROOT:-}" \
        bash "${STEP}"
else
    bash "${STEP}"
fi
