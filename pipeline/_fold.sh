# Shared prologue for per-fold pipeline steps. Sourced, not executed.
#
# A fold is (seed, held-out subject): 3 seeds x 18 subjects = 54 runs.
#   bash pipeline/02_train_event_snn.sh 1 0          # seed 1, subject 0
#   sbatch --array=1-54 cluster/array.sbatch pipeline/02_train_event_snn.sh
# Array task t maps to seed = (t-1)/18 + 1, subject = (t-1)%18.
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../setup_env.sh
source "${_here}/../setup_env.sh"

if [[ $# -ge 2 ]]; then
    SEED="$1"; SBJ="$2"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SEED=$(( (SLURM_ARRAY_TASK_ID - 1) / 18 + 1 ))
    SBJ=$(( (SLURM_ARRAY_TASK_ID - 1) % 18 ))
else
    echo "usage: $0 <seed 1-3> <held-out subject 0-17>   (or run as a Slurm array 1-54)" >&2
    exit 2
fi
SPLIT=$(( SBJ + 1 ))          # WEAR wear_split_<n>.json holds out subject n-1
TRAIN_SUBJECTS="$(seq 0 17 | grep -vx "${SBJ}" | tr '\n' ' ')"
ANN_JSON="${NWEAR_DATA}/wear_dataset_annotations/60fps/wear_split_${SPLIT}.json"
DEVICE="${DEVICE:-cuda}"
echo "[fold] seed=${SEED} held-out=sbj_${SBJ} (wear_split_${SPLIT})"
cd "${NWEAR_REPO}"
