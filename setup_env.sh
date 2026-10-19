# Source this file before running anything:  source setup_env.sh
# Edit the four paths for your machine (see nwear_paths.py for the expected layout).

export NWEAR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NWEAR_DATA="${NWEAR_DATA:-$NWEAR_REPO/data}"              # WEAR data + event clips
export NWEAR_OUTPUT="${NWEAR_OUTPUT:-$NWEAR_REPO/outputs}"        # checkpoints, caches, results
export WEAR_REPO="${WEAR_REPO:-$NWEAR_REPO/third_party/wear}"     # patched WEAR checkout (SIM A)
export NEUROBENCH_ROOT="${NEUROBENCH_ROOT:-$NWEAR_REPO/third_party/neurobench}"  # optional

# Repo root makes `import nwear_paths` and `import event_snn` work from every script.
# TriDet's compiled NMS extension lives inside the WEAR checkout.
export PYTHONPATH="$NWEAR_REPO:$WEAR_REPO/camera_baseline/tridet/libs/utils${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$NEUROBENCH_ROOT" ] && export PYTHONPATH="$PYTHONPATH:$NEUROBENCH_ROOT"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
