#!/bin/bash
set -euo pipefail
# Step 3: convert 128x128 clips (*_dvs128.mp4) to v2e event HDF5 files.
#
#   Input:  $NWEAR_DATA/sbj_<k>_clips_dvs128${SUFFIX}/<segment>/<stem>_dvs128.mp4
#   Output: $NWEAR_DATA/sbj_<k>_events_dvs128${SUFFIX}/<segment>/<stem>_dvs128.h5
#   H5:     dataset "events", shape (N, 4), uint32, columns (t_us, x, y, p)
#
#   SBJ_ID=0 bash data_preparation/clips_dvs128_to_v2e.sh               # GPU recommended
#   SBJ_ID=16 SUFFIX=_new bash data_preparation/clips_dvs128_to_v2e.sh  # re-conversion
#
# Needs v2e (paper: commit cac638444557a224ba6ee5c29260916f152c00bd, v1.5.1) and
# $NWEAR_DATA/SuperSloMo39.ckpt (md5 0fd4fb0a1e002d2aa8bd7c7e01001626, linked from
# the v2e README). Existing valid .h5 files are skipped.

export KMP_DUPLICATE_LIB_OK=TRUE
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

if [[ -n "${PYTHON:-}" ]]; then
    :
else
    PYTHON="$(command -v python3 || true)"
    if [[ -z "$PYTHON" ]]; then
        echo "Error: v2e Python not found. Install v2e and export PYTHON=/path/to/python3" >&2
        exit 1
    fi
fi
BINDIR="$(dirname "$PYTHON")"
if [[ -f "$BINDIR/v2e.py" ]]; then
    V2E=( "$PYTHON" "$BINDIR/v2e.py" )
elif [[ -x "$BINDIR/v2e" ]]; then
    V2E=( "$BINDIR/v2e" )
elif "$PYTHON" -c "import v2e" 2>/dev/null; then
    V2E=( "$PYTHON" -m v2e )
else
    echo "Error: v2e not found for interpreter: $PYTHON" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${NWEAR_DATA:?Set NWEAR_DATA (see setup_env.sh)}"
SBJ_ID="${SBJ_ID:?Set SBJ_ID (e.g. 0)}"
# Optional folder suffix, e.g. SUFFIX=_new → sbj_N_clips_dvs128_new / sbj_N_events_dvs128_new
SUFFIX="${SUFFIX:-}"

# ====================================================================================
INPUT_ROOT="${INPUT_ROOT:-$DATA_ROOT/sbj_${SBJ_ID}_clips_dvs128${SUFFIX}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/sbj_${SBJ_ID}_events_dvs128${SUFFIX}}"
SLOMO_MODEL_PATH="${SLOMO_MODEL_PATH:-$DATA_ROOT/SuperSloMo39.ckpt}"
V2E_BATCH_SIZE="${V2E_BATCH_SIZE:-2}"
OUTPUT_WIDTH="${OUTPUT_WIDTH:-128}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-128}"
# ====================================================================================

h5_events_ok() {
    local f="$1"
    [[ -f "$f" ]] || return 1
    "$PYTHON" - "$f" <<'PY' 2>/dev/null
import sys
import h5py
path = sys.argv[1]
with h5py.File(path, "r") as h:
    if "events" not in h:
        sys.exit(1)
    if h["events"].shape[0] < 1:
        sys.exit(1)
sys.exit(0)
PY
}

discover_clip_dirs() {
    find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
}

resolve_clip_dir() {
    if [[ -n "${CLIP_DIR:-}" ]]; then
        echo "$CLIP_DIR"
        return 0
    fi
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        mapfile -t _clip_dirs < <(discover_clip_dirs)
        local idx=$((SLURM_ARRAY_TASK_ID - 1))
        if (( idx < 0 || idx >= ${#_clip_dirs[@]} )); then
            echo "Array task ${SLURM_ARRAY_TASK_ID}: no clip dir (have ${#_clip_dirs[@]}); exiting."
            exit 0
        fi
        echo "${_clip_dirs[$idx]}"
        return 0
    fi
    return 1
}

process_clip_dir() {
    local clip_dir="$1"
    local clip_name output_dir
    local found_any=0 skipped=0 converted=0

    clip_name="$(basename "$clip_dir")"
    output_dir="$OUTPUT_ROOT/$clip_name"
    mkdir -p "$output_dir"

    echo "=== $clip_name ==="
    echo "    input:  $clip_dir"
    echo "    output: $output_dir"
    echo "    size:   ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}  batch_size=$V2E_BATCH_SIZE"

    for video_file in "$clip_dir"/*_dvs128.mp4; do
        [[ -f "$video_file" ]] || continue
        found_any=1

        local filename stem out_h5
        filename="$(basename "$video_file")"
        stem="${filename%.mp4}"
        out_h5="$output_dir/${stem}.h5"

        if h5_events_ok "$out_h5"; then
            echo "[skip] $clip_name/$filename -> valid ${stem}.h5"
            skipped=$((skipped + 1))
            continue
        fi
        if [[ -f "$out_h5" ]]; then
            echo "[rm partial] $out_h5"
            rm -f "$out_h5"
        fi

        echo "--------------------------------------------------------"
        echo "Processing: $clip_name/$filename"
        echo "  events h5: $out_h5"

        "${V2E[@]}" \
            --input "$video_file" \
            --output_folder "$output_dir" \
            --output_width "$OUTPUT_WIDTH" \
            --output_height "$OUTPUT_HEIGHT" \
            --skip_video_output \
            --batch_size "$V2E_BATCH_SIZE" \
            --slomo_model "$SLOMO_MODEL_PATH" \
            --no_preview \
            --pos_thres 0.2 \
            --neg_thres 0.2 \
            --sigma_thres 0.03 \
            --cutoff_hz 200 \
            --leak_rate_hz 0.1 \
            --shot_noise_rate_hz 1.0 \
            --refractory_period 0 \
            --noise_rate_cov_decades 0.1 \
            --dvs_h5 "${stem}.h5" \
            --overwrite

        echo "Finished: $clip_name/$filename"
        converted=$((converted + 1))
        echo "--------------------------------------------------------"
    done

    if [[ "$found_any" -eq 0 ]]; then
        echo "No *_dvs128.mp4 files in '$clip_dir'."
        return 1
    fi

    echo "--- Done $clip_name: converted=$converted skipped=$skipped ---"
}

if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "Error: input directory not found: $INPUT_ROOT" >&2
    exit 1
fi

if [[ ! -f "$SLOMO_MODEL_PATH" ]]; then
    echo "Error: SloMo checkpoint not found: $SLOMO_MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "--- v2e: sbj_${SBJ_ID} DVS128 clips (128x128) ---"
echo "Input root:  $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo ""

single_clip_dir=""
if single_clip_dir="$(resolve_clip_dir)"; then
    process_clip_dir "$single_clip_dir"
    exit 0
fi

found_any=0
for clip_dir in "$INPUT_ROOT"/*/; do
    [[ -d "$clip_dir" ]] || continue
    found_any=1
    process_clip_dir "$clip_dir"
done

if [[ "$found_any" -eq 0 ]]; then
    echo "No activity folders under '$INPUT_ROOT'. Exiting."
    exit 1
fi

echo "--- All activity folders processed. ---"
