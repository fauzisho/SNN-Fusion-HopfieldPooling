#!/bin/bash
set -euo pipefail
# Step 2 (as run for the paper): letterbox 240x180 clips (*_dvs240.mp4) to 128x128.
#
#   Input:  $NWEAR_DATA/sbj_<k>_clips_dvs240/<segment>/<segment>_NNN_dvs240.mp4
#   Output: $NWEAR_DATA/sbj_<k>_clips_dvs128/<segment>/<segment>_NNN_dvs128.mp4
#
#   SBJ_ID=0 bash data_preparation/clips_dvs240_to_dvs128.sh
#
# The 240x180 copies were an intermediate of an earlier DVS240 experiment
# (1920x1080 -> 240x180 -> 128x128, two lossy encodes). New work should use
# clips_to_dvs128.sh on the original clips instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${NWEAR_DATA:?Set NWEAR_DATA (see setup_env.sh)}"
SBJ_ID="${SBJ_ID:?Set SBJ_ID (e.g. 0)}"

# ====================================================================================
INPUT_ROOT="${INPUT_ROOT:-$DATA_ROOT/sbj_${SBJ_ID}_clips_dvs240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/sbj_${SBJ_ID}_clips_dvs128}"
RESIZE_SCRIPT="${RESIZE_SCRIPT:-$SCRIPT_DIR/resize_for_dvs.py}"
DVS_MODEL="${DVS_MODEL:-dvs128}"
# ====================================================================================

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
    local clip_name output_dir tmpdir
    local found_any=0 skipped=0 converted=0

    clip_name="$(basename "$clip_dir")"
    output_dir="$OUTPUT_ROOT/$clip_name"
    mkdir -p "$output_dir"
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' RETURN

    echo "=== $clip_name ==="
    echo "    input:  $clip_dir"
    echo "    output: $output_dir"
    echo "    model:  $DVS_MODEL"

    for video_file in "$clip_dir"/*_dvs240.mp4; do
        [[ -f "$video_file" ]] || continue
        found_any=1

        local filename stem base_stem out_mp4 tmp_link
        filename="$(basename "$video_file")"
        stem="${filename%.mp4}"
        base_stem="${stem%_dvs240}"
        out_mp4="$output_dir/${base_stem}_${DVS_MODEL}.mp4"

        if [[ -f "$out_mp4" ]]; then
            echo "[skip] $clip_name/$filename -> existing ${base_stem}_${DVS_MODEL}.mp4"
            skipped=$((skipped + 1))
            continue
        fi

        echo "--------------------------------------------------------"
        echo "Resizing: $clip_name/$filename"
        echo "  output: $out_mp4"

        tmp_copy="$tmpdir/${base_stem}.mp4"
        cp "$video_file" "$tmp_copy"
        python3 "$RESIZE_SCRIPT" "$tmp_copy" -o "$output_dir" -m "$DVS_MODEL" --overwrite

        wrong_mp4="$output_dir/${stem}_${DVS_MODEL}.mp4"
        if [[ -f "$wrong_mp4" && "$wrong_mp4" != "$out_mp4" ]]; then
            mv "$wrong_mp4" "$out_mp4"
        fi

        echo "Finished: $clip_name/$filename"
        converted=$((converted + 1))
        echo "--------------------------------------------------------"
    done

    if [[ "$found_any" -eq 0 ]]; then
        echo "No *_dvs240.mp4 files in '$clip_dir'."
        return 1
    fi

    echo "--- Done $clip_name: converted=$converted skipped=$skipped ---"
}

if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "Error: input directory not found: $INPUT_ROOT" >&2
    exit 1
fi

if [[ ! -f "$RESIZE_SCRIPT" ]]; then
    echo "Error: resize script not found: $RESIZE_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "--- resize: sbj_${SBJ_ID} clips_dvs240 -> ${DVS_MODEL} (128x128) ---"
echo "Input root:  $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
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
