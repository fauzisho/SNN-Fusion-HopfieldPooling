#!/bin/bash
# Step 1: WEAR video -> 2 s clips -> 128x128 -> v2e events, for one subject.
#   VIDEO=/path/to/sbj_3.mp4 bash pipeline/01_convert_events.sh 3
# Writes $NWEAR_DATA/sbj_<k>_{clips,clips_dvs128,events_dvs128}/
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../setup_env.sh"
SBJ="${1:?usage: VIDEO=<sbj video> $0 <subject 0-17>}"
VIDEO="${VIDEO:?set VIDEO to the WEAR egocentric video of this subject}"
cd "${NWEAR_REPO}"
python3 data_preparation/cut_wear_clips.py --subject "${SBJ}" --video "${VIDEO}" \
    --ann-json "${NWEAR_DATA}/wear_dataset_annotations/60fps/wear_split_1.json" --out-root "${NWEAR_DATA}"
SBJ_ID="${SBJ}" bash data_preparation/clips_to_dvs128.sh
SBJ_ID="${SBJ}" bash data_preparation/clips_dvs128_to_v2e.sh
