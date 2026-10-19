#!/bin/bash
# Download the N-WEAR DVS128 event clips (18 subjects, ~16 GB) into $NWEAR_DATA.
#
#   source setup_env.sh
#   bash data_preparation/download_events.sh            # all subjects
#   bash data_preparation/download_events.sh 0 3 16     # selected subjects
#
# Unpacks to $NWEAR_DATA/sbj_<k>_events_dvs128/<segment>/<clip>_dvs128.h5, the layout the
# event SNN and the SIM B code expect. Licence: CC BY-NC-SA 4.0 (derivative of WEAR).
set -euo pipefail
BASE_URL="${NWEAR_EVENTS_URL:-https://thesiswearsnn.nbg1.your-objectstorage.com/n-wear-events-dvs128}"
DEST="${NWEAR_DATA:?Set NWEAR_DATA (source setup_env.sh)}"
SUBJECTS=("$@"); [[ ${#SUBJECTS[@]} -eq 0 ]] && SUBJECTS=($(seq 0 17))

mkdir -p "${DEST}"; cd "${DEST}"
curl -fsSL "${BASE_URL}/SHA256SUMS" -o N-WEAR_SHA256SUMS
curl -fsSL "${BASE_URL}/README.txt" -o N-WEAR_README.txt
for k in "${SUBJECTS[@]}"; do
    f="sbj_${k}_events_dvs128.zip"
    if [[ -d "sbj_${k}_events_dvs128" ]]; then echo "[skip] sbj_${k}_events_dvs128 exists"; continue; fi
    echo "[get] ${f}"
    curl -fL --retry 3 -C - "${BASE_URL}/${f}" -o "${f}"
    grep " ${f}\$" N-WEAR_SHA256SUMS | sha256sum -c -
    unzip -q "${f}" && rm "${f}"
done
echo "done: ${DEST}"
