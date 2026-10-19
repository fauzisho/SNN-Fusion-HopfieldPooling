#!/bin/bash
# Step 2: train the event SNN (Vision SNN) for one fold, then score it (window P/R/F1, mAP).
# Paper settings: 30 epochs, no early stop, Adam 5e-3, batch 16, 8 channels, dropout 0.05,
# T=100 x 20 ms, augmentation on. ~GPU hours per fold on an RTX 2080 Ti.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_fold.sh"
OUT="${NWEAR_OUTPUT}/event_snn/loso54/seed_${SEED}/test_sbj_${SBJ}"
# shellcheck disable=SC2086
python3 -m event_snn.train_loso \
    --dataset-base "${NWEAR_DATA}" --out-dir "${OUT}" \
    --events-tier dvs128 --model allconv_adlif_egru_dvs128 --feat-neurons 512 \
    --train-subjects ${TRAIN_SUBJECTS} --test-subject "${SBJ}" --val-on-held-out --augment \
    --seed "${SEED}" --epochs 30 --early-stop-patience 0 --early-stop-min-epochs 10 \
    --batch-size 16 --lr 0.005 --channels 8 --dropout 0.05 \
    --sampling-time-ms 20 --sample-length-ms 2000 \
    --num-workers 8 --prefetch-factor 2 \
    --cache-dir "${NWEAR_OUTPUT}/event_snn/tensor_cache" --device "${DEVICE}"
python3 -m event_snn.eval_loso_map --out-root "${NWEAR_OUTPUT}/event_snn/loso54" \
    --seed "${SEED}" --test-subject "${SBJ}" --device "${DEVICE}" --batch-size 16 --mv-kernel 5 --no-aggregate
