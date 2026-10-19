#!/bin/bash
# Step 3: train the IMU SNN (EGRU 128, 19 outputs incl. NULL) for one fold.
# Paper settings are the script defaults: 30 epochs, Adam 5e-3, batch 64, step decay x0.2.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_fold.sh"
python3 imu_snn/pipeline_inertial_paper.py \
    --feat-dir "${NWEAR_DATA}/wear_dataset/processed/inertial_features/50hz/120_frames_60_stride" \
    --ann-dir "${NWEAR_DATA}/wear_dataset_annotations/60fps" \
    --imu-hz 50 --seed "${SEED}" --split "${SPLIT}" \
    --out-dir "${NWEAR_OUTPUT}/imu_snn/loso54/seed_${SEED}/train_split_${SPLIT}"
# 18-class activity-only scores of this fold (argmax over the 18 activity logits)
python3 imu_snn/eval_imu_activity_only.py --seed "${SEED}" --val-subject "${SBJ}" --device "${DEVICE}"
