#!/bin/bash
# Step 6 (SIM B): fuse the two frozen spiking encoders for one fold.
#   1. aligned held-out feature cache (read by every SIM B evaluation)
#   2. train single + double Hopfield heads (single: d=64, 30 ep, 1e-3; double: d=128, W=5, 50 ep, 5e-3)
#   3. native-window re-run: event SNN on the 2 s crop centred on every inertial window
# Needs steps 2 and 3 for this fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_fold.sh"
DEV="${DEVICE}"; [[ "${DEV}" == cuda ]] && DEV=cuda:0
python3 simb/build_aligned_cache.py --seed "${SEED}" --val-subject "${SBJ}" --device "${DEV}" --skip-if-done
python3 simb/train_hopfield_dvs_imu_fusion.py --seed "${SEED}" --val-subject "${SBJ}" --device "${DEV}" \
    --dvs-out-root "${NWEAR_OUTPUT}/event_snn/loso54" --imu-out-root "${NWEAR_OUTPUT}/imu_snn/loso54" \
    --dvs-cache-dir "${NWEAR_OUTPUT}/event_snn/tensor_cache" \
    --out-dir "${NWEAR_OUTPUT}/simb/heads/seed_${SEED}/sbj_${SBJ}"
python3 simb/eval_simb_native_fusion.py --seed "${SEED}" --val-subject "${SBJ}" --device "${DEV}" \
    --dvs-ckpt "${NWEAR_OUTPUT}/event_snn/loso54/seed_${SEED}/test_sbj_${SBJ}/model.pt" \
    --imu-ckpt "${NWEAR_OUTPUT}/imu_snn/loso54/seed_${SEED}/train_split_${SPLIT}/model.pt" \
    --batch-size 4 --skip-if-done
