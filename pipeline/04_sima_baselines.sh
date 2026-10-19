#!/bin/bash
# Step 4 (SIM A): retrain the conventional WEAR baselines for one fold and dump their
# 50 Hz frame predictions. Needs $WEAR_REPO = WEAR checkout with sima/wear_patches applied.
#   BF1 tree: Attend-and-Discriminate (IMU) + camera TriDet (I3D)
#   BF2 tree: inertial TriDet + the same camera TriDet dump (symlinked)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_fold.sh"
CFG="${NWEAR_REPO}/sima/configs"
BF1="${NWEAR_OUTPUT}/sima/bf1_ad_camera/seed_${SEED}/split_${SPLIT}"
BF2="${NWEAR_OUTPUT}/sima/bf2_imu_camera/seed_${SEED}/split_${SPLIT}"
mkdir -p "${BF1}" "${BF2}"

( cd "${WEAR_REPO}"
  python3 cluster/run_ad_split.py --config "${CFG}/attendanddiscriminate_long_120_60.yaml" \
      --split "${SPLIT}" --seed "${SEED}" --out-dir "${BF1}/attendanddiscriminate" --run-id "ad2s_s${SPLIT}_seed${SEED}"
  python3 cluster/run_tridet_split.py --config "${CFG}/tridet_camera_120_60.yaml" \
      --split "${SPLIT}" --seed "${SEED}" --out-dir "${BF1}/tridet_i3d" --run-id "tridet_i3d_2s_s${SPLIT}_seed${SEED}"
  python3 cluster/run_tridet_split.py --config "${CFG}/tridet_inertial_120_60.yaml" \
      --split "${SPLIT}" --seed "${SEED}" --out-dir "${BF2}/tridet_imu" --run-id "tridet_imu_2s_s${SPLIT}_seed${SEED}" )

python3 sima/infer_ad_frames.py --config "${CFG}/attendanddiscriminate_long_120_60.yaml" \
    --checkpoint "${BF1}/attendanddiscriminate/ckpts/final_wear_split_${SPLIT}.pth.tar" \
    --ann-json "${ANN_JSON}" --out-npz "${BF1}/ad_frames.npz"
# TriDet dumps use score threshold 0.1 (default), so TriDet can output NULL.
python3 sima/infer_tridet_frames.py --config "${CFG}/tridet_camera_120_60.yaml" \
    --checkpoint "${BF1}/tridet_i3d/ckpts/final_wear_split_${SPLIT}.pth.tar" \
    --ann-json "${ANN_JSON}" --out-npz "${BF1}/i3d_frames.npz"
python3 sima/infer_tridet_frames.py --config "${CFG}/tridet_inertial_120_60.yaml" \
    --checkpoint "${BF2}/tridet_imu/ckpts/final_wear_split_${SPLIT}.pth.tar" \
    --ann-json "${ANN_JSON}" --out-npz "${BF2}/imu_frames.npz"
ln -sfn "${BF1}/i3d_frames.npz" "${BF2}/i3d_frames.npz"
