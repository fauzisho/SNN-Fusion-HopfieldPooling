#!/bin/bash
# Step 5 (SIM A): late-fusion heads on the frozen frame dumps of one fold, for BF1 and BF2.
#   score_oracle_fusion/   score-pick + oracle at frame level (fusion_preds.npz feeds the duty cycle)
#   hopfield_late_fusion/  single Hopfield pooling (30 epochs, batch 256, lr 1e-3)
#   hopfield_double_fusion/ double Hopfield pooling (W=100 frames, stride 50, 60 epochs, lr 5e-3)
# BF1 smooths the A&D stream with a 501-frame majority vote for the score-pick/duty-cycle
# files only (MV 501); the paper's Table I scoring (evaluation/) uses the raw frames.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_fold.sh"
run_heads() {  # <tree dir> <imu npz name> <mv kernel> <imu label>
    local base="$1" imu="$1/$2"
    python3 sima/eval_ad_score_oracle_fusion.py --ad-npz "${imu}" --i3d-npz "${base}/i3d_frames.npz" \
        --out-dir "${base}/score_oracle_fusion" --mv-kernel "$3"
    python3 sima/train_hopfield_late_fusion_tridet.py --imu-npz "${imu}" --i3d-npz "${base}/i3d_frames.npz" \
        --out-dir "${base}/hopfield_late_fusion" --epochs 30 --batch-size 256 --imu-label "$4" --i3d-label "TriDet I3D 2s"
    python3 sima/train_hopfield_double_fusion_tridet.py --imu-npz "${imu}" --i3d-npz "${base}/i3d_frames.npz" \
        --out-dir "${base}/hopfield_double_fusion" --seq-len 100 --train-stride 50 --epochs 60 \
        --batch-size 128 --lr 5e-3 --hidden 128 --seed "${SEED}" --imu-label "$4" --i3d-label "TriDet I3D 2s"
}
run_heads "${NWEAR_OUTPUT}/sima/bf1_ad_camera/seed_${SEED}/split_${SPLIT}" ad_frames.npz 501 "A&D IMU"
run_heads "${NWEAR_OUTPUT}/sima/bf2_imu_camera/seed_${SEED}/split_${SPLIT}" imu_frames.npz 0 "TriDet IMU"
