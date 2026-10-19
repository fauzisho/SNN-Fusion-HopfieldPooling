#!/bin/bash
# Step 8 (after all 54 folds): Table I, mAP, duty cycle and energy.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../setup_env.sh"
cd "${NWEAR_REPO}"
DEVICE="${DEVICE:-cuda}"

# Table I inputs (one row per fold each)
python3 evaluation/eval_sima_single_modality_vote10.py
python3 evaluation/eval_sima_fusion_native_vote10.py
python3 evaluation/eval_sima_fusion_aligned_vote10.py
python3 simb/eval_simb_native_fusion.py --combine
python3 evaluation/eval_simb_aligned_vote10.py
python3 simb/eval_float_fusion_simb_map.py           # SIM B mAP after the 5-window vote
python3 evaluation/build_table1.py --results-dir "${NWEAR_OUTPUT}/results"

# IMU-gated schedule (Table IV)
python3 sima/eval_bf1_duty_cycle_loso54.py
python3 sima/eval_bf2_duty_cycle_loso54.py
python3 sima/eval_hopfield_duty_cycle_loso54.py --pairing bf1 --device "${DEVICE}"
python3 sima/eval_hopfield_duty_cycle_loso54.py --pairing bf2 --device "${DEVICE}"
python3 sima/combine_hopfield_duty_cycle.py
python3 simb/eval_simb_duty_cycle_loso54.py --device "${DEVICE}"

# Energy (Tables II and III)
python3 -m event_snn.aggregate_neurobench_dvs_parts --out-root "${NWEAR_OUTPUT}/event_snn/loso54"
python3 -m event_snn.aggregate_profile_dvs_parts --out-root "${NWEAR_OUTPUT}/event_snn/loso54"
python3 imu_snn/aggregate_neurobench_imu_parts.py --out-root "${NWEAR_OUTPUT}/imu_snn/loso54"
python3 sima/benchmark_sima_baseline_energy.py
python3 simb/benchmark_fusion_neurobench.py
python3 evaluation/build_energy_table.py
