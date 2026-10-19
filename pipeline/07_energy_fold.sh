#!/bin/bash
# Step 7 (per fold): NeuroBench workload of the two spiking encoders.
# NeuroBench 2.3.0 is used unmodified; AdLIF and the EGRU recurrence are added by
# event_snn/neurobench_integration and the analytical EGRU SOP term (see the paper, Sec. III-E).
# Measured on 20 batches of the held-out subject per fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_fold.sh"
EV="${NWEAR_OUTPUT}/event_snn/loso54"
IMU="${NWEAR_OUTPUT}/imu_snn/loso54"
TAG="seed${SEED}_sbj${SBJ}"
python3 -m event_snn.benchmark_dvs_neurobench --out-root "${EV}" --run-dir "${EV}/seed_${SEED}/test_sbj_${SBJ}" \
    --dataset-base "${NWEAR_DATA}" --eval-split val --max-batches 20 --batch-size 4 --quiet --device "${DEVICE}" \
    --csv-out "${EV}/neurobench_parts/neurobench_${TAG}.csv" --json-out "${EV}/neurobench_parts/neurobench_${TAG}.json"
python3 -m event_snn.profile_dvs_compute_spikes --out-root "${EV}" --run-dir "${EV}/seed_${SEED}/test_sbj_${SBJ}" \
    --dataset-base "${NWEAR_DATA}" --eval-split val --max-batches 20 --device "${DEVICE}" \
    --csv-out "${EV}/profile_parts/profile_${TAG}.csv" --json-out "${EV}/profile_parts/profile_${TAG}.json"
python3 imu_snn/benchmark_imu_neurobench.py --out-root "${IMU}" --run-dir "${IMU}/seed_${SEED}/train_split_${SPLIT}" \
    --batch-size 64 --quiet --device "${DEVICE}" \
    --csv-out "${IMU}/neurobench_parts/neurobench_${TAG}.csv" --json-out "${IMU}/neurobench_parts/neurobench_${TAG}.json"
