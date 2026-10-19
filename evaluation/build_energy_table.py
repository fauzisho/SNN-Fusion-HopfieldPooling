#!/usr/bin/env python3
"""Rebuild the main rows of Table II (workload and energy per 2 s window).

Energy proxies (paper, Sec. III-E):

    E_dense  = N_MAC_dense * E_MAC
    E_sparse = N_MAC_eff * E_MAC + (N_AC_eff + N_SOP_EGRU) * E_SOP + N_neuron * E_neuron

with E_MAC = 4.6 pJ, E_SOP = 10 pJ, E_neuron = 5 pJ. Counts are per-window means over
the 54 LOSO runs. The IMU NeuroBench CSV also contains the six extra held-out test
splits of WEAR (subjects 18-23, ``test_split_*``); only the 54 ``train_split_*`` runs
enter the paper, so they are filtered here.

Inputs:
    $NWEAR_OUTPUT/event_snn/loso54/results_neurobench_loso54.csv      (step 7 + 8)
    $NWEAR_OUTPUT/imu_snn/loso54/results_neurobench_egru_2s.csv
    $NWEAR_OUTPUT/sima/energy/summary.json                            (sima/benchmark_sima_baseline_energy.py)

The fusion-head rows of SIM B are analytical (simb/benchmark_fusion_neurobench.py:
85,632 MACs single, 1,797,376 MACs double).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import nwear_paths as P

E_MAC, E_SOP, E_NEURON = 4.6e-12, 10e-12, 5e-12
SIMB_SINGLE_MACS, SIMB_DOUBLE_MACS = 85_632, 1_797_376


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(r[key]) for r in rows)


def mj(joules: float) -> float:
    return joules * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event-csv", type=Path, default=P.EVENT_RUNS / f"results_neurobench_{P.EVENT_RUNS.name}.csv")
    ap.add_argument("--imu-csv", type=Path, default=P.IMU_RUNS / "results_neurobench_egru_2s.csv")
    ap.add_argument("--sima-json", type=Path, default=P.SIMA_ENERGY / "summary.json")
    ap.add_argument("--from-shipped", action="store_true", help="use the copies in results/ (no GPU runs needed)")
    args = ap.parse_args()
    if args.from_shipped:
        res = P.REPO_ROOT / "results"
        args.event_csv = res / "energy_neurobench_event_snn.csv"
        args.imu_csv = res / "energy_neurobench_imu_snn.csv"
        args.sima_json = res / "energy_sima_summary.json"

    rows = []
    sima = json.loads(args.sima_json.read_text()) if args.sima_json.is_file() else None
    if sima:
        for key, label in (("attend_and_discriminate", "A&D (IMU)"), ("tridet_imu", "TriDet (IMU)"), ("tridet_i3d", "TriDet (camera)")):
            d = sima[key]
            macs = d.get("dense_macs_per_window", d.get("dense_macs_per_window_macro_mean"))
            rows.append((label, macs, None, None, None, mj(macs * E_MAC), None))

    ev = list(csv.DictReader(args.event_csv.open()))
    if len(ev) != 54:
        print(f"warning: {args.event_csv.name} has {len(ev)} runs, expected 54")
    dense = mean(ev, "Analytical_MACs_total")
    eff_mac = mean(ev, "NB_Effective_MACs")
    eff_ac = mean(ev, "NB_Total_ACs_with_EGRU")  # NeuroBench ACs + analytical EGRU SOPs
    neuron = mean(ev, "NB_NeuronOps_Effective")
    sparse = eff_mac * E_MAC + eff_ac * E_SOP + neuron * E_NEURON
    rows.append(("Event SNN", dense, eff_mac, eff_ac, 100 * mean(ev, "NB_ActivationSparsity"), mj(dense * E_MAC), mj(sparse)))

    imu = [r for r in csv.DictReader(args.imu_csv.open()) if r["Run"].startswith("train_split")]
    if len(imu) != 54:
        print(f"warning: {args.imu_csv.name} has {len(imu)} train_split runs, expected 54")
    dense = mean(imu, "Analytical_MACs_total")
    eff_mac = mean(imu, "NB_Effective_MACs")
    eff_ac = mean(imu, "NB_Total_ACs_with_EGRU")
    sparse = eff_mac * E_MAC + eff_ac * E_SOP  # EGRU neuron ops are not registered by the harness
    rows.append(("IMU SNN (EGRU)", dense, eff_mac, eff_ac, 100 * mean(imu, "NB_ActivationSparsity"), mj(dense * E_MAC), mj(sparse)))

    if sima:
        h = sima["sima_fusion_heads"]["bf1"]
        s_a, d_a = h["single_hopfield"]["dense_macs_per_fusion_decision"], h["double_hopfield"]["dense_macs_per_fusion_decision"]
        rows.append(("Single HP, SIM A", s_a, None, None, None, mj(s_a * E_MAC), None))
        rows.append(("Double HP, SIM A", d_a, None, None, None, mj(d_a * E_MAC), None))
    rows.append(("Single HP, SIM B", SIMB_SINGLE_MACS, None, None, None, mj(SIMB_SINGLE_MACS * E_MAC), None))
    rows.append(("Double HP, SIM B", SIMB_DOUBLE_MACS, None, None, None, mj(SIMB_DOUBLE_MACS * E_MAC), None))

    def f(v, scale=1e-6, fmt="{:.3f}"):
        return "--" if v is None else fmt.format(v * scale)

    print(f"{'Component':18s} {'Dense (M)':>10s} {'Eff MAC (M)':>11s} {'Eff AC (M)':>10s} {'Sparsity %':>10s} {'Dense mJ':>9s} {'Sparse mJ':>9s}")
    for label, dense, em, ea, sp, ed, es in rows:
        print(f"{label:18s} {f(dense):>10s} {f(em):>11s} {f(ea):>10s} {f(sp, 1, '{:.2f}'):>10s} "
              f"{f(ed, 1, '{:.5f}'):>9s} {f(es, 1, '{:.4f}'):>9s}")


if __name__ == "__main__":
    main()
