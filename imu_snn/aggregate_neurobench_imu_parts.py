#!/usr/bin/env python3
"""Merge per-task IMU NeuroBench CSVs from cluster array."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
import nwear_paths as P


def energy_mj(ops: float, e_pj: float) -> float:
    return ops * e_pj * 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=Path,
        default=P.IMU_RUNS,
    )
    ap.add_argument("--parts-dir", type=Path, default=None)
    args = ap.parse_args()

    out_root = args.out_root.expanduser().resolve()
    parts_dir = args.parts_dir or (out_root / "neurobench_parts")
    csv_paths = sorted(parts_dir.glob("neurobench_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No neurobench_*.csv under {parts_dir}")

    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)

    out_csv = out_root / "results_neurobench_egru_2s.csv"
    out_json = out_root / "results_neurobench_egru_2s_summary.json"

    fields = sorted({k for row in rows for k in row})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["Run"])):
            w.writerow(r)

    n = len(rows)

    def mean(key: str) -> float:
        return st.mean(float(r[key]) for r in rows)

    e_mac, e_sop = 4.6, 10.0
    macs = int(mean("Analytical_MACs_total"))
    total_acs = mean("NB_Total_ACs_with_EGRU")

    summary = {
        "n_runs": n,
        "framework": "NeuroBench",
        "model_type": "egru",
        "mean_spike_rate": round(mean("SpikeRate"), 4),
        "mean_NB_Total_ACs_with_EGRU_per_window": round(total_acs, 2),
        "mean_NB_ActivationSparsity": round(mean("NB_ActivationSparsity"), 4),
        "mean_Analytical_MACs_per_window": macs,
        "est_GPU_mJ_per_window_analytical_MAC": round(energy_mj(macs, e_mac), 6),
        "est_Loihi_mJ_per_window_NB_SOP_nom": round(energy_mj(total_acs, e_sop), 6),
        "merged_from": str(parts_dir),
        "n_part_files": len(csv_paths),
        "docs": "https://neurobench.readthedocs.io/en/latest/",
    }

    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {n} rows → {out_csv}")
    print(f"Summary → {out_json}")


if __name__ == "__main__":
    main()
