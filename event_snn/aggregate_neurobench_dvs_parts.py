#!/usr/bin/env python3
"""Merge per-task NeuroBench CSVs from cluster array into aggregate CSV + summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path
import nwear_paths as P


def energy_mj(ops: float, e_pj: float) -> float:
    return ops * e_pj * 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=Path,
        default=P.EVENT_RUNS,
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

    if not rows:
        raise SystemExit("No rows in part CSVs")

    tag = out_root.name
    out_csv = out_root / f"results_neurobench_{tag}.csv"
    out_json = out_root / f"results_neurobench_{tag}_summary.json"

    fields = sorted({k for row in rows for k in row})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["HoldOut"])):
            w.writerow(r)

    n = len(rows)

    def mean(key: str) -> float:
        return stats.mean(float(r[key]) for r in rows)

    e_mac = 4.6
    e_sop = 10.0
    e_neuron = 5.0
    nb_macs = mean("NB_Effective_MACs")
    nb_acs_egru = mean("NB_Total_ACs_with_EGRU")
    nb_neuron = mean("NB_NeuronOps_Effective")

    summary = {
        "n_runs": n,
        "model_name": rows[0].get("Model", "unknown"),
        "framework": "NeuroBench",
        "mean_NB_Effective_MACs_per_window": round(nb_macs, 2),
        "mean_NB_Total_ACs_with_EGRU_per_window": round(nb_acs_egru, 2),
        "mean_NB_ActivationSparsity": round(mean("NB_ActivationSparsity"), 4),
        "mean_NB_NeuronOps_Effective_per_window": round(nb_neuron, 2),
        "mean_Analytical_MACs_per_window": int(mean("Analytical_MACs_total")),
        "mean_Analytical_SOPs_total_est": int(mean("Analytical_SOPs_total_est")),
        "est_GPU_mJ_per_window_NB_MAC": round(energy_mj(nb_macs, e_mac), 6),
        "est_Loihi_mJ_per_window_NB_total_nom": round(
            energy_mj(nb_acs_egru, e_sop) + energy_mj(nb_neuron, e_neuron), 6
        ),
        "merged_from": str(parts_dir),
        "n_part_files": len(csv_paths),
        "docs": "https://neurobench.readthedocs.io/en/latest/",
    }

    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {n} rows → {out_csv}")
    print(f"Summary → {out_json}")


if __name__ == "__main__":
    main()
