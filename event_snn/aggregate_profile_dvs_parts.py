#!/usr/bin/env python3
"""Merge per-task profile CSVs from cluster array into aggregate CSV + summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path
import nwear_paths as P


def energy_mj(ops: int, e_pj: float) -> float:
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
    parts_dir = args.parts_dir or (out_root / "profile_parts")
    csv_paths = sorted(parts_dir.glob("profile_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No profile_*.csv under {parts_dir}")

    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)

    if not rows:
        raise SystemExit("No rows in part CSVs")

    tag = out_root.name
    out_csv = out_root / f"results_compute_profile_{tag}.csv"
    out_json = out_root / f"results_compute_profile_{tag}_summary.json"

    fields = sorted({k for row in rows for k in row})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["HoldOut"])):
            w.writerow(r)

    n = len(rows)

    def mean(key: str) -> float:
        return stats.mean(float(r[key]) for r in rows)

    macs = int(mean("MACs_total"))
    sops = int(mean("SOPs_total"))
    nops = int(mean("NeuronOps_AdLIF"))
    e_mac = 4.6
    e_sop = 10.0
    e_neuron = 5.0

    summary = {
        "n_runs": n,
        "model_name": rows[0].get("Model", "unknown"),
        "mean_input_nz_rate": round(mean("InputNZRate"), 4),
        "mean_egru_spike_rate": round(mean("EGRUSpikeRate"), 4),
        "mean_MACs_per_window": macs,
        "mean_SOPs_per_window": sops,
        "mean_neuron_ops_per_window": nops,
        "mean_latency_ms": round(mean("Latency_ms"), 3),
        "est_GPU_mJ_per_window_MAC": round(energy_mj(macs, e_mac), 6),
        "est_Loihi_mJ_per_window_SOP_nom": round(energy_mj(sops, e_sop), 6),
        "est_Loihi_mJ_per_window_neuron_nom": round(energy_mj(nops, e_neuron), 6),
        "est_Loihi_mJ_per_window_total_nom": round(
            energy_mj(sops, e_sop) + energy_mj(nops, e_neuron), 6
        ),
        "sparsity_savings_vs_dense_MAC": round(1.0 - sops / max(macs, 1), 4),
        "out_root": str(out_root),
        "merged_from": str(parts_dir),
        "n_part_files": len(csv_paths),
    }

    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {n} rows from {len(csv_paths)} parts → {out_csv}")
    print(f"Summary → {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
