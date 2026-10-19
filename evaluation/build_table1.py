#!/usr/bin/env python3
"""Rebuild Table I of the paper (and the SIM B mAP numbers) from the per-fold CSVs.

Every value is the mean +- sample standard deviation over the 54 LOSO runs
(18 held-out subjects x 3 seeds). F1 is 18-class macro F1, both in %.

Inputs (written by the evaluation scripts, shipped in ``results/``):

    results_sima_single_modality_vote10.csv   evaluation/eval_sima_single_modality_vote10.py
    results_sima_fusion_native_vote10.csv     evaluation/eval_sima_fusion_native_vote10.py
    results_sima_fusion_aligned_vote10.csv    evaluation/eval_sima_fusion_aligned_vote10.py
    results_simb_native_vote10.csv            simb/eval_simb_native_fusion.py --combine
    results_simb_fusion_vote10.csv            evaluation/eval_simb_aligned_vote10.py
    results_simb_fusion_pr_map.csv            simb/eval_float_fusion_simb_map.py

    python evaluation/build_table1.py                    # reads results/
    python evaluation/build_table1.py --results-dir "$NWEAR_OUTPUT/results"
    python evaluation/build_table1.py --markdown > table1.md
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETTINGS = (("native", "none"), ("native", "vote"), ("aligned", "none"), ("aligned", "vote"))

# (row label, {grid: (csv file, column prefix for "no vote", column prefix for "10 s")})
SIMA_SINGLE = "results_sima_single_modality_vote10.csv"
SIMA_NAT = "results_sima_fusion_native_vote10.csv"
SIMA_ALI = "results_sima_fusion_aligned_vote10.csv"
SIMB_NAT = "results_simb_native_vote10.csv"
SIMB_ALI = "results_simb_fusion_vote10.csv"


def single(key: str) -> dict:
    return {
        "native": (SIMA_SINGLE, f"native_{key}_none", f"native_{key}_vote"),
        "aligned": (SIMA_SINGLE, f"aligned_{key}_none", f"aligned_{key}_vote"),
    }


def sima_fusion(key: str) -> dict:
    return {
        "native": (SIMA_NAT, f"{key}_none", f"{key}_vote"),
        "aligned": (SIMA_ALI, f"{key}_none", f"{key}_vote"),
    }


def simb(key: str) -> dict:
    return {
        "native": (SIMB_NAT, f"{key}_none", f"{key}_vote"),
        "aligned": (SIMB_ALI, key, f"{key}_vote"),
    }


TABLE = [
    ("SIM A single modality", [
        ("A&D^18 (IMU)", single("ad_act")),
        ("A&D^null (IMU)", single("ad")),
        ("TriDet (IMU)", single("tridet_imu")),
        ("TriDet (camera)", single("camera")),
    ]),
    ("SIM A fusion, BF1 = A&D^null + camera TriDet", [
        ("Score-pick", sima_fusion("bf1_score_pick")),
        ("Single Hopfield", sima_fusion("bf1_single_hp")),
        ("Double Hopfield", sima_fusion("bf1_double_hp")),
        ("Oracle", sima_fusion("bf1_oracle")),
    ]),
    ("SIM A fusion, BF2 = inertial TriDet + camera TriDet", [
        ("Score-pick", sima_fusion("bf2_score_pick")),
        ("Single Hopfield", sima_fusion("bf2_single_hp")),
        ("Double Hopfield", sima_fusion("bf2_double_hp")),
        ("Oracle", sima_fusion("bf2_oracle")),
    ]),
    ("SIM B (spiking) single modality", [
        ("IMU SNN", simb("imu")),
        ("Vision SNN", simb("dvs")),
    ]),
    ("SIM B fusion, IMU SNN + Vision SNN", [
        ("Score-pick", simb("score_pick")),
        ("Single Hopfield", simb("single_hp")),
        ("Double Hopfield", simb("double_hp")),
        ("Oracle", simb("oracle")),
    ]),
]


def load(results_dir: Path, name: str) -> list[dict]:
    with (results_dir / name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 54:
        raise SystemExit(f"{name}: expected 54 folds, found {len(rows)}")
    return rows


def mean_sd(rows: list[dict], column: str) -> tuple[float, float]:
    vals = [float(r[column]) for r in rows]
    return statistics.mean(vals), statistics.stdev(vals)


def table1(results_dir: Path) -> list[tuple[str, str, list[tuple[float, float]]]]:
    cache: dict[str, list[dict]] = {}
    out = []
    for block, rows in TABLE:
        for label, spec in rows:
            cells = []
            for metric in ("F1", "Acc"):
                for grid, vote in SETTINGS:
                    name, none_col, vote_col = spec[grid]
                    data = cache.setdefault(name, load(results_dir, name))
                    col = none_col if vote == "none" else vote_col
                    cells.append(mean_sd(data, f"{col}_{metric}"))
            out.append((block, label, cells))
    return out


def map_rows(results_dir: Path) -> list[tuple[str, float, float]]:
    rows = load(results_dir, "results_simb_fusion_pr_map.csv")
    keys = sorted({c[: -len("_mAP_avg")] for c in rows[0] if c.endswith("_mAP_avg")})
    return [(k, *mean_sd(rows, f"{k}_mAP_avg")) for k in keys]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = ap.parse_args()

    header = ["F1 nat", "F1 nat 10s", "F1 ali", "F1 ali 10s", "Acc nat", "Acc nat 10s", "Acc ali", "Acc ali 10s"]
    rows = table1(args.results_dir)
    if args.markdown:
        print("| Model | " + " | ".join(header) + " |")
        print("| --- |" + " ---: |" * len(header))
    last = None
    for block, label, cells in rows:
        if block != last:
            print(f"| ***{block}*** |" + " |" * len(header) if args.markdown else f"\n{block}")
            last = block
        if args.markdown:
            print(f"| {label} | " + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in cells) + " |")
        else:
            print(f"  {label:18s} " + " ".join(f"{m:6.2f}±{s:5.2f}" for m, s in cells))

    print("\nSIM B aligned, 5-window vote: average mAP over tIoU 0.3-0.7")
    for key, m, s in map_rows(args.results_dir):
        print(f"  {key:12s} {m:6.2f} ± {s:5.2f}")


if __name__ == "__main__":
    main()
