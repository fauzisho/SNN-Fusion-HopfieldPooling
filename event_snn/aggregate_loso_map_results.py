#!/usr/bin/env python3
"""Aggregate per-fold map_metrics.json into paper-style table."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
import nwear_paths as P

TIOU_KEYS = ("0.3", "0.4", "0.5", "0.6", "0.7")
DEFAULT_OUT = P.EVENT_RUNS


def mean_std(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mv-kernel", type=int, default=5)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.out_root.glob("seed_*/test_sbj_*/map_metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No map_metrics.json under {args.out_root}")

    p = [r["P"] for r in rows]
    r = [r["R"] for r in rows]
    f1 = [r["F1"] for r in rows]
    acc = [r["Acc"] for r in rows]
    maps = {k: [r[f"mAP_{k}"] for r in rows] for k in TIOU_KEYS}
    avg = [r["mAP_avg"] for r in rows]

    lines = [
        "# experimentv2e — paper-style results (WEAR format)",
        "",
        "Format matches WEAR (`dataset/3699776.pdf`): **Model · CL · P · R · F1 · mAP@0.3–0.7 · Avg**.",
        "",
        f"LOSO mean over **{len(rows)}** folds. Clip **2s**. "
        f"Segment mAP via `eval_loso_map.py` (MV={args.mv_kernel} windows ≈ {args.mv_kernel * 2}s).",
        "",
        "| Model | CL | P | R | F1 | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg |",
        "|-------|----|--:|--:|--:|--------:|--------:|--------:|--------:|--------:|----:|",
        (
            f"| AllConvAdLIFEGRU+aug (LOSO18) | 2s | "
            f"{statistics.mean(p):.2f} | {statistics.mean(r):.2f} | {statistics.mean(f1):.2f} | "
            f"{statistics.mean(maps['0.3']):.2f} | {statistics.mean(maps['0.4']):.2f} | "
            f"{statistics.mean(maps['0.5']):.2f} | {statistics.mean(maps['0.6']):.2f} | "
            f"{statistics.mean(maps['0.7']):.2f} | **{statistics.mean(avg):.2f}** |"
        ),
        "",
        f"Acc **{mean_std(acc)}**. Avg mAP **{mean_std(avg)}**.",
        "",
        "CSV: `results_dvs128_paper_table_map.csv`",
        "",
    ]
    (args.out_root / "RESULTS_DVS128_PAPER_TABLE.md").write_text("\n".join(lines) + "\n")

    fields = ["Seed", "TestSubject", "Acc", "P", "R", "F1"] + [f"mAP_{k}" for k in TIOU_KEYS] + [
        "mAP_avg",
        "n_windows",
        "n_pred_segs",
    ]
    csv_path = args.out_root / "results_dvs128_paper_table_map.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=lambda x: (x["Seed"], x["TestSubject"])):
            w.writerow(row)
    print(f"Aggregated {len(rows)} folds -> {csv_path}")
    print("\n".join(lines[7:12]))


if __name__ == "__main__":
    main()
