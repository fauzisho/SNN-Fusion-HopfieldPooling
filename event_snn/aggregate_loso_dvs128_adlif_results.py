#!/usr/bin/env python3
"""Aggregate LOSO DVS128 AdLIF+aug cluster results into CSV + summary stats."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
import nwear_paths as P


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        type=Path,
        default=P.EVENT_RUNS,
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Default: <out-root>/results_loso18_adlif_3seed.csv",
    )
    args = p.parse_args()
    out_root: Path = args.out_root
    csv_out = args.csv_out or (out_root / "results_loso18_adlif_3seed.csv")

    rows: list[dict] = []
    for seed_dir in sorted(out_root.glob("seed_*")):
        seed = seed_dir.name.replace("seed_", "")
        for run_dir in sorted(seed_dir.glob("test_sbj_*")):
            test_sbj = run_dir.name.replace("test_sbj_", "")
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                continue
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            test = data.get("test", {})
            rows.append(
                {
                    "Seed": seed,
                    "TestSubject": test_sbj,
                    "Accuracy": f"{100.0 * test.get('accuracy', 0.0):.2f}%",
                    "Precision": f"{100.0 * test.get('precision', 0.0):.2f}%",
                    "Recall": f"{100.0 * test.get('recall', 0.0):.2f}%",
                    "F1": f"{100.0 * test.get('f1', 0.0):.2f}%",
                    "AccuracyRaw": test.get("accuracy", 0.0),
                    "F1Raw": test.get("f1", 0.0),
                    "BestEpoch": data.get("best_epoch", ""),
                    "TrainWindows": data.get("split_sizes", {}).get("train", ""),
                    "TestWindows": data.get("split_sizes", {}).get("test", ""),
                    "OutDir": str(run_dir),
                }
            )

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Seed",
        "TestSubject",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AccuracyRaw",
        "F1Raw",
        "BestEpoch",
        "TrainWindows",
        "TestWindows",
        "OutDir",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), int(x["TestSubject"]))):
            w.writerow(r)

    print(f"Wrote {len(rows)} rows -> {csv_out}")

    if rows:
        accs = [r["AccuracyRaw"] for r in rows]
        f1s = [r["F1Raw"] for r in rows]
        print(f"Overall mean accuracy: {100 * statistics.mean(accs):.2f}% ± {100 * statistics.stdev(accs) if len(accs) > 1 else 0:.2f}%")
        print(f"Overall mean F1:       {100 * statistics.mean(f1s):.2f}% ± {100 * statistics.stdev(f1s) if len(f1s) > 1 else 0:.2f}%")

        by_seed: dict[str, list[float]] = {}
        for r in rows:
            by_seed.setdefault(r["Seed"], []).append(r["AccuracyRaw"])
        for seed in sorted(by_seed, key=int):
            vals = by_seed[seed]
            print(f"  seed {seed}: mean acc {100 * statistics.mean(vals):.2f}% (n={len(vals)})")


if __name__ == "__main__":
    main()
