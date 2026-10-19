#!/usr/bin/env python3
"""Aggregate the 54 primary activity-only IMU evaluation folds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "map_avg",
)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=here / "output_inertial_all_paper_2s_cluster" / "activity_only_eval",
    )
    args = parser.parse_args()

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.root.glob("seed_*/sbj_*.json"))
    ]
    if not rows:
        raise SystemExit(f"No fold JSON files under {args.root}")

    csv_path = args.root / "results_imu_activity_only.csv"
    fields = [
        "seed",
        "val_subject",
        "activity_windows",
        "null_windows_dropped",
        *METRICS,
        "map_0.3",
        "map_0.4",
        "map_0.5",
        "map_0.6",
        "map_0.7",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x["seed"], x["val_subject"])):
            flat = {key: row[key] for key in fields if key in row}
            for threshold in ("0.3", "0.4", "0.5", "0.6", "0.7"):
                flat[f"map_{threshold}"] = row["map_by_tiou"][threshold]
            writer.writerow(flat)

    lines = [
        "# IMU activity-only evaluation",
        "",
        f"Folds: {len(rows)}/54",
        "",
        "Saved 19-class checkpoint(s); no retraining. Ground-truth NULL windows",
        "are retained as timeline separators but excluded from record metrics and",
        "segment AP. Predictions use argmax over the 18 activity logits.",
        f"Model type(s): {', '.join(sorted({row.get('model_type', 'egru') for row in rows}))}.",
        "",
        "| Metric | mean ± sample SD (%) |",
        "|---|---:|",
    ]
    for key, label in (
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("macro_f1", "Macro F1"),
    ):
        values = [100 * float(row[key]) for row in rows]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {label} | {statistics.mean(values):.2f} ± {sd:.2f} |")
    for threshold in ("0.3", "0.4", "0.5", "0.6", "0.7"):
        values = [100 * float(row["map_by_tiou"][threshold]) for row in rows]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(
            f"| mAP@{threshold} | {statistics.mean(values):.2f} ± {sd:.2f} |"
        )
    values = [100 * float(row["map_avg"]) for row in rows]
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    lines.append(f"| Average mAP | {statistics.mean(values):.2f} ± {sd:.2f} |")
    lines.extend(
        [
            "",
            f"Activity windows: {sum(row['activity_windows'] for row in rows)}",
            f"NULL windows dropped: {sum(row['null_windows_dropped'] for row in rows)}",
            "",
            "Limitation: this result is conditional on the ground-truth activity mask.",
            "It does not measure online rejection of NULL, and the model was trained",
            "with a nineteenth NULL output.",
        ]
    )
    summary_path = args.root / "RESULTS_IMU_ACTIVITY_ONLY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
