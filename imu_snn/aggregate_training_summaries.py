#!/usr/bin/env python3
"""Collect metrics from summary.txt files into results_all_splits_paper_2s.csv (same idea as run_all_splits_paper_2s.ps1)."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import nwear_paths as P


def parse_summary(path: Path) -> dict[str, str] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    row: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("Final val accuracy:") and "(raw)" not in line:
            row["Accuracy"] = line.split(":", 1)[1].strip()
        elif line.startswith("Final val macro-F1:") and "(raw)" not in line:
            row["MacroF1"] = line.split(":", 1)[1].strip()
        elif line.startswith("Final val mAP avg:"):
            row["mAP"] = line.split(":", 1)[1].strip()
        elif line.startswith("Final val mAP by tIoU:"):
            row["mAPByTiou"] = line.split(":", 1)[1].strip()
        elif re.match(r"^Val\s+subjects\s*:", line):
            row["ValSubject"] = line.strip()
    if "Accuracy" in row and "MacroF1" in row:
        return row
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        type=Path,
        default=P.IMU_RUNS,
        help="Directory containing seed_*/train_split_* / test_split_*",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV path (default: <out-root>/results_all_splits_paper_2s.csv)",
    )
    args = p.parse_args()
    out_root: Path = args.out_root
    csv_out = args.csv_out or (out_root / "results_all_splits_paper_2s.csv")

    rows: list[dict[str, str]] = []
    for seed_dir in sorted(out_root.glob("seed_*")):
        m_s = re.match(r"seed_(\d+)$", seed_dir.name)
        if not m_s:
            continue
        seed = m_s.group(1)
        for kind, nmax in (("train", 18), ("test", 6)):
            for i in range(1, nmax + 1):
                summ = seed_dir / f"{kind}_split_{i}" / "summary.txt"
                if not summ.is_file():
                    continue
                parsed = parse_summary(summ)
                if not parsed:
                    continue
                rows.append(
                    {
                        "Seed": seed,
                        "Run": f"{kind}_split_{i}",
                        **parsed,
                    }
                )

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Seed", "Run", "ValSubject", "Accuracy", "MacroF1", "mAP", "mAPByTiou"]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["Run"])):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Wrote {len(rows)} rows -> {csv_out}")


if __name__ == "__main__":
    main()
