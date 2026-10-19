#!/usr/bin/env python3
"""
Fusion energy from NeuroBench encoder metrics + analytical Hopfield heads.

Per completed fusion fold ($NWEAR_OUTPUT/simb/heads):
  - DVS encoder: NeuroBench results (job 14576)
  - IMU encoder: NeuroBench results (IMU array job)
  - Hopfield heads: analytical MACs (off-chip, dense)

Deployable pipelines:
  - score_pick: DVS + IMU encoders only
  - single_hp / double_hp: encoders + Hopfield head (GPU MAC proxy)

Usage:
  python3 benchmark_fusion_neurobench.py
  python3 simb/benchmark_fusion_neurobench.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import nwear_paths as P

# Hopfield MACs per 2 s window (analytical, train_hopfield_dvs_imu_fusion.py)
HP_SINGLE_MACS = 85_632
HP_DOUBLE_MACS = 1_797_376
E_MAC_PJ = 4.6
E_SOP_PJ = 10.0


def energy_mj(ops: int | float, e_pj: float) -> float:
    return float(ops) * e_pj * 1e-9


def load_dvs_nb(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["Seed"], row["HoldOut"])] = row
    return rows


def load_imu_nb(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["Seed"], row["Run"])] = row
    return rows


def fusion_folds(fusion_root: Path) -> list[Path]:
    return sorted(
        p.parent for p in fusion_root.glob("seed_*/sbj_*/metrics.json")
    )


def benchmark_fusion(
    fold_dir: Path,
    *,
    dvs_nb: dict[tuple[str, str], dict],
    imu_nb: dict[tuple[str, str], dict],
) -> dict[str, str | float] | None:
    metrics_path = fold_dir / "metrics.json"
    if not metrics_path.is_file():
        return None

    seed = fold_dir.parent.name.replace("seed_", "")
    sbj = int(fold_dir.name.replace("sbj_", ""))
    hold_out = f"sbj_{sbj}"
    imu_run = f"train_split_{sbj + 1}"

    dvs_row = dvs_nb.get((seed, hold_out))
    imu_row = imu_nb.get((seed, imu_run))
    if dvs_row is None or imu_row is None:
        missing = []
        if dvs_row is None:
            missing.append(f"DVS seed={seed} {hold_out}")
        if imu_row is None:
            missing.append(f"IMU seed={seed} {imu_run}")
        print(f"  SKIP {fold_dir.name}: missing {', '.join(missing)}")
        return None

    meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    res = meta.get("results", {})

    dvs_gpu = float(dvs_row["Analytical_MACs_total"])  # dense MAC bound
    dvs_loihi = float(dvs_row["Est_Loihi_mJ_NB_total_nom"])
    imu_gpu = float(imu_row["Analytical_MACs_total"])
    imu_loihi = float(imu_row["Est_Loihi_mJ_NB_total_nom"])

    hp_single_gpu = energy_mj(HP_SINGLE_MACS, E_MAC_PJ)
    hp_double_gpu = energy_mj(HP_DOUBLE_MACS, E_MAC_PJ)

    row: dict[str, str | float] = {
        "Seed": seed,
        "HoldOut": hold_out,
        "IMU_run": imu_run,
        "DVS_NB_Loihi_mJ": round(dvs_loihi, 6),
        "IMU_NB_Loihi_mJ": round(imu_loihi, 6),
        "DVS_analytical_GPU_mJ": round(energy_mj(dvs_gpu, E_MAC_PJ), 6),
        "IMU_analytical_GPU_mJ": round(energy_mj(imu_gpu, E_MAC_PJ), 6),
        "HP_single_GPU_mJ": round(hp_single_gpu, 6),
        "HP_double_GPU_mJ": round(hp_double_gpu, 6),
        "ScorePick_GPU_mJ": round(energy_mj(dvs_gpu + imu_gpu, E_MAC_PJ), 6),
        "ScorePick_Loihi_mJ": round(dvs_loihi + imu_loihi, 6),
        "SingleHP_GPU_mJ": round(energy_mj(dvs_gpu + imu_gpu, E_MAC_PJ) + hp_single_gpu, 6),
        "SingleHP_Loihi_mJ": round(dvs_loihi + imu_loihi, 6),
        "DoubleHP_GPU_mJ": round(energy_mj(dvs_gpu + imu_gpu, E_MAC_PJ) + hp_double_gpu, 6),
        "DoubleHP_Loihi_mJ": round(dvs_loihi + imu_loihi, 6),
        "Acc_score_pick": round(float(res.get("score_pick", {}).get("acc", 0)) * 100, 2),
        "Acc_single_hp": round(float(res.get("single_hp", {}).get("acc", 0)) * 100, 2),
        "Acc_double_hp": round(float(res.get("double_hp", {}).get("acc", 0)) * 100, 2),
        "Acc_dvs_only": round(float(res.get("dvs_only", {}).get("acc", 0)) * 100, 2),
        "Acc_imu_only": round(float(res.get("imu_only", {}).get("acc", 0)) * 100, 2),
    }
    return row


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_folds": 0}

    n = len(rows)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n

    return {
        "n_folds": n,
        "framework": "NeuroBench encoders + analytical Hopfield",
        "mean_ScorePick_GPU_mJ": round(mean("ScorePick_GPU_mJ"), 6),
        "mean_ScorePick_Loihi_mJ": round(mean("ScorePick_Loihi_mJ"), 6),
        "mean_SingleHP_GPU_mJ": round(mean("SingleHP_GPU_mJ"), 6),
        "mean_SingleHP_Loihi_mJ": round(mean("SingleHP_Loihi_mJ"), 6),
        "mean_DoubleHP_GPU_mJ": round(mean("DoubleHP_GPU_mJ"), 6),
        "mean_DoubleHP_Loihi_mJ": round(mean("DoubleHP_Loihi_mJ"), 6),
        "mean_Acc_score_pick": round(mean("Acc_score_pick"), 2),
        "mean_Acc_single_hp": round(mean("Acc_single_hp"), 2),
        "mean_Acc_double_hp": round(mean("Acc_double_hp"), 2),
        "note": "Loihi = DVS NB + IMU NB encoders; Hopfield heads GPU-only (off-chip).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fusion-root",
        type=Path,
        default=P.SIMB_HEADS,
    )
    ap.add_argument(
        "--dvs-nb-csv",
        type=Path,
        default=P.EVENT_RUNS / f"results_neurobench_{P.EVENT_RUNS.name}.csv",
    )
    ap.add_argument(
        "--imu-nb-csv",
        type=Path,
        default=P.IMU_RUNS / "results_neurobench_egru_2s.csv",
    )
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    fusion_root = args.fusion_root.expanduser().resolve()
    dvs_nb = load_dvs_nb(args.dvs_nb_csv.expanduser().resolve())
    imu_nb = load_imu_nb(args.imu_nb_csv.expanduser().resolve())

    print(f"Fusion NeuroBench: {fusion_root}")
    print(f"  DVS NB rows: {len(dvs_nb)}, IMU NB rows: {len(imu_nb)}")

    rows: list[dict] = []
    for fold_dir in fusion_folds(fusion_root):
        print(f"  {fold_dir.relative_to(fusion_root)}", flush=True)
        row = benchmark_fusion(fold_dir, dvs_nb=dvs_nb, imu_nb=imu_nb)
        if row:
            rows.append(row)

    csv_out = args.csv_out or fusion_root / "results_neurobench_fusion.csv"
    json_out = args.json_out or fusion_root / "results_neurobench_fusion_summary.json"

    fields = sorted({k for r in rows for k in r})
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["HoldOut"])):
            w.writerow(r)

    summary = summarize(rows)
    summary["fusion_root"] = str(fusion_root)
    summary["dvs_nb_csv"] = str(args.dvs_nb_csv)
    summary["imu_nb_csv"] = str(args.imu_nb_csv)
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {len(rows)} folds → {csv_out}")
    print(f"Summary → {json_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
