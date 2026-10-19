#!/usr/bin/env python3
"""
Run NeuroBench workload metrics on IMU EGRUClassifier checkpoints.

Metrics: SynapticOperations, ActivationSparsity
Energy: analytical dense MACs (GPU) + NeuroBench ACs/SOPs (Loihi)

EGRU recurrence SOPs are supplemental (EGRU uses internal Parameters, not hooked nn modules).

Usage:
  python3 benchmark_imu_neurobench.py \\
    --out-root output_inertial_all_paper_2s_cluster

  python3 benchmark_imu_neurobench.py \\
    --out-root output_inertial_all_paper_2s_cluster \\
    --run-dir output_inertial_all_paper_2s_cluster/seed_1/test_split_1
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
import nwear_paths as P

import torch
from torch.utils.data import DataLoader, TensorDataset

_IMU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_IMU_DIR))

_NB_ROOT = P.NEUROBENCH_ROOT
if _NB_ROOT.is_dir() and str(_NB_ROOT) not in sys.path:
    sys.path.insert(0, str(_NB_ROOT))

from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import ActivationSparsity, SynapticOperations

from eval_precision_recall_from_checkpoints import load_val_tensors
from neurobench_integration import EGRUIMUNeuroBenchModel
from pipeline_inertial import EGRUClassifier
from profile_compute_spikes import (
    DEFAULT_E_MAC_PJ,
    DEFAULT_E_SOP_PJ_HIGH,
    DEFAULT_E_SOP_PJ_LOW,
    DEFAULT_E_SOP_PJ_NOM,
    MacBreakdown,
    add_energy_columns,
    build_model,
    energy_mj_from_ops,
    estimate_sops,
    iter_run_dirs,
    measure_spike_rate_egru,
)


def _identity_preprocess(data: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return data


def _identity_postprocess(preds: torch.Tensor) -> torch.Tensor:
    return preds


def benchmark_one_run(
    run_dir: Path,
    *,
    feat_dir: Path,
    ann_dir: Path,
    device: torch.device,
    batch_size: int,
    max_windows: int | None,
    quiet: bool,
    e_mac_pj: float,
    e_sop_nom: float,
) -> dict[str, str | float] | None:
    ckpt = run_dir / "model.pt"
    if not ckpt.is_file():
        return None

    m = re.match(r"seed_(\d+)/(train|test)_split_(\d+)$", "/".join(run_dir.parts[-2:]))
    if not m:
        return None
    seed, kind, split = m.group(1), m.group(2), int(m.group(3))

    x_val, y_val, _, meta = load_val_tensors(
        feat_dir, ann_dir, kind, split,
        include_null=True,
        min_overlap=0.6,
        imu_hz=50,
        video_fps=60,
    )
    wp = meta["wp"]
    T = wp["T_imu"]
    F = meta["in_features"]
    C = meta["num_classes"]
    hidden = 128

    dataset = TensorDataset(x_val, torch.as_tensor(y_val, dtype=torch.long))
    if max_windows is not None:
        n = min(len(dataset), max_windows)
        dataset = torch.utils.data.Subset(dataset, range(n))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_model("egru", F, C, hidden, 0, None, 0)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    macs = MacBreakdown.from_dims(T, F, hidden, C)
    spike_rate = measure_spike_rate_egru(model, x_val, device)
    supplemental_egru_sops = estimate_sops(T, F, hidden, spike_rate)

    nb_model = EGRUIMUNeuroBenchModel(model)
    benchmark = Benchmark(
        nb_model,
        loader,
        [_identity_preprocess],
        [_identity_postprocess],
        [[], [SynapticOperations, ActivationSparsity]],
    )
    results = benchmark.run(device=str(device), quiet=quiet)

    syn = results["SynapticOperations"]
    nb_acs = float(syn["Effective_ACs"])
    nb_macs = float(syn["Effective_MACs"])
    nb_dense = float(syn["Dense"])
    nb_sparsity = float(results["ActivationSparsity"])
    total_acs = nb_acs + supplemental_egru_sops

    row: dict[str, str | float] = {
        "Seed": seed,
        "Run": f"{kind}_split_{split}",
        "T_imu": T,
        "F": F,
        "H": hidden,
        "C": C,
        "SpikeRate": round(spike_rate, 4),
        "NB_Effective_MACs": round(nb_macs, 2),
        "NB_Effective_ACs": round(nb_acs, 2),
        "NB_Dense_SynOps": round(nb_dense, 2),
        "NB_ActivationSparsity": round(nb_sparsity, 4),
        "Supplemental_EGRU_SOPs": supplemental_egru_sops,
        "NB_Total_ACs_with_EGRU": round(total_acs, 2),
        "Analytical_MACs_total": macs.total_macs,
        "Analytical_SOPs_est": supplemental_egru_sops,
        "Val_windows": len(dataset),
        "Est_GPU_mJ_analytical_MAC": round(energy_mj_from_ops(macs.total_macs, e_mac_pj), 6),
        "Est_Loihi_mJ_NB_SOP_nom": round(energy_mj_from_ops(int(total_acs), e_sop_nom), 6),
        "Est_Loihi_mJ_NB_total_nom": round(energy_mj_from_ops(int(total_acs), e_sop_nom), 6),
        "neurobench_raw": json.dumps(results),
    }
    row["MACs_total"] = macs.total_macs
    row["SOPs_est"] = int(total_acs)
    add_energy_columns(row, e_mac_pj, DEFAULT_E_SOP_PJ_LOW, e_sop_nom, DEFAULT_E_SOP_PJ_HIGH)
    return row


def summarize(rows: list[dict], e_mac_pj: float, e_sop_nom: float) -> dict:
    if not rows:
        return {}

    n = len(rows)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n

    macs = int(mean("Analytical_MACs_total"))
    total_acs = mean("NB_Total_ACs_with_EGRU")
    return {
        "n_runs": n,
        "framework": "NeuroBench",
        "model_type": "egru",
        "mean_spike_rate": round(mean("SpikeRate"), 4),
        "mean_NB_Total_ACs_with_EGRU_per_window": round(total_acs, 2),
        "mean_NB_ActivationSparsity": round(mean("NB_ActivationSparsity"), 4),
        "mean_Analytical_MACs_per_window": macs,
        "mean_Analytical_SOPs_est": int(mean("Analytical_SOPs_est")),
        "est_GPU_mJ_per_window_analytical_MAC": round(energy_mj_from_ops(macs, e_mac_pj), 6),
        "est_Loihi_mJ_per_window_NB_SOP_nom": round(energy_mj_from_ops(int(total_acs), e_sop_nom), 6),
        "note_egru": (
            "NeuroBench hooks nn.Linear head only. EGRU recurrence SOPs added analytically."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument(
        "--feat-dir",
        type=Path,
        default=P.IMU_FEAT_DIR,
    )
    ap.add_argument(
        "--ann-dir",
        type=Path,
        default=P.ANN_DIR,
    )
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--sample-runs", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--e-mac-pj", type=float, default=DEFAULT_E_MAC_PJ)
    ap.add_argument("--e-sop-pj-nom", type=float, default=DEFAULT_E_SOP_PJ_NOM)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_root = args.out_root.expanduser().resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    csv_out = args.csv_out or out_root / "results_neurobench_egru_2s.csv"
    json_out = args.json_out or out_root / "results_neurobench_egru_2s_summary.json"

    print(f"NeuroBench IMU benchmark: {out_root.name} on {device}")
    run_dirs = iter_run_dirs(out_root, args.sample_runs)
    if args.run_dir is not None:
        run_dirs = [args.run_dir.expanduser().resolve()]
    print(f"  runs: {len(run_dirs)}")

    rows: list[dict] = []
    for run_dir in run_dirs:
        print(f"  {run_dir.relative_to(out_root)}", flush=True)
        row = benchmark_one_run(
            run_dir,
            feat_dir=args.feat_dir.resolve(),
            ann_dir=args.ann_dir.resolve(),
            device=device,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            quiet=args.quiet,
            e_mac_pj=args.e_mac_pj,
            e_sop_nom=args.e_sop_pj_nom,
        )
        if row:
            rows.append(row)

    fields = [
        "Seed", "Run", "T_imu", "F", "H", "C", "SpikeRate",
        "NB_Effective_MACs", "NB_Effective_ACs", "NB_Dense_SynOps",
        "NB_ActivationSparsity", "Supplemental_EGRU_SOPs", "NB_Total_ACs_with_EGRU",
        "Analytical_MACs_total", "Analytical_SOPs_est", "Val_windows",
        "Est_GPU_mJ_analytical_MAC", "Est_Loihi_mJ_NB_SOP_nom", "Est_Loihi_mJ_NB_total_nom",
        "Est_GPU_mJ_MAC", "Est_Loihi_mJ_SOP_low", "Est_Loihi_mJ_SOP_nom", "Est_Loihi_mJ_SOP_high",
        "neurobench_raw",
    ]
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["Run"])):
            w.writerow(r)

    summary = summarize(rows, args.e_mac_pj, args.e_sop_pj_nom)
    summary["out_root"] = str(out_root)
    summary["docs"] = "https://neurobench.readthedocs.io/en/latest/"
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(rows)} rows → {csv_out}")
    print(f"Summary → {json_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
