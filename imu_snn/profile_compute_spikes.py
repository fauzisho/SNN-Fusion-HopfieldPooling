#!/usr/bin/env python3
"""
Profile compute (MACs), EGRU spike sparsity, and estimated energy from saved checkpoints.

No retraining. Uses the same validation windows as training (see eval_precision_recall_from_checkpoints.py).

Outputs per-run CSV + JSON summary with GPU / Loihi energy estimates (configurable pJ/MAC, pJ/SOP).
See REPORT.md §5 for methodology and references.

Usage:
  python3 imu_snn/profile_compute_spikes.py --out-root "$NWEAR_OUTPUT/imu_snn/loso54" --model-type egru
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch

from eval_precision_recall_from_checkpoints import load_val_tensors
from pipeline_inertial import EGRUClassifier


# Default energy constants (pJ); see REPORT.md §5.2 — cite Horowitz (2014), Davies et al. (2018, 2021).
DEFAULT_E_MAC_PJ = 4.6          # Horowitz 2014, 45 nm FP MAC order-of-magnitude
DEFAULT_E_SOP_PJ_NOM = 10.0     # Loihi-class synaptic op (nominal; range 1–23 pJ in literature)
DEFAULT_E_SOP_PJ_LOW = 1.0
DEFAULT_E_SOP_PJ_HIGH = 23.0


@dataclass
class MacBreakdown:
    egru_macs: int
    head_macs: int
    total_macs: int

    @classmethod
    def from_dims(cls, T: int, F: int, H: int, C: int) -> MacBreakdown:
        """Analytical MAC count per window (batch size 1)."""
        # EGRU: 3 gates; input proj F→3H + recurrent H→3H per timestep
        egru = int(6 * T * H * (F + H))
        # linear head each timestep + mean pool (negligible)
        head = int(T * H * C)
        return cls(egru_macs=egru, head_macs=head, total_macs=egru + head)


def estimate_sops(T: int, F: int, H: int, spike_rate: float) -> int:
    """
    Synaptic operations (SOPs) per window for EGRU recurrence on Loihi-style hardware.

    SOPs ≈ 6 * T * H * (F + H) * spike_rate
    (active hidden units gate input/recurrent MACs each timestep).
    """
    return int(6 * T * H * (F + H) * spike_rate)


def energy_mj_from_ops(ops: int, e_pj: float) -> float:
    return ops * e_pj * 1e-9  # pJ → mJ


def gpu_power_watts() -> float | None:
    """Read current GPU power draw (W) via nvidia-smi, if available."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        line = out.strip().split("\n")[0]
        return float(line)
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None


@torch.no_grad()
def measure_spike_rate_egru(model: torch.nn.Module, x_val: torch.Tensor, device: torch.device) -> float:
    """Fraction of nonzero EGRU outputs (proxy for spike activity)."""
    egru = model.egru if hasattr(model, "egru") else model
    x = x_val.to(device)
    out = egru(x)  # (B, T, H)
    active = (out.detach().abs() > 1e-6).float()
    return float(active.mean().cpu())


@torch.no_grad()
def measure_inference_ms(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    device: torch.device,
    n_warmup: int = 5,
    n_iter: int = 50,
) -> float:
    """Mean forward-pass latency (ms) for one window (batch size 1)."""
    model.eval()
    x = x_val[:1].to(device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    return elapsed_ms


def build_model(in_features: int, num_classes: int, hidden: int) -> torch.nn.Module:
    return EGRUClassifier(
        input_size=in_features,
        hidden_size=hidden,
        num_classes=num_classes,
    )


def iter_run_dirs(out_root: Path, sample_runs: int | None):
    dirs: list[Path] = []
    for seed_dir in sorted(out_root.glob("seed_*")):
        for kind, nmax in (("train", 18), ("test", 6)):
            for i in range(1, nmax + 1):
                d = seed_dir / f"{kind}_split_{i}"
                if (d / "model.pt").is_file():
                    dirs.append(d)
    if sample_runs is not None and sample_runs < len(dirs):
        # Prefer test splits from seed_1, then fill
        preferred = [d for d in dirs if "seed_1/test_" in str(d)]
        rest = [d for d in dirs if d not in preferred]
        dirs = (preferred + rest)[:sample_runs]
    return dirs


def profile_one_run(
    run_dir: Path,
    *,
    model_type: str,
    feat_dir: Path,
    ann_dir: Path,
    device: torch.device,
    measure_gpu: bool,
) -> dict[str, str | float] | None:
    ckpt = run_dir / "model.pt"
    if not ckpt.is_file():
        return None

    m = re.match(r"seed_(\d+)/(train|test)_split_(\d+)$", "/".join(run_dir.parts[-2:]))
    if not m:
        return None
    seed, kind, split = m.group(1), m.group(2), int(m.group(3))

    x_val, _, _, meta = load_val_tensors(
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
    macs = MacBreakdown.from_dims(T, F, hidden, C)

    model = build_model(F, C, hidden)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    spike_rate = measure_spike_rate_egru(model, x_val, device)
    sops = estimate_sops(T, F, hidden, spike_rate)
    latency_ms = measure_inference_ms(model, x_val, device)

    n_val = len(x_val)
    gpu_power_w: float | None = None
    gpu_mj_measured: float | None = None
    if measure_gpu and device.type == "cuda":
        gpu_power_w = gpu_power_watts()
        if gpu_power_w is not None:
            # energy for one full-val forward pass
            gpu_mj_measured = gpu_power_w * (latency_ms / 1000.0)

    row: dict[str, str | float] = {
        "Seed": seed,
        "Run": f"{kind}_split_{split}",
        "T_imu": T,
        "F": F,
        "H": hidden,
        "C": C,
        "SpikeRate": round(spike_rate, 4),
        "MACs_total": macs.total_macs,
        "MACs_egru": macs.egru_macs,
        "MACs_head": macs.head_macs,
        "SOPs_est": sops,
        "Latency_ms": round(latency_ms, 3),
        "Val_windows": n_val,
    }
    if gpu_power_w is not None:
        row["GPU_power_W"] = round(gpu_power_w, 1)
    if gpu_mj_measured is not None:
        row["GPU_mJ_measured_forward"] = round(gpu_mj_measured, 6)
    return row


def add_energy_columns(
    row: dict[str, str | float],
    e_mac_pj: float,
    e_sop_low: float,
    e_sop_nom: float,
    e_sop_high: float,
) -> None:
    macs = int(row["MACs_total"])
    sops = int(row["SOPs_est"])
    row["Est_GPU_mJ_MAC"] = round(energy_mj_from_ops(macs, e_mac_pj), 6)
    row["Est_Loihi_mJ_SOP_low"] = round(energy_mj_from_ops(sops, e_sop_low), 6)
    row["Est_Loihi_mJ_SOP_nom"] = round(energy_mj_from_ops(sops, e_sop_nom), 6)
    row["Est_Loihi_mJ_SOP_high"] = round(energy_mj_from_ops(sops, e_sop_high), 6)


def summarize(rows: list[dict], e_mac_pj: float, e_sop_nom: float) -> dict:
    if not rows:
        return {}
    n = len(rows)
    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n

    macs = int(mean("MACs_total"))
    sops = int(mean("SOPs_est"))
    return {
        "n_runs": n,
        "mean_spike_rate": round(mean("SpikeRate"), 4),
        "mean_MACs_per_window": macs,
        "mean_SOPs_per_window": sops,
        "mean_latency_ms": round(mean("Latency_ms"), 3),
        "est_GPU_mJ_per_window_MAC": round(energy_mj_from_ops(macs, e_mac_pj), 6),
        "est_Loihi_mJ_per_window_SOP_nom": round(energy_mj_from_ops(sops, e_sop_nom), 6),
        "est_Loihi_sparsity_vs_dense_MAC_ratio": round(mean("SpikeRate"), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--model-type", choices=("egru",), default="egru")
    p.add_argument("--feat-dir", type=Path,
                   default=P.IMU_FEAT_DIR)
    p.add_argument("--ann-dir", type=Path,
                   default=P.ANN_DIR)
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--sample-runs", type=int, default=None,
                   help="Profile only N runs (default: all with model.pt)")
    p.add_argument("--measure-gpu-power", action="store_true")
    p.add_argument("--e-mac-pj", type=float, default=DEFAULT_E_MAC_PJ)
    p.add_argument("--e-sop-pj-nom", type=float, default=DEFAULT_E_SOP_PJ_NOM)
    p.add_argument("--e-sop-pj-low", type=float, default=DEFAULT_E_SOP_PJ_LOW)
    p.add_argument("--e-sop-pj-high", type=float, default=DEFAULT_E_SOP_PJ_HIGH)
    p.add_argument("--reference-macs-transformer", type=int, default=0,
                   help="Optional: literature MAC estimate for TriDet/ActionFormer row in summary")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    out_root = args.out_root.expanduser().resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    csv_out = args.csv_out or out_root / f"results_compute_profile_{args.model_type}_2s.csv"
    json_out = args.json_out or out_root / f"results_compute_profile_{args.model_type}_2s_summary.json"

    print(f"Profiling {out_root.name} ({args.model_type}) on {device}")
    run_dirs = iter_run_dirs(out_root, args.sample_runs)
    print(f"  runs: {len(run_dirs)}")

    rows: list[dict] = []
    for run_dir in run_dirs:
        print(f"  {run_dir.relative_to(out_root)}", flush=True)
        row = profile_one_run(
            run_dir,
            model_type=args.model_type,
            feat_dir=args.feat_dir.resolve(),
            ann_dir=args.ann_dir.resolve(),
            device=device,
            measure_gpu=args.measure_gpu_power,
        )
        if row:
            add_energy_columns(
                row,
                args.e_mac_pj,
                args.e_sop_pj_low,
                args.e_sop_pj_nom,
                args.e_sop_pj_high,
            )
            rows.append(row)

    fields = [
        "Seed", "Run", "T_imu", "F", "H", "C",
        "SpikeRate", "MACs_total", "MACs_egru", "MACs_head",
        "SOPs_est", "Latency_ms", "Val_windows",
        "Est_GPU_mJ_MAC", "Est_Loihi_mJ_SOP_low", "Est_Loihi_mJ_SOP_nom", "Est_Loihi_mJ_SOP_high",
        "GPU_power_W", "GPU_mJ_measured_forward",
    ]
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["Run"])):
            w.writerow(r)

    summary = summarize(rows, args.e_mac_pj, args.e_sop_pj_nom)
    summary["model_type"] = args.model_type
    summary["out_root"] = str(out_root)
    summary["energy_constants_pJ"] = {
        "E_MAC": args.e_mac_pj,
        "E_SOP_low": args.e_sop_pj_low,
        "E_SOP_nom": args.e_sop_pj_nom,
        "E_SOP_high": args.e_sop_pj_high,
    }
    if args.reference_macs_transformer > 0:
        summary["reference_transformer_MACs"] = args.reference_macs_transformer
        summary["reference_transformer_est_GPU_mJ"] = round(
            energy_mj_from_ops(args.reference_macs_transformer, args.e_mac_pj), 3
        )

    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(rows)} rows -> {csv_out}")
    print(f"Summary -> {json_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
