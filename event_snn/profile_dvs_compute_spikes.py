#!/usr/bin/env python3
"""
Profile compute (MACs), spike sparsity, and estimated energy for DVS128 SNN checkpoints.

Mirrors wearsnn/2.sensor_fusion/1.imu/profile_compute_spikes.py for experimentv2e
AllConvAdLIFEGRU_DVS128 (and compatible allconv_*_dvs128 models).

No retraining. Loads saved model.pt, runs val/test windows, reports:
  - Analytical MACs (conv + EGRU + head)
  - Measured spike rates (input, AdLIF blocks, EGRU)
  - Estimated SOPs (Loihi-class sparse synaptic ops)
  - Proxy energy: GPU (E_MAC) and neuromorphic (E_SOP low/nom/high)

Usage:
  python3 profile_dvs_compute_spikes.py \\
    --out-root output/loso_dvs128_adlif_egru_aug_valfix_loso18_cluster

  python3 profile_dvs_compute_spikes.py \\
    --out-root output/loso_dvs128_adlif_egru_aug_valfix_loso18_cluster \\
    --sample-runs 6 --eval-split val
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass, fields
from pathlib import Path
import nwear_paths as P

import torch
from torch.utils.data import DataLoader

from event_snn.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from event_snn.models.adlif import AdLIF
from event_snn.training.trainer import V2EAicoTrainer

# Literature energy constants (pJ) — see wearsnn/2.sensor_fusion/1.imu/REPORT.md §5.2
DEFAULT_E_MAC_PJ = 4.6
DEFAULT_E_SOP_PJ_NOM = 10.0
DEFAULT_E_SOP_PJ_LOW = 1.0
DEFAULT_E_SOP_PJ_HIGH = 23.0
# AdLIF has adaptive dynamics (u, w states); ~8 abstract ops/update (between Leaky and Synaptic)
DEFAULT_E_NEURON_OP_PJ = 5.0
DEFAULT_NEURON_OPS_PER_ADLIF_UPDATE = 8


@dataclass
class MacBreakdown:
    conv_macs: int
    conv_macs_per_layer: list[int]
    egru_macs: int
    head_macs: int
    total_macs: int
    flat_dim: int
    T: int
    H: int
    C: int


def _config_from_checkpoint(raw: dict, out_dir: Path) -> ExperimentConfig:
    data_raw = dict(raw["data"])
    data_raw["dataset_root"] = Path(data_raw["dataset_root"])
    if "events_tier" not in data_raw:
        data_raw["events_tier"] = "dvs128" if "dvs128" in str(out_dir) else "dvs240"
    data = DataConfig(**data_raw)
    model = ModelConfig(**raw["model"])
    train_defaults = {k.name: getattr(TrainConfig(), k.name) for k in fields(TrainConfig)}
    train_defaults.update(raw["train"])
    train = TrainConfig(**train_defaults)
    if train.cache_dir is not None:
        train.cache_dir = Path(train.cache_dir)
    return ExperimentConfig(data=data, model=model, train=train, out_dir=out_dir)


def _conv_macs_per_step(cin: int, cout: int, h: int, w: int, k: int, s: int, p: int) -> tuple[int, int, int]:
    hout = (h + 2 * p - k) // s + 1
    wout = (w + 2 * p - k) // s + 1
    macs = cout * hout * wout * k * k * cin
    return macs, hout, wout


def analytical_macs(
    *,
    channels: int = 8,
    feat_neurons: int = 512,
    num_classes: int = 18,
    input_height: int = 128,
    input_width: int = 128,
    time_bins: int = 100,
) -> MacBreakdown:
    """Dense MAC count per window (batch=1) for AllConvAdLIFEGRU_DVS128."""
    layer_specs = [
        (2, channels, 3, 2, 1),
        (channels, 2 * channels, 3, 2, 1),
        (2 * channels, 4 * channels, 3, 2, 1),
        (4 * channels, 8 * channels, 3, 2, 0),
        (8 * channels, 8 * channels, 3, 1, 1),
    ]
    h, w = input_height, input_width
    per_layer: list[int] = []
    for cin, cout, k, s, p in layer_specs:
        macs, h, w = _conv_macs_per_step(cin, cout, h, w, k, s, p)
        per_layer.append(macs)

    flat_dim = layer_specs[-1][1] * h * w
    T, H, C = time_bins, feat_neurons, num_classes
    conv_per_step = sum(per_layer)
    conv_macs = conv_per_step * T
    egru_macs = int(6 * T * H * (flat_dim + H))
    head_macs = int(T * H * C)
    total = conv_macs + egru_macs + head_macs
    return MacBreakdown(conv_macs, per_layer, egru_macs, head_macs, total, flat_dim, T, H, C)


def adlif_neuron_counts(channels: int = 8, h: int = 128, w: int = 128) -> list[int]:
    """Neurons per AdLIF block after each conv (spatial sizes match AllConv stack)."""
    sizes = [
        (channels, (h + 2 - 3) // 2 + 1, (w + 2 - 3) // 2 + 1),
        (2 * channels, 0, 0),
        (4 * channels, 0, 0),
        (8 * channels, 0, 0),
        (8 * channels, 0, 0),
    ]
    cur_h, cur_w = h, w
    counts: list[int] = []
    for i, (ch, _, _) in enumerate(sizes):
        if i == 0:
            cur_h = (h + 2 - 3) // 2 + 1
            cur_w = (w + 2 - 3) // 2 + 1
        elif i == 1:
            cur_h = (cur_h + 2 - 3) // 2 + 1
            cur_w = (cur_w + 2 - 3) // 2 + 1
        elif i == 2:
            cur_h = (cur_h + 2 - 3) // 2 + 1
            cur_w = (cur_w + 2 - 3) // 2 + 1
        elif i == 3:
            cur_h = (cur_h - 3) // 2 + 1
            cur_w = (cur_w - 3) // 2 + 1
        counts.append(ch * cur_h * cur_w)
    return counts


def estimate_conv_sops(conv_macs_per_layer: list[int], T: int, activity_rates: list[float]) -> int:
    """
    Sparse conv SOPs: each layer scaled by input activity rate.

    activity_rates[0] = input non-zero rate; activity_rates[1..5] = AdLIF spike rates.
    """
    rate = activity_rates[0]
    sops = 0
    for i, layer_macs in enumerate(conv_macs_per_layer):
        sops += int(layer_macs * T * rate)
        if i + 1 < len(activity_rates):
            rate = activity_rates[i + 1]
    return sops


def estimate_sops_egru(T: int, F: int, H: int, spike_rate: float) -> int:
    return int(6 * T * H * (F + H) * spike_rate)


def estimate_neuron_ops(T: int, neuron_counts: list[int], ops_per_update: int) -> int:
    return int(sum(n * T * ops_per_update for n in neuron_counts))


def energy_mj_from_ops(ops: int, e_pj: float) -> float:
    return ops * e_pj * 1e-9


def gpu_power_watts() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(out.strip().split("\n")[0])
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None


@torch.no_grad()
def measure_spike_rates(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 20,
) -> dict[str, float]:
    """Measure input/AdLIF/EGRU activity rates on real windows."""
    model.eval()
    input_rates: list[float] = []
    adlif_rates: list[list[float]] = []
    egru_rates: list[float] = []

    for batch_idx, (batch_x, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch_x.to(device, non_blocking=True)
        input_rates.append(float((x != 0).float().mean().cpu()))

        x_t = x.permute(4, 0, 1, 2, 3).contiguous()
        block_adlif: list[float] = []
        for block in model.feature_extractor:
            x_t = block(x_t)
            if isinstance(block, AdLIF):
                block_adlif.append(float((x_t != 0).float().mean().cpu()))
        adlif_rates.append(block_adlif)

        x_flat = x_t.flatten(2)
        egru_out = model.egru(x_flat)
        egru_rates.append(float((egru_out.abs() > 1e-6).float().mean().cpu()))

    n_adlif = max((len(r) for r in adlif_rates), default=0)
    mean_adlif = [
        sum(r[i] for r in adlif_rates if i < len(r)) / max(1, sum(1 for r in adlif_rates if i < len(r)))
        for i in range(n_adlif)
    ]
    return {
        "input_nz_rate": sum(input_rates) / max(1, len(input_rates)),
        "adlif_spike_rates": mean_adlif,
        "egru_spike_rate": sum(egru_rates) / max(1, len(egru_rates)),
        "n_batches": len(input_rates),
    }


@torch.no_grad()
def measure_inference_ms(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_warmup: int = 3,
    n_iter: int = 30,
) -> float:
    model.eval()
    batch_x, _ = next(iter(loader))
    x = batch_x[:1].to(device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter


def iter_run_dirs(out_root: Path, sample_runs: int | None, run_dir: Path | None = None) -> list[Path]:
    if run_dir is not None:
        return [run_dir.expanduser().resolve()]
    dirs = sorted(p.parent for p in out_root.glob("seed_*/test_sbj_*/model.pt"))
    if sample_runs is not None and sample_runs < len(dirs):
        preferred = [d for d in dirs if "/seed_1/" in str(d)]
        rest = [d for d in dirs if d not in preferred]
        dirs = (preferred + rest)[:sample_runs]
    return dirs


def _resolve_cache(cfg: ExperimentConfig, out_dir: Path) -> None:
    if cfg.train.cache_dir is not None:
        return
    candidates = [
        P.EVENT_TENSOR_CACHE,
        out_dir / "tensor_cache",
        out_dir.parent / "loso_dvs128_tensor_cache",
    ]
    for c in candidates:
        if c.is_dir():
            cfg.train.cache_dir = c
            return


def profile_one_run(
    run_dir: Path,
    *,
    dataset_base: Path,
    device: torch.device,
    eval_split: str,
    max_batches: int,
    measure_gpu: bool,
) -> dict[str, str | float] | None:
    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.is_file():
        return None

    m = re.match(r"seed_(\d+)/test_sbj_(\d+)$", "/".join(run_dir.parts[-2:]))
    if not m:
        return None
    seed, test_sbj = m.group(1), int(m.group(2))

    device_obj = device
    checkpoint = torch.load(ckpt_path, map_location=device_obj)

    cfg = _config_from_checkpoint(checkpoint["config"], run_dir)
    cfg.out_dir = run_dir
    cfg.train.device = str(device_obj)
    _resolve_cache(cfg, run_dir)

    metrics_path = run_dir / "metrics.json"
    meta: dict = {}
    if metrics_path.is_file():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    train_subjects = meta.get("train_subjects", [i for i in range(18) if i != test_sbj])
    val_on_held_out = meta.get("val", {}).get("accuracy") == meta.get("test", {}).get("accuracy")

    trainer = V2EAicoTrainer(cfg)
    split, _, _ = trainer.build_loso_splits(
        dataset_base,
        train_subjects=train_subjects,
        test_subject=test_sbj,
        val_on_held_out=val_on_held_out,
    )
    _, val_loader, test_loader = trainer.build_loaders(split)
    eval_loader = val_loader if eval_split == "val" else test_loader

    model = trainer.build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device_obj)

    macs = analytical_macs(
        channels=cfg.model.channels,
        feat_neurons=cfg.model.feat_neurons,
        num_classes=cfg.model.num_classes,
        input_height=cfg.data.input_height,
        input_width=cfg.data.input_width,
        time_bins=cfg.data.time_bins,
    )
    rates = measure_spike_rates(model, eval_loader, device_obj, max_batches=max_batches)
    activity = [rates["input_nz_rate"], *rates["adlif_spike_rates"]]
    sops_conv = estimate_conv_sops(macs.conv_macs_per_layer, macs.T, activity)
    sops_egru = estimate_sops_egru(macs.T, macs.flat_dim, macs.H, rates["egru_spike_rate"])
    sops_total = sops_conv + sops_egru

    neuron_counts = adlif_neuron_counts(cfg.model.channels, cfg.data.input_height, cfg.data.input_width)
    neuron_ops = estimate_neuron_ops(macs.T, neuron_counts, DEFAULT_NEURON_OPS_PER_ADLIF_UPDATE)

    latency_ms = measure_inference_ms(model, eval_loader, device_obj)

    row: dict[str, str | float] = {
        "Seed": seed,
        "HoldOut": f"sbj_{test_sbj}",
        "Model": cfg.model.model_name,
        "T": macs.T,
        "FlatDim": macs.flat_dim,
        "H": macs.H,
        "C": macs.C,
        "InputNZRate": round(rates["input_nz_rate"], 4),
        "EGRUSpikeRate": round(rates["egru_spike_rate"], 4),
        "MACs_total": macs.total_macs,
        "MACs_conv": macs.conv_macs,
        "MACs_egru": macs.egru_macs,
        "MACs_head": macs.head_macs,
        "SOPs_conv": sops_conv,
        "SOPs_egru": sops_egru,
        "SOPs_total": sops_total,
        "NeuronOps_AdLIF": neuron_ops,
        "Latency_ms": round(latency_ms, 3),
        "Eval_windows": len(eval_loader.dataset),
        "Profile_batches": rates["n_batches"],
    }
    for i, r in enumerate(rates["adlif_spike_rates"]):
        row[f"AdLIF{i+1}_rate"] = round(r, 4)

    if measure_gpu and device_obj.type == "cuda":
        pw = gpu_power_watts()
        if pw is not None:
            row["GPU_power_W"] = round(pw, 1)
            row["GPU_mJ_measured_forward"] = round(pw * (latency_ms / 1000.0), 6)

    return row


def add_energy_columns(
    row: dict[str, str | float],
    *,
    e_mac_pj: float,
    e_sop_low: float,
    e_sop_nom: float,
    e_sop_high: float,
    e_neuron_op_pj: float,
) -> None:
    macs = int(row["MACs_total"])
    sops = int(row["SOPs_total"])
    nops = int(row["NeuronOps_AdLIF"])
    row["Est_GPU_mJ_MAC"] = round(energy_mj_from_ops(macs, e_mac_pj), 6)
    row["Est_Loihi_mJ_SOP_low"] = round(energy_mj_from_ops(sops, e_sop_low), 6)
    row["Est_Loihi_mJ_SOP_nom"] = round(energy_mj_from_ops(sops, e_sop_nom), 6)
    row["Est_Loihi_mJ_SOP_high"] = round(energy_mj_from_ops(sops, e_sop_high), 6)
    row["Est_Loihi_mJ_neuron_nom"] = round(energy_mj_from_ops(nops, e_neuron_op_pj), 6)
    row["Est_Loihi_mJ_total_nom"] = round(
        energy_mj_from_ops(sops, e_sop_nom) + energy_mj_from_ops(nops, e_neuron_op_pj), 6
    )


def summarize(rows: list[dict], e_mac_pj: float, e_sop_nom: float, e_neuron_op_pj: float) -> dict:
    if not rows:
        return {}
    n = len(rows)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n

    macs = int(mean("MACs_total"))
    sops = int(mean("SOPs_total"))
    nops = int(mean("NeuronOps_AdLIF"))
    return {
        "n_runs": n,
        "model_name": rows[0].get("Model", "unknown"),
        "mean_input_nz_rate": round(mean("InputNZRate"), 4),
        "mean_egru_spike_rate": round(mean("EGRUSpikeRate"), 4),
        "mean_MACs_per_window": macs,
        "mean_SOPs_per_window": sops,
        "mean_neuron_ops_per_window": nops,
        "mean_latency_ms": round(mean("Latency_ms"), 3),
        "est_GPU_mJ_per_window_MAC": round(energy_mj_from_ops(macs, e_mac_pj), 6),
        "est_Loihi_mJ_per_window_SOP_nom": round(energy_mj_from_ops(sops, e_sop_nom), 6),
        "est_Loihi_mJ_per_window_neuron_nom": round(energy_mj_from_ops(nops, e_neuron_op_pj), 6),
        "est_Loihi_mJ_per_window_total_nom": round(
            energy_mj_from_ops(sops, e_sop_nom) + energy_mj_from_ops(nops, e_neuron_op_pj), 6
        ),
        "sparsity_savings_vs_dense_MAC": round(1.0 - sops / max(macs, 1), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, default=None, help="Profile a single LOSO run directory")
    ap.add_argument("--dataset-base", type=Path, default=P.DATA_ROOT)
    ap.add_argument("--eval-split", choices=("val", "test"), default="val")
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--sample-runs", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--measure-gpu-power", action="store_true")
    ap.add_argument("--e-mac-pj", type=float, default=DEFAULT_E_MAC_PJ)
    ap.add_argument("--e-sop-pj-nom", type=float, default=DEFAULT_E_SOP_PJ_NOM)
    ap.add_argument("--e-sop-pj-low", type=float, default=DEFAULT_E_SOP_PJ_LOW)
    ap.add_argument("--e-sop-pj-high", type=float, default=DEFAULT_E_SOP_PJ_HIGH)
    ap.add_argument("--e-neuron-op-pj", type=float, default=DEFAULT_E_NEURON_OP_PJ)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_root = args.out_root.expanduser().resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = out_root.name.replace("/", "_")
    csv_out = args.csv_out or out_root / f"results_compute_profile_{tag}.csv"
    json_out = args.json_out or out_root / f"results_compute_profile_{tag}_summary.json"

    print(f"Profiling {out_root.name} on {device} (eval={args.eval_split})")
    run_dirs = iter_run_dirs(out_root, args.sample_runs, args.run_dir)
    print(f"  runs: {len(run_dirs)}")

    rows: list[dict] = []
    for run_dir in run_dirs:
        rel = run_dir.relative_to(out_root)
        print(f"  {rel}", flush=True)
        row = profile_one_run(
            run_dir,
            dataset_base=args.dataset_base.resolve(),
            device=device,
            eval_split=args.eval_split,
            max_batches=args.max_batches,
            measure_gpu=args.measure_gpu_power,
        )
        if row:
            add_energy_columns(
                row,
                e_mac_pj=args.e_mac_pj,
                e_sop_low=args.e_sop_pj_low,
                e_sop_nom=args.e_sop_pj_nom,
                e_sop_high=args.e_sop_pj_high,
                e_neuron_op_pj=args.e_neuron_op_pj,
            )
            rows.append(row)

    base_fields = [
        "Seed", "HoldOut", "Model", "T", "FlatDim", "H", "C",
        "InputNZRate", "EGRUSpikeRate",
        "MACs_total", "MACs_conv", "MACs_egru", "MACs_head",
        "SOPs_conv", "SOPs_egru", "SOPs_total", "NeuronOps_AdLIF",
        "Latency_ms", "Eval_windows", "Profile_batches",
        "Est_GPU_mJ_MAC", "Est_Loihi_mJ_SOP_low", "Est_Loihi_mJ_SOP_nom",
        "Est_Loihi_mJ_SOP_high", "Est_Loihi_mJ_neuron_nom", "Est_Loihi_mJ_total_nom",
        "GPU_power_W", "GPU_mJ_measured_forward",
    ]
    adlif_fields = sorted({k for r in rows for k in r if k.startswith("AdLIF")})
    fields = base_fields + adlif_fields

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["HoldOut"])):
            w.writerow(r)

    summary = summarize(rows, args.e_mac_pj, args.e_sop_pj_nom, args.e_neuron_op_pj)
    summary["out_root"] = str(out_root)
    summary["eval_split"] = args.eval_split
    summary["energy_constants_pJ"] = {
        "E_MAC": args.e_mac_pj,
        "E_SOP_low": args.e_sop_pj_low,
        "E_SOP_nom": args.e_sop_pj_nom,
        "E_SOP_high": args.e_sop_pj_high,
        "E_neuron_op": args.e_neuron_op_pj,
        "neuron_ops_per_adlif_update": DEFAULT_NEURON_OPS_PER_ADLIF_UPDATE,
    }
    summary["methodology"] = (
        "Dense MACs: analytical conv+EGRU+head. "
        "SOPs: conv layers scaled by measured input/AdLIF spike rates; "
        "EGRU SOPs = 6*T*H*(F+H)*egru_spike_rate. "
        "Neuron ops: AdLIF membrane updates (dense). "
        "Energy = ops × literature pJ constants (proxy, not measured Joules)."
    )

    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(rows)} rows → {csv_out}")
    print(f"Summary → {json_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
