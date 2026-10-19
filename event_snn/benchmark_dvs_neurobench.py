#!/usr/bin/env python3
"""
Run NeuroBench workload metrics on DVS128 AllConvAdLIFEGRU_DVS128 LOSO checkpoints.

Uses the official NeuroBench harness:
  https://neurobench.readthedocs.io/en/latest/

Metrics:
  - SynapticOperations (Effective_MACs, Effective_ACs, Dense)
  - ActivationSparsity
  - NeuronOperations (AdLIF patch)

Energy (proxy, same constants as profile_dvs_compute_spikes.py):
  - GPU: Effective_MACs × E_MAC
  - Loihi: Effective_ACs × E_SOP + Effective Neuron Ops × E_neuron_op
  - Supplemental EGRU SOPs (analytical; EGRU is not a hooked nn.Linear)

Usage:
  python3 benchmark_dvs_neurobench.py \\
    --out-root output/loso_dvs128_adlif_egru_aug_valfix_loso18_cluster

  python3 benchmark_dvs_neurobench.py \\
    --out-root output/loso_dvs128_adlif_egru_aug_valfix_loso18_cluster \\
    --run-dir output/.../seed_1/test_sbj_0 --max-batches 20
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
import nwear_paths as P

import torch
from torch.utils.data import DataLoader, Subset

# Optional local NeuroBench checkout ($NEUROBENCH_ROOT); otherwise the pip package.
_NB_ROOT = P.NEUROBENCH_ROOT
if _NB_ROOT.is_dir() and str(_NB_ROOT) not in sys.path:
    sys.path.insert(0, str(_NB_ROOT))

from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import (
    ActivationSparsity,
    NeuronOperations,
    SynapticOperations,
)

from event_snn.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from event_snn.neurobench_integration import (
    SpikingJellyDVSModel,
    apply_adlif_neurobench_patches,
)
from event_snn.profile_dvs_compute_spikes import (
    DEFAULT_E_MAC_PJ,
    DEFAULT_E_NEURON_OP_PJ,
    DEFAULT_E_SOP_PJ_HIGH,
    DEFAULT_E_SOP_PJ_LOW,
    DEFAULT_E_SOP_PJ_NOM,
    analytical_macs,
    energy_mj_from_ops,
    estimate_sops_egru,
    iter_run_dirs,
    measure_spike_rates,
)
from event_snn.training.trainer import V2EAicoTrainer


def _config_from_checkpoint(raw: dict, out_dir: Path) -> ExperimentConfig:
    data_raw = dict(raw["data"])
    data_raw["dataset_root"] = Path(data_raw["dataset_root"])
    if "events_tier" not in data_raw:
        data_raw["events_tier"] = "dvs128" if "dvs128" in str(out_dir) else "dvs240"
    data = DataConfig(**data_raw)
    model = ModelConfig.from_dict(raw["model"])
    train_defaults = {k.name: getattr(TrainConfig(), k.name) for k in fields(TrainConfig)}
    train_defaults.update(raw["train"])
    train = TrainConfig(**train_defaults)
    if train.cache_dir is not None:
        train.cache_dir = Path(train.cache_dir)
    return ExperimentConfig(data=data, model=model, train=train, out_dir=out_dir)


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


def _identity_preprocess(data: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return data


def _identity_postprocess(preds: torch.Tensor) -> torch.Tensor:
    return preds


@dataclass
class EnergyBreakdown:
    nb_effective_macs: float
    nb_effective_acs: float
    nb_dense_synops: float
    nb_activation_sparsity: float
    nb_neuron_ops_effective: float
    nb_neuron_ops_dense: float
    supplemental_egru_sops: int
    total_sops_with_egru: float
    est_gpu_mj_mac: float
    est_loihi_mj_sop_nom: float
    est_loihi_mj_neuron_nom: float
    est_loihi_mj_total_nom: float


def neurobench_to_energy(
    results: dict,
    *,
    supplemental_egru_sops: int,
    e_mac_pj: float,
    e_sop_nom: float,
    e_sop_low: float,
    e_sop_high: float,
    e_neuron_op_pj: float,
) -> EnergyBreakdown:
    syn = results["SynapticOperations"]
    neuron = results.get("NeuronOperations", {})
    macs = float(syn["Effective_MACs"])
    acs = float(syn["Effective_ACs"])
    dense = float(syn["Dense"])
    n_eff = float(neuron.get("Effective Neuron Ops", 0.0))
    n_dense = float(neuron.get("Neuron Dense Ops", 0.0))
    total_acs = acs + supplemental_egru_sops
    return EnergyBreakdown(
        nb_effective_macs=macs,
        nb_effective_acs=acs,
        nb_dense_synops=dense,
        nb_activation_sparsity=float(results["ActivationSparsity"]),
        nb_neuron_ops_effective=n_eff,
        nb_neuron_ops_dense=n_dense,
        supplemental_egru_sops=supplemental_egru_sops,
        total_sops_with_egru=total_acs,
        est_gpu_mj_mac=energy_mj_from_ops(int(macs), e_mac_pj),
        est_loihi_mj_sop_nom=energy_mj_from_ops(int(total_acs), e_sop_nom),
        est_loihi_mj_neuron_nom=energy_mj_from_ops(int(n_eff), e_neuron_op_pj),
        est_loihi_mj_total_nom=(
            energy_mj_from_ops(int(total_acs), e_sop_nom)
            + energy_mj_from_ops(int(n_eff), e_neuron_op_pj)
        ),
    )


def benchmark_one_run(
    run_dir: Path,
    *,
    dataset_base: Path,
    device: torch.device,
    eval_split: str,
    max_batches: int | None,
    batch_size: int,
    quiet: bool,
    e_mac_pj: float,
    e_sop_nom: float,
    e_sop_low: float,
    e_sop_high: float,
    e_neuron_op_pj: float,
) -> dict[str, str | float] | None:
    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.is_file():
        return None

    m = re.match(r"seed_(\d+)/test_sbj_(\d+)$", "/".join(run_dir.parts[-2:]))
    if not m:
        return None
    seed, test_sbj = m.group(1), int(m.group(2))

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = _config_from_checkpoint(checkpoint["config"], run_dir)
    cfg.out_dir = run_dir
    cfg.train.device = str(device)
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
    base_loader = val_loader if eval_split == "val" else test_loader

    dataset = base_loader.dataset
    if max_batches is not None:
        n = min(len(dataset), max_batches * batch_size)
        dataset = Subset(dataset, range(n))

    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
    )

    model = trainer.build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    macs = analytical_macs(
        channels=cfg.model.channels,
        feat_neurons=cfg.model.feat_neurons,
        num_classes=cfg.model.num_classes,
        input_height=cfg.data.input_height,
        input_width=cfg.data.input_width,
        time_bins=cfg.data.time_bins,
    )
    rates = measure_spike_rates(
        model, eval_loader, device, max_batches=max_batches or 20
    )
    supplemental_egru_sops = estimate_sops_egru(
        macs.T, macs.flat_dim, macs.H, rates["egru_spike_rate"]
    )

    apply_adlif_neurobench_patches()
    nb_model = SpikingJellyDVSModel(model)

    benchmark = Benchmark(
        nb_model,
        eval_loader,
        [_identity_preprocess],
        [_identity_postprocess],
        [[], [SynapticOperations, ActivationSparsity, NeuronOperations]],
    )
    results = benchmark.run(device=str(device), quiet=quiet)
    energy = neurobench_to_energy(
        results,
        supplemental_egru_sops=supplemental_egru_sops,
        e_mac_pj=e_mac_pj,
        e_sop_nom=e_sop_nom,
        e_sop_low=e_sop_low,
        e_sop_high=e_sop_high,
        e_neuron_op_pj=e_neuron_op_pj,
    )

    row: dict[str, str | float] = {
        "Seed": seed,
        "HoldOut": f"sbj_{test_sbj}",
        "Model": cfg.model.model_name,
        "Eval_split": eval_split,
        "Eval_windows": len(dataset),
        "Profile_batches": rates["n_batches"],
        "InputNZRate": round(rates["input_nz_rate"], 4),
        "EGRUSpikeRate": round(rates["egru_spike_rate"], 4),
        "NB_Effective_MACs": round(energy.nb_effective_macs, 2),
        "NB_Effective_ACs": round(energy.nb_effective_acs, 2),
        "NB_Dense_SynOps": round(energy.nb_dense_synops, 2),
        "NB_ActivationSparsity": round(energy.nb_activation_sparsity, 4),
        "NB_NeuronOps_Effective": round(energy.nb_neuron_ops_effective, 2),
        "NB_NeuronOps_Dense": round(energy.nb_neuron_ops_dense, 2),
        "Supplemental_EGRU_SOPs": supplemental_egru_sops,
        "NB_Total_ACs_with_EGRU": round(energy.total_sops_with_egru, 2),
        "Analytical_MACs_total": macs.total_macs,
        "Est_GPU_mJ_NB_MAC": round(energy.est_gpu_mj_mac, 6),
        "Est_Loihi_mJ_NB_SOP_nom": round(
            energy_mj_from_ops(int(energy.total_sops_with_egru), e_sop_nom), 6
        ),
        "Est_Loihi_mJ_NB_neuron_nom": round(energy.est_loihi_mj_neuron_nom, 6),
        "Est_Loihi_mJ_NB_total_nom": round(energy.est_loihi_mj_total_nom, 6),
        "neurobench_raw": json.dumps(results),
    }
    # Analytical total SOP estimate for cross-check (same formula as profile script).
    from event_snn.profile_dvs_compute_spikes import (
        estimate_conv_sops,
    )

    activity = [rates["input_nz_rate"], *rates["adlif_spike_rates"]]
    sops_conv = estimate_conv_sops(macs.conv_macs_per_layer, macs.T, activity)
    row["Analytical_SOPs_conv_est"] = sops_conv
    row["Analytical_SOPs_egru_est"] = supplemental_egru_sops
    row["Analytical_SOPs_total_est"] = sops_conv + supplemental_egru_sops
    row["Analytical_MACs_total"] = macs.total_macs

    for i, r in enumerate(rates["adlif_spike_rates"], start=1):
        row[f"AdLIF{i}_rate"] = round(r, 4)

    return row


def summarize(rows: list[dict], e_mac_pj: float, e_sop_nom: float, e_neuron_op_pj: float) -> dict:
    if not rows:
        return {}

    n = len(rows)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n

    nb_macs = mean("NB_Effective_MACs")
    nb_acs = mean("NB_Effective_ACs")
    nb_acs_egru = mean("NB_Total_ACs_with_EGRU")
    nb_neuron = mean("NB_NeuronOps_Effective")
    return {
        "n_runs": n,
        "model_name": rows[0].get("Model", "unknown"),
        "framework": "NeuroBench",
        "mean_NB_Effective_MACs_per_window": round(nb_macs, 2),
        "mean_NB_Effective_ACs_per_window": round(nb_acs, 2),
        "mean_NB_Total_ACs_with_EGRU_per_window": round(nb_acs_egru, 2),
        "mean_NB_ActivationSparsity": round(mean("NB_ActivationSparsity"), 4),
        "mean_NB_NeuronOps_Effective_per_window": round(nb_neuron, 2),
        "mean_Analytical_MACs_per_window": int(mean("Analytical_MACs_total")),
        "mean_Analytical_SOPs_total_est": int(mean("Analytical_SOPs_total_est")),
        "est_GPU_mJ_per_window_NB_MAC": round(energy_mj_from_ops(int(nb_macs), e_mac_pj), 6),
        "est_Loihi_mJ_per_window_NB_SOP_nom": round(
            energy_mj_from_ops(int(nb_acs_egru), e_sop_nom), 6
        ),
        "est_Loihi_mJ_per_window_NB_neuron_nom": round(
            energy_mj_from_ops(int(nb_neuron), e_neuron_op_pj), 6
        ),
        "est_Loihi_mJ_per_window_NB_total_nom": round(
            energy_mj_from_ops(int(nb_acs_egru), e_sop_nom)
            + energy_mj_from_ops(int(nb_neuron), e_neuron_op_pj),
            6,
        ),
        "note_egru": (
            "NeuroBench SynapticOperations hooks spikingjelly Conv2d/Linear only. "
            "EGRU synaptic ops are added analytically (Supplemental_EGRU_SOPs) "
            "using measured EGRU spike rate, matching profile_dvs_compute_spikes.py."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--dataset-base", type=Path, default=P.DATA_ROOT)
    ap.add_argument("--eval-split", choices=("val", "test"), default="val")
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--sample-runs", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--quiet", action="store_true")
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
    csv_out = args.csv_out or out_root / f"results_neurobench_{tag}.csv"
    json_out = args.json_out or out_root / f"results_neurobench_{tag}_summary.json"

    print(f"NeuroBench DVS benchmark: {out_root.name} on {device} (eval={args.eval_split})")
    run_dirs = iter_run_dirs(out_root, args.sample_runs, args.run_dir)
    print(f"  runs: {len(run_dirs)}")

    rows: list[dict] = []
    for run_dir in run_dirs:
        rel = run_dir.relative_to(out_root)
        print(f"  {rel}", flush=True)
        row = benchmark_one_run(
            run_dir,
            dataset_base=args.dataset_base.resolve(),
            device=device,
            eval_split=args.eval_split,
            max_batches=args.max_batches,
            batch_size=args.batch_size,
            quiet=args.quiet,
            e_mac_pj=args.e_mac_pj,
            e_sop_nom=args.e_sop_pj_nom,
            e_sop_low=args.e_sop_pj_low,
            e_sop_high=args.e_sop_pj_high,
            e_neuron_op_pj=args.e_neuron_op_pj,
        )
        if row:
            rows.append(row)

    base_fields = [
        "Seed", "HoldOut", "Model", "Eval_split", "Eval_windows", "Profile_batches",
        "InputNZRate", "EGRUSpikeRate",
        "NB_Effective_MACs", "NB_Effective_ACs", "NB_Dense_SynOps",
        "NB_ActivationSparsity", "NB_NeuronOps_Effective", "NB_NeuronOps_Dense",
        "Supplemental_EGRU_SOPs", "NB_Total_ACs_with_EGRU",
        "Analytical_MACs_total", "Analytical_SOPs_conv_est", "Analytical_SOPs_egru_est",
        "Analytical_SOPs_total_est",
        "Est_GPU_mJ_NB_MAC", "Est_Loihi_mJ_NB_SOP_nom",
        "Est_Loihi_mJ_NB_neuron_nom", "Est_Loihi_mJ_NB_total_nom",
    ]
    adlif_fields = sorted({k for r in rows for k in r if k.startswith("AdLIF")})
    fields = base_fields + adlif_fields + ["neurobench_raw"]

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
    }
    summary["docs"] = "https://neurobench.readthedocs.io/en/latest/"

    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(rows)} rows → {csv_out}")
    print(f"Summary → {json_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
