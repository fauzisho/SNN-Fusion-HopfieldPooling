#!/usr/bin/env python3
"""Profile SIM A dense MACs for RQ2.1.

TriDet is profiled at its convolution/linear synapses with the same dense-MAC
definition used by NeuroBench.  Attend-and-Discriminate is first run through
NeuroBench for supported layers, then supplemented for its unhooked GRU and
functional attention matrix products.

TriDet consumes precomputed 2 s feature windows.  Its subject-level workload is
therefore reported both per sequence and amortized per valid input window.
I3D feature-extractor cost is outside this profile because the distributed WEAR
archives contain features rather than the I3D model or raw-video inference path.
"""

from __future__ import annotations

import os
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import nwear_paths as P

import numpy as np
import torch
import torch.nn as nn
import yaml

HOPFIELD_ROOT = Path(__file__).resolve().parent
WEAR_REPO = P.WEAR_REPO
if str(WEAR_REPO) not in sys.path:
    sys.path.insert(0, str(WEAR_REPO))

from camera_baseline.tridet.libs.core.config import _update_config  # noqa: E402
from camera_baseline.tridet.libs.modeling import make_meta_arch  # noqa: E402
from inertial_baseline.AttendAndDiscriminate import (  # noqa: E402
    AttendAndDiscriminate,
)
from train_hopfield_double_fusion_tridet import (  # noqa: E402
    TriDetDoubleHopfieldFusion,
    enrich_soft,
)
from train_hopfield_late_fusion_tridet import (  # noqa: E402
    HopfieldLateFusion,
    preds_to_soft,
)
try:
    from neurobench.hooks.layer import LayerHook  # noqa: E402
    from neurobench.metrics.workload import SynapticOperations  # noqa: E402
    from neurobench.models import TorchModel  # noqa: E402

    NEUROBENCH_AVAILABLE = True
except ModuleNotFoundError:
    # The TriDet Singularity image does not ship snntorch, which NeuroBench
    # imports even for non-spiking models. The shape-based fallback below uses
    # the identical dense Conv/Linear MAC definition.
    NEUROBENCH_AVAILABLE = False

PJ_PER_MAC = 4.6


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.load(os.path.expandvars(handle.read()), Loader=yaml.FullLoader)  # ${NWEAR_DATA} in configs


def profile_ad(config_path: Path, device: torch.device) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    ds = cfg["dataset"]
    mc = cfg["model"]
    # The YAML stores 18 activity classes; A-and-D is trained with an additional
    # NULL output, as constructed in infer_ad_frames.py.
    output_classes = ds["num_classes"] + 1
    model = AttendAndDiscriminate(
        channels=ds["input_dim"],
        classes=output_classes,
        hidden_dim=mc["hidden_dim"],
        conv_kernels=mc["conv_kernels"],
        conv_kernel_size=mc["conv_kernel_size"],
        enc_num_layers=mc["enc_layers"],
        enc_is_bidirectional=mc["enc_is_bidirectional"],
        dropout=mc["dropout"],
        dropout_rnn=mc["dropout_rnn"],
        dropout_cls=mc["dropout_cls"],
        activation=mc["activation"],
        sa_div=mc["sa_div"],
        gpu=str(device),
    ).to(device).eval()

    # NeuroBench 2.3.0 declares GRU supported, but its recurrent MAC helper does
    # not support nn.GRU. Register only layers it can count correctly and add the
    # GRU analytically below.
    x = torch.randn(
        1,
        ds["window_size"],
        ds["input_dim"],
        device=device,
    )
    if NEUROBENCH_AVAILABLE:
        wrapped = TorchModel(model)
        for layer in model.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                wrapped.connection_hooks.append(LayerHook(layer))
        with torch.no_grad():
            prediction = wrapped(x)
        nb = SynapticOperations()(
            wrapped,
            prediction,
            (x, torch.zeros(1, dtype=torch.long, device=device)),
        )
        for hook in wrapped.connection_hooks:
            hook.close()
        nb_dense = float(nb["Dense"])
        effective_macs = float(nb["Effective_MACs"])
        supported_counter = "NeuroBench SynapticOperations"
    else:
        hooks = DenseMacHooks(model)
        with torch.no_grad():
            model(x)
        nb_dense = float(hooks.total)
        hooks.close()
        effective_macs = None
        supported_counter = "shape fallback matching NeuroBench Dense"

    conv_time = ds["window_size"] - 4 * (mc["conv_kernel_size"] - 1)
    if conv_time <= 0:
        raise ValueError(f"Invalid A-and-D convolution output length: {conv_time}")
    directions = 2 if mc["enc_is_bidirectional"] else 1
    hidden = mc["hidden_dim"]
    gru_input = mc["conv_kernels"] * ds["input_dim"]
    layers = mc["enc_layers"]
    if layers != 1:
        raise NotImplementedError("The supplemental GRU formula currently assumes one layer")

    # Three GRU affine gates per direction and timestep. Bias adds and nonlinear
    # activation costs are not MACs and are excluded, consistently with
    # NeuroBench SynapticOperations.
    gru_macs = (
        3
        * conv_time
        * directions
        * hidden
        * (gru_input + hidden)
    )
    # SelfAttention executes two BMMs at every post-convolution timestep:
    # (S x C)@(C x S), then (C x S)@(S x S), where S is the 12-axis width.
    sensor_width = ds["input_dim"]
    attention_bmm_macs = (
        2 * conv_time * sensor_width * sensor_width * mc["conv_kernels"]
    )
    # TemporalAttention's FC is already in NeuroBench; this is only weights*x.
    temporal_weight_macs = conv_time * hidden * directions

    supplemental = float(gru_macs + attention_bmm_macs + temporal_weight_macs)
    total = nb_dense + supplemental
    return {
        "model": "Attend-and-Discriminate IMU",
        "window_seconds": 2.0,
        "input_shape": [1, ds["window_size"], ds["input_dim"]],
        "neurobench_supported_dense_macs": nb_dense,
        "neurobench_effective_macs": effective_macs,
        "supported_layer_counter": supported_counter,
        "supplemental_gru_macs": float(gru_macs),
        "supplemental_attention_bmm_macs": float(attention_bmm_macs),
        "supplemental_temporal_weight_macs": float(temporal_weight_macs),
        "dense_macs_per_window": total,
        "mac_proxy_mj_per_window": total * PJ_PER_MAC / 1e9,
        "notes": [
            "Dense MAC proxy; activation sparsity is not credited on conventional dense hardware.",
            "GRU and functional attention products are supplemented because NeuroBench does not count them correctly.",
        ],
    }


class DenseMacHooks:
    """Shape-based dense MAC accumulator for NeuroBench-supported ANN layers."""

    def __init__(self, model: nn.Module):
        self.total = 0
        self.by_type: dict[str, int] = defaultdict(int)
        self.handles = []
        for layer in model.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self.handles.append(layer.register_forward_hook(self._conv))
            elif isinstance(layer, nn.Linear):
                self.handles.append(layer.register_forward_hook(self._linear))

    def _conv(self, module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        kernel = math.prod(module.kernel_size)
        macs = output.numel() * (module.in_channels // module.groups) * kernel
        self.total += int(macs)
        self.by_type[type(module).__name__] += int(macs)

    def _linear(self, module: nn.Linear, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        macs = output.numel() * module.in_features
        self.total += int(macs)
        self.by_type[type(module).__name__] += int(macs)

    def reset(self) -> None:
        self.total = 0
        self.by_type.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def make_tridet(config_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    cfg = _update_config(load_yaml(config_path))
    model = make_meta_arch(cfg["model"]["model_name"], **cfg["model"])
    return model.to(device).eval(), cfg


def run_tridet_core(model: nn.Module, feat: torch.Tensor) -> int:
    video = {
        "video_id": "profile",
        "feats": feat,
        "segments": None,
        "labels": None,
        "fps": 60.0,
        "duration": float(feat.shape[-1]),
        "feat_stride": 60,
        "feat_num_frames": 120,
    }
    with torch.inference_mode():
        batched_inputs, masks = model.preprocessing([video])
        feats, masks = model.backbone(batched_inputs, masks)
        fpn_feats, fpn_masks = model.neck(feats, masks)
        model.cls_head(fpn_feats, fpn_masks)
        if model.use_trident_head:
            model.start_head(fpn_feats, fpn_masks)
            model.end_head(fpn_feats, fpn_masks)
        model.reg_head(fpn_feats, fpn_masks)
    return int(batched_inputs.shape[-1])


def profile_tridet(
    name: str,
    config_path: Path,
    feature_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, cfg = make_tridet(config_path, device)
    hooks = DenseMacHooks(model)
    rows: list[dict[str, Any]] = []
    for subject in range(18):
        path = feature_dir / f"sbj_{subject}.npy"
        shape = np.load(path, mmap_mode="r").shape
        if len(shape) != 2 or shape[1] != cfg["dataset"]["input_dim"]:
            raise ValueError(f"Unexpected feature shape at {path}: {shape}")
        valid_windows = int(shape[0])
        hooks.reset()
        # Values do not affect dense MAC count, so avoid reading the full archive.
        feat = torch.zeros(
            cfg["dataset"]["input_dim"],
            valid_windows,
            dtype=torch.float32,
            device=device,
        )
        padded_windows = run_tridet_core(model, feat)
        dense_macs = int(hooks.total)
        rows.append(
            {
                "model": name,
                "subject": subject,
                "valid_windows": valid_windows,
                "padded_windows": padded_windows,
                "sequence_dense_macs": dense_macs,
                "dense_macs_per_window": dense_macs / valid_windows,
                "mac_proxy_mj_per_window": dense_macs / valid_windows * PJ_PER_MAC / 1e9,
            }
        )
        del feat

    hooks.close()
    per_window = np.array([r["dense_macs_per_window"] for r in rows])
    sequence = np.array([r["sequence_dense_macs"] for r in rows])
    total_macs = float(sequence.sum())
    total_windows = int(sum(r["valid_windows"] for r in rows))
    macro = float(per_window.mean())
    summary = {
        "model": name,
        "window_seconds": 2.0,
        "input_dim": cfg["dataset"]["input_dim"],
        "subjects": 18,
        "dense_macs_per_window_macro_mean": macro,
        "dense_macs_per_window_macro_std": float(per_window.std(ddof=1)),
        "dense_macs_per_window_micro_mean": total_macs / total_windows,
        "mac_proxy_mj_per_window_macro_mean": macro * PJ_PER_MAC / 1e9,
        "total_profiled_windows": total_windows,
        "notes": [
            "Counts Conv/Linear synaptic MACs, matching NeuroBench Dense semantics.",
            "Subject-level TriDet inference is amortized over valid 2 s feature windows.",
            "Padding performed by the released TriDet preprocessing is included.",
            "NMS and other non-synaptic postprocessing are excluded.",
            "For I3D input, upstream RGB-to-I3D feature extraction is not included.",
        ],
    }
    return summary, rows


def profile_fusion_head(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    model = model.cpu().eval()
    inputs = tuple(x.cpu() for x in inputs)
    with torch.no_grad(), torch.profiler.profile(with_flops=True) as profiler:
        model(*inputs)
    matrix_ops = {"aten::addmm", "aten::mm", "aten::bmm", "aten::matmul"}
    flops = sum(
        float(event.flops)
        for event in profiler.key_averages()
        if event.key in matrix_ops and event.flops
    )
    # PyTorch reports two FLOPs for one multiply-accumulate in these operators.
    dense_macs = flops / 2.0
    return {
        "dense_macs_per_fusion_decision": dense_macs,
        "mac_proxy_mj_per_fusion_decision": dense_macs * PJ_PER_MAC / 1e9,
        "method": "torch.profiler matrix FLOPs / 2",
        "notes": [
            "One model call (one fusion decision); execution-rate scaling is not included.",
            "Softmax, normalization, and standalone elementwise operations are excluded from MACs.",
        ],
    }


def profile_saved_sima_heads(
    pipeline: str,
    fold_dir: Path,
    imu_npz_name: str,
) -> dict[str, Any]:
    """Profile one frozen fold on deterministic inputs from that same fold."""
    imu = np.load(fold_dir / imu_npz_name)
    i3d = np.load(fold_dir / "i3d_frames.npz")

    active_val = np.flatnonzero(imu["is_val"] & (imu["gt"] > 0))
    val = np.flatnonzero(imu["is_val"])
    if len(active_val) == 0 or len(val) < 100:
        raise ValueError(f"Insufficient validation inputs under {fold_dir}")
    single_idx = int(active_val[0])
    seq_start = int(val[0])
    single_slice = slice(single_idx, single_idx + 1)
    imu_single = preds_to_soft(
        imu["preds"][single_slice],
        imu["scores"][single_slice],
    )
    i3d_single = preds_to_soft(
        i3d["preds"][single_slice],
        i3d["scores"][single_slice],
    )

    single_ckpt = torch.load(
        fold_dir / "hopfield_late_fusion" / "model.pt",
        map_location="cpu",
    )
    single_model = HopfieldLateFusion().eval()
    single_model.load_state_dict(single_ckpt["model_state_dict"])
    single = profile_fusion_head(
        single_model,
        (
            torch.from_numpy(imu_single),
            torch.from_numpy(i3d_single),
        ),
    )

    double_ckpt = torch.load(
        fold_dir / "hopfield_double_fusion" / "model.pt",
        map_location="cpu",
    )
    hidden = int(double_ckpt["hidden"])
    seq_len = int(double_ckpt["seq_len"])
    seq_slice = slice(seq_start, seq_start + seq_len)
    imu_seq_soft = preds_to_soft(
        imu["preds"][seq_slice],
        imu["scores"][seq_slice],
    )
    i3d_seq_soft = preds_to_soft(
        i3d["preds"][seq_slice],
        i3d["scores"][seq_slice],
    )
    imu_enriched = enrich_soft(imu_seq_soft, imu["scores"][seq_slice])
    i3d_enriched = enrich_soft(i3d_seq_soft, i3d["scores"][seq_slice])
    double_model = TriDetDoubleHopfieldFusion(in_dim=20, hidden=hidden).eval()
    double_model.load_state_dict(double_ckpt["model_state_dict"])
    double = profile_fusion_head(
        double_model,
        (
            torch.from_numpy(imu_enriched).unsqueeze(0),
            torch.from_numpy(i3d_enriched).unsqueeze(0),
        ),
    )
    provenance = {
        "pipeline": pipeline,
        "fold": str(fold_dir),
        "single_input_frame_index": single_idx,
        "double_input_start_frame_index": seq_start,
        "double_input_sequence_length": seq_len,
        "checkpoint_weights_loaded": True,
    }
    return {
        "single_hopfield": {**single, "provenance": provenance},
        "double_hopfield": {**double, "provenance": provenance},
    }


def profile_sima_heads() -> dict[str, Any]:
    specs = {
        "bf1": (
            "BF1: A-and-D IMU + TriDet I3D",
            HOPFIELD_P.SIMA_BF1,
            "ad_frames.npz",
        ),
        "bf2": (
            "BF2: TriDet IMU + TriDet I3D",
            HOPFIELD_P.SIMA_BF2,
            "imu_frames.npz",
        ),
    }
    result: dict[str, Any] = {}
    for key, (label, root, imu_name) in specs.items():
        folds = []
        for seed in range(1, 4):
            for split in range(1, 19):
                profiled = profile_saved_sima_heads(
                    label,
                    root / f"seed_{seed}" / f"split_{split}",
                    imu_name,
                )
                folds.append(
                    {
                        "seed": seed,
                        "split": split,
                        "single_macs": profiled["single_hopfield"][
                            "dense_macs_per_fusion_decision"
                        ],
                        "double_macs": profiled["double_hopfield"][
                            "dense_macs_per_fusion_decision"
                        ],
                        "provenance": profiled["double_hopfield"]["provenance"],
                    }
                )

        def aggregate(field: str) -> dict[str, Any]:
            values = np.asarray([fold[field] for fold in folds], dtype=np.float64)
            return {
                "dense_macs_per_fusion_decision": float(values.mean()),
                "dense_macs_std": float(values.std(ddof=1)),
                "dense_macs_min": float(values.min()),
                "dense_macs_max": float(values.max()),
                "mac_proxy_mj_per_fusion_decision": float(
                    values.mean() * PJ_PER_MAC / 1e9
                ),
                "method": "torch.profiler matrix FLOPs / 2",
                "n_folds": len(values),
                "notes": [
                    "Mean over 54 frozen LOSO heads.",
                    "Each fold uses its first active validation frame (single) or first validation sequence (double).",
                    "Softmax, normalization, and standalone elementwise operations are excluded from MACs.",
                ],
            }

        result[key] = {
            "single_hopfield": aggregate("single_macs"),
            "double_hopfield": aggregate("double_macs"),
            "per_fold": folds,
        }
    return result


def write_report(
    output_dir: Path,
    ad: dict[str, Any],
    tridet_imu: dict[str, Any],
    tridet_i3d: dict[str, Any],
    heads: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sima_ad = (
        ad["dense_macs_per_window"]
        + tridet_i3d["dense_macs_per_window_macro_mean"]
    )
    sima_tridet = (
        tridet_imu["dense_macs_per_window_macro_mean"]
        + tridet_i3d["dense_macs_per_window_macro_mean"]
    )
    bf1_single = heads["bf1"]["single_hopfield"]["dense_macs_per_fusion_decision"]
    bf1_double = heads["bf1"]["double_hopfield"]["dense_macs_per_fusion_decision"]
    bf2_single = heads["bf2"]["single_hopfield"]["dense_macs_per_fusion_decision"]
    bf2_double = heads["bf2"]["double_hopfield"]["dense_macs_per_fusion_decision"]
    summary = {
        "energy_constant_pj_per_dense_mac": PJ_PER_MAC,
        "attend_and_discriminate": ad,
        "tridet_imu": tridet_imu,
        "tridet_i3d": tridet_i3d,
        "sima_fusion_heads": heads,
        "sima_encoder_totals": {
            "ad_imu_plus_tridet_i3d_dense_macs_per_window": sima_ad,
            "ad_imu_plus_tridet_i3d_mac_proxy_mj_per_window": sima_ad * PJ_PER_MAC / 1e9,
            "tridet_imu_plus_tridet_i3d_dense_macs_per_window": sima_tridet,
            "tridet_imu_plus_tridet_i3d_mac_proxy_mj_per_window": sima_tridet * PJ_PER_MAC / 1e9,
        },
        "sima_pipeline_lower_bounds_one_fusion_decision": {
            "bf1_ad_i3d_single_hopfield_macs": sima_ad + bf1_single,
            "bf1_ad_i3d_single_hopfield_mj": (sima_ad + bf1_single) * PJ_PER_MAC / 1e9,
            "bf1_ad_i3d_double_hopfield_macs": sima_ad + bf1_double,
            "bf1_ad_i3d_double_hopfield_mj": (sima_ad + bf1_double) * PJ_PER_MAC / 1e9,
            "bf2_tridet_imu_i3d_single_hopfield_macs": sima_tridet + bf2_single,
            "bf2_tridet_imu_i3d_single_hopfield_mj": (sima_tridet + bf2_single) * PJ_PER_MAC / 1e9,
            "bf2_tridet_imu_i3d_double_hopfield_macs": sima_tridet + bf2_double,
            "bf2_tridet_imu_i3d_double_hopfield_mj": (sima_tridet + bf2_double) * PJ_PER_MAC / 1e9,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "tridet_per_subject.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SIM A dense-operation benchmark",
        "",
        f"Energy proxy: {PJ_PER_MAC:.1f} pJ per dense MAC.",
        "",
        "| Model | Dense MAC/window | MAC proxy (mJ/window) |",
        "|---|---:|---:|",
        (
            f"| Attend-and-Discriminate IMU | {ad['dense_macs_per_window']/1e6:.3f} M "
            f"| {ad['mac_proxy_mj_per_window']:.6f} |"
        ),
        (
            f"| TriDet IMU | {tridet_imu['dense_macs_per_window_macro_mean']/1e6:.3f} M "
            f"| {tridet_imu['mac_proxy_mj_per_window_macro_mean']:.6f} |"
        ),
        (
            f"| TriDet I3D head only | {tridet_i3d['dense_macs_per_window_macro_mean']/1e6:.3f} M "
            f"| {tridet_i3d['mac_proxy_mj_per_window_macro_mean']:.6f} |"
        ),
        (
            f"| SIM A BF1 single Hopfield head (one decision) | {bf1_single/1e6:.3f} M "
            f"| {bf1_single*PJ_PER_MAC/1e9:.6f} |"
        ),
        (
            f"| SIM A BF1 double Hopfield head (one decision) | {bf1_double/1e6:.3f} M "
            f"| {bf1_double*PJ_PER_MAC/1e9:.6f} |"
        ),
        (
            f"| SIM A BF2 single Hopfield head (one decision) | {bf2_single/1e6:.3f} M "
            f"| {bf2_single*PJ_PER_MAC/1e9:.6f} |"
        ),
        (
            f"| SIM A BF2 double Hopfield head (one decision) | {bf2_double/1e6:.3f} M "
            f"| {bf2_double*PJ_PER_MAC/1e9:.6f} |"
        ),
        (
            f"| A-and-D IMU + TriDet I3D | {sima_ad/1e6:.3f} M "
            f"| {sima_ad*PJ_PER_MAC/1e9:.6f} |"
        ),
        (
            f"| TriDet IMU + TriDet I3D | {sima_tridet/1e6:.3f} M "
            f"| {sima_tridet*PJ_PER_MAC/1e9:.6f} |"
        ),
        "",
        "TriDet I3D is **head only**: the distributed WEAR input is a precomputed I3D",
        "feature archive, so RGB decoding and I3D feature extraction are not available to",
        "this benchmark. Consequently, its visual and fused values are lower bounds.",
        "",
        "Fusion-head values are per fusion decision. They are not multiplied by the",
        "frame-level execution rate used when the exported TriDet scores are rasterized.",
        "",
        "TriDet processes one full subject sequence. The table reports the macro mean of",
        "subject-level MACs divided by each subject's valid 2 s input-window count, including",
        "the padding imposed by the released inference code.",
    ]
    (output_dir / "RESULTS_SIMA_BASELINE_ENERGY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    cfg_dir = HOPFIELD_ROOT / "configs"
    ap.add_argument(
        "--ad-config",
        type=Path,
        default=cfg_dir / "attendanddiscriminate_long_120_60.yaml",
    )
    ap.add_argument(
        "--tridet-imu-config",
        type=Path,
        default=cfg_dir / "tridet_inertial_120_60.yaml",
    )
    ap.add_argument(
        "--tridet-i3d-config",
        type=Path,
        default=cfg_dir / "tridet_camera_120_60.yaml",
    )
    ap.add_argument(
        "--imu-features",
        type=Path,
        default=P.IMU_FEAT_DIR,
    )
    ap.add_argument(
        "--i3d-features",
        type=Path,
        default=P.I3D_FEAT_DIR,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=P.SIMA_ENERGY,
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("TriDet profiling requires CUDA; submit the companion SLURM job")
    device = torch.device("cuda")
    ad = profile_ad(args.ad_config, device)
    tri_imu, imu_rows = profile_tridet(
        "TriDet IMU",
        args.tridet_imu_config,
        args.imu_features,
        device,
    )
    tri_i3d, i3d_rows = profile_tridet(
        "TriDet I3D head only",
        args.tridet_i3d_config,
        args.i3d_features,
        device,
    )
    heads = profile_sima_heads()
    write_report(args.output_dir, ad, tri_imu, tri_i3d, heads, imu_rows + i3d_rows)
    print((args.output_dir / "RESULTS_SIMA_BASELINE_ENERGY.md").read_text())


if __name__ == "__main__":
    main()
