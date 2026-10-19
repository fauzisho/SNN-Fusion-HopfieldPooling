#!/usr/bin/env python3
"""Aggregate paper-style mAP@tIoU {0.3..0.7} + Avg for A&D × TriDet I3D fusion LOSO.

Sources
-------
- A&D / TriDet: FINAL mAP lines in training logs
- Score-pick / Oracle: recompute from fusion_preds.npz (frame → segments)
- Single / Double Hopfield: reload model.pt, infer val frames, then mAP

Writes:
  results_ad_fusion_map_loso.csv
  RESULTS_AD_FUSION_MAP.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch
import torch.nn as nn

try:
    from hflayers import HopfieldPooling
except ImportError as exc:
    raise ImportError("Need hflayers (use wearsnn_v2e.sif)") from exc

WEAR_REPO = P.WEAR_REPO
sys.path.insert(0, str(WEAR_REPO))
from utils.data_utils import convert_samples_to_segments  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = P.SIMA_BF1
ANN_DIR = P.ANN_DIR
TIOU = [0.3, 0.4, 0.5, 0.6, 0.7]
TIOU_KEYS = ("0.3", "0.4", "0.5", "0.6", "0.7")
SAMPLING_RATE = 50
NUM_CLASSES = 18

MAP_LINE = re.compile(
    r"Avg\. mAP\s+([\d.]+)\s+\(%\)\s+"
    r"mAP@0\.3\s+([\d.]+).*mAP@0\.4\s+([\d.]+).*"
    r"mAP@0\.5\s+([\d.]+).*mAP@0\.6\s+([\d.]+).*mAP@0\.7\s+([\d.]+)"
)
SAMPLE_LINE = re.compile(
    r"Acc\s+([\d.]+)\s+\(%\)\s+Prec\s+([\d.]+)\s+\(%\)\s+"
    r"Rec\s+([\d.]+)\s+\(%\)\s+F1\s+([\d.]+)"
)


def _load_wear_pp():
    path = WEAR_REPO / "postprocessing" / "utils.py"
    spec = importlib.util.spec_from_file_location("wear_postprocessing_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_pp = _load_wear_pp()
ANETdetection = _pp.ANETdetection
combine_predictions = _pp.combine_predictions


def parse_final_map(log_path: Path) -> dict[str, float] | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "FINAL AVERAGED RESULTS" in text:
        chunk = text.split("FINAL AVERAGED RESULTS")[-1]
    elif "FINAL RESULTS SUBJECT" in text:
        chunk = text.split("FINAL RESULTS SUBJECT")[-1]
    else:
        chunk = text
    map_m = sample_m = None
    for line in chunk.splitlines():
        if map_m is None and "Avg. mAP" in line:
            map_m = MAP_LINE.search(line)
        if sample_m is None and line.strip().startswith("Acc"):
            sample_m = SAMPLE_LINE.search(line)
    if not map_m:
        return None
    avg, *tiou = map_m.groups()
    out = {
        "mAP_avg": float(avg),
        **{f"mAP_{k}": float(v) for k, v in zip(TIOU_KEYS, tiou)},
    }
    if sample_m:
        out["acc"] = float(sample_m.group(1))
        out["f1"] = float(sample_m.group(4))
    return out


def majority_vote(preds: np.ndarray, subject_ids: np.ndarray, kernel: int) -> np.ndarray:
    """Per-subject majority filter (WEAR postprocess ~10s = 500 @ 50 Hz)."""
    if kernel <= 1:
        return preds.astype(np.int64, copy=True)
    out = preds.copy().astype(np.int64)
    half = kernel // 2
    for sbj in np.unique(subject_ids):
        idx = np.where(subject_ids == sbj)[0]
        arr = out[idx]
        smoothed = np.empty_like(arr)
        n = len(arr)
        for i in range(n):
            left = max(0, i - half)
            right = min(n, i + half + 1)
            if right - left < kernel:
                if left == 0:
                    right = min(n, kernel)
                else:
                    left = max(0, n - kernel)
            smoothed[i] = np.bincount(arr[left:right]).argmax()
        out[idx] = smoothed
    return out


def map_from_frame_preds(
    subject_ids: np.ndarray,
    preds: np.ndarray,
    ann_path: Path,
    mv_kernel: int = 0,
) -> dict[str, float]:
    preds_i = preds.astype(int)
    if mv_kernel > 1:
        preds_i = majority_vote(preds_i, subject_ids, mv_kernel)
    segs = convert_samples_to_segments(subject_ids, preds_i, SAMPLING_RATE)
    det = ANETdetection(str(ann_path), "validation", tiou_thresholds=TIOU, num_workers=4)
    v_map, _ = det.evaluate(segs)
    v_map = np.asarray(v_map, dtype=np.float64)
    return {
        "mAP_avg": float(np.mean(v_map) * 100.0),
        **{f"mAP_{k}": float(v * 100.0) for k, v in zip(TIOU_KEYS, v_map)},
    }


def preds_to_soft(preds: np.ndarray, scores: np.ndarray) -> np.ndarray:
    soft = np.zeros((len(preds), NUM_CLASSES), dtype=np.float32)
    for i, (p, s) in enumerate(zip(preds, scores)):
        cls = int(p)
        if 1 <= cls <= NUM_CLASSES:
            soft[i, cls - 1] = max(float(s), 1e-6)
        else:
            soft[i, :] = 1.0 / NUM_CLASSES
    soft /= np.maximum(soft.sum(axis=1, keepdims=True), 1e-8)
    return soft


def enrich_soft(soft: np.ndarray, scores: np.ndarray) -> np.ndarray:
    conf = scores.astype(np.float32).reshape(-1, 1)
    max_prob = soft.max(axis=1, keepdims=True)
    return np.concatenate([soft, conf, max_prob], axis=1)


class HopfieldLateFusion(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 64):
        super().__init__()
        self.proj_imu = nn.Linear(num_classes, hidden)
        self.proj_i3d = nn.Linear(num_classes, hidden)
        self.pool = HopfieldPooling(
            input_size=hidden,
            output_size=hidden,
            hidden_size=max(hidden // 2, 16),
            num_heads=1,
            scaling=0.1,
            update_steps_max=3,
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, imu_soft: torch.Tensor, i3d_soft: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack((self.proj_imu(imu_soft), self.proj_i3d(i3d_soft)), dim=1)
        return self.classifier(self.pool(tokens))


def _hp_kwargs(hidden: int) -> dict:
    return dict(
        input_size=hidden,
        output_size=hidden,
        hidden_size=max(hidden // 2, 32),
        num_heads=1,
        dropout=0.1,
        scaling=0.1,
        update_steps_max=3,
    )


class TriDetDoubleHopfieldFusion(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        hp_kw = _hp_kwargs(hidden)
        self.proj_imu = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.proj_i3d = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.hp_time_imu = HopfieldPooling(**hp_kw)
        self.hp_time_i3d = HopfieldPooling(**hp_kw)
        self.hp_fusion = HopfieldPooling(**hp_kw)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))

    def forward(self, imu_seq: torch.Tensor, i3d_seq: torch.Tensor) -> torch.Tensor:
        s_imu = self.hp_time_imu(self.proj_imu(imu_seq))
        s_i3d = self.hp_time_i3d(self.proj_i3d(i3d_seq))
        fused = self.hp_fusion(torch.stack((s_imu, s_i3d), dim=1))
        return self.classifier(fused)


@torch.no_grad()
def infer_single_hp(
    ckpt_path: Path,
    imu_soft: np.ndarray,
    i3d_soft: np.ndarray,
    is_val: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = HopfieldLateFusion().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    preds = np.zeros(len(is_val), dtype=np.int64)
    va = np.where(is_val)[0]
    bs = 4096
    for start in range(0, len(va), bs):
        idx = va[start : start + bs]
        logits = model(
            torch.tensor(imu_soft[idx], dtype=torch.float32, device=device),
            torch.tensor(i3d_soft[idx], dtype=torch.float32, device=device),
        )
        preds[idx] = logits.argmax(dim=1).cpu().numpy() + 1
    return preds


@torch.no_grad()
def infer_double_hp(
    ckpt_path: Path,
    imu_feat: np.ndarray,
    i3d_feat: np.ndarray,
    subject_ids: np.ndarray,
    val_subjects: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden = int(ckpt.get("hidden", 128))
    seq_len = int(ckpt.get("seq_len", 60))
    half = seq_len // 2
    model = TriDetDoubleHopfieldFusion(in_dim=imu_feat.shape[1], hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_frames = len(subject_ids)
    preds = np.zeros(n_frames, dtype=np.int64)
    filled = np.zeros(n_frames, dtype=bool)
    for sid in val_subjects:
        idx = np.where(subject_ids == sid)[0]
        if len(idx) < seq_len:
            continue
        imu_s = imu_feat[idx]
        i3d_s = i3d_feat[idx]
        batch_imu, batch_i3d, batch_centers = [], [], []
        batch_size = 512
        last_start = len(idx) - seq_len
        for start in range(0, last_start + 1):
            end = start + seq_len
            batch_imu.append(imu_s[start:end])
            batch_i3d.append(i3d_s[start:end])
            batch_centers.append(int(idx[start + half]))
            if len(batch_imu) == batch_size or start == last_start:
                logits = model(
                    torch.tensor(np.stack(batch_imu), dtype=torch.float32, device=device),
                    torch.tensor(np.stack(batch_i3d), dtype=torch.float32, device=device),
                )
                pred = logits.argmax(dim=1).cpu().numpy() + 1
                for frame_idx, p in zip(batch_centers, pred):
                    preds[frame_idx] = p
                    filled[frame_idx] = True
                batch_imu, batch_i3d, batch_centers = [], [], []

        for local_i, frame_idx in enumerate(idx):
            if filled[frame_idx]:
                continue
            left = idx[:local_i][filled[idx[:local_i]]]
            right = idx[local_i + 1 :][filled[idx[local_i + 1 :]]]
            if len(left):
                preds[frame_idx] = preds[left[-1]]
            elif len(right):
                preds[frame_idx] = preds[right[0]]
            else:
                preds[frame_idx] = 0
            filled[frame_idx] = True
    return preds


def mean_std(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-hopfield", action="store_true")
    ap.add_argument("--skip-fusion-recompute", action="store_true")
    ap.add_argument(
        "--mv-kernel",
        type=int,
        default=500,
        help="Majority-vote kernel for frame→segment mAP on Hopfield/oracle "
        "(0=off). Score-pick A&D already used MV=500 when saved. Default 500.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-folds", type=int, default=0, help="Debug: limit folds (0=all)")
    ap.add_argument(
        "--imu-npz-name",
        default="ad_frames.npz",
        help="Per-fold IMU frame export (ad_frames.npz or imu_frames.npz).",
    )
    ap.add_argument(
        "--imu-log-subdir",
        default="attendanddiscriminate",
        help="Folder containing the IMU training log.txt.",
    )
    args = ap.parse_args()
    out_root: Path = args.out_root
    device = torch.device(args.device)
    print(f"Device: {device} | MV kernel: {args.mv_kernel}")

    methods = [
        "ad",
        "i3d",
        "score_pick",
        "single_hp",
        "double_hp",
        "oracle",
    ]
    rows: list[dict] = []
    folds = sorted(out_root.glob("seed_*/split_*"))
    if args.max_folds > 0:
        folds = folds[: args.max_folds]

    for fold_dir in folds:
        seed = int(fold_dir.parent.name.replace("seed_", ""))
        split = int(fold_dir.name.replace("split_", ""))
        val_sbj = split - 1
        ann = ANN_DIR / f"wear_split_{split}.json"
        print(f"\n=== seed {seed} split {split} (sbj_{val_sbj}) ===")

        row: dict = {
            "Seed": seed,
            "Split": split,
            "ValSubject": f"sbj_{val_sbj}",
        }

        imu_log = parse_final_map(fold_dir / args.imu_log_subdir / "log.txt")
        i3d_log = parse_final_map(fold_dir / "tridet_i3d" / "log.txt")
        if imu_log:
            prefix = "ad" if args.imu_log_subdir == "attendanddiscriminate" else "imu"
            for k, v in imu_log.items():
                row[f"{prefix}_{k}"] = v
        if i3d_log:
            for k, v in i3d_log.items():
                row[f"i3d_{k}"] = v

        if not args.skip_fusion_recompute:
            fp = fold_dir / "score_oracle_fusion" / "fusion_preds.npz"
            z = np.load(fp)
            mask = z["is_val"]
            ids = z["subject_ids"][mask]
            for name, key in [
                ("score_pick", "score_pick"),
                ("ad_frames", "ad_preds"),
                ("i3d_frames", "i3d_preds"),
            ]:
                m = map_from_frame_preds(ids, z[key][mask], ann)
                prefix = "score_pick" if name == "score_pick" else name
                for k, v in m.items():
                    row[f"{prefix}_{k}"] = v
                print(f"  {prefix} mAP avg {m['mAP_avg']:.2f}")

            # Oracle = GT-correct pick between A&D and I3D (same as combine_predictions)
            oracle_preds = combine_predictions(
                z["ad_preds"][mask].astype(int),
                z["i3d_preds"][mask].astype(int),
                z["gt"][mask].astype(int),
            )
            m = map_from_frame_preds(ids, oracle_preds, ann, mv_kernel=args.mv_kernel)
            for k, v in m.items():
                row[f"oracle_{k}"] = v
            print(f"  oracle mAP avg {m['mAP_avg']:.2f}")

        if not args.skip_hopfield:
            imu_npz = fold_dir / args.imu_npz_name
            if not imu_npz.is_file():
                imu_npz = fold_dir / "imu_frames.npz"
            i3d_npz = fold_dir / "i3d_frames.npz"
            if not i3d_npz.is_file():
                alt = (
                    P.SIMA_BF1
                    / fold_dir.parent.name
                    / fold_dir.name
                    / "i3d_frames.npz"
                )
                if alt.is_file():
                    i3d_npz = alt
            imu = np.load(imu_npz)
            i3d = np.load(i3d_npz)
            imu_soft = preds_to_soft(imu["preds"], imu["scores"])
            i3d_soft = preds_to_soft(i3d["preds"], i3d["scores"])
            is_val = imu["is_val"]
            subject_ids = imu["subject_ids"]
            val_subjects = imu["val_subjects"]

            single_ckpt = fold_dir / "hopfield_late_fusion" / "model.pt"
            single_preds = infer_single_hp(single_ckpt, imu_soft, i3d_soft, is_val, device)
            m = map_from_frame_preds(
                subject_ids[is_val], single_preds[is_val], ann, mv_kernel=args.mv_kernel
            )
            for k, v in m.items():
                row[f"single_hp_{k}"] = v
            print(f"  single_hp mAP avg {m['mAP_avg']:.2f}")

            imu_feat = enrich_soft(imu_soft, imu["scores"])
            i3d_feat = enrich_soft(i3d_soft, i3d["scores"])
            double_ckpt = fold_dir / "hopfield_double_fusion" / "model.pt"
            double_preds = infer_double_hp(
                double_ckpt, imu_feat, i3d_feat, subject_ids, val_subjects, device
            )
            m = map_from_frame_preds(
                subject_ids[is_val], double_preds[is_val], ann, mv_kernel=args.mv_kernel
            )
            for k, v in m.items():
                row[f"double_hp_{k}"] = v
            print(f"  double_hp mAP avg {m['mAP_avg']:.2f}")

        rows.append(row)

    # CSV
    fieldnames = ["Seed", "Split", "ValSubject"]
    for method in methods:
        fieldnames += [f"{method}_acc", f"{method}_f1"] if method in ("ad", "i3d") else []
        fieldnames += [f"{method}_mAP_{k}" for k in TIOU_KEYS] + [f"{method}_mAP_avg"]
    # also keep frame-recomputed baselines
    for extra in ("ad_frames", "i3d_frames"):
        fieldnames += [f"{extra}_mAP_{k}" for k in TIOU_KEYS] + [f"{extra}_mAP_avg"]
    # unique preserve order
    seen = set()
    ordered = []
    for f in fieldnames:
        if f not in seen:
            # only keep columns that appear
            if any(f in r for r in rows) or f in ("Seed", "Split", "ValSubject"):
                ordered.append(f)
                seen.add(f)
    # dynamic: all keys present
    all_keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    # stable preferred order first
    csv_fields = [k for k in ordered if k in all_keys] + [k for k in all_keys if k not in seen]

    csv_path = out_root / "results_ad_fusion_map_loso.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["Seed"], x["Split"])):
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")

    # Markdown (paper-style overall table)
    def collect(prefix: str, key: str) -> list[float]:
        return [float(r[f"{prefix}_{key}"]) for r in rows if f"{prefix}_{key}" in r]

    md_methods = [
        ("A&D / IMU (log)", "ad"),
        ("TriDet IMU (log)", "imu"),
        ("TriDet I3D (log)", "i3d"),
        ("A&D (frames→segs)", "ad_frames"),
        ("I3D (frames→segs)", "i3d_frames"),
        ("Score-pick", "score_pick"),
        ("Single HP", "single_hp"),
        ("Double HP", "double_hp"),
        ("Oracle", "oracle"),
    ]
    lines = [
        "# A&D × TriDet I3D — LOSO mAP (WEAR paper metric)",
        "",
        f"**Root:** `{out_root}`",
        "",
        "Segment mAP at tIoU thresholds **0.3 / 0.4 / 0.5 / 0.6 / 0.7** and **Avg** "
        "(mean of the five), matching WEAR / `dataset/3699776.pdf`.",
        "",
        f"Folds: **{len(rows)}** (seed × split). "
        f"Frame→segment methods use majority-vote kernel **{args.mv_kernel}** "
        "(~10 s @ 50 Hz), matching WEAR inertial postprocessing / our score-pick export.",
        "",
        "## Overall (mean ± std)",
        "",
        "| Method | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | **Avg** |",
        "|--------|--------:|--------:|--------:|--------:|--------:|--------:|",
    ]
    for label, prefix in md_methods:
        vals = {k: collect(prefix, f"mAP_{k}") for k in TIOU_KEYS}
        avg = collect(prefix, "mAP_avg")
        if not avg:
            continue
        cells = [mean_std(vals[k]) for k in TIOU_KEYS] + [f"**{mean_std(avg)}**"]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Notes",
        "",
        "- **A&D / TriDet (log):** FINAL Avg. mAP from training logs (online TAL eval; A&D usually low without MV).",
        "- **A&D/I3D frames→segs:** from `fusion_preds.npz` (A&D already MV=500 in that export; I3D raw frames).",
        "- **Score-pick / Oracle / Hopfield:** frame preds → segments via `ANETdetection`. "
        "Hopfield re-inferred from `model.pt`, then MV applied before mAP.",
        "- TriDet **log** mAP is the fair camera number (native segments). Frame-rasterized I3D mAP is lower.",
        "- Frame Acc/F1 remain in `RESULTS_AD_FUSION.md`; this file is the paper-aligned **mAP** view.",
        "",
        f"CSV: `{csv_path.name}`",
        "",
    ]
    md_path = out_root / "RESULTS_AD_FUSION_MAP.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
