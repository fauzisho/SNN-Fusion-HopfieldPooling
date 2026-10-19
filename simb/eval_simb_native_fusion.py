#!/usr/bin/env python3
"""SIM B fusion on the native inertial grid.

For each leave-one-subject-out fold, score the frozen event SNN, IMU SNN,
score-pick, single Hopfield, double Hopfield, and oracle on the same 2 s
windows used by the native IMU evaluation (stride 1 s, activity windows
only). An event window is the 2 s of events centered on that inertial
window. Windows whose 2 s event crop does not fit inside an event segment
are left out. The 10 s vote stays inside a contiguous run of scored windows
and does not cross a rest gap.

The double Hopfield head was trained on five tokens spaced 2 s apart, so
each sequence is the windows at -4, -2, 0, +2, +4 s on this 1 s grid.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
import nwear_paths as P

import h5py
import numpy as np
import torch

P.add_to_path(P.SIMB_DIR, P.IMU_SNN_DIR)

from dvs_imu_fusion_data import (  # noqa: E402
    NUM_CLASSES,
    frame_features,
    infer_imu_subject,
    load_dvs_checkpoint,
    load_imu_checkpoint,
    load_imu_train_norm,
    parse_segment_start_sec,
)
from event_snn.dataset.representation import (  # noqa: E402
    events_to_mycount_tensor,
    load_h5_events,
)
from event_snn.dataset.splits import discover_h5_files  # noqa: E402
from event_snn.dataset.window_index import _group_by_segment  # noqa: E402
from pipeline_inertial import detect_window_params, get_split_subjects, load_npy  # noqa: E402
from train_hopfield_dvs_imu_fusion import (  # noqa: E402
    DoubleHopfieldDVSIMU,
    SingleHopfieldDVSIMU,
    oracle_pick,
    score_pick,
)

WINDOW_SEC = 2.0
STRIDE_SEC = 1.0
FILTER_SEC = 10.0
SAMPLE_US = 2_000_000


def macro_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precs, recs, f1s = [], [], []
    for c in range(NUM_CLASSES):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "Acc": acc * 100.0,
        "P": float(np.mean(precs)) * 100.0,
        "R": float(np.mean(recs)) * 100.0,
        "F1": float(np.mean(f1s)) * 100.0,
    }


def vote_spans(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = pred.copy()
    half = int(round(FILTER_SEC / STRIDE_SEC)) // 2
    i, n = 0, len(pred)
    while i < n:
        while i < n and not mask[i]:
            i += 1
        if i == n:
            break
        j = i + 1
        while j < n and mask[j]:
            j += 1
        seq = pred[i:j]
        smooth = np.empty_like(seq)
        for k in range(len(seq)):
            lo, hi = max(0, k - half), min(len(seq), k + half + 1)
            smooth[k] = int(np.bincount(seq[lo:hi].astype(np.int64)).argmax())
        out[i:j] = smooth
        i = j
    return out


def index_segments(dataset_root: Path) -> list[dict]:
    files = discover_h5_files(dataset_root)
    segments = []
    for _key, clip_paths in sorted(_group_by_segment(files).items()):
        start = parse_segment_start_sec(str(clip_paths[0].parent))
        if start is None:
            continue
        clips = tuple(clip_paths)
        segments.append(
            {
                "start": float(start),
                "end": float(start) + len(clips) * WINDOW_SEC,
                "clips": clips,
            }
        )
    segments.sort(key=lambda item: item["start"])
    return segments


def load_segment_events(clips: tuple[Path, ...]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for idx, path in enumerate(clips):
        with h5py.File(path, "r") as handle:
            raw = np.asarray(handle["events"][:])
        events = load_h5_events(raw)
        if len(events):
            events = events.copy()
            events[:, 2] += idx * SAMPLE_US
        parts.append(events)
    if not parts:
        return np.empty((0, 4), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def window_start_us(center_sec: float, segment: dict) -> int | None:
    local_us = int(round((center_sec - segment["start"]) * 1_000_000.0))
    start_us = local_us - SAMPLE_US // 2
    duration_us = int(round((segment["end"] - segment["start"]) * 1_000_000.0))
    if start_us < 0 or start_us + SAMPLE_US > duration_us:
        return None
    return start_us


def find_segment(segments: list[dict], center_sec: float) -> dict | None:
    for segment in segments:
        if segment["start"] + 1.0 <= center_sec <= segment["end"] - 1.0:
            return segment
    return None


@torch.no_grad()
def encode_dvs(
    model: torch.nn.Module,
    events: np.ndarray,
    starts: list[int],
    data_cfg,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    out = np.empty((len(starts), NUM_CLASSES), dtype=np.float32)
    batch: list[np.ndarray] = []
    batch_at: list[int] = []
    for i, start_us in enumerate(starts):
        cropped = events[(events[:, 2] >= start_us) & (events[:, 2] < start_us + SAMPLE_US)].copy()
        if len(cropped):
            cropped[:, 2] -= start_us
        tensor = events_to_mycount_tensor(
            cropped,
            sensor_height=data_cfg.sensor_height,
            sensor_width=data_cfg.sensor_width,
            ds_factor=data_cfg.ds_factor,
            model_width=data_cfg.model_width,
            sampling_time_us=data_cfg.sampling_time_ms * 1000,
            sample_length_us=SAMPLE_US,
            time_bins=data_cfg.time_bins,
        )
        batch.append(tensor)
        batch_at.append(i)
        if len(batch) == batch_size or i == len(starts) - 1:
            x = torch.from_numpy(np.stack(batch)).to(device)
            logits = model(x)
            if logits.dim() == 3:
                logits = logits.mean(dim=-1)
            probs = torch.softmax(logits, dim=1)[:, :NUM_CLASSES].cpu().numpy()
            out[np.asarray(batch_at)] = probs
            batch, batch_at = [], []
    return out


@torch.no_grad()
def hopfield_preds(
    single_path: Path,
    double_path: Path,
    dvs_feat: np.ndarray,
    imu_feat: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(valid)
    single_pred = np.full(n, -1, dtype=np.int64)
    double_pred = np.full(n, -1, dtype=np.int64)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return single_pred, double_pred

    single = SingleHopfieldDVSIMU().to(device)
    single.load_state_dict(torch.load(single_path, map_location=device, weights_only=False))
    single.eval()
    logits = single(
        torch.tensor(dvs_feat[idx], dtype=torch.float32, device=device),
        torch.tensor(imu_feat[idx], dtype=torch.float32, device=device),
    )
    single_pred[idx] = logits.argmax(dim=1).cpu().numpy()

    double = DoubleHopfieldDVSIMU().to(device)
    double.load_state_dict(torch.load(double_path, map_location=device, weights_only=False))
    double.eval()
    centers, seq_dvs, seq_imu = [], [], []
    for i in idx:
        offs = (i - 4, i - 2, i, i + 2, i + 4)
        if offs[0] < 0 or offs[-1] >= n or not valid[list(offs)].all():
            continue
        centers.append(int(i))
        seq_dvs.append(dvs_feat[list(offs)])
        seq_imu.append(imu_feat[list(offs)])
        if len(centers) == 256:
            _flush_double(double, centers, seq_dvs, seq_imu, double_pred, device)
            centers, seq_dvs, seq_imu = [], [], []
    if centers:
        _flush_double(double, centers, seq_dvs, seq_imu, double_pred, device)

    filled = np.flatnonzero(double_pred >= 0)
    if len(filled):
        for i in idx:
            if double_pred[i] >= 0:
                continue
            double_pred[i] = int(double_pred[filled[np.argmin(np.abs(filled - i))]])
    return single_pred, double_pred


def _flush_double(model, centers, seq_dvs, seq_imu, dest, device) -> None:
    logits = model(
        torch.tensor(np.stack(seq_dvs), dtype=torch.float32, device=device),
        torch.tensor(np.stack(seq_imu), dtype=torch.float32, device=device),
    )
    pred = logits.argmax(dim=1).cpu().numpy()
    for center, label in zip(centers, pred):
        dest[center] = int(label)


def score_fold(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    ann = args.ann_dir / f"wear_split_{args.val_subject + 1}.json"
    train_ids, val_ids = get_split_subjects(ann)
    if args.val_subject not in val_ids:
        raise SystemExit(f"sbj_{args.val_subject} is not the validation subject of {ann.name}")

    wp = detect_window_params(args.feat_dir)
    if abs(wp["window_sec"] - WINDOW_SEC) > 1e-6 or abs(wp["stride_sec"] - STRIDE_SEC) > 1e-6:
        raise SystemExit(f"Expected 2 s / 1 s windows, got {wp}")

    mean, std = load_imu_train_norm(
        args.feat_dir,
        ann,
        train_ids,
        T_imu=wp["T_imu"],
        window_sec=wp["window_sec"],
        stride_sec=wp["stride_sec"],
        min_overlap=0.6,
    )
    imu_model = load_imu_checkpoint(
        args.imu_ckpt,
        in_features=int(load_npy(args.feat_dir / f"sbj_{train_ids[0]}.npy", wp["T_imu"])[1]),
        hidden=args.imu_hidden,
        include_null=True,
        device=device,
    )
    imu = infer_imu_subject(
        imu_model,
        args.feat_dir,
        ann,
        args.val_subject,
        mean,
        std,
        min_overlap=0.6,
        device=device,
    )
    labels = imu.labels.astype(np.int64)
    times = imu.times.astype(np.float64)
    activity = labels >= 0
    imu_probs = imu.probs[:, :NUM_CLASSES]

    dvs_model, data_cfg, _train_cfg = load_dvs_checkpoint(args.dvs_ckpt, device)
    segments = index_segments(args.dataset_base / f"sbj_{args.val_subject}_events_dvs128")
    dvs_probs = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    valid = np.zeros(len(labels), dtype=bool)

    by_segment: dict[int, list[int]] = {}
    for i, (is_act, center) in enumerate(zip(activity, times)):
        if not is_act:
            continue
        segment = find_segment(segments, float(center))
        if segment is None:
            continue
        by_segment.setdefault(id(segment), []).append(i)
        segment.setdefault("_id", id(segment))

    seg_by_id = {id(segment): segment for segment in segments}
    for seg_id, rows in by_segment.items():
        segment = seg_by_id[seg_id]
        starts, keep = [], []
        for i in rows:
            start_us = window_start_us(float(times[i]), segment)
            if start_us is None:
                continue
            starts.append(start_us)
            keep.append(i)
        if not keep:
            continue
        print(f"  segment {Path(segment['clips'][0]).parent.name}: {len(keep)} windows", flush=True)
        events = load_segment_events(segment["clips"])
        probs = encode_dvs(dvs_model, events, starts, data_cfg, device, args.batch_size)
        dvs_probs[np.asarray(keep)] = probs
        valid[np.asarray(keep)] = True
        del events

    dvs_feat = frame_features(dvs_probs)
    imu_feat = frame_features(imu_probs)
    head_dir = args.head_root / f"seed_{args.seed}" / f"sbj_{args.val_subject}"
    single_pred, double_pred = hopfield_preds(
        head_dir / "single_hopfield.pt",
        head_dir / "double_hopfield.pt",
        dvs_feat,
        imu_feat,
        valid,
        device,
    )
    dvs_pred = dvs_probs.argmax(axis=1)
    imu_pred = imu_probs.argmax(axis=1)
    methods = {
        "dvs": dvs_pred,
        "imu": imu_pred,
        "score_pick": score_pick(dvs_feat, imu_feat),
        "single_hp": single_pred,
        "double_hp": double_pred,
        "oracle": oracle_pick(labels, dvs_pred, imu_pred),
    }
    y = labels[valid]
    row: dict = {
        "seed": args.seed,
        "sbj": args.val_subject,
        "n_windows": int(len(labels)),
        "n_activity": int(activity.sum()),
        "n_scored": int(valid.sum()),
    }
    for name, pred in methods.items():
        voted = vote_spans(pred.astype(np.int64), valid)
        raw = macro_prf(y, pred[valid].astype(np.int64))
        smooth = macro_prf(y, voted[valid].astype(np.int64))
        for key, value in raw.items():
            row[f"{name}_none_{key}"] = value
        for key, value in smooth.items():
            row[f"{name}_vote_{key}"] = value
    return row


def write_fold(row: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed_{row['seed']}_sbj_{row['sbj']:02d}.json"
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return path


def combine(fold_dir: Path, out_csv: Path) -> None:
    import csv

    rows = [json.loads(path.read_text()) for path in sorted(fold_dir.glob("seed_*_sbj_*.json"))]
    if len(rows) != 54:
        raise SystemExit(f"expected 54 folds, found {len(rows)} in {fold_dir}")
    n_scored = sum(int(row["n_scored"]) for row in rows)
    n_activity = sum(int(row["n_activity"]) for row in rows)
    print(f"FOLDS 54 SCORED {n_scored} ACTIVITY {n_activity} COVERAGE {100 * n_scored / n_activity:.2f}%")
    keys = [key for key in rows[0] if key.endswith(("Acc", "P", "R", "F1"))]
    for key in keys:
        vals = [float(row[key]) for row in rows]
        print(f"{key:28} {statistics.mean(vals):.2f}±{statistics.stdev(vals):.2f}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--val-subject", type=int)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--imu-hidden", type=int, default=128)
    ap.add_argument("--feat-dir", type=Path, default=P.IMU_FEAT_DIR)
    ap.add_argument("--ann-dir", type=Path, default=P.ANN_DIR)
    ap.add_argument("--dataset-base", type=Path, default=P.DATA_ROOT)
    ap.add_argument("--dvs-ckpt", type=Path)
    ap.add_argument("--imu-ckpt", type=Path)
    ap.add_argument("--head-root", type=Path, default=P.SIMB_HEADS)
    ap.add_argument("--out-dir", type=Path, default=P.SIMB_HEADS / "native_loso54" / "folds")
    ap.add_argument("--out-csv", type=Path, default=P.RESULTS_DIR / "results_simb_native_vote10.csv")
    ap.add_argument("--skip-if-done", action="store_true")
    args = ap.parse_args()
    if args.combine:
        combine(args.out_dir, args.out_csv)
        return
    if args.seed is None or args.val_subject is None:
        raise SystemExit("Need --seed and --val-subject, or --combine")
    out_path = args.out_dir / f"seed_{args.seed}_sbj_{args.val_subject:02d}.json"
    if args.skip_if_done and out_path.is_file():
        print(f"skip existing {out_path}")
        return
    print(f"native SIM B fusion seed={args.seed} sbj={args.val_subject} device={args.device}", flush=True)
    row = score_fold(args)
    path = write_fold(row, args.out_dir)
    print(
        f"scored {row['n_scored']}/{row['n_activity']} "
        f"double F1 {row['double_hp_none_F1']:.2f} -> {row['double_hp_vote_F1']:.2f}",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
