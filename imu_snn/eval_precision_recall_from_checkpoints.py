#!/usr/bin/env python3
"""
Recompute macro precision / recall / F1 from saved model.pt checkpoints (no retraining).

Loads validation windows with the same protocol as training (annotation JSON,
per-channel normalization from train split, 10 s majority vote), runs a forward
pass, and writes per-run metrics plus an aggregate CSV.

Usage:
  python3 imu_snn/eval_precision_recall_from_checkpoints.py \\
    --out-root "$NWEAR_OUTPUT/imu_snn/loso54" --model-type egru
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch
import torch.nn.functional as F

from pipeline_inertial import (
    ACTIVITY_CLASSES,
    NULL_CLASS_NAME,
    EGRUClassifier,
    detect_window_params,
    get_split_subjects,
    labels_from_json,
    load_npy,
    macro_f1_score,
    majority_vote_filter,
    normalize_per_channel,
)


def macro_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> tuple[float, float, float]:
    """Unweighted macro precision, recall, and F1 over class indices."""
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precs.append(float(p))
        recs.append(float(r))
        if p + r == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * p * r / (p + r))
    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def ann_path_for_run(ann_dir: Path, kind: str, split: int) -> Path:
    if kind == "train":
        return ann_dir / f"wear_split_{split}.json"
    return ann_dir / f"wear_test_split_{split}.json"


def load_val_tensors(
    feat_dir: Path,
    ann_dir: Path,
    kind: str,
    split: int,
    *,
    include_null: bool,
    min_overlap: float,
    imu_hz: int | None,
    video_fps: int | None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict]:
    wp = detect_window_params(feat_dir, imu_hz, video_fps)
    T_imu = wp["T_imu"]
    ann_path = ann_path_for_run(ann_dir, kind, split)
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    class_names = list(ACTIVITY_CLASSES)
    null_label_idx: int | None = None
    if include_null:
        null_label_idx = len(class_names)
        class_names = class_names + [NULL_CLASS_NAME]
    num_classes = len(class_names)

    npy_files = sorted(
        feat_dir.glob("sbj_*.npy"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    subject_ids = [int(p.stem.split("_")[1]) for p in npy_files]
    train_sids, val_sids = get_split_subjects(ann_path)
    sid_set = set(subject_ids)
    train_sids = [s for s in train_sids if s in sid_set]
    val_sids = [s for s in val_sids if s in sid_set]

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_sid: list[np.ndarray] = []
    in_features: int | None = None

    for sid in subject_ids:
        X, F = load_npy(feat_dir / f"sbj_{sid}.npy", T_imu)
        if in_features is None:
            in_features = F
        elif F != in_features:
            continue
        y = labels_from_json(
            ann_path, sid, len(X),
            wp["window_sec"], wp["stride_sec"],
            min_overlap=min_overlap,
        )
        if include_null:
            y_proc = np.where(y >= 0, y, null_label_idx)
            mask = np.ones_like(y_proc, dtype=bool)
        else:
            y_proc = y
            mask = y_proc >= 0
        all_X.append(X[mask])
        all_y.append(y_proc[mask])
        all_sid.append(np.full(int(mask.sum()), sid, dtype=np.int64))

    all_X = np.concatenate(all_X)
    all_y = np.concatenate(all_y)
    all_sid = np.concatenate(all_sid)

    tr_mask = np.isin(all_sid, train_sids)
    va_mask = np.isin(all_sid, val_sids)
    x_tr_raw, y_tr_raw = all_X[tr_mask], all_y[tr_mask]
    x_va_raw, y_va_raw = all_X[va_mask], all_y[va_mask]
    sid_val = all_sid[va_mask]

    x_tr_n, mean, std = normalize_per_channel(x_tr_raw)
    x_va_n, _, _ = normalize_per_channel(x_va_raw, mean, std)

    meta = {
        "wp": wp,
        "num_classes": num_classes,
        "in_features": in_features,
        "ann_path": ann_path,
        "val_subjects": val_sids,
    }
    return (
        torch.tensor(x_va_n, dtype=torch.float32),
        y_va_raw,
        sid_val,
        meta,
    )


@torch.no_grad()
def predict_egru(model: EGRUClassifier, x_val: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(x_val.to(device))
    pred = logits.mean(dim=-1).argmax(dim=1).detach().cpu()
    return np.asarray(pred, dtype=np.int64)


def eval_run_dir(
    run_dir: Path,
    *,
    model_type: str,
    feat_dir: Path,
    ann_dir: Path,
    device: torch.device,
    postprocess_sec: float,
    include_null: bool,
    imu_hz: int | None,
    video_fps: int | None,
) -> dict[str, str] | None:
    ckpt = run_dir / "model.pt"
    if not ckpt.is_file():
        return None

    m = re.match(r"seed_(\d+)/(train|test)_split_(\d+)$", "/".join(run_dir.parts[-2:]))
    if not m:
        return None
    seed, kind, split = m.group(1), m.group(2), int(m.group(3))

    x_val, y_val, sid_val, meta = load_val_tensors(
        feat_dir, ann_dir, kind, split,
        include_null=include_null,
        min_overlap=0.6,
        imu_hz=imu_hz,
        video_fps=video_fps,
    )
    num_classes = meta["num_classes"]
    in_features = meta["in_features"]
    wp = meta["wp"]

    hidden = 128
    model = EGRUClassifier(
        input_size=in_features,
        hidden_size=hidden,
        num_classes=num_classes,
    )
    pred_fn = predict_egru

    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    pred = pred_fn(model, x_val, device)
    if postprocess_sec > 0:
        pred = majority_vote_filter(pred, sid_val, wp["stride_sec"], postprocess_sec)

    acc = float((pred == y_val).mean())
    macro_p, macro_r, macro_f1 = macro_precision_recall(y_val, pred, num_classes)
    f1_check = macro_f1_score(y_val, pred, num_classes=num_classes)

    return {
        "Seed": seed,
        "Run": f"{kind}_split_{split}",
        "ValSubject": f"Val   subjects : {meta['val_subjects']}",
        "Accuracy": f"{acc:.3f}",
        "MacroPrecision": f"{macro_p:.3f}",
        "MacroRecall": f"{macro_r:.3f}",
        "MacroF1": f"{macro_f1:.3f}",
        "MacroF1_check": f"{f1_check:.3f}",
    }


def collect_runs(out_root: Path, model_type: str, **kwargs) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed_dir in sorted(out_root.glob("seed_*")):
        for kind, nmax in (("train", 18), ("test", 6)):
            for i in range(1, nmax + 1):
                run_dir = seed_dir / f"{kind}_split_{i}"
                print(f"  {run_dir.relative_to(out_root)} ...", flush=True)
                row = eval_run_dir(run_dir, model_type=model_type, **kwargs)
                if row:
                    rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = [
        "Seed", "Run", "ValSubject",
        "Accuracy", "MacroPrecision", "MacroRecall", "MacroF1", "MacroF1_check",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (int(x["Seed"]), x["Run"])):
            w.writerow(r)


def print_means(rows: list[dict[str, str]], label: str) -> None:
    if not rows:
        print(f"{label}: no rows")
        return
    n = len(rows)
    for key in ("Accuracy", "MacroPrecision", "MacroRecall", "MacroF1"):
        mean = sum(float(r[key]) for r in rows) / n
        print(f"  {key}: {mean * 100:.2f}%")
    # sanity: F1 from PR vs existing macro_f1
    diff = [
        abs(float(r["MacroF1"]) - float(r["MacroF1_check"]))
        for r in rows
    ]
    print(f"  max |MacroF1 - MacroF1_check|: {max(diff):.6f}")


def main() -> None:
    imu_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Cluster output root (seed_*/train_split_* / test_split_*)",
    )
    p.add_argument(
        "--model-type",
        choices=("egru",),
        default="egru",
    )
    p.add_argument(
        "--feat-dir",
        type=Path,
        default=P.IMU_FEAT_DIR,
    )
    p.add_argument(
        "--ann-dir",
        type=Path,
        default=P.ANN_DIR,
    )
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--postprocess-majority-sec", type=float, default=10.0)
    p.add_argument("--imu-hz", type=int, default=50)
    p.add_argument("--video-fps", type=int, default=60)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    out_root = args.out_root.expanduser().resolve()
    if not out_root.is_dir():
        raise SystemExit(f"out-root not found: {out_root}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    csv_out = args.csv_out or (
        out_root / f"results_precision_recall_{args.model_type}_2s.csv"
    )

    print(f"Evaluating {out_root.name} ({args.model_type}) on {device} ...")
    rows = collect_runs(
        out_root,
        model_type=args.model_type,
        feat_dir=args.feat_dir.resolve(),
        ann_dir=args.ann_dir.resolve(),
        device=device,
        postprocess_sec=args.postprocess_majority_sec,
        include_null=True,
        imu_hz=args.imu_hz,
        video_fps=args.video_fps,
    )
    write_csv(rows, csv_out)
    print(f"Wrote {len(rows)} rows -> {csv_out}")
    print("Means (72 runs):")
    print_means(rows, out_root.name)


if __name__ == "__main__":
    main()
