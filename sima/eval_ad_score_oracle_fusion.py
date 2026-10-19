#!/usr/bin/env python3
"""Score-pick + oracle late fusion for Attend-and-Discriminate IMU + TriDet I3D.

No learned fusion head — only the two reference late-fusion baselines used in the
Hopfield comparison ladder.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import nwear_paths as P

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

WEAR_REPO = P.WEAR_REPO
NUM_CLASSES = 18


def _load_combine_predictions():
    path = WEAR_REPO / "postprocessing" / "utils.py"
    spec = importlib.util.spec_from_file_location("wear_postprocessing_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.combine_predictions


combine_predictions = _load_combine_predictions()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true > 0
    y_true = y_true[mask] - 1
    y_pred = y_pred[mask] - 1
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=range(NUM_CLASSES))
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=range(NUM_CLASSES))
        ),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=range(NUM_CLASSES))),
        "n": int(len(y_true)),
    }


def score_pick_pred(
    imu_preds: np.ndarray,
    i3d_preds: np.ndarray,
    imu_scores: np.ndarray,
    i3d_scores: np.ndarray,
) -> np.ndarray:
    pick_imu = imu_scores >= i3d_scores
    return np.where(pick_imu, imu_preds, i3d_preds)


def majority_vote(preds: np.ndarray, subject_ids: np.ndarray, kernel: int) -> np.ndarray:
    """Per-subject majority filter (paper uses ~10s = 500 samples at 50 Hz)."""
    out = preds.copy()
    half = kernel // 2
    for sbj in np.unique(subject_ids):
        idx = np.where(subject_ids == sbj)[0]
        arr = preds[idx].astype(np.int64)
        smoothed = np.empty_like(arr)
        n = len(arr)
        for i in range(n):
            left = max(0, i - half)
            right = min(n, i + half + 1)
            # keep window size when near edges (same idea as wear majorityVoting)
            if right - left < kernel:
                if left == 0:
                    right = min(n, kernel)
                else:
                    left = max(0, n - kernel)
            smoothed[i] = np.bincount(arr[left:right]).argmax()
        out[idx] = smoothed
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A&D + TriDet I3D score-pick / oracle fusion.")
    ap.add_argument("--ad-npz", type=Path, required=True, help="A&D frame export (imu stream)")
    ap.add_argument("--i3d-npz", type=Path, required=True, help="TriDet camera frame export")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--mv-kernel",
        type=int,
        default=500,
        help="Majority-vote kernel on A&D preds before fusion (0 disables). Default 500 = 10s @ 50Hz.",
    )
    args = ap.parse_args()

    ad = np.load(args.ad_npz)
    i3d = np.load(args.i3d_npz)
    for key in ("preds", "gt", "scores", "is_val"):
        if len(ad[key]) != len(i3d[key]):
            raise SystemExit(f"A&D/I3D length mismatch on {key}: {len(ad[key])} vs {len(i3d[key])}")

    ad_preds = ad["preds"].astype(np.int64)
    if args.mv_kernel and args.mv_kernel > 1:
        ad_preds = majority_vote(ad_preds, ad["subject_ids"], args.mv_kernel)

    gt = ad["gt"]
    is_val = ad["is_val"]
    val_mask = is_val & (gt > 0)

    ad_m = metrics(gt[is_val], ad_preds[is_val])
    i3d_m = metrics(gt[is_val], i3d["preds"][is_val])

    score_pick = score_pick_pred(
        ad_preds,
        i3d["preds"],
        ad["scores"],
        i3d["scores"],
    )
    sp_m = metrics(gt[is_val], score_pick[is_val])

    oracle_val = combine_predictions(
        ad_preds[val_mask].astype(int),
        i3d["preds"][val_mask].astype(int),
        gt[val_mask].astype(int),
    )
    oracle_m = {
        "acc": float(accuracy_score(gt[val_mask] - 1, oracle_val - 1)),
        "f1": float(
            f1_score(
                gt[val_mask] - 1,
                oracle_val - 1,
                average="macro",
                zero_division=0,
                labels=range(NUM_CLASSES),
            )
        ),
        "n": int(val_mask.sum()),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        "Attend-and-Discriminate + TriDet I3D — score-pick / oracle late fusion\n"
        f"A&D npz:  {args.ad_npz}\n"
        f"I3D npz:  {args.i3d_npz}\n"
        f"MV kernel on A&D: {args.mv_kernel}\n"
        f"A&D (IMU) only:         acc={ad_m['acc']*100:.2f}%  F1={ad_m['f1']*100:.2f}%  (n={ad_m['n']})\n"
        f"TriDet I3D only:        acc={i3d_m['acc']*100:.2f}%  F1={i3d_m['f1']*100:.2f}%  (n={i3d_m['n']})\n"
        f"Score-pick (deployable): acc={sp_m['acc']*100:.2f}%  F1={sp_m['f1']*100:.2f}%  (n={sp_m['n']})\n"
        f"Oracle (GT ceiling):    acc={oracle_m['acc']*100:.2f}%  F1={oracle_m['f1']*100:.2f}%  [not deployable]\n"
    )
    (args.out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "fusion_preds.npz",
        ad_preds=ad_preds.astype(np.int64),
        i3d_preds=i3d["preds"].astype(np.int64),
        score_pick=score_pick.astype(np.int64),
        gt=gt.astype(np.int64),
        is_val=is_val.astype(np.bool_),
        subject_ids=ad["subject_ids"].astype(np.int64),
    )
    print(summary)


if __name__ == "__main__":
    main()
