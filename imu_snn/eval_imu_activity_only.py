#!/usr/bin/env python3
"""Evaluate saved 19-output IMU checkpoints on activity windows only.

No weights are retrained. Validation windows are loaded exactly as in the
original 19-class run so that train-split normalisation is unchanged. At
evaluation time:

* ground-truth NULL windows are excluded;
* predictions use argmax over the 18 activity logits only;
* majority voting is applied separately inside each contiguous activity span,
  so removing NULL does not bridge two activities across an unlabeled gap;
* NULL positions remain separators when segments are assembled for mAP.

This is a conditional, activity-only diagnostic. It is not an online
activity-versus-NULL evaluation and must not replace the WEAR-matched 19-class
comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import nwear_paths as P

import numpy as np
import torch
import torch.nn.functional as F

from eval_precision_recall_from_checkpoints import (
    load_val_tensors,
    macro_precision_recall,
    parse_summary_hopfield,
)
from pipeline_inertial import (
    ACTIVITY_CLASSES,
    EGRUClassifier,
    macro_f1_score,
    temporal_map,
)
try:  # EGRU + Hopfield readout was a thesis variant; not shipped in this release
    from pipeline_inertial_hopfield import EGRUHopfieldClassifier
except ImportError:
    EGRUHopfieldClassifier = None


NUM_ACTIVITIES = len(ACTIVITY_CLASSES)
TRAINED_NUM_CLASSES = NUM_ACTIVITIES + 1
TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def majority_vote_inside_activity_spans(
    pred: np.ndarray,
    activity_mask: np.ndarray,
    sid: np.ndarray,
    *,
    stride_sec: float,
    filter_sec: float,
) -> np.ndarray:
    """Smooth predictions without voting across a removed NULL interval."""
    out = np.full(pred.shape, -1, dtype=np.int64)
    win = max(1, int(round(filter_sec / max(stride_sec, 1e-8))))
    half = win // 2
    for subject in np.unique(sid):
        idx = np.where(sid == subject)[0]
        mask = activity_mask[idx]
        start = 0
        while start < len(idx):
            while start < len(idx) and not mask[start]:
                start += 1
            if start == len(idx):
                break
            end = start + 1
            while end < len(idx) and mask[end]:
                end += 1
            seq = pred[idx[start:end]]
            smooth = np.empty_like(seq)
            for i in range(len(seq)):
                lo = max(0, i - half)
                hi = min(len(seq), i + half + 1)
                values, counts = np.unique(seq[lo:hi], return_counts=True)
                smooth[i] = values[np.argmax(counts)]
            out[idx[start:end]] = smooth
            start = end
    return out


def parse_run(run_dir: Path) -> tuple[int, int]:
    match = re.match(r"seed_(\d+)/train_split_(\d+)$", "/".join(run_dir.parts[-2:]))
    if not match:
        raise ValueError(f"Expected seed_N/train_split_N, got {run_dir}")
    return int(match.group(1)), int(match.group(2))


@torch.no_grad()
def evaluate_fold(args: argparse.Namespace) -> dict:
    run_dir = args.out_root / f"seed_{args.seed}" / f"train_split_{args.val_subject + 1}"
    checkpoint = run_dir / "model.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    seed, split = parse_run(run_dir)

    # Keep include_null=True here: this reproduces the original validation
    # tensor and train-split normalisation before the evaluation-only mask.
    x_val, y_full, sid, meta = load_val_tensors(
        args.feat_dir,
        args.ann_dir,
        "train",
        split,
        include_null=True,
        min_overlap=args.min_overlap,
        imu_hz=args.imu_hz,
        video_fps=args.video_fps,
    )
    if int(meta["num_classes"]) != TRAINED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected {TRAINED_NUM_CLASSES} trained classes, got {meta['num_classes']}"
        )

    if args.model_type == "hopfield":
        hp = parse_summary_hopfield(run_dir / "summary.txt")
        model = EGRUHopfieldClassifier(
            input_size=int(meta["in_features"]),
            hidden_size=int(hp["hidden"]),
            num_classes=TRAINED_NUM_CLASSES,
            hopfield_quantity=int(hp["hopfield_quantity"]),
            hopfield_scaling=hp["hopfield_scaling"],
            hopfield_update_steps=int(hp["hopfield_update_steps"]),
        )
    else:
        model = EGRUClassifier(
            input_size=int(meta["in_features"]),
            hidden_size=args.hidden,
            num_classes=TRAINED_NUM_CLASSES,
        )
    try:
        state = torch.load(checkpoint, map_location=args.device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device).eval()

    logits = model(x_val.to(args.device))
    if logits.ndim == 3:
        logits = logits.mean(dim=-1)
    activity_probs = F.softmax(logits[:, :NUM_ACTIVITIES], dim=1).cpu().numpy()
    pred_raw = activity_probs.argmax(axis=1).astype(np.int64)

    null_index = NUM_ACTIVITIES
    activity_mask = y_full != null_index
    y_timeline = np.where(activity_mask, y_full, -1).astype(np.int64)
    pred_voted = majority_vote_inside_activity_spans(
        pred_raw,
        activity_mask,
        sid,
        stride_sec=float(meta["wp"]["stride_sec"]),
        filter_sec=args.postprocess_majority_sec,
    )

    y_eval = y_timeline[activity_mask]
    pred_eval = pred_voted[activity_mask]
    accuracy = float(np.mean(y_eval == pred_eval))
    precision, recall, macro_f1 = macro_precision_recall(
        y_eval, pred_eval, NUM_ACTIVITIES
    )
    f1_check = macro_f1_score(y_eval, pred_eval, num_classes=NUM_ACTIVITIES)

    map_by_tiou, map_avg = temporal_map(
        y_true=y_timeline,
        y_pred=pred_voted,
        sid=sid,
        probs=activity_probs,
        num_classes=NUM_ACTIVITIES,
        window_sec=float(meta["wp"]["window_sec"]),
        stride_sec=float(meta["wp"]["stride_sec"]),
        tiou_thresholds=TIOU_THRESHOLDS,
        exclude_class=None,
    )

    result = {
        "protocol": "activity_only_eval_of_19_class_checkpoint",
        "model_type": args.model_type,
        "seed": seed,
        "val_subject": args.val_subject,
        "split": split,
        "checkpoint": str(checkpoint),
        "trained_num_classes": TRAINED_NUM_CLASSES,
        "eval_num_classes": NUM_ACTIVITIES,
        "null_windows_dropped": int((~activity_mask).sum()),
        "activity_windows": int(activity_mask.sum()),
        "total_windows": int(len(y_full)),
        "postprocess_majority_sec": args.postprocess_majority_sec,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "macro_f1_check": f1_check,
        "map_by_tiou": {str(k): v for k, v in map_by_tiou.items()},
        "map_avg": map_avg,
        "note": (
            "Evaluation-only: NULL ground-truth windows are separators and are not "
            "scored; argmax is restricted to the 18 activity logits. The checkpoint "
            "was trained with NULL and this is not an online NULL-rejection result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val-subject", type=int, required=True, choices=range(18))
    parser.add_argument(
        "--model-type", choices=("egru", "hopfield"), default="egru"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=P.IMU_RUNS,
    )
    parser.add_argument(
        "--feat-dir",
        type=Path,
        default=P.IMU_FEAT_DIR,
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=P.ANN_DIR,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--min-overlap", type=float, default=0.6)
    parser.add_argument("--postprocess-majority-sec", type=float, default=10.0)
    parser.add_argument("--imu-hz", type=int, default=50)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.output is None:
        args.output = (
            args.out_root
            / "activity_only_eval"
            / f"seed_{args.seed}"
            / f"sbj_{args.val_subject}.json"
        )

    result = evaluate_fold(args)
    print(
        f"seed={args.seed} sbj_{args.val_subject} "
        f"n={result['activity_windows']} dropped={result['null_windows_dropped']} "
        f"Acc={100 * result['accuracy']:.2f} "
        f"P={100 * result['macro_precision']:.2f} "
        f"R={100 * result['macro_recall']:.2f} "
        f"F1={100 * result['macro_f1']:.2f} "
        f"mAP={100 * result['map_avg']:.2f}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
