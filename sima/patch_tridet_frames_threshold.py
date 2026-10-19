#!/usr/bin/env python3
"""Apply the WEAR score threshold to already-exported TriDet frame NPZs.

``infer_tridet_frames.py`` used to default to ``--score-threshold 0.0``. TriDet
emits up to 2000 segments per video, so at 0.0 every frame is covered by some
segment and the NULL class is never predicted (WEAR, Sec. 4.2). Re-running the
GPU export is unnecessary: ``convert_segments_to_samples`` paints segments in
ascending score order, so a frame's stored score is the score of the single
highest-scoring segment covering it. Dropping segments at or below the
threshold therefore leaves a frame untouched when its score is above the
threshold, and empties it otherwise. The same argument applies per class to
``class_scores``, which holds a per-class max over segments.

The rewrite is idempotent and atomic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

FIELDS_REQUIRED = ("preds", "scores", "class_scores", "subject_ids", "is_val")


def patch_file(path: Path, threshold: float, dry_run: bool) -> tuple[str, float, float]:
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}

    if "score_threshold" in data and float(data["score_threshold"]) >= threshold:
        return "already-patched", float("nan"), float("nan")

    missing = [f for f in FIELDS_REQUIRED if f not in data]
    if missing:
        raise KeyError(f"{path}: missing {missing}")

    is_val = data["is_val"].astype(bool)
    before = float((data["preds"][is_val] == 0).mean() * 100.0)

    drop = data["scores"] <= threshold
    data["preds"] = np.where(drop, 0, data["preds"]).astype(np.int64)
    data["scores"] = np.where(drop, 0.0, data["scores"]).astype(np.float32)
    cs = data["class_scores"]
    data["class_scores"] = np.where(cs <= threshold, 0.0, cs).astype(np.float32)

    after = float((data["preds"][is_val] == 0).mean() * 100.0)

    # The reference implementation falls back to the unthresholded segments if
    # thresholding would leave a subject with nothing at all. Surface that.
    for sbj in np.unique(data["subject_ids"][is_val]):
        sel = is_val & (data["subject_ids"] == sbj)
        if bool((data["preds"][sel] == 0).all()):
            return f"REFUSED (subject {int(sbj)} emptied)", before, after

    if not dry_run:
        data["score_threshold"] = np.float32(threshold)
        # np.savez_compressed appends '.npz' unless the name already ends in it,
        # so the staging name has to keep that extension last.
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)

    return "patched", before, after


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.1)
    # Only TriDet exports may be thresholded. ad_frames.npz comes from the
    # Attend-and-Discriminate window classifier, whose per-frame scores are
    # softmax probabilities, not segment confidences -- masking those at 0.1
    # would silently delete valid low-confidence predictions.
    ap.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Repeatable. Defaults to the TriDet i3d/imu frame exports only.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patterns = args.glob or [
        "seed_*/split_*/i3d_frames.npz",
        "seed_*/split_*/imu_frames.npz",
    ]

    targets: list[Path] = []
    seen: set[Path] = set()
    for root in args.roots:
        for pattern in patterns:
            for p in sorted(root.glob(pattern)):
                real = p.resolve()
                if real.name == "ad_frames.npz":
                    raise SystemExit(f"refusing to threshold A&D export: {real}")
                if real in seen:
                    continue
                seen.add(real)
                targets.append(real)

    print(f"{len(targets)} unique NPZ files; threshold {args.threshold}")
    refused = 0
    for p in targets:
        status, before, after = patch_file(p, args.threshold, args.dry_run)
        if status.startswith("REFUSED"):
            refused += 1
        rel = p.relative_to(p.parents[3])
        if status == "already-patched":
            print(f"  skip     {rel}")
        else:
            print(f"  {status:8s} {rel}  NULL% {before:5.2f} -> {after:5.2f}")

    print("refused:", refused)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
