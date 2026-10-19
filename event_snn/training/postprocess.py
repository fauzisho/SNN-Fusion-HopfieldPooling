"""Evaluation post-processing for temporal smoothing."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def apply_majority_vote(
    segment_keys: list[str],
    center_us: list[int],
    preds: list[int],
    labels: list[int],
    k: int,
) -> tuple[list[int], list[int]]:
    """Sliding majority vote within each activity segment (IMU-style)."""
    if k <= 1:
        return list(preds), list(labels)

    groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for seg, center, pred, label in zip(segment_keys, center_us, preds, labels):
        groups[seg].append((center, pred, label))

    voted_preds: list[int] = []
    voted_labels: list[int] = []
    for seg in sorted(groups):
        items = sorted(groups[seg], key=lambda row: row[0])
        seg_preds = [row[1] for row in items]
        seg_labels = [row[2] for row in items]
        for i in range(len(seg_preds)):
            chunk = seg_preds[i : i + k]
            chunk_labels = seg_labels[i : i + k]
            counts = np.bincount(chunk, minlength=18)
            voted_preds.append(int(np.argmax(counts)))
            voted_labels.append(int(chunk_labels[len(chunk_labels) // 2]))

    return voted_preds, voted_labels


def macro_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
