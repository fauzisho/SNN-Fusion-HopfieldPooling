"""Stratified train/val/test split for single-subject pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from event_snn.dataset.labels import label_from_h5_path


@dataclass(frozen=True)
class SplitIndex:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def discover_h5_files(dataset_root: Path) -> list[Path]:
    files = sorted(dataset_root.rglob("*.h5"))
    labeled = [p for p in files if label_from_h5_path(str(p)) is not None]
    if not labeled:
        raise FileNotFoundError(f"No labeled H5 files under {dataset_root}")
    return labeled


def stratified_split(
    files: list[Path],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitIndex:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Path]] = {}
    for path in files:
        label = label_from_h5_path(str(path))
        if label is None:
            continue
        by_label.setdefault(label, []).append(path)

    train, val, test = [], [], []
    for label_paths in by_label.values():
        paths = list(label_paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = max(0, n - n_train)
        train.extend(paths[:n_train])
        val.extend(paths[n_train : n_train + n_val])
        test.extend(paths[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitIndex(train=train, val=val, test=test)
