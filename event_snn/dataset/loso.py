"""Multi-subject LOSO splits for DVS240 v2e H5 data."""

from __future__ import annotations

from pathlib import Path

from event_snn.dataset.splits import SplitIndex, discover_h5_files, stratified_split


def subject_events_dir(
    dataset_base: Path,
    subject_id: int,
    events_tier: str = "dvs240",
) -> Path:
    """Resolve subject H5 root; prefer ``*_new`` when present (e.g. refreshed sbj_16/17).

    Also searches ``dataset_base/validation/`` for held-out subjects (e.g. sbj_18+).
    """
    name = f"sbj_{subject_id}_events_{events_tier}"
    refreshed_name = f"{name}_new"
    for root in (dataset_base, dataset_base / "validation"):
        refreshed = root / refreshed_name
        if refreshed.is_dir():
            return refreshed
        base = root / name
        if base.is_dir():
            return base
    return dataset_base / name


def discover_subject_files(
    dataset_base: Path,
    subject_id: int,
    events_tier: str = "dvs240",
) -> list[Path]:
    return discover_h5_files(subject_events_dir(dataset_base, subject_id, events_tier))


def discover_multi_subject_files(
    dataset_base: Path,
    subject_ids: list[int],
    events_tier: str = "dvs240",
) -> list[Path]:
    files: list[Path] = []
    for sid in subject_ids:
        files.extend(discover_subject_files(dataset_base, sid, events_tier))
    return files


def loso_train_test_split(
    dataset_base: Path,
    *,
    train_subjects: list[int],
    test_subject: int,
    events_tier: str = "dvs240",
    val_ratio: float = 0.10,
    val_on_held_out: bool = False,
    seed: int = 1,
) -> tuple[SplitIndex, list[int], int]:
    """
    LOSO split for train_subjects vs held-out test_subject.

    val_on_held_out=False (legacy): stratified val holdout from train_subjects only.
    val_on_held_out=True (paper/IMU style): val = all clips from test_subject.
    """
    train_pool = discover_multi_subject_files(dataset_base, train_subjects, events_tier)
    test_files = discover_subject_files(dataset_base, test_subject, events_tier)
    if not train_pool:
        raise FileNotFoundError(f"No train files for subjects {train_subjects}")
    if not test_files:
        raise FileNotFoundError(f"No test files for subject {test_subject}")

    if val_on_held_out:
        return (
            SplitIndex(train=train_pool, val=test_files, test=test_files),
            train_subjects,
            test_subject,
        )

    train_val = stratified_split(
        train_pool,
        train_ratio=1.0 - val_ratio,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
    )
    return (
        SplitIndex(train=train_val.train, val=train_val.val, test=test_files),
        train_subjects,
        test_subject,
    )
