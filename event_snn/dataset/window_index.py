"""Build clip or sliding-window sample lists from DVS240 H5 paths."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from event_snn.dataset.labels import label_from_h5_path


@dataclass(frozen=True)
class WindowSample:
    clip_paths: tuple[Path, ...]
    start_us: int
    label: int
    segment_key: str
    center_us: int


def _group_by_segment(files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[str(path.parent)].append(path)
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


def build_window_samples(
    files: list[Path],
    *,
    sample_length_ms: int,
    stride_ms: int,
) -> list[WindowSample]:
    """
    Build training/eval windows from H5 clip paths.

    stride_ms <= 0: one window per H5 (legacy clip mode).
    stride_ms > 0: slide within each activity segment folder (concat adjacent clips).
    """
    sample_length_us = sample_length_ms * 1000
    stride_us = max(stride_ms, 0) * 1000
    clip_us = sample_length_us
    samples: list[WindowSample] = []

    for segment_key, clip_paths in sorted(_group_by_segment(files).items()):
        label = label_from_h5_path(str(clip_paths[0]))
        if label is None:
            continue
        paths = tuple(clip_paths)

        if stride_ms <= 0:
            for idx, path in enumerate(paths):
                start_us = idx * clip_us
                center_us = start_us + clip_us // 2
                samples.append(
                    WindowSample(
                        clip_paths=paths,
                        start_us=start_us,
                        label=label,
                        segment_key=segment_key,
                        center_us=center_us,
                    )
                )
            continue

        segment_us = len(paths) * clip_us
        if segment_us < sample_length_us:
            continue
        start = 0
        while start + sample_length_us <= segment_us:
            center_us = start + sample_length_us // 2
            samples.append(
                WindowSample(
                    clip_paths=paths,
                    start_us=start,
                    label=label,
                    segment_key=segment_key,
                    center_us=center_us,
                )
            )
            start += stride_us

    return samples
