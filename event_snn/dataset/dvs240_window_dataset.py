"""PyTorch dataset for DVS240 v2e clips and sliding segment windows."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from event_snn.config import DataConfig
from event_snn.dataset.augmentation import augment_mycount, normalize_clip
from event_snn.dataset.representation import (
    crop_events_range,
    events_to_mycount_tensor,
    load_h5_events,
)
from event_snn.dataset.window_index import WindowSample, build_window_samples


def _cache_tag(data_cfg: DataConfig) -> str:
    tag = (
        f"h{data_cfg.sensor_height}_w{data_cfg.sensor_width}_"
        f"ds{data_cfg.ds_factor}_mw{data_cfg.model_width}_"
        f"t{data_cfg.sampling_time_ms}_len{data_cfg.sample_length_ms}"
    )
    if data_cfg.window_stride_ms > 0:
        tag += f"_stride{data_cfg.window_stride_ms}"
    return tag


def _cache_file_path(cache_dir: Path, sample: WindowSample, data_cfg: DataConfig) -> Path:
    tag = _cache_tag(data_cfg)
    name = f"{Path(sample.segment_key).name}_t{sample.start_us // 1000}ms.npy"
    return cache_dir / tag / name


def _legacy_clip_cache_path(cache_dir: Path, h5_path: Path, data_cfg: DataConfig) -> Path:
    tag = (
        f"h{data_cfg.sensor_height}_w{data_cfg.sensor_width}_"
        f"ds{data_cfg.ds_factor}_mw{data_cfg.model_width}_"
        f"t{data_cfg.sampling_time_ms}_len{data_cfg.sample_length_ms}"
    )
    return cache_dir / tag / f"{h5_path.stem}.npy"


class DVS240WindowDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        data_cfg: DataConfig,
        *,
        augment: bool = False,
        normalize: bool = False,
        cache_dir: Path | None = None,
        return_meta: bool = False,
        seed: int = 1,
    ) -> None:
        self.data_cfg = data_cfg
        self.augment = augment
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.return_meta = return_meta
        self.rng = np.random.default_rng(seed + int(augment))
        self.sampling_time_us = data_cfg.sampling_time_ms * 1000
        self.sample_length_us = data_cfg.sample_length_ms * 1000
        self.time_bins = data_cfg.time_bins
        self.samples = build_window_samples(
            files,
            sample_length_ms=data_cfg.sample_length_ms,
            stride_ms=data_cfg.window_stride_ms,
        )
        if not self.samples:
            raise ValueError("No window samples built from input files")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_segment_events(self, sample: WindowSample) -> np.ndarray:
        clip_us = self.sample_length_us
        parts: list[np.ndarray] = []
        for idx, path in enumerate(sample.clip_paths):
            with h5py.File(path, "r") as handle:
                raw = handle["events"][:]
            events = load_h5_events(np.asarray(raw))
            if len(events):
                events = events.copy()
                events[:, 2] += idx * clip_us
            parts.append(events)
        if not parts:
            return np.empty((0, 4), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    def _try_load_cache(self, sample: WindowSample) -> np.ndarray | None:
        if self.cache_dir is None:
            return None

        paths: list[Path] = []
        if self.data_cfg.window_stride_ms <= 0:
            clip_idx = sample.start_us // self.sample_length_us
            if 0 <= clip_idx < len(sample.clip_paths):
                paths.append(
                    _legacy_clip_cache_path(
                        self.cache_dir, sample.clip_paths[clip_idx], self.data_cfg
                    )
                )
        paths.append(_cache_file_path(self.cache_dir, sample, self.data_cfg))

        for cache_path in paths:
            if not cache_path.is_file():
                continue
            try:
                return np.load(cache_path, mmap_mode="r")
            except ValueError as exc:
                # Truncated/corrupt from a concurrent write race — drop and rebuild.
                if "mmap length is greater than file size" in str(exc) or "cannot reshape" in str(exc):
                    try:
                        cache_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                raise
        return None

    def _load_tensor(self, sample: WindowSample) -> np.ndarray:
        cached = self._try_load_cache(sample)
        if cached is not None:
            return cached

        events = self._load_segment_events(sample)
        events = crop_events_range(events, sample.start_us, self.sample_length_us)
        tensor = events_to_mycount_tensor(
            events,
            sensor_height=self.data_cfg.sensor_height,
            sensor_width=self.data_cfg.sensor_width,
            ds_factor=self.data_cfg.ds_factor,
            model_width=self.data_cfg.model_width,
            sampling_time_us=self.sampling_time_us,
            sample_length_us=self.sample_length_us,
            time_bins=self.time_bins,
        )

        if self.cache_dir is not None:
            cache_path = _cache_file_path(self.cache_dir, sample, self.data_cfg)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: avoid truncated .npy when multiple jobs share the cache.
            tmp_path = cache_path.parent / f".{cache_path.stem}.{os.getpid()}.tmp.npy"
            try:
                np.save(tmp_path, tensor.astype(np.float16, copy=False))
                os.replace(tmp_path, cache_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        return tensor

    def __getitem__(self, index: int) -> tuple:
        sample = self.samples[index]
        tensor = self._load_tensor(sample)
        if isinstance(tensor, np.memmap):
            tensor = np.array(tensor, dtype=np.float32)
        else:
            tensor = tensor.astype(np.float32, copy=False)

        if self.normalize:
            tensor = normalize_clip(tensor)
        if self.augment:
            tensor = augment_mycount(tensor, self.rng)

        label = torch.tensor(sample.label, dtype=torch.long)
        x = torch.from_numpy(tensor)
        if self.return_meta:
            return x, label, sample.segment_key, sample.center_us
        return x, label


# Backward-compatible alias
DVS240H5Dataset = DVS240WindowDataset
