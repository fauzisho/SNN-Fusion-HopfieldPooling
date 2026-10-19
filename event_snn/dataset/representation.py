"""Event-to-tensor encoding (emmanuela myCount style)."""

from __future__ import annotations

import numpy as np


def load_h5_events(raw: np.ndarray) -> np.ndarray:
    """Return events as (N, 4) int64 columns x, y, t_us, p."""
    if raw.dtype.names and {"x", "y", "t", "p"}.issubset(set(raw.dtype.names)):
        out = np.empty(len(raw), dtype=np.int64)
        out = np.column_stack(
            [
                raw["x"].astype(np.int64),
                raw["y"].astype(np.int64),
                raw["t"].astype(np.int64),
                raw["p"].astype(np.int64),
            ]
        )
        return out

    if raw.ndim == 2 and raw.shape[1] == 4:
        # wearsnn DVS240 tabular layout: t, x, y, p
        out = np.empty((raw.shape[0], 4), dtype=np.int64)
        out[:, 0] = raw[:, 1]
        out[:, 1] = raw[:, 2]
        out[:, 2] = raw[:, 0]
        out[:, 3] = raw[:, 3]
        return out

    raise ValueError(f"Unsupported H5 event layout: shape={raw.shape}, dtype={raw.dtype}")


def crop_events_to_window(events: np.ndarray, sample_length_us: int) -> np.ndarray:
    if len(events) == 0:
        return events
    t0 = int(events[:, 2].min())
    return crop_events_range(events, t0, sample_length_us)


def crop_events_range(events: np.ndarray, start_us: int, sample_length_us: int) -> np.ndarray:
    if len(events) == 0:
        return events
    t_end = start_us + sample_length_us
    mask = (events[:, 2] >= start_us) & (events[:, 2] < t_end)
    cropped = events[mask].copy()
    if len(cropped):
        cropped[:, 2] -= start_us
    return cropped


def events_to_mycount_tensor(
    events: np.ndarray,
    *,
    sensor_height: int,
    sensor_width: int,
    ds_factor: int,
    model_width: int,
    sampling_time_us: int,
    sample_length_us: int,
    time_bins: int,
) -> np.ndarray:
    """
    Build myCount representation.

    Returns float32 tensor (2, H, W, T) with event counts per polarity/bin.
    Native DVS240 is downsampled by ds_factor, then width-padded to model_width.
    """
    height = sensor_height // ds_factor
    width_ds = sensor_width // ds_factor
    pad_left = max(0, (model_width - width_ds) // 2)

    frames = np.zeros((2, height, model_width, time_bins), dtype=np.float32)
    if len(events) == 0:
        return frames

    x = np.floor(events[:, 0] / ds_factor).astype(np.int32)
    y = np.floor(events[:, 1] / ds_factor).astype(np.int32)
    t_bin = np.floor(events[:, 2] / sampling_time_us).astype(np.int32)
    pol = events[:, 3].astype(np.int32)

    valid = (
        (x >= 0)
        & (x < width_ds)
        & (y >= 0)
        & (y < height)
        & (t_bin >= 0)
        & (t_bin < time_bins)
    )
    if not valid.any():
        return frames

    x = x[valid] + pad_left
    y = y[valid]
    t_bin = t_bin[valid]
    pol = np.clip(pol, 0, 1)

    pos = pol > 0
    if pos.any():
        np.add.at(frames, (0, y[pos], x[pos], t_bin[pos]), 1)
    neg = ~pos
    if neg.any():
        np.add.at(frames, (1, y[neg], x[neg], t_bin[neg]), 1)

    return frames
