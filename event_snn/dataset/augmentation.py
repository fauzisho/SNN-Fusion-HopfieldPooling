"""On-the-fly augmentation for myCount DVS tensors (2, H, W, T)."""

from __future__ import annotations

import numpy as np


def normalize_clip(tensor: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Divide by per-clip max count to reduce subject event-rate bias."""
    peak = float(np.max(tensor))
    if peak <= 0:
        return tensor
    return tensor / (peak + eps)


def augment_mycount(
    tensor: np.ndarray,
    rng: np.random.Generator,
    *,
    event_drop_prob: float = 0.05,
    max_shift_x: int = 4,
    max_shift_y: int = 4,
    flip_prob: float = 0.5,
) -> np.ndarray:
    """Apply spatial/temporal jitter on train samples. Input shape (2, H, W, T)."""
    out = tensor.copy()

    if rng.random() < flip_prob:
        out = out[:, :, ::-1, :]

    if max_shift_x > 0 or max_shift_y > 0:
        shift_y = int(rng.integers(-max_shift_y, max_shift_y + 1))
        shift_x = int(rng.integers(-max_shift_x, max_shift_x + 1))
        out = np.roll(out, shift_y, axis=1)
        out = np.roll(out, shift_x, axis=2)

    if event_drop_prob > 0:
        drop = rng.random(out.shape) < event_drop_prob
        out = out * (~drop)

    return out
