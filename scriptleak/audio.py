"""Audio helpers (numpy + scipy)."""

from __future__ import annotations

from math import gcd

import numpy as np


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling. Returns the input unchanged if rates match."""
    sr, target_sr = int(sr), int(target_sr)
    if sr == target_sr:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(sr, target_sr)
    return resample_poly(x.astype(np.float64), target_sr // g, sr // g).astype(np.float32)
