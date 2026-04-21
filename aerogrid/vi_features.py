"""V-I trajectory feature extraction for NILM.

One 50 Hz mains cycle at 16 kHz is 320 samples. We resample to a fixed grid
(default 64 points) and concatenate voltage and current into a single feature
vector so cosine similarity directly measures V-I trajectory similarity.
"""
from __future__ import annotations

import numpy as np

from aerogrid.config import NILM, UKDALE_HF_HZ

MAINS_HZ = 50.0
SAMPLES_PER_CYCLE = int(round(UKDALE_HF_HZ / MAINS_HZ))   # 320 at 16 kHz


def extract_single_cycle(voltage: np.ndarray, current: np.ndarray,
                         start_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (v, i) spanning one full mains cycle starting at start_idx.

    If start_idx is None, finds the next zero-crossing of voltage after the
    midpoint of the window.
    """
    n = len(voltage)
    if start_idx is None:
        mid = n // 2
        zc = _next_zero_crossing(voltage, start=mid)
        start_idx = zc if zc is not None else mid
    end = start_idx + SAMPLES_PER_CYCLE
    if end > n:
        start_idx = n - SAMPLES_PER_CYCLE
        end = n
    return voltage[start_idx:end], current[start_idx:end]


def _next_zero_crossing(x: np.ndarray, start: int = 0) -> int | None:
    s = x[start:]
    sign = np.sign(s)
    # first index where sign flips from <=0 to >0
    flips = np.where((sign[:-1] <= 0) & (sign[1:] > 0))[0]
    if len(flips) == 0:
        return None
    return start + int(flips[0])


def vi_trajectory_descriptor(voltage: np.ndarray, current: np.ndarray,
                             n_points: int = NILM.vi_trajectory_points
                             ) -> np.ndarray:
    """Resample a cycle to n_points on each axis, normalize V and I separately.

    Normalizing V and I separately preserves the V-I Lissajous shape as the
    discriminative feature — otherwise voltage (~230 V) dominates current
    (~10 A) and every appliance ends up with near-identical descriptors.

    Returns a (2 * n_points,) vector [v_norm_resampled, i_norm_resampled].
    """
    if len(voltage) != len(current):
        raise ValueError("voltage and current length mismatch")
    x_src = np.linspace(0, 1, len(voltage), dtype=np.float64)
    x_dst = np.linspace(0, 1, n_points, dtype=np.float64)
    v_r = np.interp(x_dst, x_src, voltage.astype(np.float64))
    i_r = np.interp(x_dst, x_src, current.astype(np.float64))
    v_n = v_r / (np.linalg.norm(v_r) + 1e-9)
    i_n = i_r / (np.linalg.norm(i_r) + 1e-9)
    return np.concatenate([v_n, i_n])


def average_signature(cycles: list[tuple[np.ndarray, np.ndarray]],
                      n_points: int = NILM.vi_trajectory_points) -> np.ndarray:
    """Average a list of (v, i) cycles into a single signature descriptor.

    Re-normalizes V and I halves separately after averaging so the two halves
    remain comparable.
    """
    if not cycles:
        raise ValueError("need at least one cycle")
    feats = np.stack([vi_trajectory_descriptor(v, i, n_points) for v, i in cycles])
    sig = feats.mean(axis=0)
    v_half, i_half = sig[:n_points], sig[n_points:]
    v_half = v_half / (np.linalg.norm(v_half) + 1e-9)
    i_half = i_half / (np.linalg.norm(i_half) + 1e-9)
    return np.concatenate([v_half, i_half])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))
