"""DSP-first NILM: wavelet transient detection + V-I signature matching.

Given a short 16 kHz stereo window (voltage, current) the SignalWatcher:
1. Decomposes the current signal with a Daubechies wavelet (db4, level 4).
2. Reconstructs the high-frequency detail bands (D1 + D2) to isolate switching
   transients.
3. Marks onsets where the rolling energy of the detail reconstruction spikes
   above a z-score threshold.
4. Around each onset, extracts a V-I trajectory descriptor and matches it to
   the signature library by cosine similarity.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pywt

from aerogrid.config import CACHE_DIR, NILM, UKDALE_HF_HZ
from aerogrid.types import ApplianceOnset
from aerogrid.vi_features import (
    SAMPLES_PER_CYCLE,
    cosine_similarity,
    extract_single_cycle,
    vi_trajectory_descriptor,
)


class SignalWatcher:
    """NILM front-end. Stateless: call process_window() with each new chunk."""

    def __init__(
        self,
        signatures: dict[str, np.ndarray] | None = None,
        fs: float = UKDALE_HF_HZ,
        wavelet: str = NILM.wavelet,
        level: int = NILM.dwt_level,
        onset_z_threshold: float = NILM.onset_energy_threshold,
        min_gap_s: float = NILM.min_onset_gap_s,
        match_threshold: float = NILM.signature_match_threshold,
    ):
        self.fs = float(fs)
        self.wavelet = wavelet
        self.level = level
        self.onset_z_threshold = onset_z_threshold
        self.min_gap_s = min_gap_s
        self.match_threshold = match_threshold

        if signatures is None:
            signatures = self._load_default_signatures()
        self.signatures = signatures

    # ------------------------------------------------------------------ #
    # Initialization                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_default_signatures() -> dict[str, np.ndarray]:
        path = CACHE_DIR / "signatures.pkl"
        if not path.exists():
            return {}
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return payload.get("signatures", {})

    @classmethod
    def from_cache(cls) -> "SignalWatcher":
        return cls(signatures=cls._load_default_signatures())

    # ------------------------------------------------------------------ #
    # Transient extraction                                               #
    # ------------------------------------------------------------------ #
    def extract_transients(self, signal_1d: np.ndarray) -> np.ndarray:
        """Reconstruct D1+D2 detail bands of the 1-cycle RMS envelope.

        Operating on the envelope (rather than raw current) means we detect
        amplitude steps regardless of whether the 50 Hz phase changes — real
        dishwasher onsets are nearly in-phase with baseline current and produce
        little detail energy in the raw signal, but a clear step in the
        envelope.
        """
        cyc = SAMPLES_PER_CYCLE
        if len(signal_1d) < cyc:
            return np.zeros_like(signal_1d)
        env = np.sqrt(np.convolve(signal_1d ** 2, np.ones(cyc) / cyc, mode="same"))
        coeffs = pywt.wavedec(env, self.wavelet, level=self.level, mode="symmetric")
        detail_only = [np.zeros_like(c) for c in coeffs]
        detail_only[-1] = coeffs[-1]  # D1
        detail_only[-2] = coeffs[-2]  # D2
        recon = pywt.waverec(detail_only, self.wavelet, mode="symmetric")
        return recon[: len(signal_1d)]

    def _detect_onset_indices(self, current: np.ndarray,
                              detail: np.ndarray) -> np.ndarray:
        """Rising-edge onsets from the 1-cycle RMS envelope.

        The envelope jump over 0.5 s catches amplitude steps regardless of
        whether the 50 Hz phase changes, so both resistive (dishwasher) and
        inductive (washing machine) onsets trigger. The DWT detail band is
        still computed (by the caller) for use as a classification feature
        but is deliberately NOT used as the onset trigger — on quiet mains
        baselines its z-score has far too many spurious peaks from noise.
        """
        cyc = SAMPLES_PER_CYCLE
        if len(current) < cyc:
            return np.array([], dtype=np.int64)
        env = np.sqrt(np.convolve(current ** 2, np.ones(cyc) / cyc, mode="same"))
        lag = int(self.fs * 0.5)
        pad = np.full(lag, env[0])
        lagged = np.concatenate([pad, env[:-lag]])
        jump = env - lagged
        # Trigger on either a 2 A absolute jump or a doubled-envelope relative
        # jump (relative to a 0.3 A floor so baseline noise doesn't divide by
        # near-zero).
        trigger = (jump > 2.0) | (env > 2.0 * np.maximum(lagged, 0.3))

        prev = np.concatenate([[False], trigger[:-1]])
        rising = trigger & ~prev
        idx = np.where(rising)[0]
        if len(idx) == 0:
            return idx
        min_gap = int(self.fs * self.min_gap_s)
        keep: list[int] = [int(idx[0])]
        for i in idx[1:]:
            if i - keep[-1] >= min_gap:
                keep.append(int(i))
        return np.array(keep, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # Appliance identification                                           #
    # ------------------------------------------------------------------ #
    def identify_appliance(
        self,
        voltage_segment: np.ndarray,
        current_segment: np.ndarray,
    ) -> tuple[str, float]:
        """Match the V-I descriptor of a single phase-aligned cycle."""
        if not self.signatures:
            return ("unknown", 0.0)
        if len(voltage_segment) < 2 * SAMPLES_PER_CYCLE:
            return ("unknown", 0.0)
        v, c = extract_single_cycle(voltage_segment, current_segment)
        if len(v) < SAMPLES_PER_CYCLE:
            return ("unknown", 0.0)
        feat = vi_trajectory_descriptor(v, c)
        best_name = "unknown"
        best_score = 0.0
        for name, sig in self.signatures.items():
            score = cosine_similarity(feat, sig)
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < self.match_threshold:
            return ("unknown", best_score)
        return (best_name, best_score)

    # ------------------------------------------------------------------ #
    # Top-level: process a window                                        #
    # ------------------------------------------------------------------ #
    def process_window(
        self,
        voltage: np.ndarray,
        current: np.ndarray,
        window_start: datetime,
    ) -> list[ApplianceOnset]:
        """Find all appliance onsets inside a 16 kHz window."""
        if len(voltage) != len(current):
            raise ValueError("voltage/current length mismatch")
        detail = self.extract_transients(current)
        onset_idx = self._detect_onset_indices(current, detail)
        events: list[ApplianceOnset] = []
        # After the onset is marked we skip 0.5 s (past inrush / transient)
        # and grab 4 cycles of steady-state to extract the V-I descriptor.
        post_offset = int(self.fs * 0.5)
        seg_len = 4 * SAMPLES_PER_CYCLE
        for i in onset_idx:
            s0 = i + post_offset
            s1 = s0 + seg_len
            if s1 > len(voltage):
                continue
            label, score = self.identify_appliance(voltage[s0:s1], current[s0:s1])
            if label == "unknown":
                continue
            t = window_start + timedelta(seconds=float(i) / self.fs)
            events.append(ApplianceOnset(appliance=label, timestamp=t, confidence=score))
        return events
