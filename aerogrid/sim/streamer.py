"""Replay UK-DALE test-slice data as a mock smart meter.

The streamer produces, per 15-min tick in the test window:
  - a list of ground-truth ApplianceOnsets that occurred in the slot
    (drawn from onsets.parquet so the graph always has *some* NILM signal,
     even when 16 kHz data isn't available for that slot)
  - optionally a (voltage, current) 16 kHz chunk if the slot overlaps the
    3-day / 6-h 16 kHz FLAC slice — when present, the real SignalWatcher
    runs in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from aerogrid.behavioral_predictor import load_onsets
from aerogrid.config import (
    SLOT_MINUTES,
    UKDALE_16KHZ_END,
    UKDALE_16KHZ_START,
    UKDALE_DIR,
    UKDALE_HF_HZ,
    UKDALE_TEST_END,
    UKDALE_TEST_START,
)
from aerogrid.types import ApplianceOnset


@dataclass
class Tick:
    now: datetime
    new_onsets: list[ApplianceOnset]
    mains_chunk: tuple[np.ndarray, np.ndarray] | None
    chunk_start: datetime | None


class Streamer:
    """Iterate the test window at 15-min cadence."""

    def __init__(self, flac_path: Path | None = None):
        self.onsets = load_onsets()
        self.test_onsets = self.onsets[self.onsets["split"] == "test"].copy()
        self.test_onsets["timestamp"] = self.test_onsets["timestamp"].dt.tz_convert("UTC")

        self._flac_path = flac_path or (UKDALE_DIR / "house_1" / "mains_16khz_3day.flac")
        self._flac_cache: tuple[np.ndarray, np.ndarray, int, datetime] | None = None

    def _load_flac(self):
        if not self._flac_path.exists():
            return None
        if self._flac_cache is not None:
            return self._flac_cache
        import soundfile as sf
        data, fs = sf.read(self._flac_path, always_2d=True)
        voltage = data[:, 0] * 300.0
        current = data[:, 1] * 15.0
        self._flac_cache = (voltage.astype(np.float32),
                            current.astype(np.float32), int(fs),
                            UKDALE_16KHZ_START)
        return self._flac_cache

    # ------------------------------------------------------------------ #
    def iter_ticks(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        hf_every_n_ticks: int = 24,          # 1x per 6 h by default
    ) -> Iterator[Tick]:
        start = start or UKDALE_TEST_START
        end = end or UKDALE_TEST_END
        tick_dt = timedelta(minutes=SLOT_MINUTES)

        now = start
        idx = 0
        while now < end:
            slot_end = now + tick_dt
            mask = (
                (self.test_onsets["timestamp"] >= now)
                & (self.test_onsets["timestamp"] < slot_end)
            )
            onsets: list[ApplianceOnset] = []
            for _, row in self.test_onsets[mask].iterrows():
                onsets.append(
                    ApplianceOnset(
                        appliance=row["appliance"],
                        timestamp=row["timestamp"].to_pydatetime(),
                        confidence=1.0,
                        source="ground_truth",
                    )
                )

            chunk = None
            chunk_t = None
            hf = self._load_flac()
            if hf is not None and (idx % hf_every_n_ticks == 0):
                voltage, current, fs, flac_start = hf
                # Grab the slot's worth of HF if it's inside the FLAC window.
                if UKDALE_16KHZ_START <= now < UKDALE_16KHZ_END:
                    rel = (now - flac_start).total_seconds()
                    n_samples = fs * SLOT_MINUTES * 60
                    s0 = int(rel * fs)
                    s1 = s0 + n_samples
                    if 0 <= s0 and s1 <= len(voltage):
                        chunk = (voltage[s0:s1], current[s0:s1])
                        chunk_t = now

            yield Tick(
                now=now,
                new_onsets=onsets,
                mains_chunk=chunk,
                chunk_start=chunk_t,
            )
            now = slot_end
            idx += 1
