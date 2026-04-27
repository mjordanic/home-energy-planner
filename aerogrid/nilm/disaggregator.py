"""Per-appliance disaggregation: base interface + perfect (ground-truth) impl.

NILM disaggregation is **not** the focus of this project.  The classes below
provide a plug-in point for a real NILM model while shipping a
:class:`Disaggregator` that returns ground truth from the simulator.

All public APIs are **timestamp-aware**: inputs and outputs carry UTC
``DatetimeIndex`` so alignment is guaranteed by timestamp join — there is no
positional indexing or ``start_idx`` arithmetic anywhere.

Two entry points:

- :class:`Disaggregator` (batch) — takes a ``mains_df`` DataFrame, returns a
  per-appliance DataFrame aligned by timestamp join.
- :class:`RollingDisaggregator` (streaming) — ``append(p_w, t)`` ingests one
  sample and ``infer_latest(t)`` returns the per-appliance estimate at ``t``.

To plug in a real NILM model, subclass :class:`DisaggregatorBase` (batch)
and/or override :class:`RollingDisaggregator` with a ring-buffer path.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aerogrid.config import APPLIANCES, SCENARIO_DIR
from aerogrid.nilm.onset_detector import OnsetDetector
from aerogrid.types import ApplianceOnset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Abstract base                                                               #
# --------------------------------------------------------------------------- #
class DisaggregatorBase(ABC):
    """Abstract NILM disaggregator.

    Subclass this and implement :meth:`appliances` + :meth:`disaggregate`
    to plug in a real NILM model. The timestamp-based API ensures alignment
    is always by time, never by position.
    """

    @abstractmethod
    def appliances(self) -> list[str]:
        """Return the list of appliance names this disaggregator handles."""

    @abstractmethod
    def disaggregate(self, mains_df: pd.DataFrame) -> pd.DataFrame:
        """Batch-disaggregate an aggregate mains trace.

        Args:
            mains_df: DataFrame with at least a ``timestamp`` column
                (UTC-aware, 1 Hz) and a ``power_w`` column.

        Returns:
            DataFrame indexed by ``timestamp`` with one column per appliance,
            values in watts (float32).  Rows are aligned to the input
            timestamps via join — there is no positional dependency.
        """


# --------------------------------------------------------------------------- #
# Perfect (dummy) disaggregator                                               #
# --------------------------------------------------------------------------- #
class Disaggregator(DisaggregatorBase):
    """Perfect disaggregator backed by ground-truth per-appliance traces.

    Not a real NILM model — uses the simulator's own per-appliance Series so
    the rest of the pipeline (onset detection, triggers, MPC optimizer) can be
    developed and tested without a trained disaggregator.

    All traces are stored as ``pd.Series`` with UTC ``DatetimeIndex``.
    :meth:`disaggregate` performs a timestamp join so any time window can be
    queried without computing index offsets.

    Replace with a real :class:`DisaggregatorBase` subclass for production.
    """

    def __init__(self, traces: dict[str, pd.Series] | None = None):
        """
        Args:
            traces: Pre-loaded ``{appliance: Series}`` dict where each Series
                has a UTC-aware ``DatetimeIndex`` and float32 power values.
                Can also be populated later via :meth:`add_trace` or the
                :meth:`from_scenario` factory.
        """
        self._series: dict[str, pd.Series] = {}
        if traces:
            for name, s in traces.items():
                self.add_trace(name, s)

    def add_trace(self, name: str, series: pd.Series) -> None:
        """Register a ground-truth power trace.

        Args:
            name: Appliance name (should match a key in ``APPLIANCES``).
            series: UTC-timezone-aware ``DatetimeIndex`` Series of power
                values in watts (float32).

        Raises:
            TypeError: if ``series`` does not have a ``DatetimeIndex``.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            logger.error(
                "Disaggregator.add_trace(%r): expected DatetimeIndex, got %s",
                name, type(series.index).__name__,
            )
            raise TypeError(
                f"add_trace({name!r}): series must have a DatetimeIndex, "
                f"got {type(series.index).__name__}"
            )
        if series.index.tz is None:
            series = series.tz_localize("UTC")
        self._series[name] = series.astype(np.float32)
        logger.debug("Disaggregator.add_trace: registered %s (%d samples)", name, len(series))

    def appliances(self) -> list[str]:
        return list(self._series.keys())

    def disaggregate(self, mains_df: pd.DataFrame) -> pd.DataFrame:
        """Return ground-truth appliance power aligned to the input timestamps.

        Alignment is by timestamp join — no positional assumptions, no
        ``start_idx`` arithmetic.  Timestamps in ``mains_df`` that have no
        matching entry in the stored trace return 0.0.

        Args:
            mains_df: DataFrame with a ``timestamp`` column (UTC-aware, 1 Hz).

        Returns:
            DataFrame indexed by ``timestamp`` (UTC), columns = appliance
            names, values in watts (float32).
        """
        if "timestamp" not in mains_df.columns:
            raise ValueError("mains_df must have a 'timestamp' column")
        ts = pd.to_datetime(mains_df["timestamp"], utc=True)
        result = pd.DataFrame(index=ts)
        result.index.name = "timestamp"
        for name, series in self._series.items():
            result[name] = series.reindex(ts).fillna(0.0).to_numpy(np.float32)
        return result

    # ------------------------------------------------------------------ #
    # Factories                                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_scenario(
        cls,
        scenario_dir: Path = SCENARIO_DIR,
        appliances: Iterable[str] | None = None,
        split: str | None = None,
    ) -> "Disaggregator":
        """Load ground-truth traces from scenario parquets.

        Each ``{appliance}_1hz.parquet`` is read, optionally filtered to
        ``split``, and stored as a UTC-indexed ``pd.Series``.

        Args:
            scenario_dir: Directory containing ``{appliance}_1hz.parquet``
                files produced by ``scripts/generate_scenario.py``.
            appliances: Names to load; defaults to all cycle-based appliances.
            split: Optional ``"train"`` / ``"test"`` filter.

        Raises:
            FileNotFoundError: if the parquet for any requested appliance is absent.
        """
        names = list(appliances) if appliances else [
            a for a, spec in APPLIANCES.items() if spec.cycle_slots > 0
        ]
        logger.info(
            "Disaggregator.from_scenario: loading %d appliances from %s split=%s",
            len(names), scenario_dir, split,
        )
        traces: dict[str, pd.Series] = {}
        for name in names:
            path = scenario_dir / f"{name}_1hz.parquet"
            if not path.exists():
                logger.error(
                    "Disaggregator.from_scenario: trace not found at %s — run generate_scenario.py", path,
                )
                raise FileNotFoundError(
                    f"no ground-truth trace at {path}. "
                    "Run scripts/generate_scenario.py first."
                )
            df = pd.read_parquet(path)
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split]
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            series = (
                df.set_index("timestamp")["power_w"]
                .rename(name)
                .astype(np.float32)
            )
            traces[name] = series
            logger.debug(
                "Disaggregator.from_scenario: %s loaded %d samples", name, len(series),
            )
        logger.info("Disaggregator.from_scenario: loaded appliances=%s", list(traces.keys()))
        return cls(traces=traces)

    @classmethod
    def from_cache(
        cls,
        appliances: Iterable[str] | None = None,
        cache_dir: Path | None = None,
    ) -> "Disaggregator":
        """Backward-compatible alias for :meth:`from_scenario`."""
        return cls.from_scenario(appliances=appliances)


# --------------------------------------------------------------------------- #
# Streaming (rolling) disaggregator                                           #
# --------------------------------------------------------------------------- #
class RollingDisaggregator:
    """Streaming disaggregation with explicit timestamp API.

    Both :meth:`append` and :meth:`infer_latest` take an explicit ``t``
    (UTC ``datetime``) so alignment is always by timestamp, never by
    monotonic counter or ring-buffer index.

    For the perfect disaggregator the power lookup is a O(1) pandas
    ``DatetimeIndex`` hashtable access.  For a real NILM model, subclass and
    override :meth:`append` (fill a ring buffer with ``p_w``) and
    :meth:`infer_latest` (run inference on the buffer — ``t`` can be ignored).
    """

    def __init__(self, disagg: Disaggregator):
        self._series: dict[str, pd.Series] = dict(disagg._series)
        self._appliance_names: list[str] = disagg.appliances()

    def append(self, p_w: float, t: datetime) -> None:
        """Ingest one aggregate-power sample.

        For the perfect disaggregator ``p_w`` is ignored (ground truth is
        retrieved by timestamp in :meth:`infer_latest`).  For a real NILM
        model, push ``p_w`` into a ring buffer here.

        Args:
            p_w: Aggregate household power in watts.
            t: UTC timestamp of this sample.
        """

    def infer_latest(self, t: datetime) -> dict[str, float]:
        """Return per-appliance power (watts) at timestamp *t*.

        Args:
            t: UTC ``datetime`` of the sample to look up.

        Returns:
            ``{appliance: power_w}`` dict; returns 0.0 for any appliance
            whose trace does not contain *t*.
        """
        ts = pd.Timestamp(t)
        out: dict[str, float] = {}
        for name, series in self._series.items():
            try:
                out[name] = float(series.loc[ts])
            except KeyError:
                out[name] = 0.0
                logger.debug(
                    "RollingDisaggregator.infer_latest: %s has no entry at %s — returning 0.0",
                    name, ts.isoformat(),
                )
        logger.debug(
            "RollingDisaggregator.infer_latest: t=%s power=%s",
            ts.isoformat(),
            {k: f"{v:.1f}W" for k, v in out.items()},
        )
        return out


# --------------------------------------------------------------------------- #
# power_to_onsets helper                                                      #
# --------------------------------------------------------------------------- #
def power_to_onsets(
    traces: dict[str, np.ndarray],
    start_time: datetime,
    *,
    detectors: dict[str, OnsetDetector] | None = None,
    sample_rate_hz: float = 1.0,
) -> list[ApplianceOnset]:
    """Offline: feed each per-appliance trace through an OnsetDetector.

    For streaming use, instantiate ``OnsetDetector`` instances directly and
    call ``.update(p_w, t)`` per sample — this helper is for batch processing
    (e.g. notebooks, tests, evaluation pipelines).

    Args:
        traces: Dict mapping appliance name → 1 Hz float32 power trace.
        start_time: UTC timestamp of the first sample.
        detectors: Optional pre-built ``{name: OnsetDetector}`` dict.  If
            ``None``, one detector is created per appliance using the
            ``on_power_threshold_w`` from ``APPLIANCES``.
        sample_rate_hz: Sampling rate; determines the time step between samples.

    Returns:
        List of :class:`~aerogrid.types.ApplianceOnset` events, sorted by time.
    """
    if detectors is None:
        detectors = {
            name: OnsetDetector(
                appliance=name,
                threshold_w=APPLIANCES[name].on_power_threshold_w
                if name in APPLIANCES else 20.0,
            )
            for name in traces
        }
    dt_s = 1.0 / sample_rate_hz
    events: list[ApplianceOnset] = []
    for name, trace in traces.items():
        det = detectors.get(name)
        if det is None:
            continue
        for i, p in enumerate(trace):
            t = start_time + timedelta(seconds=i * dt_s)
            onset = det.update(float(p), t)
            if onset is not None:
                events.append(onset)
    events.sort(key=lambda o: o.timestamp)
    return events


__all__ = [
    "DisaggregatorBase",
    "Disaggregator",
    "RollingDisaggregator",
    "power_to_onsets",
]
