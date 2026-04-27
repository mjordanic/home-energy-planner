"""Disaggregator smoke + streaming tests.

Tests the perfect (ground-truth) disaggregator and onset detection plumbing.
All disaggregator APIs are timestamp-based — no positional indexing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aerogrid.nilm import (
    Disaggregator,
    OnsetDetector,
    PerfectDisaggModel,
    RollingDisaggregator,
    power_to_onsets,
)


_BASE_TS = pd.Timestamp("2024-01-01", tz="UTC")


def _fake_traces(n: int = 3600) -> dict[str, pd.Series]:
    """Ground-truth: 0 W background + 2000 W dishwasher from sample 1200–1800."""
    ts = pd.date_range(_BASE_TS, periods=n, freq="s", tz="UTC")
    target = np.zeros(n, dtype=np.float32)
    target[1200:1800] = 2000.0
    return {"dishwasher": pd.Series(target, index=ts)}


def _fake_mains_df(n: int = 3600) -> pd.DataFrame:
    """Synthetic mains DataFrame with matching timestamps."""
    ts = pd.date_range(_BASE_TS, periods=n, freq="s", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "power_w": np.full(n, 150.0, dtype=np.float32)})


# --------------------------------------------------------------------------- #
# PerfectDisaggModel unit tests                                               #
# --------------------------------------------------------------------------- #
def test_perfect_disagg_model_returns_ground_truth():
    gt = np.array([0.0, 0.0, 500.0, 2000.0, 0.0], dtype=np.float32)
    model = PerfectDisaggModel(gt, threshold_w=20.0)

    active, power = model.at(2)
    assert active == 1.0
    assert power == 500.0

    active, power = model.at(0)
    assert active == 0.0
    assert power == 0.0

    active, power = model.at(999)
    assert active == 0.0
    assert power == 0.0


# --------------------------------------------------------------------------- #
# Disaggregator batch tests                                                   #
# --------------------------------------------------------------------------- #
def test_disaggregator_batch():
    traces = _fake_traces(3600)
    disagg = Disaggregator(traces=traces)

    out = disagg.disaggregate(_fake_mains_df(3600))

    assert "dishwasher" in out.columns
    assert len(out) == 3600
    np.testing.assert_array_equal(
        out["dishwasher"].to_numpy(), traces["dishwasher"].to_numpy()
    )


def test_disaggregator_appliances():
    disagg = Disaggregator(traces=_fake_traces())
    assert disagg.appliances() == ["dishwasher"]


def test_disaggregator_timestamp_join_subwindow():
    """Querying a sub-window by timestamp returns exactly that window — no start_idx."""
    traces = _fake_traces(3600)
    disagg = Disaggregator(traces=traces)

    # Slice from sample 1000 to 1500 — spans the on-window (1200–1800).
    sub_ts = pd.date_range(_BASE_TS + pd.Timedelta(seconds=1000), periods=500, freq="s", tz="UTC")
    sub_df = pd.DataFrame({"timestamp": sub_ts, "power_w": 0.0})
    out = disagg.disaggregate(sub_df)

    assert len(out) == 500
    # First 200 samples (1000–1199) are off; next 300 (1200–1499) are on.
    assert (out["dishwasher"].to_numpy()[:200] == 0.0).all()
    assert (out["dishwasher"].to_numpy()[200:] == 2000.0).all()


def test_disaggregator_raises_without_timestamp_column():
    disagg = Disaggregator(traces=_fake_traces())
    with pytest.raises(ValueError, match="timestamp"):
        disagg.disaggregate(pd.DataFrame({"power_w": [0.0]}))


def test_add_trace_raises_on_non_datetime_index():
    disagg = Disaggregator()
    bad_series = pd.Series([1.0, 2.0], index=[0, 1])
    with pytest.raises(TypeError):
        disagg.add_trace("dishwasher", bad_series)


# --------------------------------------------------------------------------- #
# RollingDisaggregator streaming tests                                        #
# --------------------------------------------------------------------------- #
def test_rolling_disaggregator_streaming():
    traces = _fake_traces(1500)
    disagg = Disaggregator(traces=traces)
    rolling = RollingDisaggregator(disagg)

    # Stream 1500 samples.
    for i in range(1500):
        t = (_BASE_TS + pd.Timedelta(seconds=i)).to_pydatetime()
        rolling.append(150.0, t)

    # Off window (sample 100).
    t_off = (_BASE_TS + pd.Timedelta(seconds=100)).to_pydatetime()
    assert rolling.infer_latest(t_off)["dishwasher"] == 0.0

    # On window (sample 1250).
    t_on = (_BASE_TS + pd.Timedelta(seconds=1250)).to_pydatetime()
    assert rolling.infer_latest(t_on)["dishwasher"] == 2000.0


def test_rolling_disaggregator_unknown_timestamp_returns_zero():
    disagg = Disaggregator(traces=_fake_traces(100))
    rolling = RollingDisaggregator(disagg)
    far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
    assert rolling.infer_latest(far_future)["dishwasher"] == 0.0


# --------------------------------------------------------------------------- #
# OnsetDetector tests                                                         #
# --------------------------------------------------------------------------- #
def test_onset_detector_fires_on_rising_edge():
    det = OnsetDetector(appliance="dishwasher", threshold_w=20.0)
    t0 = datetime(2024, 12, 15, 19, 0, tzinfo=timezone.utc)
    assert det.update(0.0, t0) is None
    assert det.update(10.0, t0 + timedelta(seconds=1)) is None
    onset = det.update(100.0, t0 + timedelta(seconds=2))
    assert onset is not None
    assert onset.appliance == "dishwasher"
    assert det.update(5.0, t0 + timedelta(seconds=10)) is None
    assert det.update(100.0, t0 + timedelta(seconds=20)) is None
    assert det.update(5.0, t0 + timedelta(minutes=11)) is None
    onset2 = det.update(100.0, t0 + timedelta(minutes=12))
    assert onset2 is not None


def test_power_to_onsets_emits_events():
    n = 3600
    trace = np.zeros(n, dtype=np.float32)
    trace[100:200] = 2000.0
    trace[2000:2100] = 2000.0
    events = power_to_onsets(
        {"dishwasher": trace},
        start_time=datetime(2024, 12, 15, 0, 0, tzinfo=timezone.utc),
        detectors={"dishwasher": OnsetDetector("dishwasher", threshold_w=20.0)},
    )
    assert len(events) == 2
    assert all(e.appliance == "dishwasher" for e in events)
