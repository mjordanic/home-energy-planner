"""End-to-end smoke test for the streaming digital twin.

Runs a short simulated window with both BaselineStrategy and
OptimizerStrategy through ``digital_twin.run`` against a synthetic
in-memory price feed and a single manually-injected onset.  Asserts the
basic plumbing produces a non-empty slot log + event log and that the
optimizer fired at least one replan.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerogrid.sim.digital_twin import run
from aerogrid.sim.streamer import Streamer
from aerogrid.sim.strategies import BaselineStrategy, OptimizerStrategy, Strategy


class _InMemoryPriceFeed:
    """Synthetic cheap-night / expensive-day curve for a few days."""
    def __init__(self, start: datetime, end: datetime):
        idx = pd.date_range(start - timedelta(days=7), end, freq="15min", tz="UTC")
        hod = idx.hour + idx.minute / 60.0
        lbmp = 30 + 30 * np.sin(2 * np.pi * (hod - 7) / 24)
        self.df = pd.DataFrame({"timestamp": idx, "lbmp": lbmp.astype(np.float32)})

    def history(self, now):
        return self.df[self.df["timestamp"] < now]

    def realized(self, now):
        slot_min = (now.minute // 15) * 15
        slot = now.replace(minute=slot_min, second=0, microsecond=0)
        row = self.df[self.df["timestamp"] == slot]
        return float(row["lbmp"].iloc[0]) if not row.empty else None


@pytest.fixture
def tmp_logs(tmp_path: Path):
    return tmp_path / "slot_log.parquet", tmp_path / "event_log.parquet"


def test_digital_twin_runs_both_strategies_end_to_end(tmp_logs):
    slot_log_path, event_log_path = tmp_logs
    start = datetime(2024, 12, 15, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)

    feed = _InMemoryPriceFeed(start, end)
    streamer = Streamer(realized_price_provider=feed.realized)
    # One injected onset inside the window.
    streamer.add_onset("dishwasher", start + timedelta(minutes=5))

    baseline = BaselineStrategy()
    optimizer = OptimizerStrategy(
        price_history_provider=feed.history,
        price_oracle_impl="naive",
        horizon_slots=8,
        auto_confirm=True,
    )
    strategies: list[Strategy] = [baseline, optimizer]

    summary = run(
        strategies=strategies,
        streamer=streamer,
        start=start,
        end=end,
        slot_log_path=slot_log_path,
        event_log_path=event_log_path,
    )

    assert summary["n_samples"] == 30 * 60
    assert slot_log_path.exists()
    assert event_log_path.exists()

    slot_df = pd.read_parquet(slot_log_path)
    event_df = pd.read_parquet(event_log_path)

    assert not slot_df.empty
    assert {"baseline_total_kw", "optimizer_total_kw"} <= set(slot_df.columns)
    assert (event_df["strategy"] == "stream").any()
    assert (event_df["strategy"] == "baseline").any()
    assert (event_df["strategy"] == "optimizer").any()
    # The optimizer should fire at least one replan in 30 minutes (periodic + onset).
    assert (
        (event_df["strategy"] == "optimizer")
        & (event_df["event_type"] == "replan_triggered")
    ).any()
