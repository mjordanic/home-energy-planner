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

from aerogrid.config import BatterySpec
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


class _DayNightPriceFeed:
    """Price feed: cheap at night (22:00–06:00 UTC = 15 €/MWh), expensive during
    the day (06:00–22:00 UTC = 120 €/MWh).  History extends 14 days back so the
    naive oracle can fit the seasonal pattern and forecast the day/night transition.
    """

    def __init__(self, start: datetime, end: datetime):
        idx = pd.date_range(start - timedelta(days=14), end, freq="15min", tz="UTC")
        hod = idx.hour
        lbmp = np.where((hod >= 22) | (hod < 6), 15.0, 120.0).astype(np.float32)
        self.df = pd.DataFrame({"timestamp": idx, "lbmp": lbmp})

    def history(self, now):
        return self.df[self.df["timestamp"] < now]

    def realized(self, now):
        slot_min = (now.minute // 15) * 15
        slot = now.replace(minute=slot_min, second=0, microsecond=0)
        row = self.df[self.df["timestamp"] == slot]
        return float(row["lbmp"].iloc[0]) if not row.empty else None


def test_three_strategies_battery_columns_and_cost_ordering(tmp_path: Path):
    """Three-strategy twin: baseline, optimizer_nobatt, optimizer_batt.

    Scenario: Monday 06:00 (start of expensive period, 120 €/MWh).  The battery
    strategy is pre-loaded with 5 kWh SoC, representing a battery that warmed up
    over the preceding cheap night (€15/MWh).  During the 30-minute expensive
    window the optimizer_batt discharges, reducing net grid draw and cumulative
    cost below optimizer_nobatt and baseline, which both pay full expensive prices.

    Asserts:
    - battery columns (battery_charge_kw, battery_discharge_kw, soc_kwh, net_grid_kw)
      exist for optimizer_batt and are ≥ 0 throughout.
    - soc_kwh starts above 0 (pre-loaded) and decreases (battery discharged).
    - cumulative cost ordering holds at the last slot:
        optimizer_batt ≤ optimizer_nobatt ≤ baseline
    """
    # Monday 06:00: expensive period (120 €/MWh).
    start = datetime(2024, 12, 16, 6, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)

    feed = _DayNightPriceFeed(start, end)
    streamer = Streamer(realized_price_provider=feed.realized)
    # Evening cycle onset to give optimizer something to schedule.
    streamer.add_onset("dishwasher", start + timedelta(minutes=10))

    baseline = BaselineStrategy(name="baseline")
    optimizer_nobatt = OptimizerStrategy(
        name="optimizer_nobatt",
        price_history_provider=feed.history,
        price_oracle_impl="naive",
        horizon_slots=32,           # 8h horizon (sees cheap return at 22:00 eventually)
        auto_confirm=True,
        battery_enabled=False,
    )
    optimizer_batt = OptimizerStrategy(
        name="optimizer_batt",
        price_history_provider=feed.history,
        price_oracle_impl="naive",
        horizon_slots=32,
        auto_confirm=True,
        battery_enabled=True,
    )
    # Pre-load battery to simulate overnight charging (battery warmed up).
    optimizer_batt.commit.soc_kwh = 5.0

    slot_log_path = tmp_path / "slot_log.parquet"
    event_log_path = tmp_path / "event_log.parquet"

    summary = run(
        strategies=[baseline, optimizer_nobatt, optimizer_batt],
        streamer=streamer,
        start=start,
        end=end,
        slot_log_path=slot_log_path,
        event_log_path=event_log_path,
    )

    assert slot_log_path.exists()
    slot_df = pd.read_parquet(slot_log_path)
    assert not slot_df.empty

    # Battery columns exist for the batt strategy.
    batt_cols = {
        "optimizer_batt_battery_charge_kw",
        "optimizer_batt_battery_discharge_kw",
        "optimizer_batt_soc_kwh",
        "optimizer_batt_net_grid_kw",
    }
    assert batt_cols <= set(slot_df.columns), (
        f"Missing battery columns: {batt_cols - set(slot_df.columns)}"
    )

    # Battery values are non-negative.
    for col in ("optimizer_batt_battery_charge_kw", "optimizer_batt_battery_discharge_kw",
                "optimizer_batt_soc_kwh", "optimizer_batt_net_grid_kw"):
        assert (slot_df[col] >= -1e-9).all(), f"{col} has negative values"

    # Battery SoC starts non-zero (pre-loaded) and decreases (discharging).
    assert slot_df["optimizer_batt_soc_kwh"].iloc[0] > 0.0, "Battery SoC should start > 0 (pre-loaded)"
    assert slot_df["optimizer_batt_soc_kwh"].iloc[-1] < slot_df["optimizer_batt_soc_kwh"].iloc[0], (
        "Battery SoC should decrease (battery discharged during expensive period)"
    )

    # Cost ordering at the last slot: batt ≤ nobatt ≤ baseline.
    last = slot_df.iloc[-1]
    assert last["optimizer_batt_cum_cost_eur"] <= last["optimizer_nobatt_cum_cost_eur"] + 1e-6, (
        f"batt={last['optimizer_batt_cum_cost_eur']:.4f} > "
        f"nobatt={last['optimizer_nobatt_cum_cost_eur']:.4f}"
    )
    assert last["optimizer_nobatt_cum_cost_eur"] <= last["baseline_cum_cost_eur"] + 1e-6, (
        f"nobatt={last['optimizer_nobatt_cum_cost_eur']:.4f} > "
        f"baseline={last['baseline_cum_cost_eur']:.4f}"
    )
