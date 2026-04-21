"""Tests for LangGraph assembly + HITL interrupt / resume."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from aerogrid.behavioral_predictor import HybridBehavioralPredictor
from aerogrid.config import APPLIANCES
from aerogrid.graph import build_graph, make_thread_id
from aerogrid.price_oracle import SeasonalNaiveOracle
from aerogrid.signal_watcher import SignalWatcher
from aerogrid.types import ApplianceOnset


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_onsets() -> pd.DataFrame:
    """Plausible sparse onset log spanning 40 training days + 14 test."""
    rng = np.random.default_rng(0)
    rows = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for d in range(40):
        for app, peak_h in [("dishwasher", 21), ("washing_machine", 9)]:
            h = int(np.clip(rng.normal(peak_h, 2), 0, 23))
            rows.append({
                "appliance": app,
                "timestamp": base + timedelta(days=d, hours=h),
                "split": "train",
            })
    for d in range(40, 54):
        for app, peak_h in [("dishwasher", 21), ("washing_machine", 9)]:
            h = int(np.clip(rng.normal(peak_h, 2), 0, 23))
            rows.append({
                "appliance": app,
                "timestamp": base + timedelta(days=d, hours=h),
                "split": "test",
            })
    df = pd.DataFrame(rows)
    df["split"] = df["split"].astype("category")
    return df


@pytest.fixture
def fake_prices() -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idx = pd.date_range(base, periods=54 * 96, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    hod = idx.hour + idx.minute / 60
    lbmp = 30 + 20 * np.sin(2 * np.pi * (hod - 7) / 24) + rng.normal(0, 3, len(idx))
    return pd.DataFrame({"timestamp": idx, "lbmp": lbmp.astype(np.float32)})


# --------------------------------------------------------------------------- #
# tests                                                                       #
# --------------------------------------------------------------------------- #
def _providers(prices_df: pd.DataFrame):
    def hist(now):
        return prices_df[prices_df["timestamp"] < now]

    def realized(now):
        floor = (now.minute // 15) * 15
        slot = now.replace(minute=floor, second=0, microsecond=0)
        row = prices_df[prices_df["timestamp"] == slot]
        return float(row["lbmp"].iloc[0]) if not row.empty else None

    return hist, realized


def test_graph_compiles_and_runs_once(fake_onsets, fake_prices):
    hist, realized = _providers(fake_prices)
    predictor = HybridBehavioralPredictor().fit(fake_onsets)
    builder, checkpointer = build_graph(
        watcher=SignalWatcher(signatures={}),
        price_oracle=SeasonalNaiveOracle(),
        predictor=predictor,
        price_history_provider=hist,
        realized_price_provider=realized,
        auto_confirm=True,
    )
    graph = builder.compile(checkpointer=checkpointer)
    now = datetime(2026, 2, 10, 0, 0, tzinfo=timezone.utc)   # in test window
    state = {
        "now": now,
        "mains_chunk": None,
        "chunk_start": None,
        "recent_onsets": [],
        "realized_prices": [],
        "event_log": [],
        "cumulative_cost": 0.0,
        "cumulative_baseline_cost": 0.0,
        "iteration": 0,
    }
    cfg = {"configurable": {"thread_id": make_thread_id(now)}}
    result = graph.invoke(state, config=cfg)
    assert result["iteration"] == 1
    assert result["schedule"] is not None
    assert result["schedule"].solver_status in ("optimal", "optimal_inaccurate")
    assert len(result["schedule"].tasks) == 2    # dishwasher + washing_machine


def test_interrupt_resumes_with_user_answer(fake_onsets, fake_prices):
    hist, realized = _providers(fake_prices)
    predictor = HybridBehavioralPredictor().fit(fake_onsets)
    builder, checkpointer = build_graph(
        watcher=SignalWatcher(signatures={}),
        price_oracle=SeasonalNaiveOracle(),
        predictor=predictor,
        price_history_provider=hist,
        realized_price_provider=realized,
        auto_confirm=False,        # turn HITL on
    )
    graph = builder.compile(checkpointer=checkpointer)
    now = datetime(2026, 2, 10, 0, 0, tzinfo=timezone.utc)
    state = {
        "now": now, "mains_chunk": None, "chunk_start": None,
        "recent_onsets": [], "realized_prices": [], "event_log": [],
        "cumulative_cost": 0.0, "cumulative_baseline_cost": 0.0, "iteration": 0,
    }
    cfg = {"configurable": {"thread_id": make_thread_id(now)}}
    first = graph.invoke(state, config=cfg)
    assert "__interrupt__" in first, "expected an interrupt with a HITL question"

    second = graph.invoke(Command(resume="yes"), config=cfg)
    assert second.get("user_confirmation") == "yes"
    assert second.get("iteration") == 1


def test_replan_reason_on_price_deviation(fake_onsets, fake_prices):
    """Monitor should flag a replan when realized differs from forecast."""
    # Inject a huge spike at a known slot.
    spike_time = datetime(2026, 2, 10, 3, 0, tzinfo=timezone.utc)
    fake_prices = fake_prices.copy()
    fake_prices.loc[fake_prices["timestamp"] == spike_time, "lbmp"] = 500.0
    hist, realized = _providers(fake_prices)

    predictor = HybridBehavioralPredictor().fit(fake_onsets)
    builder, checkpointer = build_graph(
        watcher=SignalWatcher(signatures={}),
        price_oracle=SeasonalNaiveOracle(),
        predictor=predictor,
        price_history_provider=hist,
        realized_price_provider=realized,
        auto_confirm=True,
    )
    graph = builder.compile(checkpointer=checkpointer)
    state = {
        "now": spike_time, "mains_chunk": None, "chunk_start": None,
        "recent_onsets": [], "realized_prices": [], "event_log": [],
        "cumulative_cost": 0.0, "cumulative_baseline_cost": 0.0, "iteration": 0,
    }
    cfg = {"configurable": {"thread_id": make_thread_id(spike_time)}}
    r = graph.invoke(state, config=cfg)
    assert r.get("replan_reason") and "price_deviation" in r["replan_reason"]
