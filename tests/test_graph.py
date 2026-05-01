"""Tests for LangGraph assembly + HITL interrupt/resume on the slow path."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from aerogrid.graph import build_graph, make_thread_id
from aerogrid.price_oracle import SeasonalNaiveOracle
from aerogrid.types import Schedule


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_prices() -> pd.DataFrame:
    base = datetime(2024, 10, 1, tzinfo=timezone.utc)
    idx = pd.date_range(base, periods=54 * 96, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    hod = idx.hour + idx.minute / 60
    lbmp = 30 + 20 * np.sin(2 * np.pi * (hod - 7) / 24) + rng.normal(0, 3, len(idx))
    return pd.DataFrame({"timestamp": idx, "lbmp": lbmp.astype(np.float32)})


def _providers(prices_df: pd.DataFrame):
    def hist(now):
        return prices_df[prices_df["timestamp"] < now]
    return hist


def _base_state(now: datetime, *, remaining_ev_kwh: float = 24.0) -> dict:
    return {
        "now": now,
        "committed_tasks": [],
        "remaining_ev_kwh": remaining_ev_kwh,
        "ev_power_setpoint_kw": 0.0,
        "previous_plan": None,
        "event_log": [],
    }


# --------------------------------------------------------------------------- #
# tests                                                                       #
# --------------------------------------------------------------------------- #
def test_graph_compiles_and_runs_once(fake_prices):
    hist = _providers(fake_prices)
    builder, checkpointer = build_graph(
        price_oracle=SeasonalNaiveOracle(),
        price_history_provider=hist,
        auto_confirm=True,
    )
    graph = builder.compile(checkpointer=checkpointer)
    # Pick a moment 5 h before the deadline so the EV must charge inside a 2 h horizon.
    now = datetime(2024, 11, 20, 2, 0, tzinfo=timezone.utc)
    state = _base_state(now)
    cfg = {"configurable": {"thread_id": make_thread_id(now)}}
    result = graph.invoke(state, config=cfg)
    plan = result["current_plan"]
    assert isinstance(plan, Schedule)
    assert plan.solver_status in ("optimal", "optimal_inaccurate")
    assert plan.horizon_slots > 0
    assert len(plan.ev_power_kw) == plan.horizon_slots


def test_first_plan_interrupts_when_auto_confirm_off(fake_prices):
    hist = _providers(fake_prices)
    builder, checkpointer = build_graph(
        price_oracle=SeasonalNaiveOracle(),
        price_history_provider=hist,
        auto_confirm=False,
    )
    graph = builder.compile(checkpointer=checkpointer)
    now = datetime(2024, 11, 20, 2, 0, tzinfo=timezone.utc)
    state = _base_state(now)
    cfg = {"configurable": {"thread_id": make_thread_id(now)}}
    first = graph.invoke(state, config=cfg)
    assert "__interrupt__" in first, "first plan should interrupt for user confirmation"
    second = graph.invoke(Command(resume="yes"), config=cfg)
    assert second.get("user_confirmation") == "yes"
    assert second.get("current_plan") is not None


def test_auto_confirms_when_small_delta(fake_prices):
    hist = _providers(fake_prices)
    builder, checkpointer = build_graph(
        price_oracle=SeasonalNaiveOracle(),
        price_history_provider=hist,
        auto_confirm=False,
    )
    graph = builder.compile(checkpointer=checkpointer)
    now = datetime(2024, 11, 20, 2, 0, tzinfo=timezone.utc)
    state = _base_state(now)
    cfg = {"configurable": {"thread_id": make_thread_id(now)}}
    first = graph.invoke(state, config=cfg)
    graph.invoke(Command(resume="yes"), config=cfg)

    # Second invocation a few seconds later with the previous plan as baseline;
    # receding-horizon MPC will produce an essentially identical plan.
    state2 = _base_state(now + timedelta(seconds=5))
    state2["previous_plan"] = first.get("current_plan")
    cfg2 = {"configurable": {"thread_id": make_thread_id(now + timedelta(seconds=5))}}
    second = graph.invoke(state2, config=cfg2)
    # Either the policy auto-accepted or there was no interrupt at all.
    assert "__interrupt__" not in second or (
        second.get("user_confirmation", "").startswith("auto")
    )
