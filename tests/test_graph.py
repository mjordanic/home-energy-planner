"""Tests for LangGraph assembly + HITL interrupt/resume on the slow path."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from aerogrid.config import BatterySpec
from aerogrid.graph import build_graph, make_thread_id
from aerogrid.price_oracle import SeasonalNaiveOracle
from aerogrid.types import ApplianceOnset, PriceForecast, ReplanTrigger, Schedule


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


# --------------------------------------------------------------------------- #
# run-now vs proposed cost on a new onset (base load + battery like for like) #
# --------------------------------------------------------------------------- #
class _StepOracle(SeasonalNaiveOracle):
    """130 €/MWh for the first 2 h, 100 €/MWh after: a dishwasher shift saves
    5 kWh × 0.03 €/kWh = €0.15, above the €0.10 ask threshold."""

    def get_15min_forecast(self, now, context, horizon_slots=96):
        median = [130.0] * 8 + [100.0] * (horizon_slots - 8)
        return PriceForecast(slot_start=now, median=median, source="step")


def _onset_state(now: datetime, *, battery: bool) -> dict:
    state = _base_state(now, remaining_ev_kwh=0.0)
    state["replan_trigger"] = ReplanTrigger(kind="new_onset", detail="dishwasher", at=now)
    state["new_onsets"] = [ApplianceOnset(appliance="dishwasher", timestamp=now, confidence=1.0)]
    if battery:
        state["battery_spec"] = BatterySpec()
        state["soc_kwh"] = 5.0
    return state


@pytest.mark.parametrize("battery", [False, True])
def test_run_now_cost_includes_base_load_and_battery(fake_prices, battery):
    builder, checkpointer = build_graph(
        price_oracle=_StepOracle(),
        price_history_provider=_providers(fake_prices),
        auto_confirm=False,
    )
    graph = builder.compile(checkpointer=checkpointer)
    now = datetime(2024, 11, 20, 10, 0, tzinfo=timezone.utc)
    cfg = {"configurable": {"thread_id": f"onset-{battery}"}}
    first = graph.invoke(_onset_state(now, battery=battery), config=cfg)
    proposal = graph.get_state(cfg).values["pending_reschedule"]
    assert proposal is not None, "dishwasher should be shifted out of the 130 €/MWh slots"
    # Pinning the cycle at slot 0 is a restriction of the same problem, so
    # the run-now plan can never be cheaper than the free optimum.
    assert proposal.cost_now_eur >= proposal.cost_proposed_eur - 1e-6
    # Declining commits the pinned run-now plan, which must be a real plan:
    # the dishwasher at slot 0 and, with a battery, the battery dispatched.
    if "__interrupt__" in first:
        graph.invoke(Command(resume="decline"), config=cfg)
    committed = graph.get_state(cfg).values["current_plan"]
    assert committed.cycle_starts["dishwasher"] == 0
    assert committed.expected_cost == pytest.approx(proposal.cost_now_eur)
    if battery:
        assert len(committed.soc_kwh) == committed.horizon_slots
        assert max(committed.battery_discharge_kw) > 0


def test_no_battery_real_saving_reaches_approval_question(fake_prices):
    builder, checkpointer = build_graph(
        price_oracle=_StepOracle(),
        price_history_provider=_providers(fake_prices),
        auto_confirm=False,
    )
    graph = builder.compile(checkpointer=checkpointer)
    now = datetime(2024, 11, 20, 10, 0, tzinfo=timezone.utc)
    cfg = {"configurable": {"thread_id": "onset-ask"}}
    first = graph.invoke(_onset_state(now, battery=False), config=cfg)
    assert "__interrupt__" in first, "a €0.15 saving should ask the user"
    payload = first["__interrupt__"][0].value
    assert payload["reschedule"]["appliance"] == "dishwasher"
    assert payload["reschedule"]["savings_eur"] == pytest.approx(0.15, abs=0.02)


def test_pending_cycle_window_is_measured_from_the_original_onset():
    from aerogrid.graph import _pending_cycles_from_onsets

    onset = datetime(2024, 11, 20, 17, 30, tzinfo=timezone.utc)
    (fresh,) = _pending_cycles_from_onsets(
        [ApplianceOnset("dishwasher", onset, 1.0)], set(), 96, now=onset,
    )
    assert fresh.latest_start_slot == 8                 # full 2 h window
    (re_offered,) = _pending_cycles_from_onsets(
        [ApplianceOnset("dishwasher", onset, 1.0)], set(), 96,
        now=onset + timedelta(hours=1, minutes=30),
    )
    assert re_offered.latest_start_slot == 2            # only 30 min left
    (expired,) = _pending_cycles_from_onsets(
        [ApplianceOnset("dishwasher", onset, 1.0)], set(), 96,
        now=onset + timedelta(hours=3),
    )
    assert expired.latest_start_slot == 0               # window over → run now
