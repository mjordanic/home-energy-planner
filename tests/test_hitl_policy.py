"""Unit tests for the HITL decision policy."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aerogrid.config import HITL_RESCHEDULE_MIN_SAVINGS_EUR
from aerogrid.hitl_policy import _in_sleep_window, decide, decide_reschedule
from aerogrid.types import RescheduleProposal, Schedule, ScheduledTask


SLOT_START = datetime(2024, 12, 20, 0, 0, tzinfo=timezone.utc)     # midnight


def _plan(
    *,
    ev_setpoint_kw: float = 3.0,
    heater_setpoint_kw: float = 1.0,
    dish_start: int = 20 * 4,    # 20:00
    wash_start: int | None = None,
    cost: float = 1.0,
    dish_committed: bool = False,
) -> Schedule:
    tasks = [
        ScheduledTask(
            appliance="dishwasher", start_slot=dish_start, slots=8, expected_kwh=5.0,
            committed=dish_committed,
        )
    ]
    if wash_start is not None:
        tasks.append(
            ScheduledTask(
                appliance="washing_machine", start_slot=wash_start, slots=6,
                expected_kwh=3.6,
            )
        )
    return Schedule(
        slot_start=SLOT_START,
        horizon_slots=96,
        ev_power_kw=[ev_setpoint_kw] * 96,
        heater_power_kw=[heater_setpoint_kw] * 96,
        tasks=tasks,
        expected_cost=cost,
        baseline_cost=cost * 1.5,
    )


# --------------------------------------------------------------------------- #
# Plan-level decisions                                                        #
# --------------------------------------------------------------------------- #
def test_first_plan_always_asks():
    d = decide(old_plan=None, new_plan=_plan())
    assert d.action == "ask"
    assert "first" in d.reason


def test_no_new_plan_auto():
    d = decide(old_plan=_plan(), new_plan=None)
    assert d.action == "auto"


def test_tiny_ev_shift_is_auto():
    old = _plan(ev_setpoint_kw=3.0)
    new = _plan(ev_setpoint_kw=3.5)  # Δ = 0.5 kW, within 1.5 kW tolerance
    assert decide(old, new).action == "auto"


def test_large_ev_shift_asks():
    old = _plan(ev_setpoint_kw=3.0)
    new = _plan(ev_setpoint_kw=6.0)
    d = decide(old, new)
    assert d.action == "ask"
    assert "EV" in d.reason


def test_small_appliance_shift_is_auto():
    old = _plan(dish_start=20 * 4)    # 20:00
    new = _plan(dish_start=20 * 4 + 1)  # 20:15 → 15-min shift, below 30-min threshold
    assert decide(old, new).action == "auto"


def test_large_appliance_shift_asks():
    old = _plan(dish_start=20 * 4)                     # 20:00
    new = _plan(dish_start=20 * 4 + 3)                 # 20:45 = 45 min shift
    d = decide(old, new)
    assert d.action == "ask"
    assert "dishwasher" in d.reason


def test_new_appliance_asks():
    old = _plan(wash_start=None)
    new = _plan(wash_start=10 * 4)
    d = decide(old, new)
    assert d.action == "ask"
    assert "washing_machine" in d.reason


def test_shift_into_sleep_window_asks():
    old = _plan(dish_start=20 * 4)
    new = _plan(dish_start=23 * 4)
    d = decide(old, new)
    assert d.action == "ask"
    assert "sleep" in d.reason.lower() or "overnight" in d.question.lower()


def test_committed_task_is_not_prompted():
    old = _plan(dish_start=20 * 4, dish_committed=True)
    new = _plan(dish_start=21 * 4, dish_committed=True)
    assert decide(old, new).action == "auto"


def test_cost_bump_asks():
    old = _plan(cost=1.00)
    new = _plan(cost=1.75)       # +0.75 > 0.50 threshold
    d = decide(old, new)
    assert d.action == "ask"
    assert "cost" in d.reason.lower()


def test_sleep_window_wrap_around():
    from datetime import time
    s = time(22, 0)
    e = time(6, 0)
    assert _in_sleep_window(time(23, 0), s, e)
    assert _in_sleep_window(time(5, 30), s, e)
    assert not _in_sleep_window(time(12, 0), s, e)
    assert not _in_sleep_window(time(21, 59), s, e)


# --------------------------------------------------------------------------- #
# Reschedule proposals                                                        #
# --------------------------------------------------------------------------- #
def _proposal(
    *,
    appliance: str = "dishwasher",
    shift_min: float = 90.0,
    cost_now: float = 1.20,
    cost_proposed: float = 0.85,
) -> RescheduleProposal:
    onset_at = SLOT_START.replace(hour=20, minute=0)
    return RescheduleProposal(
        appliance=appliance,
        onset_at=onset_at,
        proposed_start_at=onset_at + timedelta(minutes=shift_min),
        cycle_slots=8,
        rated_kw=2.5,
        cost_now_eur=cost_now,
        cost_proposed_eur=cost_proposed,
    )


def test_reschedule_no_proposal_is_auto():
    d = decide_reschedule(None)
    assert d.action == "auto"


def test_reschedule_zero_shift_is_auto():
    p = _proposal(shift_min=0.0)
    d = decide_reschedule(p)
    assert d.action == "auto"


def test_reschedule_big_savings_is_ask():
    p = _proposal(shift_min=90.0, cost_now=1.20, cost_proposed=0.85)
    d = decide_reschedule(p)
    assert d.action == "ask"
    assert "dishwasher" in d.question
    assert "0.35" in d.question or "0.4" in d.question  # approx € savings


def test_reschedule_below_threshold_is_auto():
    """Savings below HITL_RESCHEDULE_MIN_SAVINGS_EUR → auto-decline."""
    p = _proposal(
        shift_min=90.0,
        cost_now=1.00,
        cost_proposed=1.00 - HITL_RESCHEDULE_MIN_SAVINGS_EUR / 2,
    )
    d = decide_reschedule(p)
    assert d.action == "auto"


def test_reschedule_question_mentions_appliance_and_savings():
    p = _proposal(appliance="washing_machine", shift_min=60.0, cost_now=2.0, cost_proposed=1.5)
    d = decide_reschedule(p)
    assert "washing_machine" in d.question
    assert "1.0" in d.question or "1 h" in d.question
    assert "0.5" in d.question
