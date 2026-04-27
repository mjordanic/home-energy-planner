"""Unit tests for the HITL decision policy."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aerogrid.hitl_policy import _in_sleep_window, decide
from aerogrid.types import Schedule, ScheduledTask


SLOT_START = datetime(2024, 12, 20, 0, 0, tzinfo=timezone.utc)     # midnight


def _plan(
    *,
    ev_setpoint_kw: float = 3.0,
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
        tasks=tasks,
        expected_cost=cost,
        baseline_cost=cost * 1.5,
    )


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
    # Old: 20:00 → New: 23:00 (into sleep hours).
    old = _plan(dish_start=20 * 4)
    new = _plan(dish_start=23 * 4)
    d = decide(old, new)
    assert d.action == "ask"
    assert "sleep" in d.reason.lower() or "overnight" in d.question.lower()


def test_committed_task_is_not_prompted():
    old = _plan(dish_start=20 * 4, dish_committed=True)
    # Shift in the proposed plan (but new also marks it committed — the policy
    # skips when either side is committed).
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
