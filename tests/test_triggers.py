"""Unit tests for TriggerManager + CommitTracker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aerogrid.commit import CommitTracker
from aerogrid.config import EV_DAILY_NEED_KWH
from aerogrid.triggers import TriggerManager, time_to_deadline_hours
from aerogrid.types import (
    ApplianceOnset,
    PriceForecast,
    Sample,
    Schedule,
    ScheduledTask,
)


NOW = datetime(2024, 12, 20, 20, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# TriggerManager                                                              #
# --------------------------------------------------------------------------- #
def _forecast(median_price: float = 50.0) -> PriceForecast:
    return PriceForecast(slot_start=NOW, median=[median_price] * 8, source="test")


def test_initial_call_triggers_periodic():
    tm = TriggerManager()
    out = tm.evaluate(now=NOW)
    assert out is not None and out.kind == "periodic"
    assert "initial" in out.detail


def test_cooldown_suppresses_triggers():
    tm = TriggerManager(cooldown_s=30.0)
    tm.notify_replanned(NOW)
    # Within cooldown: even a new onset is suppressed.
    onset = ApplianceOnset(appliance="dishwasher", timestamp=NOW, confidence=1.0)
    out = tm.evaluate(now=NOW + timedelta(seconds=10), new_onsets=[onset])
    assert out is None


def test_new_onset_fires():
    tm = TriggerManager()
    tm.notify_replanned(NOW - timedelta(minutes=5))     # clear cooldown
    onset = ApplianceOnset(appliance="dishwasher", timestamp=NOW, confidence=1.0)
    out = tm.evaluate(now=NOW, new_onsets=[onset], committed_tasks=[])
    assert out is not None and out.kind == "new_onset"
    assert out.detail == "dishwasher"


def test_new_onset_suppressed_if_already_committed():
    tm = TriggerManager()
    tm.notify_replanned(NOW - timedelta(minutes=5))
    onset = ApplianceOnset(appliance="dishwasher", timestamp=NOW, confidence=1.0)
    committed = [ScheduledTask(appliance="dishwasher", start_slot=0, slots=8,
                               expected_kwh=5.0, committed=True)]
    # No price surprise, not near deadline → periodic fires only if timer elapsed.
    tm.notify_replanned(NOW)    # fresh
    out = tm.evaluate(
        now=NOW + timedelta(seconds=60),   # past cooldown
        new_onsets=[onset],
        committed_tasks=committed,
    )
    # Not a new-onset trigger (committed) and not-yet-periodic, so no trigger.
    assert out is None or out.kind != "new_onset"


def test_price_surprise_fires():
    tm = TriggerManager(price_deviation=0.25)
    tm.notify_replanned(NOW - timedelta(minutes=5))
    sample = Sample(t=NOW, p_mains_w=500.0, realized_price=100.0)   # 2x forecast
    out = tm.evaluate(
        now=NOW,
        latest_sample=sample,
        price_forecast=_forecast(50.0),
    )
    assert out is not None and out.kind == "price_surprise"


def test_deadline_slip_fires_when_rate_insufficient():
    tm = TriggerManager(deadline_safety=1.0)
    slip_now = datetime(2024, 12, 20, 5, 0, tzinfo=timezone.utc)   # 2 h to 07:00
    tm.notify_replanned(slip_now - timedelta(minutes=5))
    # 24 kWh to go, 2 hours left, EV only charging 1 kW — clearly slipping.
    out = tm.evaluate(
        now=slip_now,
        remaining_ev_kwh=24.0,
        ev_power_setpoint_kw=1.0,
    )
    assert out is not None and out.kind == "deadline_slip"


def test_deadline_slip_not_fired_when_charging_fast_enough():
    tm = TriggerManager(deadline_safety=1.0)
    slip_now = datetime(2024, 12, 20, 5, 0, tzinfo=timezone.utc)
    tm.notify_replanned(slip_now - timedelta(minutes=5))
    # 5 kWh left, 2 h to go, charging 7 kW → fine (7 kW >> 2.5 kW required).
    out = tm.evaluate(
        now=slip_now,
        remaining_ev_kwh=5.0,
        ev_power_setpoint_kw=7.0,
    )
    # Should only fire if periodic timer elapsed — last replan was 5 min ago so no
    # periodic either.
    assert out is None or out.kind != "deadline_slip"


def test_periodic_fires_after_resync_minutes():
    tm = TriggerManager(periodic_minutes=15)
    tm.notify_replanned(NOW)
    assert tm.evaluate(now=NOW + timedelta(minutes=14)) is None
    out = tm.evaluate(now=NOW + timedelta(minutes=16))
    assert out is not None and out.kind == "periodic"


def test_time_to_deadline_hours_rolls_over():
    # At 02:00, deadline is 07:00 → 5 h away.
    t = datetime(2024, 12, 20, 2, 0, tzinfo=timezone.utc)
    assert abs(time_to_deadline_hours(t) - 5.0) < 1e-6
    # At 08:00, deadline is 07:00 tomorrow → 23 h away.
    t = datetime(2024, 12, 20, 8, 0, tzinfo=timezone.utc)
    assert abs(time_to_deadline_hours(t) - 23.0) < 1e-6


# --------------------------------------------------------------------------- #
# CommitTracker                                                               #
# --------------------------------------------------------------------------- #
def test_commit_tracker_decrements_ev_kwh():
    ct = CommitTracker(remaining_ev_kwh=7.0, ev_power_setpoint_kw=7.0)
    now = NOW
    for i in range(3600):           # 1 h at 1 Hz
        ct.tick(now + timedelta(seconds=i))
    # 7 kW × 1 h = 7 kWh delivered → remaining ≈ 0.
    assert ct.remaining_ev_kwh == pytest.approx(0.0, abs=1e-3)


def test_commit_tracker_resets_at_deadline():
    ct = CommitTracker(remaining_ev_kwh=0.0, ev_power_setpoint_kw=0.0)
    # Step through 06:59 → 07:00 boundary. Tick at exactly hour=7, minute=0, second=0.
    reset_time = NOW.replace(hour=7, minute=0, second=0, microsecond=0)
    ct.tick(reset_time)
    assert ct.remaining_ev_kwh == EV_DAILY_NEED_KWH


def test_commit_tracker_adopts_plan_and_retires_task():
    ct = CommitTracker(remaining_ev_kwh=10.0)
    plan = Schedule(
        slot_start=NOW,
        horizon_slots=8,
        ev_power_kw=[3.0] * 8,
        tasks=[
            ScheduledTask(appliance="dishwasher", start_slot=0, slots=4, expected_kwh=2.0),
            ScheduledTask(appliance="washing_machine", start_slot=3, slots=6, expected_kwh=3.0),
        ],
        expected_cost=1.0,
        baseline_cost=2.0,
    )
    ct.adopt_plan(plan, NOW)
    assert ct.ev_power_setpoint_kw == 3.0
    # Only dishwasher (start_slot=0) is committed.
    assert {t.appliance for t in ct.committed_tasks} == {"dishwasher"}

    # After the dishwasher's 4×15 = 60 min cycle, it retires on tick.
    ct.tick(NOW + timedelta(minutes=61))
    assert len(ct.committed_tasks) == 0
