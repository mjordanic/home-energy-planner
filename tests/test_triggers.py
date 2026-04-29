"""Unit tests for TriggerManager + CommitTracker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aerogrid.commit import CommitTracker
from aerogrid.config import EV_AVAILABLE_FROM_HOUR, EV_DAILY_NEED_KWH, HEATER_DEADLINES
from aerogrid.triggers import (
    TriggerManager,
    ev_charging_window_hours,
    time_to_deadline_hours,
)
from aerogrid.types import (
    ApplianceOnset,
    PriceForecast,
    Sample,
    Schedule,
    ScheduledTask,
)


NOW = datetime(2024, 12, 20, 21, 0, tzinfo=timezone.utc)   # 21:00 — inside EV window


def _forecast(median_price: float = 50.0) -> PriceForecast:
    return PriceForecast(slot_start=NOW, median=[median_price] * 8, source="test")


# --------------------------------------------------------------------------- #
# TriggerManager                                                              #
# --------------------------------------------------------------------------- #
def test_initial_call_triggers_periodic():
    tm = TriggerManager()
    out = tm.evaluate(now=NOW)
    assert out is not None and out.kind == "periodic"
    assert "initial" in out.detail


def test_cooldown_suppresses_triggers():
    tm = TriggerManager(cooldown_s=30.0)
    tm.notify_replanned(NOW)
    onset = ApplianceOnset(appliance="dishwasher", timestamp=NOW, confidence=1.0)
    out = tm.evaluate(now=NOW + timedelta(seconds=10), new_onsets=[onset])
    assert out is None


def test_new_onset_fires():
    tm = TriggerManager()
    tm.notify_replanned(NOW - timedelta(minutes=5))
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
    tm.notify_replanned(NOW)    # fresh
    out = tm.evaluate(
        now=NOW + timedelta(seconds=60),   # past cooldown
        new_onsets=[onset],
        committed_tasks=committed,
    )
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


def test_deadline_slip_fires_when_rate_insufficient_inside_window():
    """At 05:00 (inside EV window), 24 kWh in 2 h needs 12 kW; 1 kW current → slip."""
    tm = TriggerManager(deadline_safety=1.0)
    slip_now = datetime(2024, 12, 20, 5, 0, tzinfo=timezone.utc)
    tm.notify_replanned(slip_now - timedelta(minutes=5))
    out = tm.evaluate(
        now=slip_now,
        remaining_ev_kwh=24.0,
        ev_power_setpoint_kw=1.0,
    )
    assert out is not None and out.kind == "deadline_slip"


def test_deadline_slip_suppressed_outside_ev_window():
    """At 14:00 (between 07:00 deadline and 20:00 plug-in), the EV literally
    cannot charge so a 0 kW setpoint is *correct*, not a slip — the trigger
    must not fire.
    """
    tm = TriggerManager(deadline_safety=1.0)
    slip_now = datetime(2024, 12, 20, 14, 0, tzinfo=timezone.utc)
    tm.notify_replanned(slip_now - timedelta(minutes=5))
    out = tm.evaluate(
        now=slip_now,
        remaining_ev_kwh=24.0,
        ev_power_setpoint_kw=0.0,
    )
    assert out is None or out.kind != "deadline_slip"


def test_deadline_slip_uses_charging_window_inside_ev_window():
    """Inside the plug-in window the slip math must use the actual charging
    window, not naive clock time-to-deadline.

    At 22:00 the EV has 9 h to its 07:00 deadline. ev_charging_window_hours
    correctly returns 9 h. Required rate is 24 / 9 ≈ 2.67 kW — so a 0.5 kW
    current rate is a real slip.
    """
    tm = TriggerManager(deadline_safety=1.0)
    slip_now = datetime(2024, 12, 20, 22, 0, tzinfo=timezone.utc)
    tm.notify_replanned(slip_now - timedelta(minutes=5))
    out = tm.evaluate(
        now=slip_now,
        remaining_ev_kwh=24.0,
        ev_power_setpoint_kw=0.5,
    )
    assert out is not None and out.kind == "deadline_slip"


def test_deadline_slip_not_fired_when_charging_fast_enough():
    tm = TriggerManager(deadline_safety=1.0)
    slip_now = datetime(2024, 12, 20, 5, 0, tzinfo=timezone.utc)
    tm.notify_replanned(slip_now - timedelta(minutes=5))
    out = tm.evaluate(
        now=slip_now,
        remaining_ev_kwh=5.0,
        ev_power_setpoint_kw=7.0,
    )
    assert out is None or out.kind != "deadline_slip"


def test_periodic_fires_after_resync_minutes():
    tm = TriggerManager(periodic_minutes=15)
    tm.notify_replanned(NOW)
    assert tm.evaluate(now=NOW + timedelta(minutes=14)) is None
    out = tm.evaluate(now=NOW + timedelta(minutes=16))
    assert out is not None and out.kind == "periodic"


def test_time_to_deadline_hours_rolls_over():
    t = datetime(2024, 12, 20, 2, 0, tzinfo=timezone.utc)
    assert abs(time_to_deadline_hours(t) - 5.0) < 1e-6
    t = datetime(2024, 12, 20, 8, 0, tzinfo=timezone.utc)
    assert abs(time_to_deadline_hours(t) - 23.0) < 1e-6


def test_ev_charging_window_hours_inside_window():
    """At 22:00 we're 2 h into the 20:00→07:00 window → 9 h left."""
    t = datetime(2024, 12, 20, 22, 0, tzinfo=timezone.utc)
    assert abs(ev_charging_window_hours(t) - 9.0) < 1e-6


def test_ev_charging_window_hours_outside_window():
    """At 14:00 the next window is 20:00→07:00 → 11 h."""
    t = datetime(2024, 12, 20, 14, 0, tzinfo=timezone.utc)
    assert abs(ev_charging_window_hours(t) - 11.0) < 1e-6


# --------------------------------------------------------------------------- #
# CommitTracker                                                               #
# --------------------------------------------------------------------------- #
def test_commit_tracker_decrements_ev_kwh():
    ct = CommitTracker(remaining_ev_kwh=7.0, ev_power_setpoint_kw=7.0)
    now = NOW
    for i in range(3600):           # 1 h at 1 Hz
        ct.tick(now + timedelta(seconds=i))
    assert ct.remaining_ev_kwh == pytest.approx(0.0, abs=1e-3)


def test_commit_tracker_resets_at_ev_deadline():
    ct = CommitTracker(remaining_ev_kwh=0.0, ev_power_setpoint_kw=0.0)
    reset_time = NOW.replace(hour=7, minute=0, second=0, microsecond=0)
    ct.tick(reset_time)
    assert ct.remaining_ev_kwh == EV_DAILY_NEED_KWH


def test_commit_tracker_decrements_heater_kwh_in_active_window():
    """Heater setpoint debits the active window's kWh counter."""
    # 02:00 → next deadline is 07:00 → window 7 is active.
    now = datetime(2024, 12, 20, 2, 0, tzinfo=timezone.utc)
    ct = CommitTracker(
        remaining_ev_kwh=0.0,
        heater_power_setpoint_kw=2.0,
    )
    initial = ct.remaining_heater_kwh_by_window[7]
    for i in range(1800):           # 30 min at 1 Hz
        ct.tick(now + timedelta(seconds=i))
    # 2 kW × 0.5 h = 1 kWh delivered to window 7.
    assert ct.remaining_heater_kwh_by_window[7] == pytest.approx(initial - 1.0, abs=1e-2)
    # Other window untouched.
    assert ct.remaining_heater_kwh_by_window[18] == pytest.approx(2.0, abs=1e-3)


def test_commit_tracker_resets_heater_window_at_deadline():
    """Each heater window resets to its required kWh as the deadline passes."""
    ct = CommitTracker(remaining_ev_kwh=0.0)
    ct.remaining_heater_kwh_by_window[7] = 0.0   # window already satisfied
    deadline_passing = NOW.replace(hour=7, minute=0, second=0, microsecond=0)
    ct.tick(deadline_passing)
    expected = next(d.kwh_required for d in HEATER_DEADLINES if d.hour == 7)
    assert ct.remaining_heater_kwh_by_window[7] == pytest.approx(expected, abs=1e-3)


def test_commit_tracker_adopts_plan_with_heater_setpoint():
    """``adopt_plan`` copies both EV and heater first-slot setpoints."""
    ct = CommitTracker(remaining_ev_kwh=10.0)
    plan = Schedule(
        slot_start=NOW,
        horizon_slots=8,
        ev_power_kw=[3.0] * 8,
        heater_power_kw=[1.5] * 8,
        tasks=[],
        expected_cost=1.0,
        baseline_cost=2.0,
    )
    ct.adopt_plan(plan, NOW)
    assert ct.ev_power_setpoint_kw == 3.0
    assert ct.heater_power_setpoint_kw == 1.5


def test_adopt_cycle_start_pins_future_cycle():
    """A future-shifted dishwasher cycle becomes a committed task."""
    ct = CommitTracker(remaining_ev_kwh=0.0)
    start_at = NOW + timedelta(minutes=30)
    ct.adopt_cycle_start(
        appliance="dishwasher",
        slots=8,
        expected_kwh=5.0,
        start_at=start_at,
    )
    assert len(ct.committed_tasks) == 1
    # Cycle ends at start_at + 8 × 15 min = start_at + 2 h.
    ct.tick(start_at + timedelta(minutes=121))
    assert len(ct.committed_tasks) == 0
