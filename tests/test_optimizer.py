"""Tests for the receding-horizon LP optimizer.

After the April 2026 refactor the optimiser controls two continuous loads
(EV charger, heater) and no longer schedules cycle-shaped dishwasher /
washing-machine starts (those moved to the HITL reschedule path). The
tests below pin the new semantics:

* EV-availability gate (no charging before EV_AVAILABLE_FROM_HOUR).
* EV deadline regimes (inside vs. outside horizon).
* Heater per-window energy delivery.
* House power cap couples EV and heater.
* Soft slack absorbs infeasibility.
* Committed cycle tasks consume cap headroom.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from aerogrid.config import (
    APPLIANCES,
    EV_AVAILABLE_FROM_HOUR,
    EV_DAILY_NEED_KWH,
    HEATER_DEADLINES,
    SHORT_HORIZON_SLOTS,
    HeaterEnergyDeadline,
)
from aerogrid.optimizer import solve_receding_horizon
from aerogrid.types import PendingCycle, ScheduledTask


def _pc(appliance: str, *, latest_start: int = 8) -> PendingCycle:
    """Build a PendingCycle for ``appliance`` allowing shifts up to slot ``latest_start``."""
    spec = APPLIANCES[appliance]
    return PendingCycle(
        appliance=appliance,
        cycle_slots=int(spec.cycle_slots),
        rated_kw=float(spec.rated_kw),
        earliest_start_slot=0,
        latest_start_slot=int(min(latest_start, H_DAY - spec.cycle_slots)),
    )


H_DAY = SHORT_HORIZON_SLOTS                # 96 slots = 24 h
H_SHORT = 8                                 # 2 h, used for tight-deadline tests


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _flat_prices(n: int = H_DAY, val: float = 50.0) -> np.ndarray:
    return np.full(n, val, dtype=float)


def _cheap_overnight(n: int = H_DAY) -> np.ndarray:
    """Price profile: cheap 20:00–06:00, expensive 06:00–20:00.

    Slot 0 = 00:00 UTC. Cheap span is slots 0..24 and 80..95 (overnight halves).
    """
    p = np.full(n, 80.0)
    p[0:24] = 20.0
    p[80:96] = 20.0
    return p


# --------------------------------------------------------------------------- #
# EV: availability gate                                                       #
# --------------------------------------------------------------------------- #
def test_ev_zero_outside_charging_window():
    """EV charging is forbidden before EV_AVAILABLE_FROM_HOUR each day."""
    now = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)   # 14:00 UTC
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=24.0,
    )
    ev = np.asarray(sched.ev_power_kw)
    # Slots 0..23 cover 14:00–20:00 → all closed.
    assert ev[:24].sum() == pytest.approx(0.0, abs=1e-6)
    # Some power must flow afterwards to satisfy the deadline.
    assert ev[24:].sum() > 0.0


def test_ev_full_charge_inside_window():
    """24 kWh is delivered entirely inside the 11-hour overnight window."""
    now = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=24.0,
    )
    delivered = np.asarray(sched.ev_power_kw).sum() * 0.25
    assert delivered == pytest.approx(24.0, abs=1e-3)


def test_ev_overnight_window_carries_full_night():
    """At midnight (already inside the 20:00–07:00 window) the optimiser uses
    the full remaining 7 hours, not just the slots after the next 20:00."""
    now = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)   # 00:00 UTC
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=10.0,
    )
    ev = np.asarray(sched.ev_power_kw)
    # 00:00–07:00 is open → slots 0..27. 07:00–20:00 closed → 28..79.
    # 20:00 next day onwards open again → 80..95.
    assert ev[28:80].sum() == pytest.approx(0.0, abs=1e-6)
    delivered = ev.sum() * 0.25
    assert delivered == pytest.approx(10.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# EV: deadline regimes                                                        #
# --------------------------------------------------------------------------- #
def test_ev_deadline_inside_short_horizon():
    """If deadline is inside a short horizon the constraint is hard-tight."""
    now = datetime(2026, 4, 15, 5, 0, tzinfo=timezone.utc)   # 2 h to 07:00
    prices = np.full(H_SHORT, 40.0)
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_SHORT, remaining_ev_kwh=10.0,
    )
    delivered = np.asarray(sched.ev_power_kw).sum() * 0.25
    assert delivered == pytest.approx(10.0, abs=1e-3)


def test_ev_soft_slack_absorbs_infeasibility():
    """If the available kWh exceeds the physical max in the window, slack saves us."""
    now = datetime(2026, 4, 15, 5, 0, tzinfo=timezone.utc)   # 2 h to deadline
    prices = np.full(H_SHORT, 40.0)
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_SHORT, remaining_ev_kwh=100.0,    # impossible
    )
    assert sched.solver_status in ("optimal", "optimal_inaccurate")
    # Plan charges at rated power the whole horizon (open from 05:00–07:00).
    ev = np.asarray(sched.ev_power_kw)
    assert ev.sum() > 0.0


# --------------------------------------------------------------------------- #
# Heater: per-window energy delivery                                          #
# --------------------------------------------------------------------------- #
def test_heater_delivers_required_per_window():
    """Default deadlines: 4 kWh by 07:00, 2 kWh by 18:00."""
    now = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
    )
    # We expect both deadlines to be met within the 24h horizon.
    assert sched.heater_window_kwh[7] == pytest.approx(4.0, abs=1e-3)
    assert sched.heater_window_kwh[18] == pytest.approx(2.0, abs=1e-3)


def test_heater_window_partial_remaining():
    """If a window has only part of its kWh outstanding, only that kWh is delivered."""
    now = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        remaining_heater_kwh_by_window={7: 1.5, 18: 0.0},
    )
    assert sched.heater_window_kwh[7] == pytest.approx(1.5, abs=1e-3)
    assert sched.heater_window_kwh[18] == pytest.approx(0.0, abs=1e-3)


def test_heater_picks_cheapest_slots_in_window():
    """Within an overnight window, the heater concentrates power in the cheapest hour."""
    now = datetime(2026, 4, 15, 18, 0, tzinfo=timezone.utc)
    # Overnight window = 18:00 → 07:00. Slots 0..51 of the 24h horizon.
    # Make slot 16 (=22:00) the cheapest.
    prices = np.full(H_DAY, 80.0)
    prices[14:18] = 10.0
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
    )
    heat = np.asarray(sched.heater_power_kw)
    cheap_kwh = heat[14:18].sum() * 0.25
    # The 4 kWh need fits inside 4 slots × 2 kW × 0.25 h = 2 kWh per slot.
    # 4 cheap slots can hold 2 kWh; remaining 2 kWh land in next-cheapest slots.
    assert cheap_kwh >= 1.99


def test_heater_zero_when_window_satisfied():
    """If both windows already hold ≥ required kWh, the heater stays off."""
    now = datetime(2026, 4, 15, 6, 0, tzinfo=timezone.utc)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        remaining_heater_kwh_by_window={7: 0.0, 18: 0.0},
    )
    heat = np.asarray(sched.heater_power_kw)
    assert heat.sum() == pytest.approx(0.0, abs=1e-3)


def test_heater_soft_slack_when_window_too_short():
    """If kwh_required > rated × window_h, slack makes the LP feasible."""
    now = datetime(2026, 4, 15, 5, 0, tzinfo=timezone.utc)
    # Custom deadline: 100 kWh required by 07:00 — physically impossible
    # (2 kW × 2 h = 4 kWh max).
    custom = (HeaterEnergyDeadline(hour=7, kwh_required=100.0),)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        heater_deadlines=custom,
    )
    assert sched.solver_status in ("optimal", "optimal_inaccurate")
    heat = np.asarray(sched.heater_power_kw)
    assert heat.max() <= APPLIANCES["heater"].rated_kw + 1e-6


# --------------------------------------------------------------------------- #
# House cap                                                                   #
# --------------------------------------------------------------------------- #
def test_house_cap_couples_ev_and_heater():
    """EV + heater + committed cycles never exceed the house cap."""
    now = datetime(2026, 4, 15, 18, 0, tzinfo=timezone.utc)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=24.0,
    )
    ev = np.asarray(sched.ev_power_kw)
    heat = np.asarray(sched.heater_power_kw)
    assert (ev + heat).max() <= 10.0 + 1e-6


def test_committed_cycle_consumes_cap_headroom():
    """A committed dishwasher cycle reduces EV+heater cap headroom."""
    now = datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc)
    committed = [
        ScheduledTask(
            appliance="dishwasher", start_slot=0, slots=8,
            expected_kwh=APPLIANCES["dishwasher"].rated_kw * 8 * 0.25,
            committed=True,
        )
    ]
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY,
        remaining_ev_kwh=24.0, committed_tasks=committed,
        house_cap_kw=8.5,
    )
    # During the committed-cycle slots (0..7) cap is 8.5 − 2.5 = 6.0 kW for
    # everything else (EV + heater).
    ev = np.asarray(sched.ev_power_kw)
    heat = np.asarray(sched.heater_power_kw)
    dish_rated = APPLIANCES["dishwasher"].rated_kw
    for t in range(8):
        assert ev[t] + heat[t] + dish_rated <= 8.5 + 1e-6


# --------------------------------------------------------------------------- #
# Schedule fields                                                             #
# --------------------------------------------------------------------------- #
def test_schedule_includes_heater_window_kwh():
    """``Schedule.heater_window_kwh`` echoes per-deadline kWh planned."""
    now = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
    )
    assert set(sched.heater_window_kwh.keys()) == {d.hour for d in HEATER_DEADLINES}
    for h in sched.heater_window_kwh:
        assert sched.heater_window_kwh[h] >= 0.0


def test_baseline_cost_uses_naive_window_charging():
    """Baseline ≥ optimal whenever prices are non-flat."""
    now = datetime(2026, 4, 15, 18, 0, tzinfo=timezone.utc)
    prices = _cheap_overnight()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=24.0,
    )
    # We'd expect savings > 0 because the optimiser shifts EV into cheap slots.
    assert sched.expected_cost <= sched.baseline_cost + 1e-6


# --------------------------------------------------------------------------- #
# Joint MIP — pending cycle placement                                         #
# --------------------------------------------------------------------------- #
def test_pending_cycle_runs_exactly_once_in_window():
    """A pending cycle must run exactly once inside its allowed window."""
    now = datetime(2026, 4, 15, 19, 45, tzinfo=timezone.utc)
    prices = _flat_prices()
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    assert "dishwasher" in sched.cycle_starts
    s = sched.cycle_starts["dishwasher"]
    assert 0 <= s <= 8


def test_pending_cycle_shifts_into_cheap_window():
    """Joint MIP shifts a pending cycle into the cheap forward window —
    analogous to Scenario G in the notebook.

    Onset at 19:45 with a sharp peak at slots 0..3 followed by a deeper
    valley narrow enough that only one 8-slot start lies entirely in cheap
    territory.
    """
    now = datetime(2026, 4, 15, 19, 45, tzinfo=timezone.utc)
    prices = np.full(H_DAY, 60.0)
    prices[0:4] = 250.0
    prices[4:12] = 25.0
    prices[12:20] = 200.0    # bracket the valley so only slot 4 fits the whole cycle
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    # Only slot 4 keeps the full 8-slot cycle inside the cheap valley.
    assert sched.cycle_starts["dishwasher"] == 4


def test_pending_cycle_runs_now_when_no_cheaper_option():
    """If no candidate slot is cheaper than slot 0, the MIP runs at slot 0."""
    now = datetime(2026, 4, 15, 19, 45, tzinfo=timezone.utc)
    prices = np.full(H_DAY, 60.0)
    prices[0:8] = 20.0       # cheapest 8 slots align with run-now
    prices[8:16] = 200.0
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    assert sched.cycle_starts["dishwasher"] == 0


def test_joint_solve_respects_house_cap_with_pending_cycle():
    """EV + heater + pending dishwasher never exceed the house cap.

    With a tight 8 kW cap, EV (rated 7 kW) + dishwasher (2.5 kW) cannot
    run flat-out simultaneously. The MIP must reduce one of them in any
    overlapping slot.
    """
    now = datetime(2026, 4, 15, 21, 0, tzinfo=timezone.utc)
    prices = np.full(H_DAY, 80.0)
    prices[4:8] = 20.0       # cheapest hour: slots 4–7
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY,
        remaining_ev_kwh=12.0, house_cap_kw=8.0,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    s = sched.cycle_starts["dishwasher"]
    cycle_slots = APPLIANCES["dishwasher"].cycle_slots
    rated = APPLIANCES["dishwasher"].rated_kw
    ev = np.asarray(sched.ev_power_kw)
    heat = np.asarray(sched.heater_power_kw)
    dish = np.zeros(H_DAY)
    dish[s : s + cycle_slots] = rated
    assert (ev + heat + dish).max() <= 8.0 + 1e-6


def test_joint_solve_avoids_cap_violation_via_shift():
    """When the cap is so tight that running the cycle now would force EV slack,
    the MIP shifts the cycle far enough that EV can complete at full power.
    """
    # Cap = 7 kW (only enough for EV alone). EV needs 7 kWh in the next 1 h
    # (= 7 kW for 4 slots). Cycle slots 0..7 are blocked for the cycle if
    # it would force EV to 4.5 kW, missing the deadline.
    now = datetime(2026, 4, 15, 4, 0, tzinfo=timezone.utc)
    prices = np.full(H_DAY, 80.0)
    sched = solve_receding_horizon(
        now, prices,
        horizon_slots=H_DAY,
        remaining_ev_kwh=7.0, time_to_deadline_h=1.0,
        house_cap_kw=7.0,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    # The MIP must place the cycle after the EV finishes (slot 4 onward).
    assert sched.cycle_starts["dishwasher"] >= 4
    # And the EV deadline is honoured (no slack).
    ev_kwh = float(np.asarray(sched.ev_power_kw)[:4].sum() * 0.25)
    assert ev_kwh >= 7.0 - 1e-3


def test_pending_cycle_cost_is_in_expected_cost():
    """``expected_cost`` includes the pending cycle's energy cost contribution."""
    now = datetime(2026, 4, 15, 19, 45, tzinfo=timezone.utc)
    prices = np.full(H_DAY, 60.0)
    sched_with = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    sched_without = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
    )
    cycle_slots = APPLIANCES["dishwasher"].cycle_slots
    rated = APPLIANCES["dishwasher"].rated_kw
    expected_cycle_cost = cycle_slots * rated * 0.25 * 60.0 / 1000.0  # = 0.30
    delta = sched_with.expected_cost - sched_without.expected_cost
    assert abs(delta - expected_cycle_cost) < 1e-3


def test_pending_cycle_cost_is_in_baseline_cost():
    """``baseline_cost`` must include the pending cycle's energy so savings are not artificially negative."""
    now = datetime(2026, 4, 15, 19, 45, tzinfo=timezone.utc)
    prices = np.full(H_DAY, 60.0)
    sched_with = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    sched_without = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
    )
    cycle_slots = APPLIANCES["dishwasher"].cycle_slots
    rated = APPLIANCES["dishwasher"].rated_kw
    expected_cycle_cost = cycle_slots * rated * 0.25 * 60.0 / 1000.0
    delta = sched_with.baseline_cost - sched_without.baseline_cost
    assert abs(delta - expected_cycle_cost) < 1e-3, (
        f"baseline_cost delta {delta:.4f} should equal cycle cost {expected_cycle_cost:.4f}; "
        "savings will be wrong without it"
    )


def test_pending_cycle_filtered_when_already_committed():
    """If the same appliance is already in committed_tasks, the pending entry is skipped."""
    now = datetime(2026, 4, 15, 19, 45, tzinfo=timezone.utc)
    prices = _flat_prices()
    cycle_slots = APPLIANCES["dishwasher"].cycle_slots
    committed = [
        ScheduledTask(
            appliance="dishwasher", start_slot=2, slots=cycle_slots,
            expected_kwh=APPLIANCES["dishwasher"].rated_kw * cycle_slots * 0.25,
            committed=True,
        )
    ]
    sched = solve_receding_horizon(
        now, prices, horizon_slots=H_DAY, remaining_ev_kwh=0.0,
        committed_tasks=committed,
        pending_cycles=[_pc("dishwasher", latest_start=8)],
    )
    # The pending entry was filtered → no cycle_starts entry emitted.
    assert "dishwasher" not in sched.cycle_starts
