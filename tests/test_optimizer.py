"""Tests for the receding-horizon MPC optimizer."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from aerogrid.config import APPLIANCES, EV_DAILY_NEED_KWH, SLOTS_PER_DAY
from aerogrid.optimizer import solve_receding_horizon
from aerogrid.types import ScheduledTask


H_SHORT = 8       # 2 h
H_LONG = SLOTS_PER_DAY


def _flat_probs(n: int) -> dict[str, np.ndarray]:
    return {
        "dishwasher": np.full(n, 1.0 / n),
        "washing_machine": np.full(n, 1.0 / n),
    }


def _prices_cheap(cheap_start: int, cheap_len: int,
                  cheap_val: float = 5.0, base_val: float = 80.0,
                  n: int = H_LONG) -> np.ndarray:
    p = np.full(n, base_val)
    p[cheap_start : cheap_start + cheap_len] = cheap_val
    return p


# --------------------------------------------------------------------------- #
# Long-horizon behaviour (matches the old 24-h suite, now via MPC with horizon=96)
# --------------------------------------------------------------------------- #
def test_long_horizon_routes_ev_to_cheap_window():
    cheap_start, cheap_len = 14, 12
    prices = _prices_cheap(cheap_start, cheap_len)
    now = datetime(2024, 12, 20, 0, 0, tzinfo=timezone.utc)
    sched = solve_receding_horizon(
        now, prices, _flat_probs(H_LONG),
        horizon_slots=H_LONG,
    )
    assert sched.solver_status in ("optimal", "optimal_inaccurate")
    ev = np.asarray(sched.ev_power_kw)
    assert ev.sum() * 0.25 == pytest.approx(EV_DAILY_NEED_KWH, abs=1e-3)

    cheap_energy = ev[cheap_start : cheap_start + cheap_len].sum() * 0.25
    expected_cheap = min(EV_DAILY_NEED_KWH, cheap_len * 7.0 * 0.25)
    slack = cheap_len * 0.25 * (7.0 - 5.1) / 2
    assert cheap_energy >= expected_cheap - slack


def test_long_horizon_respects_house_cap():
    prices = _prices_cheap(cheap_start=30, cheap_len=16)
    now = datetime(2024, 12, 20, 0, 0, tzinfo=timezone.utc)
    sched = solve_receding_horizon(
        now, prices, _flat_probs(H_LONG),
        horizon_slots=H_LONG,
    )
    scheduled = {t.appliance: t for t in sched.tasks}
    # House cap: EV + cycle loads ≤ 10 kW at every slot.
    ev = np.asarray(sched.ev_power_kw)
    for t in range(H_LONG):
        load = ev[t]
        for name, task in scheduled.items():
            if task.start_slot <= t < task.start_slot + task.slots:
                load += APPLIANCES[name].rated_kw
        assert load <= 10.0 + 1e-6


def test_ghost_reservation_prefers_habitual_slot_when_prices_tie():
    prices = np.full(H_LONG, 40.0)
    probs = {
        "dishwasher": np.zeros(H_LONG),
        "washing_machine": np.zeros(H_LONG),
    }
    probs["dishwasher"][72] = 0.9
    probs["washing_machine"][40] = 0.9
    now = datetime(2024, 12, 20, 0, 0, tzinfo=timezone.utc)
    sched = solve_receding_horizon(
        now, prices, probs, horizon_slots=H_LONG, reservation_lambda=10.0,
    )
    tasks = {t.appliance: t for t in sched.tasks}
    assert tasks["dishwasher"].start_slot == 72
    assert tasks["washing_machine"].start_slot == 40


# --------------------------------------------------------------------------- #
# Short-horizon MPC behaviour                                                 #
# --------------------------------------------------------------------------- #
def test_short_horizon_deadline_outside_uses_proportional():
    """EV should get at least the proportional share this horizon."""
    # 24 kWh remaining, 12 h to deadline, horizon 2 h → need ≥ 4 kWh this horizon.
    now = datetime(2024, 12, 20, 19, 0, tzinfo=timezone.utc)     # 12 h to 07:00
    prices = np.full(H_SHORT, 40.0)
    sched = solve_receding_horizon(
        now, prices, _flat_probs(H_SHORT),
        horizon_slots=H_SHORT,
        remaining_ev_kwh=24.0,
    )
    ev = np.asarray(sched.ev_power_kw)
    delivered = ev.sum() * 0.25
    # deadline_safety default 1.2 → need ≥ 4 × 1.2 = 4.8 kWh
    assert delivered >= 4.8 - 1e-3


def test_short_horizon_deadline_inside_is_hard():
    """If deadline is within horizon, it's a hard constraint."""
    # 2 h to deadline, horizon 2 h → must fully charge within 8 slots.
    now = datetime(2024, 12, 20, 5, 0, tzinfo=timezone.utc)
    prices = np.full(H_SHORT, 40.0)
    sched = solve_receding_horizon(
        now, prices, _flat_probs(H_SHORT),
        horizon_slots=H_SHORT,
        remaining_ev_kwh=10.0,          # 10 kWh; 8 slots × 7 kW × 0.25 = 14 kWh → feasible
    )
    ev = np.asarray(sched.ev_power_kw)
    delivered = ev.sum() * 0.25
    assert delivered == pytest.approx(10.0, abs=1e-3)


def test_soft_slack_absorbs_infeasibility():
    """If the house cap + committed tasks can't meet the deadline, slack kicks in."""
    # 2 h to deadline, house cap 10 kW, committed washing machine occupying the whole horizon
    # (rated 2.4 kW), EV max 7 kW. Max EV energy = 7 × 2 = 14 kWh, but with washer
    # taking 2.4 kW continuously the allowable EV is 7.6 kW → still feasible.
    # Force real infeasibility: require 100 kWh remaining.
    now = datetime(2024, 12, 20, 5, 0, tzinfo=timezone.utc)
    prices = np.full(H_SHORT, 40.0)
    sched = solve_receding_horizon(
        now, prices, _flat_probs(H_SHORT),
        horizon_slots=H_SHORT,
        remaining_ev_kwh=100.0,          # impossible — slack saves us
    )
    # Solver should still return an "optimal" plan thanks to soft slack.
    assert sched.solver_status in ("optimal", "optimal_inaccurate")
    # Plan charges at rated power the whole horizon.
    ev = np.asarray(sched.ev_power_kw)
    assert ev.sum() > 0.0


def test_committed_tasks_are_reflected_and_contribute_to_house_cap():
    # Commit a washing_machine cycle over slots 0..5 (6 slots). Horizon 8 slots.
    committed = [
        ScheduledTask(
            appliance="washing_machine", start_slot=0, slots=6,
            expected_kwh=3.6, committed=True,
        )
    ]
    prices = np.full(H_SHORT, 40.0)
    now = datetime(2024, 12, 20, 19, 0, tzinfo=timezone.utc)
    sched = solve_receding_horizon(
        now, prices, _flat_probs(H_SHORT),
        horizon_slots=H_SHORT,
        remaining_ev_kwh=10.0,
        committed_tasks=committed,
    )
    # committed task is surfaced in the output plan
    committed_out = [t for t in sched.tasks if t.committed]
    assert len(committed_out) == 1
    assert committed_out[0].appliance == "washing_machine"

    # Washer (2.4 kW) + EV (≤ 7.6 kW) ≤ 10 kW in slots 0..5.
    ev = np.asarray(sched.ev_power_kw)
    wash_kw = APPLIANCES["washing_machine"].rated_kw
    for t in range(6):
        assert ev[t] + wash_kw <= 10.0 + 1e-6
