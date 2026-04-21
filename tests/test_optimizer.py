"""Tests for the proactive MILP scheduler."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from aerogrid.config import APPLIANCES, EV_DAILY_NEED_KWH, SLOTS_PER_DAY
from aerogrid.optimizer import solve_proactive_schedule


def _flat_probs(n: int = SLOTS_PER_DAY) -> dict[str, np.ndarray]:
    return {
        "dishwasher": np.full(n, 1.0 / n),
        "washing_machine": np.full(n, 1.0 / n),
    }


def _prices_with_cheap_window(cheap_start: int, cheap_len: int,
                              cheap_val: float = 5.0,
                              base_val: float = 80.0) -> np.ndarray:
    p = np.full(SLOTS_PER_DAY, base_val)
    p[cheap_start : cheap_start + cheap_len] = cheap_val
    return p


def test_optimizer_routes_ev_to_cheap_window():
    # Cheap window must be before the EV deadline (07:00 = slot 28 when now=0:00).
    # 12 slots of cheap × 7 kW × 0.25 h = 21 kWh < 24 kWh needed, so the
    # optimizer will fill the cheap window and take at most 2 slots elsewhere.
    cheap_start, cheap_len = 14, 12
    prices = _prices_with_cheap_window(cheap_start=cheap_start, cheap_len=cheap_len)
    now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    sched = solve_proactive_schedule(now, prices, _flat_probs())

    assert sched.solver_status in ("optimal", "optimal_inaccurate")
    ev = np.asarray(sched.ev_power_kw)
    assert abs(ev.sum() * 0.25 - EV_DAILY_NEED_KWH) < 1e-4

    # Cheap window should absorb most of the energy — the 10 kW house cap can
    # steal a few kWh in slots where both cycle-based appliances also run.
    cheap_energy = ev[cheap_start : cheap_start + cheap_len].sum() * 0.25
    expected_cheap = min(EV_DAILY_NEED_KWH, cheap_len * 7.0 * 0.25)   # kWh
    # Allow up to 6 slots × 0.25 h × worst-case appliance steal (7 − 5.1 kW).
    slack = cheap_len * 0.25 * (7.0 - 5.1) / 2
    assert cheap_energy >= expected_cheap - slack, (
        f"ev only charged {cheap_energy:.2f} kWh in cheap window "
        f"(expected ~{expected_cheap:.2f} minus at most {slack:.2f})"
    )


def test_optimizer_schedules_contiguous_cycles_and_respects_cap():
    prices = _prices_with_cheap_window(cheap_start=30, cheap_len=16)
    now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    sched = solve_proactive_schedule(now, prices, _flat_probs())
    assert sched.solver_status in ("optimal", "optimal_inaccurate")

    scheduled = {t.appliance: t for t in sched.tasks}
    assert "dishwasher" in scheduled
    assert "washing_machine" in scheduled

    # cycle lengths should match config
    assert scheduled["dishwasher"].slots == APPLIANCES["dishwasher"].cycle_slots
    assert scheduled["washing_machine"].slots == APPLIANCES["washing_machine"].cycle_slots

    # house cap: at every slot, EV + scheduled appliance power ≤ 10 kW
    ev = np.asarray(sched.ev_power_kw)
    for t in range(SLOTS_PER_DAY):
        load = ev[t]
        for name, task in scheduled.items():
            if task.start_slot <= t < task.start_slot + task.slots:
                load += APPLIANCES[name].rated_kw
        assert load <= APPLIANCES["ev_charger"].rated_kw + 2 * 2.5 + 0.01
        assert load <= 10.0 + 1e-6, f"slot {t}: load {load:.2f} exceeds house cap"


def test_ghost_reservation_prefers_habitual_slot_when_prices_tie():
    # With flat prices, the reservation utility should choose the slot with
    # highest onset probability.
    prices = np.full(SLOTS_PER_DAY, 40.0)
    probs = {
        "dishwasher": np.zeros(SLOTS_PER_DAY),
        "washing_machine": np.zeros(SLOTS_PER_DAY),
    }
    probs["dishwasher"][72] = 0.9   # high probability at 18:00
    probs["washing_machine"][40] = 0.9  # 10:00

    now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    sched = solve_proactive_schedule(now, prices, probs, reservation_lambda=10.0)
    tasks = {t.appliance: t for t in sched.tasks}
    assert tasks["dishwasher"].start_slot == 72
    assert tasks["washing_machine"].start_slot == 40
