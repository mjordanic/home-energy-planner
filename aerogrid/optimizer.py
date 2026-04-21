"""Proactive MILP scheduler.

Schedules EV charging (continuous power) plus any number of cycle-based
bufferable loads (dishwasher, washing machine) over a 96-slot horizon at
15 min resolution. The objective is

    minimize  actual_cost  −  λ · Σ_t  Σ_app  z_app[t] · onset_prob_app[t]

where the reservation-utility term nudges the scheduler to align cycle starts
with slots where the user historically tends to start them. With λ set from
config, the optimizer will time-shift for cheap prices when it can, but still
prefers slots that match user behavior if the savings are small.

Bufferable cycle-based appliances are modelled via a start-indicator binary
variable plus the "cycle contiguity" convolution, so every selected cycle is
contiguous.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import cvxpy as cp
import numpy as np
import pandas as pd

from aerogrid.config import (
    APPLIANCES,
    EV_DAILY_NEED_KWH,
    EV_DEADLINE_SLOT,
    HOUSE_POWER_CAP_KW,
    RESERVATION_LAMBDA,
    SLOT_MINUTES,
    SLOTS_PER_DAY,
)
from aerogrid.types import Schedule, ScheduledTask

# ---- cost helpers ---------------------------------------------------------- #
# prices in $/MWh; 1 slot = 15 min = 0.25 h; 1 kWh = 1e-3 MWh
_SLOT_HOURS = SLOT_MINUTES / 60.0
_PER_SLOT_FACTOR = _SLOT_HOURS / 1000.0   # kW × $/MWh × factor → $


def _deadline_slot_from_now(now: datetime, deadline_hour_local: int = 7) -> int:
    """Number of slots from `now` until the next local-clock deadline_hour.

    We use UTC everywhere else; for this domain the convention is that the EV
    must be at target SoC by 07:00 local time regardless of DST/timezone. The
    twin is run in UTC so we approximate with UTC here too — adjust if you
    deploy against real local-time requirements.
    """
    t = now
    target = t.replace(hour=deadline_hour_local, minute=0, second=0, microsecond=0)
    if target <= t:
        target = target + timedelta(days=1)
    mins = (target - t).total_seconds() / 60.0
    return min(SLOTS_PER_DAY, max(1, int(mins // SLOT_MINUTES)))


def solve_proactive_schedule(
    now: datetime,
    prices: np.ndarray,                   # (96,) $/MWh
    onset_probs: dict[str, np.ndarray],   # per-appliance (96,), e.g. from BehavioralPredictor
    ev_need_kwh: float = EV_DAILY_NEED_KWH,
    house_cap_kw: float = HOUSE_POWER_CAP_KW,
    reservation_lambda: float = RESERVATION_LAMBDA,
    horizon_slots: int = SLOTS_PER_DAY,
    appliances: dict | None = None,
) -> Schedule:
    """Solve the MILP. Returns a Schedule (or a degenerate one on infeasibility)."""
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if prices.size < horizon_slots:
        # pad with mean if forecast is shorter than horizon (shouldn't happen).
        pad = np.full(horizon_slots - prices.size, prices.mean() if prices.size else 0.0)
        prices = np.concatenate([prices, pad])
    prices = prices[:horizon_slots]

    appliances = appliances or APPLIANCES
    ev_spec = appliances["ev_charger"]
    cycle_apps = {
        name: spec for name, spec in appliances.items()
        if spec.cycle_slots > 0 and spec.bufferable and name in onset_probs
    }

    ev_deadline = min(horizon_slots, max(1, _deadline_slot_from_now(now)))

    # --- Decision variables ------------------------------------------------ #
    p_ev = cp.Variable(horizon_slots, nonneg=True)
    starts: dict[str, cp.Variable] = {}
    z: dict[str, cp.Expression] = {}

    constraints: list = [p_ev <= ev_spec.rated_kw]
    # EV must be charged by deadline
    constraints.append(cp.sum(p_ev[:ev_deadline]) * _SLOT_HOURS >= ev_need_kwh)

    # Each cycle-based appliance: start-indicator + contiguity via convolution.
    # Exactly one cycle per horizon per appliance. The optimizer decides WHEN,
    # the reservation utility term encourages alignment with user habits.
    for name, spec in cycle_apps.items():
        L = spec.cycle_slots
        s = cp.Variable(horizon_slots, boolean=True, name=f"start_{name}")
        starts[name] = s
        constraints.append(cp.sum(s) == 1)
        # Can't start so late that the cycle won't complete in the horizon.
        constraints.append(s[horizon_slots - L + 1 :] == 0)
        # z[t] = Σ_{k=0..L-1} s[t-k] — is appliance on at slot t.
        rows: list[cp.Expression] = []
        for t in range(horizon_slots):
            terms = [s[t - k] for k in range(L) if 0 <= t - k < horizon_slots]
            rows.append(cp.sum(terms) if terms else cp.Constant(0))
        z[name] = cp.reshape(cp.vstack(rows), (horizon_slots,), order="C")

    # House power cap at each slot.
    for t in range(horizon_slots):
        load_t = p_ev[t]
        for name, spec in cycle_apps.items():
            load_t = load_t + z[name][t] * spec.rated_kw
        constraints.append(load_t <= house_cap_kw)

    # --- Objective --------------------------------------------------------- #
    actual_cost = cp.sum(cp.multiply(p_ev, prices)) * _PER_SLOT_FACTOR
    for name, spec in cycle_apps.items():
        actual_cost = actual_cost + (
            cp.sum(cp.multiply(z[name], prices)) * spec.rated_kw * _PER_SLOT_FACTOR
        )

    reservation_utility = 0.0
    for name, spec in cycle_apps.items():
        probs = onset_probs[name].astype(float)
        # Map the prob at the CYCLE START slot — encourage s[t] * prob[t].
        reservation_utility = reservation_utility + cp.sum(
            cp.multiply(starts[name], probs)
        )

    obj = cp.Minimize(actual_cost - reservation_lambda * reservation_utility)
    prob = cp.Problem(obj, constraints)

    # Prefer HiGHS (bundled with modern scipy); fall back cleanly.
    status = "none"
    for solver in ("HIGHS", "GLPK_MI", "SCIPY"):
        try:
            prob.solve(solver=solver)
            status = prob.status
            break
        except cp.SolverError:
            continue
        except Exception as e:  # noqa: BLE001
            print(f"optimizer: solver {solver} raised {e!r}")
            continue

    slot0 = now.replace(minute=(now.minute // SLOT_MINUTES) * SLOT_MINUTES,
                        second=0, microsecond=0)
    sched = Schedule(
        slot_start=slot0,
        horizon_slots=horizon_slots,
        ev_power_kw=[],
        tasks=[],
        solver_status=status,
    )
    if prob.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        # Fallback: charge EV ASAP, skip bufferable loads.
        p = np.zeros(horizon_slots)
        remaining = ev_need_kwh
        for t in range(horizon_slots):
            if remaining <= 0:
                break
            add = min(ev_spec.rated_kw, remaining / _SLOT_HOURS)
            p[t] = add
            remaining -= add * _SLOT_HOURS
        sched.ev_power_kw = [float(x) for x in p]
        sched.expected_cost = float((p * prices).sum() * _PER_SLOT_FACTOR)
        sched.baseline_cost = sched.expected_cost
        sched.solver_status = f"fallback:{status}"
        return sched

    sched.ev_power_kw = [float(v) for v in p_ev.value]
    for name, spec in cycle_apps.items():
        sv = np.asarray(starts[name].value).round().astype(int)
        if sv.sum() == 0:
            continue
        start_slot = int(np.argmax(sv))
        sched.tasks.append(
            ScheduledTask(
                appliance=name,
                start_slot=start_slot,
                slots=spec.cycle_slots,
                expected_kwh=spec.rated_kw * spec.cycle_slots * _SLOT_HOURS,
            )
        )
    sched.expected_cost = float(actual_cost.value)
    sched.baseline_cost = _baseline_cost(prices, onset_probs, ev_need_kwh, appliances)
    return sched


def _baseline_cost(prices: np.ndarray, onset_probs: dict[str, np.ndarray],
                   ev_need_kwh: float, appliances: dict) -> float:
    """Naïve: EV charges ASAP; dishwasher/washer run at their most-likely hour."""
    horizon = len(prices)
    ev_spec = appliances["ev_charger"]
    p = np.zeros(horizon)
    remaining = ev_need_kwh
    for t in range(horizon):
        if remaining <= 0:
            break
        add = min(ev_spec.rated_kw, remaining / _SLOT_HOURS)
        p[t] = add
        remaining -= add * _SLOT_HOURS
    cost = float((p * prices).sum() * _PER_SLOT_FACTOR)
    for name, probs in onset_probs.items():
        spec = appliances[name]
        L = spec.cycle_slots
        if L <= 0:
            continue
        start = int(np.argmax(probs[: horizon - L + 1])) if horizon > L else 0
        cost += float(
            prices[start : start + L].sum() * spec.rated_kw * _PER_SLOT_FACTOR
        )
    return cost
