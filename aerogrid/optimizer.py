"""Receding-horizon mixed-integer scheduler (MPC core of AeroGrid).

This module implements the optimisation block of the outer LangGraph loop. It
formulates a small mixed-integer linear program (MILP) over a short rolling
horizon and is invoked by ``aerogrid.graph.n_optimize`` whenever
``TriggerManager`` decides a replan is warranted.

Modelling overview
==================

Time is discretised into 15-minute slots indexed by ``t ∈ {0, …, T-1}``
where ``T = horizon_slots`` (default ``T = 8`` ⇒ 2-hour horizon). The slot
length in hours is ``Δt = SLOT_MINUTES / 60 = 0.25 h``.

Two qualitatively different loads are co-optimised:

* **Continuous EV charging**, modelled by a non-negative real power ``p_ev[t]``
  bounded by the charger's rated power.
* **Cycle-based bufferable appliances** (dishwasher, washing machine, heater),
  each with a fixed-shape rectangular cycle of length ``L_a`` slots at rated
  power ``P_a``. Their start time is a binary decision ``s_a[t] ∈ {0, 1}``
  with at most one start per appliance per horizon.

A *committed* task pinned by :class:`aerogrid.commit.CommitTracker` is *not*
re-decided — its cycle is treated as exogenous load against the house power
cap for the slots it still occupies.

Decision variables
==================

* ``p_ev[t] ∈ [0, P_ev_max]``       — EV charging power in slot ``t`` (kW).
* ``s_a[t] ∈ {0, 1}``               — start indicator for appliance ``a``
                                      at slot ``t``.
* ``σ_ev ≥ 0``                      — soft slack on the EV energy constraint
                                      (kWh shortfall, see below).

Two derived expressions are used in the constraint and cost formulation:

* ``z_a[t] = Σ_{k=0}^{min(L_a-1, t)} s_a[t-k]`` — the "is-running" indicator
  for cycle appliance ``a`` at slot ``t``. Because ``Σ_t s_a[t] ≤ 1`` and
  cycles are not allowed to wrap past the horizon end, ``z_a[t] ∈ {0, 1}``
  is automatically integral and equals 1 iff appliance ``a`` was started in
  any of the previous ``L_a`` slots (inclusive).
* ``actual_cost``                    — total electricity cost of the plan
  (see below).

Unit conversion
===============

Prices ``π[t]`` are quoted in currency/MWh, power in kW, and slot length in
hours. The "per-slot" multiplier converts ``kW × ($/MWh) × Δt`` into ``$``::

    κ ≡ Δt / 1000   (so that  p[kW] · π[$/MWh] · κ → $)

and is stored in the module-level constant ``_PER_SLOT_FACTOR``.

Constraints
===========

C1. **EV charger rating** ::

        0 ≤ p_ev[t] ≤ P_ev_max,   ∀ t

C2. **EV energy / deadline** — state-dependent. Let ``E`` be the kWh the EV
    still needs by the next deadline at hour ``EV_DEADLINE_HOUR`` (default
    07:00 UTC), and ``H = T · Δt`` the horizon length in hours, and ``τ``
    the time to that deadline.

    * If ``τ ≤ H`` (deadline lies inside the horizon), let
      ``t_d = round(τ / Δt)`` be the deadline slot. Require::

            Δt · Σ_{t=0}^{t_d − 1} p_ev[t] + σ_ev  ≥  E

    * If ``τ > H`` (deadline beyond the horizon), require this horizon to
      deliver its proportional share of ``E`` plus a safety margin
      ``γ ≡ deadline_safety``::

            Δt · Σ_{t=0}^{T-1} p_ev[t] + σ_ev  ≥  E · (H / τ) · γ

    The non-negative slack ``σ_ev`` keeps the MILP feasible when the house
    power cap (C5) makes the right-hand side unattainable. A heavy penalty
    ``ρ · σ_ev`` (with ``ρ ≡ _SLACK_PENALTY = 1000``) drives ``σ_ev`` to
    zero whenever feasibility allows.

C3. **One start per cycle appliance** ::

        Σ_t s_a[t]  ≤  1,                 ∀ a ∈ A_cycle

    Combined with C4 this means every cycle either runs exactly once
    (entirely within the horizon) or is deferred to a later replan.

C4. **Cycle must fit in horizon** — the last allowed start is ``T − L_a``::

        s_a[T - L_a + 1 : T] = 0,         ∀ a ∈ A_cycle, L_a > 1

C5. **Comfort deadline (optional, per appliance)** — appliances declaring
    ``deadline_hours`` (e.g. heater pre-conditioning by 07:00 / 18:00) must
    *finish* their cycle by the next such deadline. Let ``t_d^a`` be the slot
    of the next deadline for appliance ``a`` and ``ℓ ≡ t_d^a − L_a`` the
    latest slot at which the cycle may start to finish in time. We add::

        s_a[ℓ + 1 : T] = 0    if    0 < ℓ < T − L_a
        s_a[1 : T] = 0        if    ℓ ≤ 0  (start immediately, best effort)

    If the deadline lies past the horizon, the constraint is vacuous and is
    re-checked at the next periodic replan.

C6. **Aggregate house power cap** — at every slot ``t``::

        p_ev[t] + Σ_{a ∈ A_cycle} z_a[t] · P_a
                + Σ_{c ∈ A_committed: t ∈ [c.start, c.start+L_c)} P_c
        ≤ P_max

    Committed tasks contribute their rated power as a constant load against
    the cap during the slots they still occupy.

Objective
=========

Three terms — a true cost we want to minimise, a soft incentive to *reserve*
likely-to-occur cycles in cheap slots (the "ghost reservation" utility), and
the slack penalty::

    minimise   C_actual(p_ev, s)  −  λ · U_reservation(s)  +  ρ · σ_ev

where

* ``C_actual`` — the realised electricity bill over the horizon::

      C_actual = κ · Σ_t  p_ev[t] · π[t]
               + κ · Σ_a  P_a · Σ_t  z_a[t] · π[t]

* ``U_reservation`` — alignment between *speculative* cycle starts and the
  behavioural predictor's onset probabilities ``P̂_a(t)``. Without this term
  the MILP would never start a *new* (uncommitted) cycle, because doing so
  strictly increases ``C_actual``. The term gives a soft bonus to cycles
  the household is statistically likely to want anyway, with weight
  ``λ ≡ RESERVATION_LAMBDA`` (currency per unit probability mass)::

      U_reservation = Σ_a Σ_t  s_a[t] · P̂_a(t)

* ``ρ · σ_ev`` — slack penalty (``ρ ≫ price_typical``) ensuring that if a
  feasible deadline-meeting plan exists, the solver finds it; otherwise it
  returns the closest feasible plan with a small, well-defined deficit.

Solver chain and fallback
=========================

CVXPY is used to express the program. Solvers are tried in the order
``HIGHS → GLPK_MI → SCIPY``. If all three fail (or return a non-optimal
status), :func:`solve_receding_horizon` returns a deterministic fallback
schedule that charges the EV ASAP without scheduling new cycles, so the
caller (the digital twin) always has an actionable plan.

Baseline cost (for savings reporting)
=====================================

``_baseline_cost`` evaluates a price-unaware "naive scheduler": EV charges
ASAP from the start of the horizon; each cycle appliance starts at the slot
of maximum onset probability (truncated to fit in the horizon). The ratio
``(baseline − expected) / baseline`` is reported as the plan's "savings"
and is the headline metric used in the notebooks.

Reproducibility
===============

The optimisation is fully deterministic given ``(now, prices, onset_probs,
remaining_ev_kwh, committed_tasks)`` plus the configuration constants in
:mod:`aerogrid.config`. The behavioural predictor and price oracle are the
only stochastic upstreams; once their outputs are fixed, the MILP is
reproducible to within solver-tolerance.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

import cvxpy as cp
import numpy as np

from aerogrid.config import (
    APPLIANCES,
    EV_DAILY_NEED_KWH,
    EV_DEADLINE_HOUR,
    HOUSE_POWER_CAP_KW,
    RESERVATION_LAMBDA,
    SHORT_HORIZON_SLOTS,
    SLOT_MINUTES,
    TRIGGER_DEADLINE_SAFETY,
)
from aerogrid.triggers import time_to_deadline_hours, time_to_next_deadline
from aerogrid.types import Schedule, ScheduledTask

logger = logging.getLogger(__name__)


# Slot length in hours (Δt). Used in every energy = power × time conversion.
_SLOT_HOURS = SLOT_MINUTES / 60.0

# Conversion factor κ such that  p[kW] · π[$/MWh] · κ → $.
# Derivation: π is in $/MWh = $/(1000 kWh), so
#     p · Δt · π/1000 = (kW · h · $/kWh) = $.
_PER_SLOT_FACTOR = _SLOT_HOURS / 1000.0

# Penalty ρ on the EV-energy slack variable. Must dominate any plausible
# horizon cost so the solver pushes σ_ev to zero whenever feasible. With
# horizon costs typically below O(1)$ and slack measured in kWh, ρ = 1000
# leaves a comfortable margin.
_SLACK_PENALTY = 1000.0


def _floor_slot(now: datetime) -> datetime:
    """Round ``now`` down to the start of the current 15-min slot.

    The MILP indexes slots from the *start* of the slot containing ``now``,
    so all wall-clock-derived quantities (price arrays, onset probabilities,
    schedule timestamps) are aligned by flooring to the nearest multiple of
    ``SLOT_MINUTES`` and zeroing sub-minute fields.
    """
    return now.replace(
        minute=(now.minute // SLOT_MINUTES) * SLOT_MINUTES,
        second=0,
        microsecond=0,
    )


def _prep_prices(prices: np.ndarray, horizon_slots: int) -> np.ndarray:
    """Coerce the price forecast to a length-``horizon_slots`` ``float64`` array.

    The price oracle may return shorter or longer forecasts than the optimiser
    needs (for example the seasonal-naive baseline returns a fixed-length
    median while the actual horizon is configurable). To keep the MILP
    well-defined we:

    * **right-pad** with the mean of available prices when the forecast is
      shorter than the horizon — using the mean is a *neutral* fill that
      makes those slots neither attractive nor repulsive to the optimiser;
    * **truncate** when the forecast is longer than the horizon.

    An empty input array degenerates to a zero-price horizon, which causes
    the optimiser to charge ASAP — a safe and conservative behaviour when
    the oracle has produced no useful signal.
    """
    p = np.asarray(prices, dtype=float).reshape(-1)
    original_size = p.size
    if p.size < horizon_slots:
        pad_val = p.mean() if p.size else 0.0
        p = np.concatenate([p, np.full(horizon_slots - p.size, pad_val)])
        logger.debug(
            "_prep_prices: padded %d → %d slots with pad_val=%.2f",
            original_size, horizon_slots, pad_val,
        )
    elif p.size > horizon_slots:
        logger.debug(
            "_prep_prices: truncated %d → %d slots", original_size, horizon_slots,
        )
    result = p[:horizon_slots]
    logger.debug(
        "_prep_prices: price array min=%.2f max=%.2f mean=%.2f",
        result.min(), result.max(), result.mean(),
    )
    return result


def solve_receding_horizon(
    now: datetime,
    prices: np.ndarray,
    onset_probs: dict[str, np.ndarray] | None = None,
    *,
    remaining_ev_kwh: float = EV_DAILY_NEED_KWH,
    time_to_deadline_h: float | None = None,
    committed_tasks: Iterable[ScheduledTask] | None = None,
    horizon_slots: int = SHORT_HORIZON_SLOTS,
    house_cap_kw: float = HOUSE_POWER_CAP_KW,
    reservation_lambda: float = RESERVATION_LAMBDA,
    deadline_safety: float = TRIGGER_DEADLINE_SAFETY,
    appliances: dict | None = None,
) -> Schedule:
    """Solve the receding-horizon MILP for the next ``horizon_slots`` slots.

    The full mathematical formulation is given in the module docstring. This
    function is a faithful CVXPY translation of that program, plus light
    bookkeeping to (a) prepare the inputs, (b) honour committed tasks, and
    (c) provide a deterministic fallback when the solver fails.

    Sets and indexes
    ----------------
    Let ``T = horizon_slots`` and ``Δt = 0.25 h``. The horizon length in hours
    is ``H = T · Δt``. The set of cycle appliances ``A_cycle`` consists of
    bufferable appliances with ``cycle_slots > 0``, an onset-probability
    forecast in ``onset_probs`` for this horizon, and which are *not*
    currently pinned by a committed task (those are handled exogenously).

    Decision variables
    ------------------
    * ``p_ev[t] ∈ [0, P_ev_max]``      — continuous EV power, kW.
    * ``s_a[t] ∈ {0, 1}``              — binary start indicator for ``a ∈ A_cycle``.
    * ``σ_ev ≥ 0``                     — soft slack on the EV energy constraint, kWh.

    Constraints
    -----------
    1. ``0 ≤ p_ev[t] ≤ P_ev_max``                                 (charger rating)
    2. EV energy / deadline (state-dependent on ``time_to_deadline_h``):

       * deadline inside horizon (``τ ≤ H``)::

             Δt · Σ_{t<t_d} p_ev[t] + σ_ev ≥ remaining_ev_kwh

       * deadline outside horizon (``τ > H``)::

             Δt · Σ_t p_ev[t] + σ_ev ≥ remaining_ev_kwh · (H/τ) · γ

         where ``γ = deadline_safety``.

    3. ``Σ_t s_a[t] ≤ 1`` for every ``a ∈ A_cycle``               (one start)
    4. ``s_a[T - L_a + 1 :] = 0``                                 (cycle fits)
    5. ``s_a[ℓ + 1 :] = 0`` when an explicit comfort deadline applies, with
       ``ℓ = (deadline-slot of a) − L_a`` (cycle finishes by deadline).
    6. House cap at every slot: EV + running cycle loads + committed
       constant-power loads ``≤ house_cap_kw``.

    Objective
    ---------
    ``minimise  C_actual − λ · U_reservation + ρ · σ_ev``

    where ``C_actual`` is the realised electricity bill of the plan over the
    horizon, ``U_reservation = Σ_a Σ_t s_a[t] · P̂_a(t)`` rewards starting
    cycles in slots the behavioural predictor finds likely (otherwise no new
    cycle would ever be chosen — every cycle strictly increases cost), and
    ``ρ = _SLACK_PENALTY = 1000`` ensures slack is only used when truly
    infeasible to meet the deadline.

    Args:
        now: Current simulation time (UTC). Used for deadline arithmetic and
            to time-stamp ``slot_start`` of the returned schedule.
        prices: 15-min price forecast in currency/MWh (e.g. EUR/MWh from the
            SMARD oracle). Length needs not equal ``horizon_slots``; it is
            normalised by :func:`_prep_prices`.
        onset_probs: ``{appliance_name: np.ndarray of shape (horizon_slots,)}``
            from :class:`aerogrid.behavioral_predictor.BehavioralPredictor`.
            Appliances missing from this dict (or whose forecast is shorter
            than the horizon) are zero-padded — i.e. they receive no
            reservation utility for those slots.
        remaining_ev_kwh: kWh the EV still needs before ``EV_DEADLINE_HOUR``.
            Maintained outside this function by :class:`CommitTracker` and
            decremented in real time.
        time_to_deadline_h: Hours until the next EV deadline. ``None``
            triggers a recompute via
            :func:`~aerogrid.triggers.time_to_deadline_hours`.
        committed_tasks: Tasks pinned by :class:`CommitTracker`. Their cycles
            are *not* re-decided; their rated power is added as a constant
            against the house cap for the slots they still occupy and the
            tasks are echoed into the returned schedule with
            ``committed=True``.
        horizon_slots: ``T``, number of 15-min decision slots in the horizon.
        house_cap_kw: ``P_max``, the aggregate household power limit.
        reservation_lambda: ``λ``, weight on the ghost-reservation utility.
            Larger ⇒ MILP is more eager to schedule cycles in high-probability
            slots even when prices are slightly suboptimal.
        deadline_safety: ``γ ≥ 1``, multiplier on the proportional EV-energy
            requirement when the deadline lies beyond the horizon. ``γ > 1``
            front-loads charging slightly to absorb forecast uncertainty.
        appliances: Override of the global appliance registry (default
            :data:`aerogrid.config.APPLIANCES`).

    Returns:
        :class:`~aerogrid.types.Schedule` populated with:

        * ``ev_power_kw`` — list of length ``horizon_slots`` (kW per slot).
        * ``tasks`` — list of :class:`ScheduledTask` for every newly chosen
          cycle start, plus all input ``committed_tasks`` echoed with
          ``committed=True``.
        * ``expected_cost`` — value of ``C_actual`` for the chosen plan.
        * ``baseline_cost`` — naive baseline cost from :func:`_baseline_cost`,
          enabling the savings ratio reported by the digital twin.
        * ``solver_status`` — CVXPY status string of the solver that
          succeeded (``"optimal"`` / ``"optimal_inaccurate"``), or
          ``"fallback:<status>"`` when no solver returned an optimal
          solution.

        On total solver failure (or non-optimal status) the function falls
        back to charging the EV ASAP at ``P_ev_max`` until ``remaining_ev_kwh``
        is satisfied or the horizon ends, with no new cycle tasks scheduled.
        This guarantees the caller always has an actionable plan.
    """
    # --- 0. Resolve defaults and partition the appliance set ---------------- #
    # The set of cycle appliances entering the MILP excludes:
    #   * non-bufferable loads (e.g. fridge),
    #   * appliances with no onset-probability forecast (so we'd give them
    #     zero reservation utility — they would never be started by the MILP
    #     anyway, and including them only inflates the variable count),
    #   * already-committed appliances (those are exogenous load on the cap).
    appliances = appliances or APPLIANCES
    onset_probs = onset_probs or {}
    committed_list = list(committed_tasks or [])
    committed_apps = {t.appliance for t in committed_list}

    ev_spec = appliances["ev_charger"]
    cycle_apps = {
        name: spec
        for name, spec in appliances.items()
        if spec.cycle_slots > 0 and spec.bufferable
        and name in onset_probs and name not in committed_apps
    }

    logger.info(
        "solve_receding_horizon: now=%s horizon=%d slots remaining_ev=%.2fkWh "
        "committed=%s cycle_apps=%s",
        now.isoformat(), horizon_slots, remaining_ev_kwh,
        [t.appliance for t in committed_list], list(cycle_apps.keys()),
    )

    # --- 1. Time / deadline arithmetic -------------------------------------- #
    prices = _prep_prices(prices, horizon_slots)
    horizon_h = horizon_slots * _SLOT_HOURS
    if time_to_deadline_h is None:
        time_to_deadline_h = time_to_deadline_hours(now, EV_DEADLINE_HOUR)
    deadline_in_horizon = time_to_deadline_h <= horizon_h
    # Slot index of the EV deadline within the horizon. Clamped to ``[1, T]``:
    # at least 1 slot of integration is required (we need somewhere to put
    # the kWh) and the deadline can't index past the horizon.
    deadline_slot = min(
        horizon_slots,
        max(1, int(round(time_to_deadline_h / _SLOT_HOURS))),
    )
    logger.debug(
        "solve_receding_horizon: time_to_deadline_h=%.2fh horizon_h=%.2fh "
        "deadline_in_horizon=%s deadline_slot=%d",
        time_to_deadline_h, horizon_h, deadline_in_horizon, deadline_slot,
    )

    # --- 2. Decision variables ---------------------------------------------- #
    p_ev = cp.Variable(horizon_slots, nonneg=True)
    slack_ev = cp.Variable(nonneg=True)
    starts: dict[str, cp.Variable] = {}
    z: dict[str, cp.Expression] = {}

    # C1: charger rating upper bound. Lower bound is implicit via ``nonneg=True``.
    constraints: list = [p_ev <= ev_spec.rated_kw]

    # --- 3. C2: EV energy / deadline constraint ----------------------------- #
    # Two regimes, switching on whether the daily 07:00 deadline falls inside
    # this 2-h horizon. Outside, we *prorate* the remaining kWh to the
    # horizon's share of the time-to-deadline and inflate it by ``γ`` so the
    # MILP charges slightly faster than strictly necessary.
    if remaining_ev_kwh > 0.0:
        if deadline_in_horizon:
            constraints.append(
                cp.sum(p_ev[:deadline_slot]) * _SLOT_HOURS + slack_ev
                >= remaining_ev_kwh
            )
        else:
            required_this_horizon = max(
                0.0,
                remaining_ev_kwh
                * (horizon_h / max(time_to_deadline_h, 1e-6))
                * deadline_safety,
            )
            constraints.append(
                cp.sum(p_ev) * _SLOT_HOURS + slack_ev >= required_this_horizon
            )

    # --- 4. C3-C5: cycle-appliance start variables and structural constraints  #
    for name, spec in cycle_apps.items():
        L = spec.cycle_slots
        if L > horizon_slots:
            # The cycle physically cannot fit in this horizon. The next
            # replan (with a different ``now``) will reconsider it.
            continue
        s = cp.Variable(horizon_slots, boolean=True, name=f"start_{name}")
        starts[name] = s

        # C3: at most one start per appliance per horizon. Combined with
        # ``s[t] ∈ {0,1}`` this gives a clean "either run once or skip"
        # decision.
        constraints.append(cp.sum(s) <= 1)

        # C4: cycle must finish before the horizon ends. Latest legal start
        # index is ``T - L``; everything strictly after must be zero.
        if L > 1:
            constraints.append(s[horizon_slots - L + 1 :] == 0)

        # C5: comfort-deadline constraint (e.g. heater pre-conditioning by
        # 07:00 / 18:00). Three cases:
        #   1. ``ℓ`` lies strictly inside the horizon — forbid starts after ℓ.
        #   2. ``ℓ ≤ 0`` but the deadline is still ahead — best effort, allow
        #      a start only at slot 0 so the cycle starts as early as possible
        #      (it cannot finish before the deadline; the operator accepts it).
        #   3. Deadline beyond the horizon — vacuous; revisited at next replan.
        if spec.deadline_hours:
            tdh = time_to_next_deadline(now, spec.deadline_hours)
            if tdh is not None:
                app_deadline_slot = int(round(tdh / _SLOT_HOURS))
                latest_start = app_deadline_slot - L
                if 0 < latest_start < horizon_slots - L:
                    constraints.append(s[latest_start + 1 :] == 0)
                elif latest_start <= 0 < horizon_slots:
                    constraints.append(s[1:] == 0)

        # Build the "is-running" expression z_a[t] = Σ_{k=0..L-1} s_a[t-k],
        # bounded by the horizon endpoints. Because Σ_t s_a[t] ≤ 1, every
        # term in this sum is in {0,1} and at most one is nonzero, so z_a[t]
        # is binary by construction (the solver does *not* see it as an
        # extra integer variable — it's purely a linear function of ``s``).
        rows = []
        for t in range(horizon_slots):
            terms = [s[t - k] for k in range(L) if 0 <= t - k < horizon_slots]
            rows.append(cp.sum(terms) if terms else cp.Constant(0))
        z[name] = cp.reshape(cp.vstack(rows), (horizon_slots,), order="C")

    # --- 5. C6: house power cap at every slot ------------------------------- #
    # The cap is the only constraint coupling the EV and cycle decisions.
    # Without it the optimiser would charge at full power and run every
    # cycle in cheap slots; the cap forces it to *negotiate* between the
    # two when prices coincide.
    for t in range(horizon_slots):
        load_t = p_ev[t]
        for name, spec in cycle_apps.items():
            if name in z:
                load_t = load_t + z[name][t] * spec.rated_kw
        # Committed tasks are *constants*, not decision variables — but they
        # consume cap headroom for the slots they still occupy.
        for task in committed_list:
            if task.start_slot <= t < task.start_slot + task.slots:
                load_t = load_t + appliances[task.appliance].rated_kw
        constraints.append(load_t <= house_cap_kw)

    # --- 6. Objective ------------------------------------------------------- #
    # Term 1: realised electricity bill (EV + cycle loads × π[t] × κ).
    actual_cost = cp.sum(cp.multiply(p_ev, prices)) * _PER_SLOT_FACTOR
    for name, spec in cycle_apps.items():
        if name in z:
            actual_cost = actual_cost + (
                cp.sum(cp.multiply(z[name], prices))
                * spec.rated_kw
                * _PER_SLOT_FACTOR
            )

    # Term 2: ghost-reservation utility. We deliberately couple to ``s_a``
    # (the *start* slot) rather than ``z_a`` because we want to reward the
    # decision moment that aligns with predicted user intent — once a cycle
    # is started the rest of its run is mechanical.
    reservation_utility: cp.Expression = cp.Constant(0.0)
    for name, s in starts.items():
        probs = np.asarray(
            onset_probs.get(name, np.zeros(horizon_slots)), dtype=float,
        )
        if probs.size < horizon_slots:
            probs = np.concatenate([probs, np.zeros(horizon_slots - probs.size)])
        probs = probs[:horizon_slots]
        reservation_utility = reservation_utility + cp.sum(cp.multiply(s, probs))

    obj = cp.Minimize(
        actual_cost
        - reservation_lambda * reservation_utility
        + _SLACK_PENALTY * slack_ev
    )
    prob = cp.Problem(obj, constraints)

    # --- 7. Solver chain ---------------------------------------------------- #
    # HiGHS is the default open-source MILP solver and handles everything
    # this program throws at it; GLPK_MI and SciPy are fallbacks for
    # environments where HiGHS is missing. We don't propagate the exception
    # because we always need an actionable plan downstream — the fallback
    # block below produces one even if every solver fails.
    status = "none"
    for solver in ("HIGHS", "GLPK_MI", "SCIPY"):
        logger.debug("solve_receding_horizon: trying solver=%s", solver)
        try:
            prob.solve(solver=solver)
            status = prob.status
            logger.info(
                "solve_receding_horizon: solver=%s status=%s value=%s",
                solver, status, f"{prob.value:.4f}" if prob.value is not None else "None",
            )
            break
        except cp.SolverError:
            logger.warning("solve_receding_horizon: solver=%s raised SolverError — trying next", solver)
            continue
        except Exception as e:                       # noqa: BLE001
            logger.error("solve_receding_horizon: solver=%s raised unexpected error: %r", solver, e)
            continue

    # --- 8. Assemble the Schedule ------------------------------------------- #
    # ``slot0`` is the wall-clock start of slot 0 — every slot index ``t`` in
    # the schedule maps to the half-open interval [slot0 + t·Δt, slot0 + (t+1)·Δt).
    slot0 = _floor_slot(now)
    sched = Schedule(
        slot_start=slot0,
        horizon_slots=horizon_slots,
        ev_power_kw=[],
        tasks=[
            ScheduledTask(
                appliance=t.appliance, start_slot=t.start_slot, slots=t.slots,
                expected_kwh=t.expected_kwh, committed=True,
            )
            for t in committed_list
            if t.start_slot < horizon_slots
        ],
        solver_status=status,
        committed_until=slot0 + timedelta(minutes=SLOT_MINUTES),
    )

    # --- 8a. Solver failure → deterministic fallback ------------------------ #
    # The fallback is a price-unaware ASAP-charge plan: this is the same
    # policy used in ``_baseline_cost``, ensuring the digital twin always
    # has a feasible setpoint to apply. ``baseline_cost == expected_cost``
    # by definition here, so the reported "savings" is zero — an honest
    # signal that the optimiser gave up.
    if prob.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        logger.warning(
            "solve_receding_horizon: no optimal solution (status=%s) — using ASAP fallback", status,
        )
        p = np.zeros(horizon_slots)
        remaining = remaining_ev_kwh
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

    # --- 8b. Solver success → unpack the optimum ---------------------------- #
    sched.ev_power_kw = [float(v) for v in p_ev.value]
    logger.debug(
        "solve_receding_horizon: EV power slots %s",
        [f"{v:.2f}" for v in sched.ev_power_kw],
    )
    for name, spec in cycle_apps.items():
        if name not in starts:
            continue
        # The boolean variables come back as floats near {0, 1}; round to
        # integers to recover the start slot. ``argmax`` works because at
        # most one entry is 1 (constraint C3).
        sv = np.asarray(starts[name].value).round().astype(int)
        if sv.sum() == 0:
            logger.debug("solve_receding_horizon: %s — not scheduled this horizon", name)
            continue
        start_slot = int(np.argmax(sv))
        expected_kwh = spec.rated_kw * spec.cycle_slots * _SLOT_HOURS
        sched.tasks.append(
            ScheduledTask(
                appliance=name,
                start_slot=start_slot,
                slots=spec.cycle_slots,
                expected_kwh=expected_kwh,
            )
        )
        logger.info(
            "solve_receding_horizon: scheduled %s start_slot=%d slots=%d expected_kwh=%.2f",
            name, start_slot, spec.cycle_slots, expected_kwh,
        )

    sched.expected_cost = float(actual_cost.value)
    sched.baseline_cost = _baseline_cost(
        prices, onset_probs, remaining_ev_kwh, appliances, horizon_slots,
    )
    logger.info(
        "solve_receding_horizon: expected_cost=%.4f baseline_cost=%.4f "
        "savings=%.1f%% solver=%s",
        sched.expected_cost,
        sched.baseline_cost,
        (1 - sched.expected_cost / max(sched.baseline_cost, 1e-9)) * 100,
        sched.solver_status,
    )
    return sched


def _baseline_cost(
    prices: np.ndarray,
    onset_probs: dict[str, np.ndarray],
    ev_need_kwh: float,
    appliances: dict,
    horizon_slots: int,
) -> float:
    """Compute a price-unaware reference cost for the same horizon.

    The baseline emulates a household that runs *without* any optimisation:

    * **EV** charges as fast as possible from the start of the horizon at
      its rated power, until ``ev_need_kwh`` is satisfied or the horizon
      ends. This is the worst-case outcome of a "plug it in and walk away"
      controller and is also the function used as the optimiser's failure
      fallback (see :func:`solve_receding_horizon`), so the two are
      consistent by construction.
    * **Cycle appliances** start at the slot of *maximum predicted onset
      probability* ``argmax_t P̂_a(t)`` (truncated to leave room for the
      whole cycle inside the horizon). This represents the user's natural
      timing — the optimisation aims to *shift* this start away from
      expensive slots.

    The MILP's ``expected_cost`` is then compared to this baseline to
    produce the "savings" ratio reported in the schedule and used as the
    headline metric in the demonstration notebooks.

    Args:
        prices: Length-``horizon_slots`` price array (currency/MWh).
        onset_probs: ``{appliance_name: np.ndarray}`` of predicted onset
            probabilities. Missing appliances simply contribute no cost
            (they're assumed not to run in the baseline).
        ev_need_kwh: kWh the EV would need over this horizon.
        appliances: Appliance registry (rated power, cycle length, …).
        horizon_slots: Number of 15-min slots.

    Returns:
        The total currency cost of the naive plan over the horizon.
    """
    ev_spec = appliances["ev_charger"]

    # EV: greedy ASAP charge, capped per-slot at the charger's rated power
    # and at the kWh remaining (so the last slot may be partial).
    p = np.zeros(horizon_slots)
    remaining = ev_need_kwh
    for t in range(horizon_slots):
        if remaining <= 0:
            break
        add = min(ev_spec.rated_kw, remaining / _SLOT_HOURS)
        p[t] = add
        remaining -= add * _SLOT_HOURS
    cost = float((p * prices).sum() * _PER_SLOT_FACTOR)

    # Cycle loads: start each at the slot of highest onset probability,
    # restricted to the slots from which the full cycle can finish before
    # the horizon end. This is the "user-naive" timing the MILP attempts
    # to improve upon.
    for name, probs in onset_probs.items():
        spec = appliances.get(name)
        if spec is None or spec.cycle_slots <= 0:
            continue
        L = spec.cycle_slots
        probs = np.asarray(probs, dtype=float)
        if probs.size < horizon_slots:
            probs = np.concatenate([probs, np.zeros(horizon_slots - probs.size)])
        probs = probs[:horizon_slots]
        # If the cycle is at least as long as the horizon, we can only start
        # at slot 0 — anywhere else would leave the cycle unfinished.
        if horizon_slots <= L:
            start = 0
        else:
            start = int(np.argmax(probs[: horizon_slots - L + 1]))
        cost += float(
            prices[start : start + L].sum() * spec.rated_kw * _PER_SLOT_FACTOR
        )
    return cost


__all__ = ["solve_receding_horizon"]
