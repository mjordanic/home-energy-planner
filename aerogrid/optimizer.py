"""Receding-horizon scheduler (MPC core of AeroGrid).

This module implements the optimisation block of the outer LangGraph loop. It
formulates a small program over a configurable rolling horizon and is
invoked by ``aerogrid.graph.n_optimize`` whenever ``TriggerManager`` decides
a replan is warranted.

The program is a **pure linear program (LP)** in the common case (no
user-triggered cycles waiting to be placed) and **collapses to a small mixed
integer program (MIP) on demand** when the caller passes ``pending_cycles``.
Each pending cycle adds at most ``HITL_RESCHEDULE_WINDOW_HOURS / 0.25 + 1``
binary start-indicator variables (typically 9 per cycle at the default 2 h
window) plus the equality ``Σ_t s_a[t] = 1``. With one pending cycle and the
default 24 h horizon, HiGHS solves the MIP in tens of milliseconds.

The MIP form lets the optimiser decide *jointly* — within the user-allowed
shift window — when a cycle should run, which EV power profile to pick, and
how to spread heater power, all under the house power cap. 

Modelling overview
==================

Time is discretised into 15-minute slots indexed by ``t ∈ {0, …, T-1}``
where ``T = horizon_slots`` (default ``T = 96`` ⇒ 24-hour horizon). The slot
length in hours is ``Δt = SLOT_MINUTES / 60 = 0.25 h``.

Two qualitatively different *continuous* loads are co-optimised:

* **EV charger** — non-negative real power ``p_ev[t]`` bounded by the
  charger's rated power and gated by an availability mask: the EV is only
  pluggable from :data:`aerogrid.config.EV_AVAILABLE_FROM_HOUR` UTC each day
  until the next deadline at :data:`aerogrid.config.EV_DEADLINE_HOUR`. Slots
  outside that window are forced to zero.
* **Heater** — non-negative real power ``p_heat[t]`` bounded by the heater's
  rated power. The heater has *energy delivery deadlines* listed in
  :data:`aerogrid.config.HEATER_DEADLINES`: by the time each deadline hour
  arrives, a configured kWh must have been delivered in the preceding
  *window* (the gap between the previous deadline and the current one, with
  wrap-around over 24 h).

The two event-driven cycle appliances (dishwasher, washing machine) are *not*
controlled by this LP. They are user-triggered; the
``aerogrid.graph.n_propose_reschedule`` node decides whether to offer a
small forward shift via the HITL gate. Their already-running cycles do
appear here as exogenous load through ``committed_tasks`` (their rated
power consumes cap headroom for the slots they still occupy).

Decision variables
==================

* ``p_ev[t] ∈ [0, P_ev_max]``       — EV charging power in slot ``t`` (kW),
                                       forced to zero outside the EV
                                       availability window.
* ``p_heat[t] ∈ [0, P_heat_max]``   — heater power in slot ``t`` (kW).
* ``σ_ev ≥ 0``                      — soft slack on the EV energy
                                       constraint (kWh shortfall).
* ``σ_heat[w] ≥ 0``                 — soft slack on the heater energy for
                                       deadline-window ``w`` (kWh).

There are no binary variables, so the program is a convex LP and HiGHS
solves it deterministically in milliseconds.

Unit conversion
===============

Prices ``π[t]`` are quoted in currency/MWh, power in kW, and slot length in
hours. The "per-slot" multiplier converts ``kW × ($/MWh) × Δt`` into ``$``::

    κ ≡ Δt / 1000   (so that  p[kW] · π[$/MWh] · κ → $)

and is stored in the module-level constant ``_PER_SLOT_FACTOR``.

Constraints
===========

C1. **EV charger rating and availability** ::

        0 ≤ p_ev[t] ≤ P_ev_max,   ∀ t
        p_ev[t] = 0,              ∀ t ∉ available_window(now)

    where ``available_window(now)`` enumerates which slot indices in the
    horizon fall in the window ``[EV_AVAILABLE_FROM_HOUR, EV_DEADLINE_HOUR)``
    (with wrap-around around midnight). When the EV is plugged in mid-window
    (``now`` already in the window), every slot up to the deadline is open.

C2. **EV energy / deadline** — state-dependent. Let ``E`` be the kWh the EV
    still needs by the next deadline at hour ``EV_DEADLINE_HOUR``, and
    ``H = T · Δt`` the horizon length in hours, and ``τ`` the time to that
    deadline. Two regimes:

    * If ``τ ≤ H`` (deadline lies inside the horizon), let
      ``t_d = round(τ / Δt)`` be the deadline slot. Require::

            Δt · Σ_{t=0}^{t_d − 1} p_ev[t] + σ_ev  ≥  E

    * If ``τ > H`` (deadline beyond the horizon), require this horizon to
      deliver its proportional share of ``E`` plus a safety margin
      ``γ ≡ deadline_safety``::

            Δt · Σ_{t=0}^{T-1} p_ev[t] + σ_ev  ≥  E · (H / τ) · γ

    The non-negative slack ``σ_ev`` keeps the LP feasible when the house
    power cap (C5) makes the right-hand side unattainable. A heavy penalty
    ``ρ · σ_ev`` (with ``ρ ≡ _SLACK_PENALTY = 1000``) drives ``σ_ev`` to
    zero whenever feasibility allows.

C3. **Heater energy per deadline window** — for every entry
    ``(hour_w, kwh_required_w)`` in :data:`HEATER_DEADLINES`, build the set
    of slots ``W_w`` that fall inside the window ending at ``hour_w`` (the
    interval from the *previous* deadline to ``hour_w``, with wrap-around).
    Require::

        Δt · Σ_{t ∈ W_w ∩ [0, T)} p_heat[t] + σ_heat[w]  ≥  remaining_w

    where ``remaining_w`` is what's still owed in window ``w`` *if* the
    deadline lies within the horizon. For deadline windows entirely beyond
    the horizon the constraint is omitted; for deadline windows that have
    just rolled over (i.e. ``hour_w`` is *behind* ``now``), the window's
    requirement has been reset by the commit tracker, and we either
    contribute nothing yet (window starts in the future) or contribute the
    full amount (window starts in the past and we are already inside the
    next iteration of the same window).

C4. **Heater rating** ::

        0 ≤ p_heat[t] ≤ P_heat_max,   ∀ t

C5. **Aggregate house power cap** — at every slot ``t``::

        p_ev[t] + p_heat[t]
            + Σ_{c ∈ committed_tasks: t ∈ [c.start, c.start+L_c)} P_c
            + Σ_{a ∈ pending_cycles} z_a[t] · P_a
        ≤ P_max

    Committed tasks contribute their rated power as a constant load against
    the cap during the slots they still occupy. *Pending* cycles
    (user-triggered onsets awaiting a HITL response) appear through the
    derived "is-running" indicator ``z_a[t] = Σ_{k=0..L_a−1} s_a[t−k]``
    where ``s_a[t]`` is the binary start indicator described under C6 and
    ``z_a[t] ∈ {0, 1}`` by construction.

C6. **Pending cycle placement** — for each pending cycle ``a`` with
    ``cycle_slots = L_a``, ``rated_kw = P_a``, and allowed-start interval
    ``[earliest_a, latest_a]``::

        s_a[t] ∈ {0, 1},   ∀ t ∈ [earliest_a, latest_a]
        Σ_t s_a[t] = 1

    The equality forces the cycle to run exactly once inside the window —
    the user has already started it, so we cannot decline to run it; we can
    only choose *when* (and the earliest choice ``t = earliest_a`` is
    "run now"). The chosen start slot is reported back through
    ``Schedule.cycle_starts``.

Objective
=========

Two terms — the realised electricity bill plus the slack penalty::

    minimise   C_actual  +  ρ · (σ_ev + Σ_w σ_heat[w])

where ``C_actual`` is the sum over slots of every controlled load times the
forecast price::

    C_actual = κ · Σ_t π[t] · (p_ev[t] + p_heat[t])
             + κ · Σ_t π[t] · Σ_{c ∈ committed} P_c · 1[t ∈ c.range]
             + κ · Σ_t π[t] · Σ_{a ∈ pending}   P_a · z_a[t]

Every controlled load strictly *needs* to deliver its energy (EV deadline,
heater windows, pending cycles via C6), so the optimiser will always start
using all of them — there is no separate "reservation" utility term.

Solver chain and fallback
=========================

CVXPY is used to express the program. Solvers are tried in the order
``HIGHS → ECOS → SCIPY`` for pure-LP problems (no pending cycles), and
``HIGHS → GLPK_MI`` for mixed-integer problems. ECOS and SCIPY's linprog
are LP-only and cannot solve the MIP form. If all candidate solvers fail
or return a non-optimal status, :func:`solve_receding_horizon` returns a
deterministic fallback schedule that:

* charges the EV ASAP at rated power inside the availability window until
  ``remaining_ev_kwh`` is satisfied or the horizon ends,
* runs the heater at rated power inside each deadline window until the
  required kWh is delivered, and
* places each pending cycle at its ``earliest_start_slot`` (i.e. "run now").

This guarantees the caller always has an actionable plan even if the
solver chain breaks.

Baseline cost (for savings reporting)
=====================================

``_baseline_cost`` evaluates a price-unaware "naive scheduler": EV charges
ASAP starting from the first available slot in the EV window, and the
heater runs at rated power starting from the first slot of each deadline
window. The ratio ``(baseline − expected) / baseline`` is reported as the
plan's "savings" and is the headline metric used in the notebooks.

Reproducibility
===============

The optimisation is fully deterministic given ``(now, prices,
remaining_ev_kwh, remaining_heater_kwh_by_window, committed_tasks)`` plus
the configuration constants in :mod:`aerogrid.config`. The price oracle is
the only stochastic upstream; once its output is fixed, the LP is
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
    EV_AVAILABLE_FROM_HOUR,
    EV_DAILY_NEED_KWH,
    EV_DEADLINE_HOUR,
    HEATER_DEADLINES,
    HOUSE_POWER_CAP_KW,
    SHORT_HORIZON_SLOTS,
    SLOT_MINUTES,
    TRIGGER_DEADLINE_SAFETY,
    HeaterEnergyDeadline,
)
from aerogrid.triggers import time_to_deadline_hours
from aerogrid.types import PendingCycle, Schedule, ScheduledTask

logger = logging.getLogger(__name__)


# Slot length in hours (Δt). Used in every energy = power × time conversion.
_SLOT_HOURS = SLOT_MINUTES / 60.0

# Conversion factor κ such that  p[kW] · π[$/MWh] · κ → $.
# Derivation: π is in $/MWh = $/(1000 kWh), so
#     p · Δt · π/1000 = (kW · h · $/kWh) = $.
_PER_SLOT_FACTOR = _SLOT_HOURS / 1000.0

# Penalty ρ on the energy slack variables (EV and per-heater-window). Must
# dominate any plausible horizon cost so the solver pushes σ to zero whenever
# feasible. With horizon costs typically below O(10)$ at 24 h and slack
# measured in kWh, ρ = 1000 leaves a comfortable margin.
_SLACK_PENALTY = 1000.0


def _floor_slot(now: datetime) -> datetime:
    """Round ``now`` down to the start of the current 15-min slot.

    The MILP indexes slots from the *start* of the slot containing ``now``,
    so all wall-clock-derived quantities (price arrays, schedule timestamps)
    are aligned by flooring to the nearest multiple of ``SLOT_MINUTES`` and
    zeroing sub-minute fields.
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
    median while the actual horizon is configurable). To keep the LP
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


def _ev_availability_mask(
    slot0: datetime,
    horizon_slots: int,
    available_from_hour: int = EV_AVAILABLE_FROM_HOUR,
    deadline_hour: int = EV_DEADLINE_HOUR,
) -> np.ndarray:
    """Boolean mask of length ``horizon_slots``: True where EV charging is allowed.

    A slot is open iff its wall-clock start falls inside *some* daily charging
    window ``[available_from_hour:00, deadline_hour:00)`` (UTC). The window
    wraps around midnight when ``available_from_hour > deadline_hour``
    (the default 20:00 → 07:00 case).
    """
    mask = np.zeros(horizon_slots, dtype=bool)
    for t in range(horizon_slots):
        slot_t = slot0 + timedelta(minutes=SLOT_MINUTES * t)
        h = slot_t.hour + slot_t.minute / 60.0
        if available_from_hour < deadline_hour:
            # No wrap (e.g. plug-in at 06:00, deadline at 18:00).
            mask[t] = available_from_hour <= h < deadline_hour
        else:
            # Wrap around midnight (e.g. plug-in at 20:00, deadline at 07:00).
            mask[t] = (h >= available_from_hour) or (h < deadline_hour)
    return mask


def _heater_window_slot_masks(
    slot0: datetime,
    horizon_slots: int,
    deadlines: tuple[HeaterEnergyDeadline, ...] = HEATER_DEADLINES,
) -> dict[int, np.ndarray]:
    """For each heater deadline, return the slot mask of its **current iteration**.

    A slot contributes to the deadline whose boundary is the *next* one strictly
    after that slot (circular over 24 h). The mask, however, only covers the
    *first* occurrence of each deadline in the horizon — i.e. the window
    iteration whose ``kwh_required`` we're about to enforce. Slots that fall
    into a later iteration of the same deadline (because the horizon spans
    multiple days) are intentionally excluded; the next replan, fired after
    each window resets, will constrain those iterations on their own. Lumping
    iterations together would let the LP satisfy the current ``kwh_required``
    using slots that physically belong to the *next* window — under-delivering
    the deadline that's actually due.
    """
    if not deadlines:
        return {}
    sorted_hours = sorted(d.hour for d in deadlines)
    masks: dict[int, np.ndarray] = {h: np.zeros(horizon_slots, dtype=bool) for h in sorted_hours}

    # First slot at which each deadline hour occurs (h:00 boundary).
    first_deadline_slot: dict[int, int] = {h: horizon_slots for h in sorted_hours}
    for t in range(horizon_slots):
        slot_t = slot0 + timedelta(minutes=SLOT_MINUTES * t)
        if slot_t.minute == 0 and slot_t.hour in first_deadline_slot and first_deadline_slot[slot_t.hour] == horizon_slots:
            first_deadline_slot[slot_t.hour] = t

    deadline_minutes = {h: int(h * 60) for h in sorted_hours}
    for t in range(horizon_slots):
        slot_t = slot0 + timedelta(minutes=SLOT_MINUTES * t)
        slot_minutes = int(slot_t.hour * 60 + slot_t.minute)
        # Minutes forward to each deadline on a 24 h ring.
        # We need "next strictly after", so distance 0 means +24 h.
        dist = {
            h: ((deadline_minutes[h] - slot_minutes) % 1440) or 1440
            for h in sorted_hours
        }
        h_star = min(sorted_hours, key=lambda h: dist[h])
        # Restrict to the first iteration: only slots strictly before the
        # first occurrence of h_star in the horizon belong to the current
        # window. Slots at or after that occurrence are in the next iteration.
        if t < first_deadline_slot[h_star]:
            masks[h_star][t] = True
    return masks


def solve_receding_horizon(
    now: datetime,
    prices: np.ndarray,
    *,
    remaining_ev_kwh: float = EV_DAILY_NEED_KWH,
    remaining_heater_kwh_by_window: dict[int, float] | None = None,
    time_to_deadline_h: float | None = None,
    committed_tasks: Iterable[ScheduledTask] | None = None,
    pending_cycles: Iterable[PendingCycle] | None = None,
    horizon_slots: int = SHORT_HORIZON_SLOTS,
    house_cap_kw: float = HOUSE_POWER_CAP_KW,
    deadline_safety: float = TRIGGER_DEADLINE_SAFETY,
    appliances: dict | None = None,
    heater_deadlines: tuple[HeaterEnergyDeadline, ...] | None = None,
    ev_available_from_hour: int = EV_AVAILABLE_FROM_HOUR,
    ev_deadline_hour: int = EV_DEADLINE_HOUR,
) -> Schedule:
    """Solve the receding-horizon LP for the next ``horizon_slots`` slots.

    The full mathematical formulation is given in the module docstring. This
    function is a faithful CVXPY translation of that program, plus light
    bookkeeping to (a) prepare the inputs, (b) honour committed tasks, and
    (c) provide a deterministic fallback when the solver fails.

    Args:
        now: Current simulation time (UTC). Used for deadline arithmetic and
            to time-stamp ``slot_start`` of the returned schedule.
        prices: 15-min price forecast in currency/MWh (e.g. EUR/MWh from the
            SMARD oracle). Length need not equal ``horizon_slots``; it is
            normalised by :func:`_prep_prices`.
        remaining_ev_kwh: kWh the EV still needs before ``ev_deadline_hour``.
            Maintained outside this function by :class:`CommitTracker` and
            decremented in real time.
        remaining_heater_kwh_by_window: Per-deadline-hour kWh still owed in
            the *current* iteration of each heater window. Defaults to the
            full ``kwh_required`` of every deadline if not supplied (i.e.
            "fresh start"). Maintained by :class:`CommitTracker` between
            replans.
        time_to_deadline_h: Hours until the next EV deadline. ``None``
            triggers a recompute via :func:`time_to_deadline_hours`.
        committed_tasks: Cycle tasks pinned by :class:`CommitTracker`
            (running dishwashers / washing machines). Their cycles are not
            re-decided; their rated power is added as a constant load
            against the house cap for the slots they still occupy and the
            tasks are echoed into the returned schedule with
            ``committed=True``.
        pending_cycles: User-triggered cycles that the optimiser should
            place jointly with the EV / heater plan. Each pending cycle
            adds binary start indicators ``s_a[t]`` for ``t`` in the
            allowed window and the equality ``Σ_t s_a[t] = 1`` (it must
            run exactly once inside the user-allowed window — the user
            already started it). When non-empty, the program is a small
            MIP rather than a pure LP. The chosen start slots are
            reported back through ``Schedule.cycle_starts``.
        horizon_slots: ``T``, number of 15-min decision slots in the horizon.
        house_cap_kw: ``P_max``, the aggregate household power limit.
        deadline_safety: ``γ ≥ 1``, multiplier on the proportional EV-energy
            requirement when the deadline lies beyond the horizon. ``γ > 1``
            front-loads charging slightly to absorb forecast uncertainty.
        appliances: Override of the global appliance registry (default
            :data:`aerogrid.config.APPLIANCES`).
        heater_deadlines: Override of :data:`aerogrid.config.HEATER_DEADLINES`.
        ev_available_from_hour, ev_deadline_hour: Override the EV charging
            window boundaries.

    Returns:
        :class:`~aerogrid.types.Schedule` populated with ``ev_power_kw``,
        ``heater_power_kw``, ``heater_window_kwh`` (per-deadline planned
        kWh), ``tasks`` (echoed committed cycles only), ``expected_cost``,
        ``baseline_cost``, and ``solver_status``.

        On total solver failure (or non-optimal status) the function falls
        back to charging the EV ASAP within its availability window and
        running the heater at rated power inside each deadline window. This
        guarantees the caller always has an actionable plan.
    """
    # --- 0. Resolve defaults ------------------------------------------------ #
    appliances = appliances or APPLIANCES
    heater_deadlines = heater_deadlines if heater_deadlines is not None else HEATER_DEADLINES
    committed_list = list(committed_tasks or [])
    pending_list = [
        pc for pc in (pending_cycles or [])
        # Skip degenerate cycles. We also skip a pending cycle whose
        # appliance is already in committed_tasks: that is a "double-fire"
        # (the same user start was committed last replan) and would
        # over-count the load against the cap.
        if pc.cycle_slots > 0
        and pc.cycle_slots <= horizon_slots
        and pc.appliance not in {t.appliance for t in committed_list}
        and pc.earliest_start_slot <= pc.latest_start_slot
        and pc.earliest_start_slot >= 0
        and pc.latest_start_slot + pc.cycle_slots <= horizon_slots
    ]
    ev_spec = appliances["ev_charger"]
    heater_spec = appliances["heater"]

    if remaining_heater_kwh_by_window is None:
        remaining_heater_kwh_by_window = {d.hour: d.kwh_required for d in heater_deadlines}

    logger.info(
        "solve_receding_horizon: now=%s horizon=%d slots remaining_ev=%.2fkWh "
        "remaining_heater=%s committed=%s",
        now.isoformat(), horizon_slots, remaining_ev_kwh,
        {h: round(v, 2) for h, v in remaining_heater_kwh_by_window.items()},
        [t.appliance for t in committed_list],
    )

    # --- 1. Price prep + window masks --------------------------------------- #
    prices = _prep_prices(prices, horizon_slots)
    horizon_h = horizon_slots * _SLOT_HOURS
    slot0 = _floor_slot(now)

    if time_to_deadline_h is None:
        time_to_deadline_h = time_to_deadline_hours(now, ev_deadline_hour)
    deadline_in_horizon = time_to_deadline_h <= horizon_h
    deadline_slot = min(
        horizon_slots,
        max(1, int(round(time_to_deadline_h / _SLOT_HOURS))),
    )

    ev_mask = _ev_availability_mask(
        slot0, horizon_slots, ev_available_from_hour, ev_deadline_hour,
    )
    heater_masks = _heater_window_slot_masks(slot0, horizon_slots, heater_deadlines)
    logger.debug(
        "solve_receding_horizon: ev_window_slots=%d heater_window_sizes=%s",
        int(ev_mask.sum()),
        {h: int(m.sum()) for h, m in heater_masks.items()},
    )

    # --- 2. Decision variables ---------------------------------------------- #
    p_ev = cp.Variable(horizon_slots, nonneg=True)
    p_heat = cp.Variable(horizon_slots, nonneg=True)
    slack_ev = cp.Variable(nonneg=True)
    slack_heat: dict[int, cp.Variable] = {h: cp.Variable(nonneg=True) for h in heater_masks}

    # C1: per-slot upper bound is rated_kw inside the EV charging window
    # and zero outside. Combined with ``nonneg=True`` this fully realises
    # the hard availability gate without introducing extra constraints.
    ev_upper_bound = ev_mask.astype(float) * ev_spec.rated_kw
    constraints: list = [
        p_ev <= ev_upper_bound,                                    # C1
        p_heat <= heater_spec.rated_kw,                            # C4 upper
    ]

    # C6: pending-cycle start indicators (one boolean variable per allowed
    # start slot). The "is-running" indicator z_a[t] is built as a linear
    # combination of the s_a[t] (no extra integer variables needed because
    # cycles cannot overlap themselves under the Σ s_a = 1 constraint).
    pending_data: list[tuple[PendingCycle, cp.Variable, list]] = []
    for pc in pending_list:
        n_starts = pc.latest_start_slot - pc.earliest_start_slot + 1
        s_a = cp.Variable(n_starts, boolean=True, name=f"s_{pc.appliance}")
        constraints.append(cp.sum(s_a) == 1)                                       # C6 equality
        # z_a[t] = sum of s_a[k] for k such that the cycle started at
        # earliest_start_slot+k is still running at slot t, i.e.
        #   earliest_start_slot + k ≤ t < earliest_start_slot + k + cycle_slots
        z_a_per_slot: list = []
        for t in range(horizon_slots):
            valid_k = [
                k for k in range(n_starts)
                if (pc.earliest_start_slot + k) <= t < (pc.earliest_start_slot + k + pc.cycle_slots)
            ]
            if valid_k:
                z_a_per_slot.append(cp.sum(s_a[valid_k]))
            else:
                z_a_per_slot.append(0.0)
        pending_data.append((pc, s_a, z_a_per_slot))

    # --- 3. C2: EV energy / deadline constraint ----------------------------- #
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

    # --- 4. C3: heater energy per deadline window --------------------------- #
    # Only constrain windows whose deadline lies within the horizon AND for
    # which the window actually overlaps the horizon. For deadlines fully
    # outside, we apply a proportional pre-charge (same logic as the EV's
    # outside-horizon regime) so the heater starts contributing early on
    # long-horizon (24 h) plans where the next overnight deadline may be
    # 25 h away after a 18:00 reset.
    for d in heater_deadlines:
        kwh_required = float(remaining_heater_kwh_by_window.get(d.hour, d.kwh_required))
        if kwh_required <= 1e-6:
            continue
        window_mask = heater_masks.get(d.hour)
        if window_mask is None or not window_mask.any():
            # Window is entirely outside the horizon — proportional regime.
            continue
        # Energy delivered in window-slots inside the horizon ≥ kwh_required − slack.
        constraints.append(
            cp.sum(cp.multiply(p_heat, window_mask.astype(float))) * _SLOT_HOURS
            + slack_heat[d.hour]
            >= kwh_required
        )

    # --- 5. C5: house power cap at every slot ------------------------------- #
    for t in range(horizon_slots):
        load_t = p_ev[t] + p_heat[t]
        for task in committed_list:
            if task.start_slot <= t < task.start_slot + task.slots:
                load_t = load_t + appliances[task.appliance].rated_kw
        for pc, _s_a, z_a_per_slot in pending_data:
            if not isinstance(z_a_per_slot[t], (int, float)) or z_a_per_slot[t] != 0.0:
                load_t = load_t + pc.rated_kw * z_a_per_slot[t]
        constraints.append(load_t <= house_cap_kw)

    # --- 6. Objective ------------------------------------------------------- #
    actual_cost = (
        cp.sum(cp.multiply(p_ev, prices)) * _PER_SLOT_FACTOR
        + cp.sum(cp.multiply(p_heat, prices)) * _PER_SLOT_FACTOR
    )
    # Pending cycles: their cost depends on which slot the MIP picks via
    # z_a[t], so it's part of the optimisation. We sum p_a · π · κ across
    # the horizon for each pending cycle.
    pending_cost: cp.Expression | float = 0.0
    for pc, _s_a, z_a_per_slot in pending_data:
        for t in range(horizon_slots):
            zt = z_a_per_slot[t]
            if isinstance(zt, (int, float)) and zt == 0.0:
                continue
            pending_cost = pending_cost + pc.rated_kw * float(prices[t]) * _PER_SLOT_FACTOR * zt
    # Committed loads: their cost is fixed (constant), so technically it
    # doesn't change the optimum, but adding it makes ``expected_cost`` a
    # truthful "what this horizon costs" number.
    committed_cost = 0.0
    for task in committed_list:
        rated = float(appliances[task.appliance].rated_kw)
        for t in range(task.start_slot, min(task.start_slot + task.slots, horizon_slots)):
            committed_cost += rated * float(prices[t]) * _PER_SLOT_FACTOR

    slack_term = _SLACK_PENALTY * (slack_ev + sum(slack_heat.values()))
    obj = cp.Minimize(actual_cost + pending_cost + slack_term)
    prob = cp.Problem(obj, constraints)

    # --- 7. Solver chain ---------------------------------------------------- #
    # ECOS and SCIPY's linprog cannot solve mixed-integer problems, so
    # restrict to MIP-capable solvers when there are boolean variables.
    if pending_data:
        solver_chain = ("HIGHS", "GLPK_MI")
    else:
        solver_chain = ("HIGHS", "ECOS", "SCIPY")
    status = "none"
    for solver in solver_chain:
        logger.debug("solve_receding_horizon: trying solver=%s", solver)
        try:
            prob.solve(solver=solver)
            status = prob.status
            logger.info(
                "solve_receding_horizon: solver=%s status=%s value=%s",
                solver, status,
                f"{prob.value:.4f}" if prob.value is not None else "None",
            )
            break
        except cp.SolverError:
            logger.warning(
                "solve_receding_horizon: solver=%s raised SolverError — trying next", solver,
            )
            continue
        except Exception as e:                       # noqa: BLE001
            logger.error(
                "solve_receding_horizon: solver=%s raised unexpected error: %r", solver, e,
            )
            continue

    # --- 8. Assemble the Schedule ------------------------------------------- #
    sched = Schedule(
        slot_start=slot0,
        horizon_slots=horizon_slots,
        ev_power_kw=[],
        heater_power_kw=[],
        heater_window_kwh={},
        tasks=[
            ScheduledTask(
                appliance=t.appliance, start_slot=t.start_slot, slots=t.slots,
                expected_kwh=t.expected_kwh, committed=True,
            )
            for t in committed_list
            if t.start_slot < horizon_slots
        ],
        cycle_starts={},
        solver_status=status,
        committed_until=slot0 + timedelta(minutes=SLOT_MINUTES),
    )

    # --- 8a. Solver failure → deterministic fallback ------------------------ #
    if prob.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        logger.warning(
            "solve_receding_horizon: no optimal solution (status=%s) — using ASAP fallback",
            status,
        )
        ev_plan = _fallback_ev_plan(remaining_ev_kwh, ev_mask, ev_spec.rated_kw, horizon_slots)
        heater_plan, window_kwh = _fallback_heater_plan(
            remaining_heater_kwh_by_window, heater_masks, heater_spec.rated_kw, horizon_slots,
        )
        # Fallback for pending cycles: place each one at its earliest
        # allowed start (i.e. "run now"). The schedule still tracks them
        # in cycle_starts for the caller's HITL machinery.
        fallback_cycle_starts: dict[str, int] = {}
        for pc in pending_list:
            fallback_cycle_starts[pc.appliance] = pc.earliest_start_slot
        sched.ev_power_kw = [float(x) for x in ev_plan]
        sched.heater_power_kw = [float(x) for x in heater_plan]
        sched.heater_window_kwh = {int(k): float(v) for k, v in window_kwh.items()}
        sched.cycle_starts = fallback_cycle_starts
        # Cost contribution from pending cycles in the fallback (running now).
        fallback_pending_cost = 0.0
        for pc in pending_list:
            for t in range(
                pc.earliest_start_slot,
                min(pc.earliest_start_slot + pc.cycle_slots, horizon_slots),
            ):
                fallback_pending_cost += pc.rated_kw * float(prices[t]) * _PER_SLOT_FACTOR
        sched.expected_cost = float(
            ((ev_plan + heater_plan) * prices).sum() * _PER_SLOT_FACTOR
            + committed_cost
            + fallback_pending_cost
        )
        sched.baseline_cost = sched.expected_cost
        sched.solver_status = f"fallback:{status}"
        return sched

    # --- 8b. Solver success → unpack the optimum ---------------------------- #
    sched.ev_power_kw = [float(v) for v in p_ev.value]
    sched.heater_power_kw = [float(v) for v in p_heat.value]
    sched.heater_window_kwh = {
        int(h): float(np.asarray(p_heat.value)[m].sum() * _SLOT_HOURS)
        for h, m in heater_masks.items()
    }
    # Recover the chosen start slot from each pending cycle's binary vector.
    chosen_starts: dict[str, int] = {}
    for pc, s_a, _z in pending_data:
        s_vals = np.asarray(s_a.value, dtype=float)
        # Round to {0, 1} (HiGHS may return e.g. 0.999999) and pick argmax.
        k_star = int(np.argmax(s_vals))
        chosen_starts[pc.appliance] = pc.earliest_start_slot + k_star
        logger.debug(
            "solve_receding_horizon: pending cycle %s placed at slot %d (s=%.3f)",
            pc.appliance, chosen_starts[pc.appliance], float(s_vals[k_star]),
        )
    sched.cycle_starts = chosen_starts
    logger.debug(
        "solve_receding_horizon: EV power slots %s",
        [f"{v:.2f}" for v in sched.ev_power_kw[: min(8, horizon_slots)]],
    )
    logger.debug(
        "solve_receding_horizon: heater power slots %s",
        [f"{v:.2f}" for v in sched.heater_power_kw[: min(8, horizon_slots)]],
    )

    sched.expected_cost = float(prob.value - _SLACK_PENALTY * (
        float(slack_ev.value or 0.0) + sum(float(s.value or 0.0) for s in slack_heat.values())
    ) + committed_cost)
    sched.baseline_cost = _baseline_cost(
        prices, remaining_ev_kwh, remaining_heater_kwh_by_window,
        ev_mask, heater_masks, ev_spec.rated_kw, heater_spec.rated_kw,
        committed_list, appliances, horizon_slots,
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


def _fallback_ev_plan(
    remaining_ev_kwh: float,
    ev_mask: np.ndarray,
    rated_kw: float,
    horizon_slots: int,
) -> np.ndarray:
    """ASAP charge inside the EV availability window until the kWh need is met."""
    p = np.zeros(horizon_slots)
    remaining = remaining_ev_kwh
    for t in range(horizon_slots):
        if remaining <= 0:
            break
        if not ev_mask[t]:
            continue
        add = min(rated_kw, remaining / _SLOT_HOURS)
        p[t] = add
        remaining -= add * _SLOT_HOURS
    return p


def _fallback_heater_plan(
    remaining_by_window: dict[int, float],
    window_masks: dict[int, np.ndarray],
    rated_kw: float,
    horizon_slots: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """ASAP heater run inside each window until that window's kWh is met."""
    p = np.zeros(horizon_slots)
    delivered: dict[int, float] = {}
    for h, mask in window_masks.items():
        need = float(remaining_by_window.get(h, 0.0))
        if need <= 0:
            delivered[h] = 0.0
            continue
        for t in np.where(mask)[0]:
            if need <= 0:
                break
            # Keep room under the rated_kw cap if a previous window already
            # requested power in this slot (unusual but possible at boundary).
            avail = max(0.0, rated_kw - p[t])
            add = min(avail, need / _SLOT_HOURS)
            p[t] += add
            need -= add * _SLOT_HOURS
        delivered[h] = float(remaining_by_window.get(h, 0.0) - need)
    return p, delivered


def _baseline_cost(
    prices: np.ndarray,
    ev_need_kwh: float,
    heater_need_by_window: dict[int, float],
    ev_mask: np.ndarray,
    heater_masks: dict[int, np.ndarray],
    ev_rated_kw: float,
    heater_rated_kw: float,
    committed_tasks: list[ScheduledTask],
    appliances: dict,
    horizon_slots: int,
) -> float:
    """Compute a price-unaware reference cost for the same horizon.

    The baseline emulates a household that runs *without* any optimisation:

    * **EV** charges as fast as possible from the first available slot in
      its charging window, until ``ev_need_kwh`` is satisfied or the window
      ends. This is the worst-case outcome of a "plug it in and walk away"
      controller.
    * **Heater** runs at rated power from the first slot of each deadline
      window until the window's required kWh is delivered.
    * **Committed cycle tasks** (dishwasher / washing machine that are
      already running) contribute their rated power × duration × price as a
      constant — same on both sides so they never change the savings ratio.

    The optimiser's ``expected_cost`` is then compared to this baseline to
    produce the "savings" ratio reported in the schedule and used as the
    headline metric in the demonstration notebooks.
    """
    ev_plan = _fallback_ev_plan(ev_need_kwh, ev_mask, ev_rated_kw, horizon_slots)
    heater_plan, _ = _fallback_heater_plan(
        heater_need_by_window, heater_masks, heater_rated_kw, horizon_slots,
    )
    cost = float(((ev_plan + heater_plan) * prices).sum() * _PER_SLOT_FACTOR)
    for task in committed_tasks:
        rated = float(appliances[task.appliance].rated_kw)
        for t in range(task.start_slot, min(task.start_slot + task.slots, horizon_slots)):
            cost += rated * float(prices[t]) * _PER_SLOT_FACTOR
    return cost


__all__ = ["solve_receding_horizon"]
