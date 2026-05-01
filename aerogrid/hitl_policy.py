"""Pure policy: should we interrupt the user about this plan change?

Inputs vary by call site:

* **Plan-level**: ``decide(old_plan, new_plan)`` — used at the end of a
  generic replan. AUTO when the new plan only nudges existing setpoints
  within tolerance; ASK when a setpoint or cost moves significantly.
* **Reschedule-proposal**: ``decide_reschedule(proposal)`` — used after a
  ``new_onset`` trigger when an event-driven cycle appliance (dishwasher /
  washing machine) was started. Returns ASK with a savings-aware question
  when the proposed shift saves ≥ ``HITL_RESCHEDULE_MIN_SAVINGS_EUR``;
  AUTO ("decline / run now") when there is no useful savings.

Plan-level AUTO when:
  - first call with no new plan → nothing to approve.
  - only EV setpoint changed by < HITL_EV_TOLERANCE_KW.
  - tentative (non-committed) task starts shifted by < HITL_SHIFT_TOLERANCE_MIN.
  - new cost ≤ old cost + a small bump.

Plan-level ASK when:
  - first plan (no prior user confirmation).
  - a new appliance is being scheduled.
  - any non-committed start moves by ≥ HITL_ASK_SHIFT_MIN.
  - a task's start crosses INTO the 22:00–06:00 sleep window.
  - new cost exceeds old by ≥ HITL_COST_BUMP_USD.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from aerogrid.config import (
    HITL_ASK_SHIFT_MIN,
    HITL_COST_BUMP_USD,
    HITL_EV_TOLERANCE_KW,
    HITL_RESCHEDULE_MIN_SAVINGS_EUR,
    SLEEP_WINDOW_END,
    SLEEP_WINDOW_START,
    SLOT_MINUTES,
)
from aerogrid.types import HITLDecision, RescheduleProposal, Schedule

logger = logging.getLogger(__name__)


def _in_sleep_window(t: time, start: time = SLEEP_WINDOW_START, end: time = SLEEP_WINDOW_END) -> bool:
    """True iff t falls in [start, end) with wrap-around across midnight."""
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def _task_start_dt(plan: Schedule, start_slot: int) -> datetime:
    """Convert a slot index to an absolute UTC datetime within the given plan."""
    return plan.slot_start + timedelta(minutes=SLOT_MINUTES * start_slot)


def decide(
    old_plan: Schedule | None,
    new_plan: Schedule | None,
    *,
    ev_tolerance_kw: float = HITL_EV_TOLERANCE_KW,
    ask_shift_min: float = HITL_ASK_SHIFT_MIN,
    cost_bump_usd: float = HITL_COST_BUMP_USD,
) -> HITLDecision:
    """Plan-level decision: auto-commit or ask the user about a plan change.

    Pure function — no side effects. The graph node ``n_hitl_gate`` is
    responsible for acting on the decision (calling ``interrupt()`` when
    ``action == "ask"``).
    """
    logger.debug(
        "hitl_policy.decide: old_plan=%s new_plan=%s",
        "None" if old_plan is None else f"tasks={len(old_plan.tasks)}",
        "None" if new_plan is None else f"tasks={len(new_plan.tasks)} cost={new_plan.expected_cost:.4f}",
    )

    if new_plan is None:
        logger.debug("hitl_policy.decide: no new plan → AUTO")
        return HITLDecision(action="auto", reason="no new plan")

    # First-time scheduling: always ask.
    if old_plan is None:
        n = len(new_plan.tasks)
        ev_kw = new_plan.ev_power_kw[0] if new_plan.ev_power_kw else 0.0
        heater_kw = new_plan.heater_power_kw[0] if new_plan.heater_power_kw else 0.0
        logger.info(
            "hitl_policy.decide: ASK — first plan (tasks=%d ev_setpoint=%.1fkW heater_setpoint=%.1fkW)",
            n, ev_kw, heater_kw,
        )
        return HITLDecision(
            action="ask",
            reason="first plan",
            question=(
                f"First plan: {n} committed task(s), EV setpoint {ev_kw:.1f} kW, "
                f"heater setpoint {heater_kw:.1f} kW. Accept? (yes/no)"
            ),
        )

    old_tasks = {t.appliance: t for t in old_plan.tasks}
    new_tasks = {t.appliance: t for t in new_plan.tasks}

    # A brand-new appliance being scheduled for the first time.
    newly = set(new_tasks) - set(old_tasks)
    if newly:
        logger.info("hitl_policy.decide: ASK — new appliance(s): %s", sorted(newly))
        return HITLDecision(
            action="ask",
            reason=f"new appliance(s): {', '.join(sorted(newly))}",
            question=f"New scheduling for {', '.join(sorted(newly))}. Accept? (yes/no)",
        )

    # Check each task for a significant shift — but only if not committed.
    ask_shift_slots = int(round(ask_shift_min / SLOT_MINUTES))
    for name, new_task in new_tasks.items():
        old_task = old_tasks[name]
        if new_task.committed or old_task.committed:
            logger.debug("hitl_policy.decide: skipping committed task %s", name)
            continue
        shift_slots = abs(new_task.start_slot - old_task.start_slot)

        new_start = _task_start_dt(new_plan, new_task.start_slot)
        old_start = _task_start_dt(old_plan, old_task.start_slot)

        logger.debug(
            "hitl_policy.decide: task=%s old_slot=%d new_slot=%d shift=%d slots (%d min)",
            name, old_task.start_slot, new_task.start_slot,
            shift_slots, shift_slots * SLOT_MINUTES,
        )

        # Crossing INTO the sleep window is always worth asking.
        if _in_sleep_window(new_start.time()) and not _in_sleep_window(old_start.time()):
            logger.info(
                "hitl_policy.decide: ASK — %s would start at %s (sleep window)",
                name, new_start.strftime("%H:%M"),
            )
            return HITLDecision(
                action="ask",
                reason=f"{name} would start at {new_start.strftime('%H:%M')} (sleep hours)",
                question=(
                    f"Run {name} at {new_start.strftime('%H:%M')} (overnight)? "
                    "Accept? (yes/no)"
                ),
            )

        if shift_slots >= ask_shift_slots:
            logger.info(
                "hitl_policy.decide: ASK — %s shifted %d min (threshold=%d min)",
                name, shift_slots * SLOT_MINUTES, ask_shift_min,
            )
            return HITLDecision(
                action="ask",
                reason=f"{name} shifted {shift_slots * SLOT_MINUTES} min",
                question=(
                    f"Shift {name}: {old_start.strftime('%H:%M')} → "
                    f"{new_start.strftime('%H:%M')}. Accept? (yes/no)"
                ),
            )

    # EV power setpoint change beyond tolerance (look at slot 0 only — the
    # slot that's actually about to be committed).
    if old_plan.ev_power_kw and new_plan.ev_power_kw:
        old_kw = float(old_plan.ev_power_kw[0])
        new_kw = float(new_plan.ev_power_kw[0])
        delta_kw = abs(new_kw - old_kw)
        logger.debug(
            "hitl_policy.decide: EV setpoint old=%.2fkW new=%.2fkW delta=%.2fkW tolerance=%.2fkW",
            old_kw, new_kw, delta_kw, ev_tolerance_kw,
        )
        if delta_kw > ev_tolerance_kw:
            logger.info(
                "hitl_policy.decide: ASK — EV setpoint Δ=%.1fkW > tolerance=%.1fkW",
                delta_kw, ev_tolerance_kw,
            )
            return HITLDecision(
                action="ask",
                reason=f"EV setpoint Δ = {delta_kw:.1f} kW > {ev_tolerance_kw}",
                question=(
                    f"Change EV charge rate {old_kw:.1f} → {new_kw:.1f} kW? Accept? (yes/no)"
                ),
            )

    # Cost-bump check — the deadline guard may force expensive charging.
    cost_delta = new_plan.expected_cost - old_plan.expected_cost
    logger.debug(
        "hitl_policy.decide: cost delta=%.4f threshold=%.4f",
        cost_delta, cost_bump_usd,
    )
    if cost_delta > cost_bump_usd:
        logger.info(
            "hitl_policy.decide: ASK — cost increases $%.2f > threshold=$%.2f",
            cost_delta, cost_bump_usd,
        )
        return HITLDecision(
            action="ask",
            reason=f"cost increases ${cost_delta:.2f}",
            question=(
                f"Plan cost rises ${cost_delta:.2f} (deadline constraints). Accept? (yes/no)"
            ),
        )

    logger.info("hitl_policy.decide: AUTO — within tolerance")
    return HITLDecision(action="auto", reason="within tolerance")


def decide_reschedule(
    proposal: RescheduleProposal | None,
    *,
    min_savings_eur: float = HITL_RESCHEDULE_MIN_SAVINGS_EUR,
) -> HITLDecision:
    """Decide whether to ask the user about a proposed appliance shift.

    A proposal is only forwarded to the user when ``savings_eur ≥
    min_savings_eur`` AND the shift is non-zero. For below-threshold
    savings or zero shifts we *auto-decline* (run the cycle at the original
    onset time) — there's nothing useful to ask about.

    Args:
        proposal: The reschedule offer from the optimiser, or ``None``.
        min_savings_eur: Below this we don't bother asking the user.

    Returns:
        ``HITLDecision(action="ask")`` with a human-readable question
        framed in terms of savings, or ``HITLDecision(action="auto",
        reason="run now")`` when the proposal isn't worth asking about.
    """
    if proposal is None:
        return HITLDecision(action="auto", reason="no reschedule proposal")
    if proposal.shift_minutes <= 0.5:
        # 30 s tolerance: anything tighter is just rounding noise.
        return HITLDecision(action="auto", reason="no useful shift")
    if proposal.savings_eur < min_savings_eur:
        logger.info(
            "hitl_policy.decide_reschedule: AUTO run-now — savings €%.3f < threshold €%.3f",
            proposal.savings_eur, min_savings_eur,
        )
        return HITLDecision(
            action="auto",
            reason=(
                f"savings €{proposal.savings_eur:.2f} < threshold €{min_savings_eur:.2f}"
            ),
        )
    shift_h = proposal.shift_minutes / 60.0
    propose_str = proposal.proposed_start_at.strftime("%H:%M")
    question = (
        f"Postpone {proposal.appliance} by {shift_h:.1f} h to {propose_str}? "
        f"You can save €{proposal.savings_eur:.2f}. (accept/decline)"
    )
    logger.info(
        "hitl_policy.decide_reschedule: ASK — %s shift %.0fmin saves €%.2f",
        proposal.appliance, proposal.shift_minutes, proposal.savings_eur,
    )
    return HITLDecision(
        action="ask",
        reason=(
            f"{proposal.appliance} shift {proposal.shift_minutes:.0f}min "
            f"saves €{proposal.savings_eur:.2f}"
        ),
        question=question,
    )


__all__ = ["decide", "decide_reschedule"]
