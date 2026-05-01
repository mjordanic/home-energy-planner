"""LangGraph state schema.

The graph is a streaming sample processor: every 1 Hz sample enters through
``latest_sample`` and, when TriggerManager says so, a replan pass populates
``current_plan`` / ``committed_until`` / ``replan_reason``. All fields are
optional (``total=False``) so nodes can write incrementally.

After the April 2026 refactor the state also tracks the heater (continuous
variable power, like the EV) and any pending event-driven appliance
reschedule proposals (dishwasher / washing machine).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from aerogrid.types import (
    ApplianceOnset,
    HITLDecision,
    PriceForecast,
    ReplanTrigger,
    RescheduleProposal,
    Sample,
    Schedule,
    ScheduledTask,
)


class AeroGridState(TypedDict, total=False):
    # ---- Streaming ---- #
    now: datetime
    latest_sample: Sample

    # ---- Onsets (updated every sample) ---- #
    new_onsets: list[ApplianceOnset]           # onsets received since last state flush
    recent_onsets: list[ApplianceOnset]        # rolling history, capped

    # ---- Commitment (updated every sample + on replan) ---- #
    committed_tasks: list[ScheduledTask]       # currently running, cannot be preempted
    ev_power_setpoint_kw: float                # current EV charge rate
    remaining_ev_kwh: float                    # kWh still owed to meet EV deadline
    heater_power_setpoint_kw: float            # current heater power
    # Per-deadline-hour remaining kWh. Keys are UTC hours from
    # ``HEATER_DEADLINES``; values are kWh still owed in the window ending
    # at that hour. Reset to ``kwh_required`` at the moment each deadline
    # passes (handled by CommitTracker.tick).
    remaining_heater_kwh_by_window: dict[int, float]

    # ---- Planning (updated on replan only) ---- #
    price_forecast: PriceForecast | None       # short-horizon forecast
    current_plan: Schedule | None              # output of last replan
    previous_plan: Schedule | None             # last confirmed plan, for HITL diff
    committed_until: datetime | None           # first slot(s) of plan now committed
    last_replan_at: datetime | None
    replan_trigger: ReplanTrigger | None

    # ---- HITL ---- #
    hitl_decision: HITLDecision | None
    pending_question: str | None
    user_confirmation: str | None
    # Pending reschedule offer for an event-driven cycle appliance
    # (dishwasher / washing machine). When non-None the HITL gate will ask
    # the user (or the simulated user, via HITL_AUTO_RESPONSES) whether to
    # accept the shift. Cleared by the HITL gate after a decision is made.
    pending_reschedule: RescheduleProposal | None
    # Alternative schedule that pins the proposed cycle at slot 0 (the
    # "run now" outcome of a HITL decline). Computed alongside the
    # reschedule proposal so the commit step can swap the EV/heater plan
    # to the cap-feasible decline-version when the user (or the simulated
    # user) chooses to decline. None whenever no proposal is pending.
    decline_plan: Schedule | None

    # ---- Monitoring / bookkeeping ---- #
    # Strategy-local running cost; populated by the OptimizerStrategy when it
    # builds the graph state. The baseline (and any future strategy) tracks
    # its own cost outside the LangGraph state.
    cumulative_cost: float
    event_log: list[dict[str, Any]]
