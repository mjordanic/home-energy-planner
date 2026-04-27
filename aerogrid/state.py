"""LangGraph state schema.

The graph is a streaming sample processor: every 1 Hz sample enters through
``latest_sample`` and, when TriggerManager says so, a replan pass populates
``current_plan`` / ``committed_until`` / ``replan_reason``. All fields are
optional (``total=False``) so nodes can write incrementally.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

import numpy as np

from aerogrid.types import (
    ApplianceOnset,
    HITLDecision,
    PriceForecast,
    ReplanTrigger,
    Sample,
    Schedule,
    ScheduledTask,
)


class AeroGridState(TypedDict, total=False):
    # ---- Streaming ---- #
    now: datetime
    latest_sample: Sample

    # ---- Disaggregation (updated every sample) ---- #
    per_appliance_power_w: dict[str, float]
    new_onsets: list[ApplianceOnset]           # onsets detected since last state flush
    recent_onsets: list[ApplianceOnset]        # rolling history, capped

    # ---- Commitment (updated every sample + on replan) ---- #
    committed_tasks: list[ScheduledTask]       # currently running, cannot be preempted
    ev_power_setpoint_kw: float                # current EV charge rate
    remaining_ev_kwh: float                    # kWh still owed to meet deadline

    # ---- Planning (updated on replan only) ---- #
    price_forecast: PriceForecast | None       # short-horizon forecast
    onset_probs: dict[str, np.ndarray]         # per-appliance behavioral priors
    current_plan: Schedule | None              # output of last replan
    previous_plan: Schedule | None             # last confirmed plan, for HITL diff
    committed_until: datetime | None           # first slot(s) of plan now committed
    last_replan_at: datetime | None
    replan_trigger: ReplanTrigger | None

    # ---- HITL ---- #
    hitl_decision: HITLDecision | None
    pending_question: str | None
    user_confirmation: str | None

    # ---- Monitoring / bookkeeping ---- #
    cumulative_cost: float
    cumulative_baseline_cost: float
    event_log: list[dict[str, Any]]
