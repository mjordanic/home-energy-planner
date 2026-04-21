"""LangGraph state schema."""
from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

import numpy as np

from aerogrid.types import ApplianceOnset, PriceForecast, Schedule


class AeroGridState(TypedDict, total=False):
    """State carried between LangGraph nodes.

    Fields are all optional (`total=False`) so nodes can populate incrementally.
    """
    # time and sensor inputs
    now: datetime
    mains_chunk: tuple[np.ndarray, np.ndarray] | None   # (voltage, current), 16 kHz
    chunk_start: datetime | None

    # NILM output + history
    new_onsets: list[ApplianceOnset]
    recent_onsets: list[ApplianceOnset]

    # forecasting + planning
    price_forecast: PriceForecast | None
    onset_probs: dict[str, np.ndarray]
    schedule: Schedule | None

    # HITL
    pending_question: str | None
    user_confirmation: str | None

    # monitor / replan
    realized_prices: list[float]            # each slot's realized price, appended
    replan_reason: str | None
    iteration: int                          # how many graph cycles we've completed

    # housekeeping
    cumulative_cost: float
    cumulative_baseline_cost: float
    event_log: list[dict[str, Any]]
