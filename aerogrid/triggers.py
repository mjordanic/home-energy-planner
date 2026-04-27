"""TriggerManager — decides when the outer (MPC) loop should fire.

The inner 1 Hz loop runs disaggregation + commit-tracking on every sample.
The outer loop (price forecast + behavioral prediction + MILP + HITL) is
expensive, so we only fire it when one of a handful of conditions is met:

1. ``new_onset``      — an appliance fired that isn't in the committed plan.
2. ``price_surprise`` — realized 15-min LBMP deviates from forecast by more
                        than ``REPLAN_PRICE_DEVIATION`` (relative).
3. ``deadline_slip``  — the EV's current charge rate cannot meet the daily
                        kWh target by 07:00 at the configured safety margin.
4. ``periodic``       — at most every ``TRIGGER_RESYNC_MINUTES`` simulated
                        minutes, regardless — a safety net for slow drift.

A ``cooldown_s`` guards against chatter (e.g. rapid onset flapping). The
evaluator is a pure function over the state you give it — the tracker just
remembers when it last fired.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from aerogrid.config import (
    EV_DEADLINE_HOUR,
    REPLAN_PRICE_DEVIATION,
    TRIGGER_COOLDOWN_S,
    TRIGGER_DEADLINE_SAFETY,
    TRIGGER_RESYNC_MINUTES,
)
from aerogrid.types import ApplianceOnset, PriceForecast, ReplanTrigger, Sample, ScheduledTask

logger = logging.getLogger(__name__)


def time_to_deadline_hours(now: datetime, deadline_hour: int = EV_DEADLINE_HOUR) -> float:
    """Hours remaining to the next deadline_hour:00 on the UTC clock.

    If now is already at or past the deadline for today, roll to tomorrow.
    """
    target = now.replace(hour=deadline_hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    hours = (target - now).total_seconds() / 3600.0
    logger.debug(
        "time_to_deadline_hours: now=%s deadline_hour=%02d:00 → %.2fh remaining",
        now.isoformat(), deadline_hour, hours,
    )
    return hours


def time_to_next_deadline(now: datetime, deadline_hours: tuple[int, ...]) -> float | None:
    """Hours to the nearest upcoming deadline from a set of daily deadline hours.

    Considers both today and tomorrow for each hour so the result is always
    the next occurrence in the future, regardless of whether any deadline hour
    has already passed today.

    Args:
        now: Current UTC datetime.
        deadline_hours: Tuple of UTC hours (e.g. ``(7, 18)`` for 07:00 and 18:00).

    Returns:
        Minimum hours to the next deadline, or ``None`` if ``deadline_hours``
        is empty.
    """
    if not deadline_hours:
        return None
    candidates: list[float] = []
    for dh in deadline_hours:
        for offset in (0, 1):
            candidate = (
                now.replace(hour=dh, minute=0, second=0, microsecond=0)
                + timedelta(days=offset)
            )
            if candidate > now:
                candidates.append((candidate - now).total_seconds() / 3600.0)
    return min(candidates) if candidates else None


@dataclass
class TriggerManager:
    price_deviation: float = REPLAN_PRICE_DEVIATION
    periodic_minutes: float = TRIGGER_RESYNC_MINUTES
    cooldown_s: float = TRIGGER_COOLDOWN_S
    deadline_safety: float = TRIGGER_DEADLINE_SAFETY
    last_replan_at: datetime | None = None

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        *,
        now: datetime,
        latest_sample: Sample | None = None,
        new_onsets: list[ApplianceOnset] | None = None,
        committed_tasks: list[ScheduledTask] | None = None,
        price_forecast: PriceForecast | None = None,
        remaining_ev_kwh: float = 0.0,
        ev_power_setpoint_kw: float = 0.0,
    ) -> ReplanTrigger | None:
        """Return a trigger if one fires at ``now``, else None."""
        logger.debug(
            "TriggerManager.evaluate: now=%s new_onsets=%d committed=%d "
            "ev_remaining=%.2fkWh ev_setpoint=%.2fkW",
            now.isoformat(),
            len(new_onsets or []),
            len(committed_tasks or []),
            remaining_ev_kwh,
            ev_power_setpoint_kw,
        )

        # Cooldown — prevents MILP thrashing on chattering signals.
        if self.last_replan_at is not None:
            elapsed_s = (now - self.last_replan_at).total_seconds()
            if elapsed_s < self.cooldown_s:
                logger.debug(
                    "TriggerManager: cooldown active (%.1fs elapsed < %.1fs threshold) — skipping",
                    elapsed_s, self.cooldown_s,
                )
                return None

        committed_apps = {t.appliance for t in (committed_tasks or [])}

        # 1. New onset for an appliance not already running under commit.
        for onset in new_onsets or []:
            if onset.appliance not in committed_apps:
                trigger = ReplanTrigger(
                    kind="new_onset", detail=onset.appliance, at=now,
                )
                logger.info(
                    "TriggerManager FIRED new_onset: appliance=%s at=%s",
                    onset.appliance, now.isoformat(),
                )
                return trigger

        # 2. Price surprise (realized 15-min LBMP vs forecast median).
        if (
            price_forecast is not None
            and latest_sample is not None
            and latest_sample.realized_price is not None
            and price_forecast.median
        ):
            realized = float(latest_sample.realized_price)
            expected = float(price_forecast.median[0])
            denom = max(abs(expected), 1e-6)
            deviation = abs(realized - expected) / denom
            logger.debug(
                "TriggerManager price check: realized=%.2f expected=%.2f deviation=%.1f%% threshold=%.1f%%",
                realized, expected, deviation * 100, self.price_deviation * 100,
            )
            if deviation > self.price_deviation:
                pct = (realized - expected) / denom * 100.0
                detail = f"realized={realized:.2f} forecast={expected:.2f} ({pct:+.0f}%)"
                logger.info("TriggerManager FIRED price_surprise: %s at=%s", detail, now.isoformat())
                return ReplanTrigger(kind="price_surprise", detail=detail, at=now)

        # 3. Deadline slip — will the current EV rate meet the daily kWh need?
        hours_left = time_to_deadline_hours(now)
        if remaining_ev_kwh > 0.1 and hours_left > 0:
            required_rate = remaining_ev_kwh / hours_left
            effective_rate = max(ev_power_setpoint_kw, 0.1) * self.deadline_safety
            logger.debug(
                "TriggerManager deadline-slip check: required_rate=%.2fkW effective_rate=%.2fkW "
                "remaining=%.2fkWh hours_left=%.2fh",
                required_rate, effective_rate, remaining_ev_kwh, hours_left,
            )
            # Allow current rate up to 1/safety of what's needed; otherwise replan.
            # default deadline_safety is 1.2
            if required_rate > effective_rate:
                detail = (
                    f"need {required_rate:.1f}kW, current {ev_power_setpoint_kw:.1f}kW, "
                    f"{remaining_ev_kwh:.1f}kWh / {hours_left:.1f}h"
                )
                logger.info("TriggerManager FIRED deadline_slip: %s at=%s", detail, now.isoformat())
                return ReplanTrigger(kind="deadline_slip", detail=detail, at=now)

        # 4. Periodic resync (or first call).
        if self.last_replan_at is None:
            logger.info("TriggerManager FIRED periodic: initial plan at=%s", now.isoformat())
            return ReplanTrigger(kind="periodic", detail="initial plan", at=now)
        minutes_since = (now - self.last_replan_at).total_seconds() / 60.0
        if minutes_since >= self.periodic_minutes:
            detail = f"{minutes_since:.1f}m since last"
            logger.info("TriggerManager FIRED periodic: %s at=%s", detail, now.isoformat())
            return ReplanTrigger(kind="periodic", detail=detail, at=now)

        logger.debug("TriggerManager: no trigger fired at=%s", now.isoformat())
        return None

    def notify_replanned(self, now: datetime) -> None:
        """Caller calls this after a replan completes, to restart the cooldown."""
        logger.debug("TriggerManager.notify_replanned: cooldown reset at=%s", now.isoformat())
        self.last_replan_at = now


__all__ = ["TriggerManager", "time_to_deadline_hours", "time_to_next_deadline"]
