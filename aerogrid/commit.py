"""CommitTracker — single source of truth for "what's actually running".

The inner 1 Hz loop calls ``.tick(now, dt)`` every sample to:
- decrement the EV's remaining kWh by its current setpoint,
- retire committed cycle tasks whose duration has elapsed,
- roll the remaining-EV counter at the daily deadline.

When the outer loop produces a plan and HITL approves it, the digital twin
calls ``.adopt_plan(plan, now)`` to commit the plan's first slot:
- the EV setpoint for the current 15-min slot is copied in,
- any task whose ``start_slot == 0`` becomes a committed task whose cycle
  cannot be rescheduled until it finishes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from aerogrid.config import EV_DAILY_NEED_KWH, EV_DEADLINE_HOUR, SLOT_MINUTES
from aerogrid.types import Schedule, ScheduledTask

logger = logging.getLogger(__name__)


@dataclass
class CommitTracker:
    remaining_ev_kwh: float = EV_DAILY_NEED_KWH
    ev_power_setpoint_kw: float = 0.0
    committed_tasks: list[ScheduledTask] = field(default_factory=list)
    last_deadline_reset: datetime | None = None
    _task_end_times: dict[str, datetime] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def tick(self, now: datetime, dt_s: float = 1.0) -> None:
        """Advance ``dt_s`` seconds of simulated wall time."""
        if self.ev_power_setpoint_kw > 0.0:
            delivered = self.ev_power_setpoint_kw * dt_s / 3600.0
            prev = self.remaining_ev_kwh
            self.remaining_ev_kwh = max(0.0, self.remaining_ev_kwh - delivered)
            logger.debug(
                "CommitTracker.tick: EV delivered=%.4fkWh remaining=%.3fkWh → %.3fkWh (setpoint=%.2fkW)",
                delivered, prev, self.remaining_ev_kwh, self.ev_power_setpoint_kw,
            )

        # Reset the daily EV target exactly at the deadline hour.
        if (
            now.hour == EV_DEADLINE_HOUR
            and now.minute == 0
            and now.second == 0
            and (
                self.last_deadline_reset is None
                or now > self.last_deadline_reset + timedelta(hours=1)
            )
        ):
            logger.info(
                "CommitTracker: daily EV reset at %s — remaining reset to %.1fkWh",
                now.isoformat(), EV_DAILY_NEED_KWH,
            )
            self.remaining_ev_kwh = EV_DAILY_NEED_KWH
            self.last_deadline_reset = now

        # Retire committed tasks whose cycle has finished.
        live: list[ScheduledTask] = []
        for task in self.committed_tasks:
            end = self._task_end_times.get(task.appliance)
            if end is None or now < end:
                live.append(task)
            else:
                self._task_end_times.pop(task.appliance, None)
                logger.info(
                    "CommitTracker: task retired appliance=%s cycle_end=%s",
                    task.appliance, end.isoformat() if end else "None",
                )
        self.committed_tasks = live

    # ------------------------------------------------------------------ #
    def adopt_plan(self, plan: Schedule, now: datetime) -> None:
        """Commit the first slot of ``plan`` — EV setpoint + any t=0 tasks."""
        logger.info(
            "CommitTracker.adopt_plan: now=%s tasks_in_plan=%d",
            now.isoformat(), len(plan.tasks),
        )
        if plan.ev_power_kw:
            new_setpoint = float(plan.ev_power_kw[0])
            logger.info(
                "CommitTracker: EV setpoint %.2fkW → %.2fkW",
                self.ev_power_setpoint_kw, new_setpoint,
            )
            self.ev_power_setpoint_kw = new_setpoint

        existing = {t.appliance for t in self.committed_tasks}
        for task in plan.tasks:
            if task.start_slot != 0:
                logger.debug(
                    "CommitTracker: skipping task appliance=%s start_slot=%d (not slot-0)",
                    task.appliance, task.start_slot,
                )
                continue
            if task.appliance in existing:
                logger.debug(
                    "CommitTracker: skipping task appliance=%s (already committed)",
                    task.appliance,
                )
                continue
            committed_task = ScheduledTask(
                appliance=task.appliance,
                start_slot=0,
                slots=task.slots,
                expected_kwh=task.expected_kwh,
                committed=True,
            )
            end_time = now + timedelta(minutes=SLOT_MINUTES * task.slots)
            self.committed_tasks.append(committed_task)
            self._task_end_times[task.appliance] = end_time
            logger.info(
                "CommitTracker: committed appliance=%s slots=%d expected_kwh=%.2f ends_at=%s",
                task.appliance, task.slots, task.expected_kwh, end_time.isoformat(),
            )

    def snapshot(self) -> dict:
        return {
            "remaining_ev_kwh": self.remaining_ev_kwh,
            "ev_power_setpoint_kw": self.ev_power_setpoint_kw,
            "committed_tasks": [t.as_dict() for t in self.committed_tasks],
        }


__all__ = ["CommitTracker"]
