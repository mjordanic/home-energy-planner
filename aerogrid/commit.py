"""CommitTracker — single source of truth for "what's actually running".

The inner 1 Hz loop calls ``.tick(now, dt)`` every sample to:
- decrement the EV's remaining kWh by its current setpoint,
- decrement each heater deadline window's remaining kWh by the heater
  setpoint applied to that window,
- retire committed cycle tasks whose duration has elapsed,
- roll the remaining-EV counter at the daily EV deadline,
- reset each heater window's remaining kWh as the deadline passes.

When the outer loop produces a plan and HITL approves it, the digital twin
calls ``.adopt_plan(plan, now)`` to commit the plan's first slot:
- the EV setpoint for the current 15-min slot is copied in,
- the heater setpoint for the current 15-min slot is copied in,
- any cycle task whose ``start_slot == 0`` becomes a committed task whose
  cycle cannot be rescheduled until it finishes.

A separate :meth:`adopt_cycle_start` is used by the HITL reschedule path
to commit a *future-shifted* cycle (e.g. user accepted "delay dishwasher
by 1 h"); the cycle is marked committed so the optimiser sees it as
exogenous load against the cap until it ends.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from aerogrid.config import (
    APPLIANCES,
    BatterySpec,
    EV_DAILY_NEED_KWH,
    EV_DEADLINE_HOUR,
    HEATER_DEADLINES,
    SLOT_MINUTES,
    HeaterEnergyDeadline,
)
from aerogrid.types import ApplianceOnset, Schedule, ScheduledTask

logger = logging.getLogger(__name__)


def _initial_heater_remaining(
    deadlines: tuple[HeaterEnergyDeadline, ...] = HEATER_DEADLINES,
) -> dict[int, float]:
    """Initial per-deadline-hour kWh: every window starts at its required total."""
    return {d.hour: float(d.kwh_required) for d in deadlines}


@dataclass
class CommitTracker:
    remaining_ev_kwh: float = EV_DAILY_NEED_KWH
    ev_power_setpoint_kw: float = 0.0
    heater_power_setpoint_kw: float = 0.0
    # Per-deadline-hour kWh remaining in the *currently active* window.
    # Reset to ``HEATER_DEADLINES[i].kwh_required`` exactly when each
    # deadline hour passes (handled inside ``tick``).
    remaining_heater_kwh_by_window: dict[int, float] = field(
        default_factory=_initial_heater_remaining,
    )
    committed_tasks: list[ScheduledTask] = field(default_factory=list)
    last_ev_deadline_reset: datetime | None = None
    last_heater_deadline_reset: dict[int, datetime] = field(default_factory=dict)
    _task_start_times: dict[str, datetime] = field(default_factory=dict)
    _task_end_times: dict[str, datetime] = field(default_factory=dict)
    # Track which heater window the current ``now`` lives in, so the heater
    # setpoint is debited from the right window.
    _heater_deadlines: tuple[HeaterEnergyDeadline, ...] = HEATER_DEADLINES

    # ---- Home Battery SoC tracking ---------------------------------------- #
    # ``battery_spec`` must be set to enable SoC tracking.  When ``None``
    # (the default) the tracker holds no battery state and ``tick()`` skips
    # all battery accounting — backward-compatible.
    battery_spec: BatterySpec | None = None
    soc_kwh: float = 0.0
    battery_charge_setpoint_kw: float = 0.0
    battery_discharge_setpoint_kw: float = 0.0
    # Applied (throttled) discharge from the most recent tick — the single
    # source of truth for "how much the battery actually delivered this slot".
    # Always ≤ battery_discharge_setpoint_kw; equals it when no throttle applies.
    battery_discharge_applied_kw: float = 0.0

    # ------------------------------------------------------------------ #
    def _active_heater_window(self, now: datetime) -> int | None:
        """Return the deadline hour whose window currently contains ``now``."""
        if not self._heater_deadlines:
            return None
        sorted_hours = sorted(d.hour for d in self._heater_deadlines)
        # Pick deadline with smallest distance ahead.
        best_hour = None
        best_dist = None
        for h in sorted_hours:
            target = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if target <= now:
                target = target + timedelta(days=1)
            dist = (target - now).total_seconds()
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_hour = h
        return best_hour

    # ------------------------------------------------------------------ #
    def tick(
        self,
        now: datetime,
        dt_s: float = 1.0,
        *,
        offsettable_load_kw: float | None = None,
    ) -> None:
        """Advance ``dt_s`` seconds of simulated wall time.

        Decrements EV and heater "remaining" counters according to current
        setpoints, retires expired committed tasks, and resets the daily
        EV need + each heater window's required kWh as the corresponding
        deadlines pass.

        When the battery is discharging and *offsettable_load_kw* is
        supplied, the applied discharge is throttled to
        ``min(setpoint, offsettable_load_kw + charge_setpoint)`` so that
        net grid draw never goes negative (no phantom export). SoC drains by
        the *applied* amount, not the raw setpoint. When omitted the behaviour
        is unchanged — every existing caller and test is unaffected.
        """
        if self.ev_power_setpoint_kw > 0.0:
            delivered = self.ev_power_setpoint_kw * dt_s / 3600.0
            prev = self.remaining_ev_kwh
            self.remaining_ev_kwh = max(0.0, self.remaining_ev_kwh - delivered)
            logger.debug(
                "CommitTracker.tick: EV delivered=%.4fkWh remaining=%.3fkWh → %.3fkWh (setpoint=%.2fkW)",
                delivered, prev, self.remaining_ev_kwh, self.ev_power_setpoint_kw,
            )

        if self.heater_power_setpoint_kw > 0.0 and self._heater_deadlines:
            active_h = self._active_heater_window(now)
            if active_h is not None:
                delivered = self.heater_power_setpoint_kw * dt_s / 3600.0
                prev = self.remaining_heater_kwh_by_window.get(active_h, 0.0)
                new_val = max(0.0, prev - delivered)
                self.remaining_heater_kwh_by_window[active_h] = new_val
                logger.debug(
                    "CommitTracker.tick: heater window=%02d:00 delivered=%.4fkWh "
                    "remaining=%.3fkWh → %.3fkWh (setpoint=%.2fkW)",
                    active_h, delivered, prev, new_val, self.heater_power_setpoint_kw,
                )

        # Update Home Battery SoC from charge/discharge setpoints.
        if self.battery_spec is not None:
            bspec = self.battery_spec
            if self.battery_charge_setpoint_kw > 0.0:
                energy_stored = bspec.eta_charge * self.battery_charge_setpoint_kw * dt_s / 3600.0
                self.soc_kwh = min(bspec.capacity_kwh, self.soc_kwh + energy_stored)
                logger.debug(
                    "CommitTracker.tick: battery charge stored=%.4f kWh soc=%.3f kWh",
                    energy_stored, self.soc_kwh,
                )
            if self.battery_discharge_setpoint_kw > 0.0:
                # No-export throttle (ADR 0001): the battery can only discharge
                # as much as the household load it offsets. When
                # offsettable_load_kw is supplied we cap applied discharge to
                # that load (plus any simultaneous charge draw). Without it we
                # apply the full setpoint — backward-compatible.
                if offsettable_load_kw is not None:
                    applied_dis_kw = min(
                        self.battery_discharge_setpoint_kw,
                        max(0.0, offsettable_load_kw + self.battery_charge_setpoint_kw),
                    )
                else:
                    applied_dis_kw = self.battery_discharge_setpoint_kw
                self.battery_discharge_applied_kw = applied_dis_kw
                energy_released = applied_dis_kw * dt_s / 3600.0 / bspec.eta_discharge
                self.soc_kwh = max(0.0, self.soc_kwh - energy_released)
                logger.debug(
                    "CommitTracker.tick: battery discharge setpoint=%.4f applied=%.4f "
                    "released=%.4f kWh soc=%.3f kWh",
                    self.battery_discharge_setpoint_kw, applied_dis_kw,
                    energy_released, self.soc_kwh,
                )
            else:
                self.battery_discharge_applied_kw = 0.0

        # Reset the daily EV target exactly at the deadline hour.
        if (
            now.hour == EV_DEADLINE_HOUR
            and now.minute == 0
            and now.second == 0
            and (
                self.last_ev_deadline_reset is None
                or now > self.last_ev_deadline_reset + timedelta(hours=1)
            )
        ):
            logger.info(
                "CommitTracker: daily EV reset at %s — remaining reset to %.1fkWh",
                now.isoformat(), EV_DAILY_NEED_KWH,
            )
            self.remaining_ev_kwh = EV_DAILY_NEED_KWH
            self.last_ev_deadline_reset = now

        # Reset each heater window at the moment its deadline passes.
        for d in self._heater_deadlines:
            if (
                now.hour == d.hour
                and now.minute == 0
                and now.second == 0
                and (
                    self.last_heater_deadline_reset.get(d.hour) is None
                    or now > self.last_heater_deadline_reset[d.hour] + timedelta(hours=1)
                )
            ):
                logger.info(
                    "CommitTracker: heater window %02d:00 reset at %s → %.1fkWh required",
                    d.hour, now.isoformat(), d.kwh_required,
                )
                self.remaining_heater_kwh_by_window[d.hour] = float(d.kwh_required)
                self.last_heater_deadline_reset[d.hour] = now

        # Retire committed tasks whose cycle has finished.
        live: list[ScheduledTask] = []
        for task in self.committed_tasks:
            end = self._task_end_times.get(task.appliance)
            if end is None or now < end:
                live.append(task)
            else:
                self._task_start_times.pop(task.appliance, None)
                self._task_end_times.pop(task.appliance, None)
                logger.info(
                    "CommitTracker: task retired appliance=%s cycle_end=%s",
                    task.appliance, end.isoformat() if end else "None",
                )
        self.committed_tasks = live

    # ------------------------------------------------------------------ #
    def adopt_plan(self, plan: Schedule, now: datetime) -> None:
        """Commit the first slot of ``plan`` — EV + heater setpoints + any t=0 tasks."""
        logger.info(
            "CommitTracker.adopt_plan: now=%s tasks_in_plan=%d ev_slots=%d heater_slots=%d",
            now.isoformat(), len(plan.tasks),
            len(plan.ev_power_kw), len(plan.heater_power_kw),
        )
        if plan.ev_power_kw:
            new_setpoint = float(plan.ev_power_kw[0])
            logger.info(
                "CommitTracker: EV setpoint %.2fkW → %.2fkW",
                self.ev_power_setpoint_kw, new_setpoint,
            )
            self.ev_power_setpoint_kw = new_setpoint
        if plan.heater_power_kw:
            new_setpoint = float(plan.heater_power_kw[0])
            logger.info(
                "CommitTracker: heater setpoint %.2fkW → %.2fkW",
                self.heater_power_setpoint_kw, new_setpoint,
            )
            self.heater_power_setpoint_kw = new_setpoint

        # Battery setpoints — only if the plan contains battery vectors.
        if plan.battery_charge_kw:
            self.battery_charge_setpoint_kw = float(plan.battery_charge_kw[0])
            logger.info(
                "CommitTracker: battery charge setpoint → %.2f kW",
                self.battery_charge_setpoint_kw,
            )
        if plan.battery_discharge_kw:
            self.battery_discharge_setpoint_kw = float(plan.battery_discharge_kw[0])
            logger.info(
                "CommitTracker: battery discharge setpoint → %.2f kW",
                self.battery_discharge_setpoint_kw,
            )

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
            self._commit_task(task.appliance, task.slots, task.expected_kwh, now)

    # ------------------------------------------------------------------ #
    def adopt_cycle_start(
        self,
        appliance: str,
        slots: int,
        expected_kwh: float,
        start_at: datetime,
        *,
        now: datetime | None = None,
    ) -> None:
        """Pin an event-driven cycle (dishwasher / washing machine) starting at ``start_at``.

        Used by the HITL reschedule path: after the user accepts a small
        forward shift, the inner loop schedules the cycle to start at
        ``start_at`` and registers it here so the optimiser sees it as
        exogenous load on the cap until it ends. ``start_at`` may be in
        the future (a deferred start) or equal to ``now`` (run immediately,
        the "decline" case is also handled this way).
        """
        if appliance in {t.appliance for t in self.committed_tasks}:
            old_start = self._task_start_times.get(appliance)
            if old_start is not None and now is not None and now < old_start:
                self._task_start_times[appliance] = start_at
                self._task_end_times[appliance] = start_at + timedelta(
                    minutes=SLOT_MINUTES * slots
                )
                logger.info(
                    "CommitTracker: updated deferred appliance=%s start_at=%s ends_at=%s",
                    appliance,
                    start_at.isoformat(),
                    self._task_end_times[appliance].isoformat(),
                )
            else:
                logger.debug(
                    "CommitTracker.adopt_cycle_start: already committed and started: %s",
                    appliance,
                )
            return
        self._commit_task(appliance, slots, expected_kwh, start_at)

    def _commit_task(
        self,
        appliance: str,
        slots: int,
        expected_kwh: float,
        start_at: datetime,
    ) -> None:
        """Internal — append a committed task and stamp its end time."""
        committed_task = ScheduledTask(
            appliance=appliance,
            start_slot=0,
            slots=int(slots),
            expected_kwh=float(expected_kwh),
            committed=True,
        )
        end_time = start_at + timedelta(minutes=SLOT_MINUTES * slots)
        self.committed_tasks.append(committed_task)
        self._task_start_times[appliance] = start_at
        self._task_end_times[appliance] = end_time
        logger.info(
            "CommitTracker: committed appliance=%s slots=%d expected_kwh=%.2f start_at=%s ends_at=%s",
            appliance, slots, expected_kwh, start_at.isoformat(), end_time.isoformat(),
        )

    def running_committed_tasks(self, now: datetime) -> list[ScheduledTask]:
        """Return committed cycle tasks that are actively running at ``now``."""
        out: list[ScheduledTask] = []
        for task in self.committed_tasks:
            s = self._task_start_times.get(task.appliance)
            e = self._task_end_times.get(task.appliance)
            if s is None or e is None:
                continue
            if s <= now < e:
                out.append(task)
        return out

    def replannable_onsets(self, now: datetime) -> list[ApplianceOnset]:
        """Expose deferred (not-yet-started) cycles as synthetic onsets."""
        out: list[ApplianceOnset] = []
        for task in self.committed_tasks:
            s = self._task_start_times.get(task.appliance)
            e = self._task_end_times.get(task.appliance)
            if s is None or e is None:
                continue
            if now < s:
                out.append(
                    ApplianceOnset(
                        appliance=task.appliance,
                        timestamp=now,
                        confidence=1.0,
                        source="injected",
                    )
                )
        return out

    def snapshot(self) -> dict:
        return {
            "remaining_ev_kwh": self.remaining_ev_kwh,
            "ev_power_setpoint_kw": self.ev_power_setpoint_kw,
            "heater_power_setpoint_kw": self.heater_power_setpoint_kw,
            "remaining_heater_kwh_by_window": dict(self.remaining_heater_kwh_by_window),
            "committed_tasks": [t.as_dict() for t in self.committed_tasks],
            "soc_kwh": self.soc_kwh,
            "battery_charge_setpoint_kw": self.battery_charge_setpoint_kw,
            "battery_discharge_setpoint_kw": self.battery_discharge_setpoint_kw,
            "battery_discharge_applied_kw": self.battery_discharge_applied_kw,
        }


__all__ = ["CommitTracker"]
