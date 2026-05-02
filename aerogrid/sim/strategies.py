"""Independent strategy agents for digital-twin comparison.

Architecture
------------
Each strategy is a self-contained agent that owns its own planning
machinery (price oracle, MPC graph, …) and is fed by the digital twin
with a uniform per-tick view of the simulation environment:

  * a 1 Hz tick stream (:class:`~aerogrid.sim.streamer.Streamer`)
    carrying the realized price on slot boundaries,
  * the same gated list of appliance onsets each tick.

Per simulation tick, the digital twin hands every strategy

  ``tick(sample, onsets, dt_s=1.0)``

and lets each strategy decide what to do.  Both strategies see the same
onsets — there is no NILM in the loop, so the comparison is symmetric.

Cross-strategy onset gating
---------------------------
The only inter-strategy coordination happens upstream in the digital twin.
For every onset due at the current sample, the digital twin checks every
strategy's :meth:`Strategy.has_pending_appliance` flag.  If any strategy
still has the appliance pending or running, the onset is suppressed for
all strategies — preventing the same household appliance from being
"started twice" while a previous cycle is still in flight.

Concrete strategies
-------------------
:class:`BaselineStrategy`
    Naive ASAP household.  Time-driven setpoints (EV charges at rated
    power inside its window until full; heater runs at rated power until
    each window's kWh requirement is met) and immediate cycle starts on
    every gated onset.

:class:`OptimizerStrategy`
    MPC + LangGraph + HITL gating.  Owns its own price oracle and compiled
    LangGraph; both are built inside its constructor.  Multiple instances
    can be run side by side with different oracles — each one is fully
    independent.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from aerogrid.commit import CommitTracker
from aerogrid.config import (
    APPLIANCES,
    EV_AVAILABLE_FROM_HOUR,
    EV_DAILY_NEED_KWH,
    EV_DEADLINE_HOUR,
    HEATER_DEADLINES,
    HITL_AUTO_RESPONSES,
    SHORT_HORIZON_SLOTS,
    SLOT_MINUTES,
    HeaterEnergyDeadline,
)
from aerogrid.graph import build_graph, make_thread_id as _default_make_thread_id
from aerogrid.price_oracle import make_oracle
from aerogrid.triggers import TriggerManager
from aerogrid.types import ApplianceOnset, Sample

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared data types                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class SlotRecord:
    """Per-strategy snapshot at one 15-minute slot boundary.

    The digital twin assembles one row per slot by flattening every
    strategy's SlotRecord (column-prefixed by ``strategy.name`` via
    :meth:`to_flat_dict`).  ``extra`` holds strategy-specific fields that
    don't fit the common schema.
    """
    timestamp: datetime
    ev_kw: float
    heater_kw: float
    cycle_kw: float
    total_kw: float
    slot_cost_eur: float
    cum_cost_eur: float
    remaining_ev_kwh: float
    active_cycles: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self, prefix: str) -> dict[str, Any]:
        """Flatten into a dict whose keys are ``{prefix}_<field>``."""
        out: dict[str, Any] = {
            f"{prefix}_ev_kw": self.ev_kw,
            f"{prefix}_heater_kw": self.heater_kw,
            f"{prefix}_cycle_kw": self.cycle_kw,
            f"{prefix}_total_kw": self.total_kw,
            f"{prefix}_slot_cost_eur": self.slot_cost_eur,
            f"{prefix}_cum_cost_eur": self.cum_cost_eur,
            f"{prefix}_remaining_ev_kwh": self.remaining_ev_kwh,
            f"{prefix}_active_cycles": ",".join(self.active_cycles),
        }
        for k, v in self.extra.items():
            out[f"{prefix}_{k}"] = v
        return out


def make_event(
    timestamp: datetime,
    strategy: str,
    event_type: str,
    *,
    appliance: str | None = None,
    power_kw: float | None = None,
    remaining_ev_kwh: float | None = None,
    cum_cost_eur: float = 0.0,
    price_eur_mwh: float | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Construct a row for ``event_log.parquet`` with the canonical schema.

    Conventions for ``strategy``:

    * ``"baseline"`` / ``"optimizer"`` / ...   — strategy-emitted events
    * ``"stream"``                              — digital-twin-level events
                                                  (``onset_permitted`` /
                                                  ``onset_suppressed``)
    """
    return {
        "timestamp": timestamp,
        "strategy": strategy,
        "event_type": event_type,
        "appliance": appliance,
        "power_kw": power_kw,
        "remaining_ev_kwh": remaining_ev_kwh,
        "cum_cost_eur": cum_cost_eur,
        "price_eur_mwh": price_eur_mwh,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _in_ev_window(h: float) -> bool:
    """True when fractional hour-of-day *h* is inside the EV availability window."""
    if EV_AVAILABLE_FROM_HOUR < EV_DEADLINE_HOUR:
        return EV_AVAILABLE_FROM_HOUR <= h < EV_DEADLINE_HOUR
    return h >= EV_AVAILABLE_FROM_HOUR or h < EV_DEADLINE_HOUR


def _active_heater_window(
    now: datetime,
    deadlines: tuple[HeaterEnergyDeadline, ...] = HEATER_DEADLINES,
) -> int | None:
    """Return the deadline hour whose window currently contains *now*."""
    if not deadlines:
        return None
    best_hour: int | None = None
    best_dist: float | None = None
    for d in deadlines:
        target = now.replace(hour=d.hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        dist = (target - now).total_seconds()
        if best_dist is None or dist < best_dist:
            best_dist, best_hour = dist, d.hour
    return best_hour


def _jsonable(x: Any) -> Any:
    """Recursively convert *x* into JSON-serialisable Python primitives."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if hasattr(x, "as_dict"):
        return x.as_dict()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    return str(x)


# --------------------------------------------------------------------------- #
# Strategy abstract base class                                                 #
# --------------------------------------------------------------------------- #

class Strategy(ABC):
    """Interface every strategy agent must implement.

    Subclasses MUST set ``self.name`` (column-prefix in the slot log and the
    ``strategy`` column in the event log) and SHOULD initialise
    ``self.cumulative_cost = 0.0`` plus ``self._pending_events: list[dict]``
    via ``super().__init__(name)``.

    Required:

    :meth:`tick`
        Process one second.  Update internal state, react to gated onsets,
        emit events.
    :meth:`has_pending_appliance`
        Used by the digital twin to gate onsets across strategies.
    :meth:`get_slot_record`
        Snapshot at a 15-min slot boundary; ALSO accrues this slot's cost.

    Provided:

    :meth:`flush_events` / :meth:`close` / :meth:`summary`
        Default implementations work for most strategies.
    """

    name: str
    cumulative_cost: float

    def __init__(self, name: str) -> None:
        self.name = name
        self.cumulative_cost = 0.0
        self._pending_events: list[dict[str, Any]] = []

    @abstractmethod
    def tick(
        self,
        sample: Sample,
        onsets: list[ApplianceOnset],
        *,
        dt_s: float = 1.0,
    ) -> None:
        """Advance one second.  ``onsets`` is already cross-strategy gated."""

    @abstractmethod
    def has_pending_appliance(self, appliance: str) -> bool:
        """True if a cycle for *appliance* is currently running OR committed-but-not-yet-started."""

    @abstractmethod
    def get_slot_record(self, now: datetime, price_eur_mwh: float) -> SlotRecord:
        """Accrue this slot's cost and return a SlotRecord snapshot."""

    def flush_events(
        self,
        *,
        price_eur_mwh: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return and clear all pending events; back-fill last-known price."""
        events = self._pending_events
        self._pending_events = []
        if price_eur_mwh is not None:
            for e in events:
                if e.get("price_eur_mwh") is None:
                    e["price_eur_mwh"] = price_eur_mwh
        return events

    def close(self) -> None:
        """Final cleanup hook.  Default: no-op."""
        return None

    def summary(self) -> dict[str, Any]:
        """End-of-run summary for printing / logging.  Default: name + cost."""
        return {"name": self.name, "cumulative_cost_eur": self.cumulative_cost}


# --------------------------------------------------------------------------- #
# BaselineStrategy                                                             #
# --------------------------------------------------------------------------- #

class BaselineStrategy(Strategy):
    """Naive ASAP household policy — price-unaware time-driven setpoints.

    EV charges at full rated power from plug-in until the battery is full,
    then stops; heater runs at full rated power at the start of each
    deadline window until that window's kWh requirement is met; cycle
    appliances start immediately at every gated onset.

    Construction is parameter-free apart from the strategy ``name``.
    """

    def __init__(self, name: str = "baseline") -> None:
        super().__init__(name)
        self.remaining_ev_kwh: float = EV_DAILY_NEED_KWH
        self.remaining_heater_kwh_by_window: dict[int, float] = {
            d.hour: float(d.kwh_required) for d in HEATER_DEADLINES
        }
        # Active cycles: (appliance, end_time, power_kw) — appliance unique.
        self._active_cycles: list[tuple[str, datetime, float]] = []

        # Current power setpoints — set in tick(), read in get_slot_record().
        self._ev_power_kw: float = 0.0
        self._heater_power_kw: float = 0.0

        # Guards: prevent repeated resets within the same clock-hour.
        self._last_ev_reset: datetime | None = None
        self._last_heater_resets: dict[int, datetime] = {}

    # ------------------------------------------------------------------ #
    def has_pending_appliance(self, appliance: str) -> bool:
        # ASAP starts immediately, so "pending" == "currently running".
        return any(a == appliance for a, _, _ in self._active_cycles)

    # ------------------------------------------------------------------ #
    def _emit(
        self,
        now: datetime,
        event_type: str,
        *,
        appliance: str | None = None,
        power_kw: float | None = None,
        detail: str | None = None,
    ) -> None:
        self._pending_events.append(make_event(
            timestamp=now,
            strategy=self.name,
            event_type=event_type,
            appliance=appliance,
            power_kw=power_kw,
            remaining_ev_kwh=self.remaining_ev_kwh,
            cum_cost_eur=self.cumulative_cost,
            price_eur_mwh=None,
            detail=detail,
        ))

    # ------------------------------------------------------------------ #
    def tick(
        self,
        sample: Sample,
        onsets: list[ApplianceOnset],
        *,
        dt_s: float = 1.0,
    ) -> None:
        """Process one second: update setpoints, react to onsets, emit events.

        Order of operations matches :meth:`CommitTracker.tick` so the baseline
        sees the same per-second discretisation behaviour as the optimizer.
        """
        now = sample.t
        prev_ev_kw = self._ev_power_kw
        prev_heater_kw = self._heater_power_kw
        prev_cycle_apps = {a for a, _, _ in self._active_cycles}

        h = now.hour + now.minute / 60.0

        # --- EV: rated power inside window while need > 0 -----------------
        ev_rated = APPLIANCES["ev_charger"].rated_kw
        if _in_ev_window(h) and self.remaining_ev_kwh > 1e-6:
            self._ev_power_kw = ev_rated
        else:
            self._ev_power_kw = 0.0
        self.remaining_ev_kwh = max(
            0.0, self.remaining_ev_kwh - self._ev_power_kw * dt_s / 3600.0
        )

        # --- Heater: rated power at window start until requirement met ----
        active_h = _active_heater_window(now)
        heater_rated = APPLIANCES["heater"].rated_kw
        need_before = (
            self.remaining_heater_kwh_by_window.get(active_h, 0.0)
            if active_h is not None else 0.0
        )
        if active_h is not None and need_before > 1e-6:
            self._heater_power_kw = heater_rated
            delivered = heater_rated * dt_s / 3600.0
            self.remaining_heater_kwh_by_window[active_h] = max(0.0, need_before - delivered)
        else:
            self._heater_power_kw = 0.0

        # --- Daily EV reset -----------------------------------------------
        if (
            now.hour == EV_DEADLINE_HOUR and now.minute == 0 and now.second == 0
            and (self._last_ev_reset is None
                 or now > self._last_ev_reset + timedelta(hours=1))
        ):
            self.remaining_ev_kwh = EV_DAILY_NEED_KWH
            self._last_ev_reset = now
            self._emit(
                now, "ev_daily_reset", appliance="ev_charger",
                detail=f"reset to {EV_DAILY_NEED_KWH:.1f} kWh",
            )

        # --- Heater window resets -----------------------------------------
        for d in HEATER_DEADLINES:
            if (
                now.hour == d.hour and now.minute == 0 and now.second == 0
                and (self._last_heater_resets.get(d.hour) is None
                     or now > self._last_heater_resets[d.hour] + timedelta(hours=1))
            ):
                self.remaining_heater_kwh_by_window[d.hour] = float(d.kwh_required)
                self._last_heater_resets[d.hour] = now
                self._emit(
                    now, "heater_window_reset", appliance="heater",
                    detail=f"window {d.hour:02d}:00 reset to {d.kwh_required:.1f} kWh",
                )

        # --- Retire finished cycles ---------------------------------------
        self._active_cycles = [
            (a, e, pwr) for a, e, pwr in self._active_cycles if e > now
        ]
        for retired in prev_cycle_apps - {a for a, _, _ in self._active_cycles}:
            self._emit(
                now, "cycle_end", appliance=retired, power_kw=0.0,
                detail="cycle duration elapsed",
            )

        # --- React to gated onsets — start each cycle ASAP ---------------
        for onset in onsets:
            self._emit(
                now, "onset_received",
                appliance=onset.appliance,
                detail=f"source={onset.source} confidence={onset.confidence:.2f}",
            )
            self._start_cycle(onset.appliance, now)

        # --- Emit setpoint transition events ------------------------------
        if prev_ev_kw < 1e-6 and self._ev_power_kw > 1e-6:
            self._emit(
                now, "ev_charging_start", appliance="ev_charger",
                power_kw=self._ev_power_kw,
                detail=(
                    f"charging at {self._ev_power_kw:.2f} kW, "
                    f"remaining {self.remaining_ev_kwh:.2f} kWh"
                ),
            )
        elif prev_ev_kw > 1e-6 and self._ev_power_kw < 1e-6:
            reason = "battery_full" if self.remaining_ev_kwh < 1e-3 else "outside_ev_window"
            self._emit(
                now, "ev_charging_stop", appliance="ev_charger", power_kw=0.0,
                detail=f"{reason}, remaining {self.remaining_ev_kwh:.3f} kWh",
            )

        if prev_heater_kw < 1e-6 and self._heater_power_kw > 1e-6:
            self._emit(
                now, "heater_on", appliance="heater",
                power_kw=self._heater_power_kw,
                detail=f"window {active_h:02d}:00, remaining {need_before:.2f} kWh",
            )
        elif prev_heater_kw > 1e-6 and self._heater_power_kw < 1e-6:
            need_after = (
                self.remaining_heater_kwh_by_window.get(active_h, 0.0)
                if active_h is not None else 0.0
            )
            reason = "window_satisfied" if need_after < 1e-3 else "no_active_need"
            self._emit(
                now, "heater_off", appliance="heater", power_kw=0.0,
                detail=f"{reason}",
            )

    # ------------------------------------------------------------------ #
    def _start_cycle(self, appliance: str, now: datetime) -> None:
        """Start *appliance* at *now* with no shift (naive policy)."""
        spec = APPLIANCES.get(appliance)
        if spec is None or spec.cycle_slots <= 0:
            return
        if any(a == appliance for a, _, _ in self._active_cycles):
            return
        duration = timedelta(minutes=SLOT_MINUTES * spec.cycle_slots)
        end_time = now + duration
        self._active_cycles.append((appliance, end_time, spec.rated_kw))
        logger.info(
            "%s: %s cycle started at %s, ends %s (%.2f kW)",
            self.name, appliance, now.isoformat(), end_time.isoformat(), spec.rated_kw,
        )
        self._emit(
            now, "cycle_start", appliance=appliance, power_kw=spec.rated_kw,
            detail=(
                f"immediate onset response, "
                f"duration {spec.cycle_slots * SLOT_MINUTES} min"
            ),
        )

    # ------------------------------------------------------------------ #
    def get_slot_record(self, now: datetime, price_eur_mwh: float) -> SlotRecord:
        cycle_kw = sum(pwr for _, _, pwr in self._active_cycles)
        total_kw = self._ev_power_kw + self._heater_power_kw + cycle_kw
        slot_cost = total_kw * (SLOT_MINUTES / 60.0) * (price_eur_mwh / 1000.0)
        self.cumulative_cost += slot_cost

        return SlotRecord(
            timestamp=now,
            ev_kw=self._ev_power_kw,
            heater_kw=self._heater_power_kw,
            cycle_kw=cycle_kw,
            total_kw=total_kw,
            slot_cost_eur=slot_cost,
            cum_cost_eur=self.cumulative_cost,
            remaining_ev_kwh=self.remaining_ev_kwh,
            active_cycles=[a for a, _, _ in self._active_cycles],
        )


# --------------------------------------------------------------------------- #
# OptimizerStrategy                                                            #
# --------------------------------------------------------------------------- #

class OptimizerStrategy(Strategy):
    """Model Predictive Control (MPC) + LangGraph + HITL — owns its price oracle 
    and compiled graph.

    Construction
    ------------
    All optimizer-private machinery is built INSIDE the constructor:

    * price oracle (selected by ``price_oracle_impl``),
    * compiled LangGraph (using the oracle + the SHARED
      ``price_history_provider`` from the digital twin's PriceServer).

    The only external dependency is ``price_history_provider`` — realized
    price history belongs to the simulation environment, not the strategy.

    Multiple OptimizerStrategy instances can be run side by side with
    different ``price_oracle_impl`` values — each owns its own graph, so
    there is no hidden cross-talk.
    """

    def __init__(
        self,
        name: str = "optimizer",
        *,
        price_history_provider: Callable,
        price_oracle_impl: str = "naive",
        horizon_slots: int = SHORT_HORIZON_SLOTS,
        auto_confirm: bool = True,
        auto_responses: dict[str, str] | None = None,
        replan_jsonl_path: Path | None = None,
    ) -> None:
        super().__init__(name)
        self.commit = CommitTracker(remaining_ev_kwh=EV_DAILY_NEED_KWH)
        self.trig = TriggerManager()
        self.auto_responses = (
            auto_responses if auto_responses is not None
            else dict(HITL_AUTO_RESPONSES)
        )

        oracle = make_oracle(price_oracle_impl)
        builder, checkpointer = build_graph(
            price_oracle=oracle,
            price_history_provider=price_history_provider,
            auto_confirm=auto_confirm,
            horizon_slots=horizon_slots,
            auto_responses=self.auto_responses,
        )
        self.graph = builder.compile(checkpointer=checkpointer)
        self._make_thread_id = _default_make_thread_id

        # Per-replan state
        self._previous_plan: Any = None
        self._last_solver_status: str | None = None
        self._last_expected_cost: float | None = None
        self._last_trigger_kind: str | None = None  # cleared after each slot record

        # Counters surfaced in summary().
        self.n_replans: int = 0
        self.n_hitl: int = 0
        self.n_reschedule_accept: int = 0
        self.n_reschedule_decline: int = 0
        self.last_replan_reason: str | None = None

        # Optional per-replan JSONL log.
        self._replan_jsonl_path = replan_jsonl_path
        self._replan_fh = None
        if replan_jsonl_path is not None:
            replan_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._replan_fh = replan_jsonl_path.open("w")

    # ------------------------------------------------------------------ #
    def has_pending_appliance(self, appliance: str) -> bool:
        # CommitTracker holds running AND deferred-future committed tasks.
        return appliance in {t.appliance for t in self.commit.committed_tasks}

    def close(self) -> None:
        if self._replan_fh is not None:
            self._replan_fh.close()
            self._replan_fh = None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cumulative_cost_eur": self.cumulative_cost,
            "n_replans": self.n_replans,
            "n_hitl": self.n_hitl,
            "n_reschedule_accept": self.n_reschedule_accept,
            "n_reschedule_decline": self.n_reschedule_decline,
            "last_replan_reason": self.last_replan_reason,
        }

    # ------------------------------------------------------------------ #
    def _emit(
        self,
        now: datetime,
        event_type: str,
        *,
        appliance: str | None = None,
        power_kw: float | None = None,
        detail: str | None = None,
    ) -> None:
        self._pending_events.append(make_event(
            timestamp=now,
            strategy=self.name,
            event_type=event_type,
            appliance=appliance,
            power_kw=power_kw,
            remaining_ev_kwh=self.commit.remaining_ev_kwh,
            cum_cost_eur=self.cumulative_cost,
            price_eur_mwh=None,
            detail=detail,
        ))

    # ------------------------------------------------------------------ #
    def tick(
        self,
        sample: Sample,
        onsets: list[ApplianceOnset],
        *,
        dt_s: float = 1.0,
    ) -> None:
        """Process one second.

        1. Tick CommitTracker.
        2. Emit ``onset_received`` for each gated onset.
        3. Trigger evaluation; if it fires, run the slow-path graph and
           adopt the resulting plan.
        """
        from langgraph.types import Command  # lazy

        now = sample.t

        # 1. Advance commit-tracker state.
        self.commit.tick(now, dt_s)

        new_onsets = list(onsets)
        for onset in new_onsets:
            self._emit(
                now, "onset_received",
                appliance=onset.appliance,
                detail=f"source={onset.source} confidence={onset.confidence:.2f}",
            )

        # 2. Trigger evaluation.
        trigger = self.trig.evaluate(
            now=now,
            latest_sample=sample,
            new_onsets=new_onsets,
            committed_tasks=self.commit.running_committed_tasks(now),
            price_forecast=None,
            remaining_ev_kwh=self.commit.remaining_ev_kwh,
            ev_power_setpoint_kw=self.commit.ev_power_setpoint_kw,
        )
        if trigger is None:
            return

        # 3. Slow path — build state, run graph, adopt plan.
        self._emit(
            now, "replan_triggered",
            detail=f"kind={trigger.kind} reason={trigger.detail!r}",
        )
        logger.info(
            "%s: TRIGGER kind=%s detail=%r at=%s (replan #%d)",
            self.name, trigger.kind, trigger.detail, now.isoformat(),
            self.n_replans + 1,
        )
        self._last_trigger_kind = trigger.kind

        state_in = {
            "now": now,
            "latest_sample": sample,
            "new_onsets": [*new_onsets, *self.commit.replannable_onsets(now)],
            "committed_tasks": list(self.commit.running_committed_tasks(now)),
            "remaining_ev_kwh": self.commit.remaining_ev_kwh,
            "ev_power_setpoint_kw": self.commit.ev_power_setpoint_kw,
            "heater_power_setpoint_kw": self.commit.heater_power_setpoint_kw,
            "remaining_heater_kwh_by_window": dict(
                self.commit.remaining_heater_kwh_by_window
            ),
            "previous_plan": self._previous_plan,
            "replan_trigger": trigger,
            "event_log": [],
            "cumulative_cost": self.cumulative_cost,
        }
        cfg = {"configurable": {"thread_id": self._make_thread_id(now)}}
        try:
            result = self.graph.invoke(state_in, config=cfg)
        except Exception as exc:
            logger.error(
                "%s: graph error at=%s: %r", self.name, now.isoformat(), exc
            )
            return

        if "__interrupt__" in result:
            self.n_hitl += 1
            logger.info(
                "%s: HITL interrupt #%d at=%s — auto-resuming",
                self.name, self.n_hitl, now.isoformat(),
            )
            result = self.graph.invoke(Command(resume="yes"), config=cfg)

        self.n_replans += 1
        self.last_replan_reason = trigger.detail or trigger.kind
        self.trig.notify_replanned(now)

        new_plan = result.get("current_plan")
        proposal = result.get("pending_reschedule")
        ans = (result.get("user_confirmation") or "").lower()

        if new_plan is not None:
            self._adopt_plan(now, new_plan)
            self._previous_plan = new_plan
            self._last_solver_status = new_plan.solver_status
            self._last_expected_cost = float(new_plan.expected_cost)

        if proposal is not None:
            self._emit(
                now, "reschedule_proposed",
                appliance=proposal.appliance,
                detail=(
                    f"shift={proposal.shift_minutes:.0f} min "
                    f"savings=€{proposal.savings_eur:.3f} "
                    f"proposed_start={proposal.proposed_start_at.isoformat()}"
                ),
            )
            self._handle_reschedule_decision(now, proposal, ans)

        self._write_replan_jsonl(
            now=now, sample=sample, trigger=trigger, result=result,
            new_plan=new_plan, proposal=proposal, ans=ans,
        )

    # ------------------------------------------------------------------ #
    def _adopt_plan(self, now: datetime, new_plan: Any) -> None:
        prev_ev_sp = self.commit.ev_power_setpoint_kw
        prev_heat_sp = self.commit.heater_power_setpoint_kw
        self.commit.adopt_plan(new_plan, now)

        if abs(self.commit.ev_power_setpoint_kw - prev_ev_sp) > 1e-6:
            self._emit(
                now, "ev_setpoint_changed", appliance="ev_charger",
                power_kw=self.commit.ev_power_setpoint_kw,
                detail=(
                    f"{prev_ev_sp:.2f} → {self.commit.ev_power_setpoint_kw:.2f} kW "
                    f"(solver={new_plan.solver_status})"
                ),
            )
        if abs(self.commit.heater_power_setpoint_kw - prev_heat_sp) > 1e-6:
            self._emit(
                now, "heater_setpoint_changed", appliance="heater",
                power_kw=self.commit.heater_power_setpoint_kw,
                detail=(
                    f"{prev_heat_sp:.2f} → {self.commit.heater_power_setpoint_kw:.2f} kW"
                ),
            )

        for appliance, slot in (new_plan.cycle_starts or {}).items():
            spec = APPLIANCES.get(appliance)
            if spec is None or spec.cycle_slots <= 0:
                continue
            cycle_kwh = spec.rated_kw * spec.cycle_slots * (SLOT_MINUTES / 60.0)
            start_at = now + timedelta(minutes=SLOT_MINUTES * int(slot))
            self.commit.adopt_cycle_start(
                appliance=appliance, slots=spec.cycle_slots,
                expected_kwh=cycle_kwh, start_at=start_at, now=now,
            )
            self._emit(
                now, "cycle_committed",
                appliance=appliance, power_kw=spec.rated_kw,
                detail=(
                    f"start_at={start_at.isoformat()} "
                    f"slot_offset={int(slot)} "
                    f"duration={spec.cycle_slots * SLOT_MINUTES} min"
                ),
            )

        logger.info(
            "%s: adopted plan at=%s solver=%s expected_cost=%.4f",
            self.name, now.isoformat(),
            new_plan.solver_status, new_plan.expected_cost,
        )

    # ------------------------------------------------------------------ #
    def _handle_reschedule_decision(
        self,
        now: datetime,
        proposal: Any,
        ans: str,
    ) -> None:
        if ans in ("no", "reject", "cancel"):
            return
        spec = APPLIANCES[proposal.appliance]
        cycle_kwh = spec.rated_kw * spec.cycle_slots * (SLOT_MINUTES / 60.0)
        if ans == "accept":
            self.commit.adopt_cycle_start(
                appliance=proposal.appliance,
                slots=proposal.cycle_slots,
                expected_kwh=cycle_kwh,
                start_at=proposal.proposed_start_at,
                now=now,
            )
            self.n_reschedule_accept += 1
            self._emit(
                now, "reschedule_accepted",
                appliance=proposal.appliance,
                detail=(
                    f"start_at={proposal.proposed_start_at.isoformat()} "
                    f"shift={proposal.shift_minutes:.0f} min "
                    f"saves=€{proposal.savings_eur:.3f}"
                ),
            )
        else:
            self.commit.adopt_cycle_start(
                appliance=proposal.appliance,
                slots=proposal.cycle_slots,
                expected_kwh=cycle_kwh,
                start_at=proposal.onset_at,
                now=now,
            )
            self.n_reschedule_decline += 1
            self._emit(
                now, "reschedule_declined",
                appliance=proposal.appliance,
                detail=f"running_immediately, lost €{proposal.savings_eur:.3f}",
            )

    # ------------------------------------------------------------------ #
    def _write_replan_jsonl(
        self,
        *,
        now: datetime,
        sample: Sample,
        trigger: Any,
        result: dict,
        new_plan: Any,
        proposal: Any,
        ans: str,
    ) -> None:
        if self._replan_fh is None:
            return
        entry = {
            "now": now.isoformat(),
            "trigger": trigger.as_dict(),
            "hitl": _jsonable(result.get("hitl_decision")),
            "reschedule": _jsonable(proposal),
            "user_answer": ans or None,
            "plan": _jsonable(new_plan),
            "commit": self.commit.snapshot(),
            "realized_price": sample.realized_price,
            "cumulative_cost": self.cumulative_cost,
        }
        self._replan_fh.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------ #
    def get_slot_record(self, now: datetime, price_eur_mwh: float) -> SlotRecord:
        running_tasks = self.commit.running_committed_tasks(now)
        cycle_kw = sum(APPLIANCES[t.appliance].rated_kw for t in running_tasks)
        ev_kw = self.commit.ev_power_setpoint_kw
        heater_kw = self.commit.heater_power_setpoint_kw
        total_kw = ev_kw + heater_kw + cycle_kw
        slot_cost = total_kw * (SLOT_MINUTES / 60.0) * (price_eur_mwh / 1000.0)
        self.cumulative_cost += slot_cost

        rec = SlotRecord(
            timestamp=now,
            ev_kw=ev_kw,
            heater_kw=heater_kw,
            cycle_kw=cycle_kw,
            total_kw=total_kw,
            slot_cost_eur=slot_cost,
            cum_cost_eur=self.cumulative_cost,
            remaining_ev_kwh=self.commit.remaining_ev_kwh,
            active_cycles=[t.appliance for t in running_tasks],
            extra={
                "trigger_kind": self._last_trigger_kind,
                "solver_status": self._last_solver_status,
                "expected_cost_eur": self._last_expected_cost,
                "n_replans_total": self.n_replans,
            },
        )
        self._last_trigger_kind = None
        return rec


__all__ = [
    "Strategy",
    "BaselineStrategy",
    "OptimizerStrategy",
    "SlotRecord",
    "make_event",
]
