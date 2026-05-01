"""Shared dataclasses used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Sample:
    """One simulation tick emitted by the streamer."""
    t: datetime                   # wall-clock of this sample (UTC)
    realized_price: float | None = None   # $/MWh, populated on 15-min slot boundaries


@dataclass(frozen=True)
class ApplianceOnset:
    appliance: str
    timestamp: datetime
    confidence: float             # [0, 1]
    source: Literal["injected"] = "injected"

    def as_dict(self) -> dict:
        """Serialise the onset to a JSON-compatible dict."""
        return {
            "appliance": self.appliance,
            "timestamp": self.timestamp.isoformat(),
            "confidence": float(self.confidence),
            "source": self.source,
        }


@dataclass
class ScheduledTask:
    """A single fixed-shape cycle task at a chosen start slot.

    Used for *event-driven* appliances (dishwasher, washing machine) once the
    user's onset has been confirmed (possibly with a HITL-approved shift).
    Continuous loads (EV, heater) are not represented as ``ScheduledTask`` —
    their per-slot powers live directly on :class:`Schedule`.
    """
    appliance: str
    start_slot: int
    slots: int                    # number of 15-min slots the task occupies
    expected_kwh: float
    committed: bool = False       # True => cannot be rescheduled without user override

    def as_dict(self) -> dict:
        """Serialise the task to a JSON-compatible dict."""
        return {
            "appliance": self.appliance,
            "start_slot": int(self.start_slot),
            "slots": int(self.slots),
            "expected_kwh": float(self.expected_kwh),
            "committed": bool(self.committed),
        }


@dataclass
class Schedule:
    """Result of a (mostly) LP solve over the receding horizon.

    The optimiser produces continuous per-slot powers for the EV charger and
    the heater plus, when the caller passes ``pending_cycles``, a chosen
    start slot for each pending cycle. Already-committed cycles (pinned by
    the commit tracker before the solve) appear in ``tasks`` with
    ``committed=True`` and consume cap headroom but are not re-decided.
    """
    slot_start: datetime                    # start of slot 0 (t=now rounded down)
    horizon_slots: int
    ev_power_kw: list[float] = field(default_factory=list)        # length=horizon_slots
    heater_power_kw: list[float] = field(default_factory=list)    # length=horizon_slots
    heater_window_kwh: dict[int, float] = field(default_factory=dict)
    tasks: list[ScheduledTask] = field(default_factory=list)
    # Map of appliance name → chosen start slot for pending cycles included
    # in the joint solve. Empty when the caller passed no ``pending_cycles``.
    cycle_starts: dict[str, int] = field(default_factory=dict)
    expected_cost: float = 0.0
    baseline_cost: float = 0.0
    solver_status: str = "unknown"
    committed_until: datetime | None = None   # first N slots committed as of this plan

    def savings(self) -> float:
        """Return fractional cost savings vs. the naive baseline: ``(baseline − expected) / baseline``."""
        if self.baseline_cost == 0:
            return 0.0
        return (self.baseline_cost - self.expected_cost) / self.baseline_cost

    def as_dict(self) -> dict:
        """Serialise the schedule (including computed savings) to a JSON-compatible dict."""
        return {
            "slot_start": self.slot_start.isoformat(),
            "horizon_slots": self.horizon_slots,
            "ev_power_kw": [float(p) for p in self.ev_power_kw],
            "heater_power_kw": [float(p) for p in self.heater_power_kw],
            "heater_window_kwh": {int(k): float(v) for k, v in self.heater_window_kwh.items()},
            "tasks": [t.as_dict() for t in self.tasks],
            "cycle_starts": {str(k): int(v) for k, v in self.cycle_starts.items()},
            "expected_cost": self.expected_cost,
            "baseline_cost": self.baseline_cost,
            "savings": self.savings(),
            "solver_status": self.solver_status,
            "committed_until": (
                self.committed_until.isoformat() if self.committed_until else None
            ),
        }


@dataclass(frozen=True)
class PendingCycle:
    """A user-triggered cycle that the optimiser must place inside the horizon.

    Passed into :func:`aerogrid.optimizer.solve_receding_horizon` to convert
    the LP into a small mixed-integer program: for each pending cycle the
    optimiser introduces binary start-indicator variables ``s_a[t]`` for
    ``t ∈ [earliest_start_slot, latest_start_slot]`` and the constraint
    ``Σ_t s_a[t] = 1`` (the cycle must run exactly once inside the
    user-allowed window). The MIP then jointly picks the cycle's start slot
    *and* the EV / heater power profile, naturally accounting for the house
    power cap and any deadline pressure.

    Attributes:
        appliance: Cycle appliance name (e.g. ``"dishwasher"``).
        cycle_slots: Number of 15-min slots the cycle occupies.
        rated_kw: Rated power of the cycle (kW).
        earliest_start_slot: Lowest allowed start slot. Defaults to ``0``
            (run now). Used in HITL bookkeeping (``earliest = onset_slot``).
        latest_start_slot: Highest allowed start slot. Typically
            ``window_slots = HITL_RESCHEDULE_WINDOW_HOURS · 60 / SLOT_MINUTES``,
            clipped to ``horizon_slots − cycle_slots`` so the cycle fits in
            the horizon.
    """
    appliance: str
    cycle_slots: int
    rated_kw: float
    earliest_start_slot: int = 0
    latest_start_slot: int = 0

    def as_dict(self) -> dict:
        return {
            "appliance": self.appliance,
            "cycle_slots": int(self.cycle_slots),
            "rated_kw": float(self.rated_kw),
            "earliest_start_slot": int(self.earliest_start_slot),
            "latest_start_slot": int(self.latest_start_slot),
        }


@dataclass(frozen=True)
class PriceForecast:
    slot_start: datetime
    median: list[float]           # length=horizon_slots, $/MWh
    q10: list[float] | None = None
    q90: list[float] | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class ReplanTrigger:
    """Why the TriggerManager decided to fire a replan."""
    kind: Literal[
        "new_onset",
        "unknown_onset",
        "price_surprise",
        "commit_boundary",
        "deadline_slip",
        "periodic",
        "manual",
    ]
    detail: str = ""
    at: datetime | None = None

    def as_dict(self) -> dict:
        """Serialise the trigger to a JSON-compatible dict."""
        return {
            "kind": self.kind,
            "detail": self.detail,
            "at": self.at.isoformat() if self.at else None,
        }


@dataclass(frozen=True)
class RescheduleProposal:
    """A proposed shift of an event-driven cycle appliance.

    Produced by the ``propose_reschedule`` graph node when the user starts an
    appliance and the optimiser found a cheaper start within the configured
    look-ahead window. The HITL gate either auto-resolves it (per-appliance
    via :data:`aerogrid.config.HITL_AUTO_RESPONSES`) or asks the real user.

    A proposal of ``shift_minutes == 0`` is *implicit*: it means "run now,
    no cheaper alternative was found"; such proposals are filtered out
    before the HITL gate and never reach the user.
    """
    appliance: str
    onset_at: datetime                     # when the user started the appliance
    proposed_start_at: datetime            # what the agent proposes instead
    cycle_slots: int                       # cycle length, used to bookkeep commit
    rated_kw: float                        # rated power, used for cap accounting
    cost_now_eur: float                    # cost if the cycle runs at onset_at
    cost_proposed_eur: float               # cost if the cycle runs at proposed_start_at

    @property
    def shift_minutes(self) -> float:
        """Minutes the proposal would push the cycle start forward."""
        return (self.proposed_start_at - self.onset_at).total_seconds() / 60.0

    @property
    def savings_eur(self) -> float:
        """Euro savings if the user accepts the shift (always ≥ 0 by construction)."""
        return max(0.0, self.cost_now_eur - self.cost_proposed_eur)

    def as_dict(self) -> dict:
        """Serialise the proposal to a JSON-compatible dict."""
        return {
            "appliance": self.appliance,
            "onset_at": self.onset_at.isoformat(),
            "proposed_start_at": self.proposed_start_at.isoformat(),
            "shift_minutes": float(self.shift_minutes),
            "cycle_slots": int(self.cycle_slots),
            "rated_kw": float(self.rated_kw),
            "cost_now_eur": float(self.cost_now_eur),
            "cost_proposed_eur": float(self.cost_proposed_eur),
            "savings_eur": float(self.savings_eur),
        }


@dataclass(frozen=True)
class HITLDecision:
    """Output of HITLPolicy.decide(old_plan, new_plan)."""
    action: Literal["auto", "ask"]
    reason: str = ""
    question: str = ""          # populated when action == "ask"

    def as_dict(self) -> dict:
        """Serialise the HITL decision to a JSON-compatible dict."""
        return {"action": self.action, "reason": self.reason, "question": self.question}
