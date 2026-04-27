"""Shared dataclasses used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Sample:
    """One 1 Hz meter reading emitted by the ScenarioStreamer."""
    t: datetime                   # wall-clock of this sample (UTC)
    p_mains_w: float              # instantaneous household aggregate power
    realized_price: float | None = None   # $/MWh, populated on 15-min slot boundaries


@dataclass(frozen=True)
class ApplianceOnset:
    appliance: str
    timestamp: datetime
    confidence: float             # [0, 1]
    source: Literal["nilm", "disaggregator", "ground_truth"] = "disaggregator"

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
    """Result of a MILP solve (receding-horizon MPC)."""
    slot_start: datetime                    # start of slot 0 (t=now rounded down)
    horizon_slots: int
    ev_power_kw: list[float] = field(default_factory=list)    # length=horizon_slots
    tasks: list[ScheduledTask] = field(default_factory=list)
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
            "tasks": [t.as_dict() for t in self.tasks],
            "expected_cost": self.expected_cost,
            "baseline_cost": self.baseline_cost,
            "savings": self.savings(),
            "solver_status": self.solver_status,
            "committed_until": (
                self.committed_until.isoformat() if self.committed_until else None
            ),
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
class HITLDecision:
    """Output of HITLPolicy.decide(old_plan, new_plan)."""
    action: Literal["auto", "ask"]
    reason: str = ""
    question: str = ""          # populated when action == "ask"

    def as_dict(self) -> dict:
        """Serialise the HITL decision to a JSON-compatible dict."""
        return {"action": self.action, "reason": self.reason, "question": self.question}
