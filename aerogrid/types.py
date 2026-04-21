"""Shared dataclasses / pydantic models used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class ApplianceOnset:
    appliance: str
    timestamp: datetime
    confidence: float             # [0, 1] cosine similarity against signature
    source: Literal["nilm", "ground_truth"] = "nilm"

    def as_dict(self) -> dict:
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

    def as_dict(self) -> dict:
        return {
            "appliance": self.appliance,
            "start_slot": int(self.start_slot),
            "slots": int(self.slots),
            "expected_kwh": float(self.expected_kwh),
        }


@dataclass
class Schedule:
    """Result of a MILP solve."""
    slot_start: datetime                    # start of slot 0 (t=now rounded down)
    horizon_slots: int
    ev_power_kw: list[float] = field(default_factory=list)    # length=horizon_slots
    tasks: list[ScheduledTask] = field(default_factory=list)
    expected_cost: float = 0.0
    baseline_cost: float = 0.0              # cost of naive "charge ASAP" schedule
    solver_status: str = "unknown"

    def savings(self) -> float:
        if self.baseline_cost == 0:
            return 0.0
        return (self.baseline_cost - self.expected_cost) / self.baseline_cost

    def as_dict(self) -> dict:
        return {
            "slot_start": self.slot_start.isoformat(),
            "horizon_slots": self.horizon_slots,
            "ev_power_kw": [float(p) for p in self.ev_power_kw],
            "tasks": [t.as_dict() for t in self.tasks],
            "expected_cost": self.expected_cost,
            "baseline_cost": self.baseline_cost,
            "savings": self.savings(),
            "solver_status": self.solver_status,
        }


@dataclass(frozen=True)
class PriceForecast:
    slot_start: datetime
    median: list[float]           # length=horizon_slots, $/MWh
    q10: list[float] | None = None
    q90: list[float] | None = None
    source: str = "unknown"       # "gridfm", "chronos", "naive"
