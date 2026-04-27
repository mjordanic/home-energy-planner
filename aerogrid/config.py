"""Static configuration for AeroGrid.

Date windows, appliance specs, paths, and model/horizon/trigger/HITL tuning
all live here so scripts, notebooks, and the digital twin agree on the same
constants.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SCENARIO_DIR = DATA_DIR / "scenario"
NYISO_DIR = DATA_DIR / "nyiso"
ENTSOE_DIR = DATA_DIR / "entsoe"
SMARD_DIR = DATA_DIR / "smard"
CACHE_DIR = DATA_DIR / "cache"

for _d in (SCENARIO_DIR, NYISO_DIR, ENTSOE_DIR, SMARD_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured data directory exists: %s", _d)

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
HOUSE_POWER_CAP_KW = 10.0
SCENARIO_MAINS_HZ = 1.0


# --------------------------------------------------------------------------- #
# Date windows (all UTC)                                                       #
# --------------------------------------------------------------------------- #
def _utc(y: int, m: int, d: int) -> datetime:
    """Construct a timezone-aware UTC :class:`~datetime.datetime` at midnight."""
    return datetime(y, m, d, tzinfo=timezone.utc)


# NYISO — 90 days, last 14 reserved for simulation.
NYISO_TRAIN_START = _utc(2024, 10, 1)
NYISO_TRAIN_END = _utc(2024, 12, 16)       # 76 train days
NYISO_TEST_START = _utc(2024, 12, 16)
NYISO_TEST_END = _utc(2024, 12, 30)        # 14 test days
NYISO_ZONE = "N.Y.C."                      # matches NYISO CSV zone name

# ENTSO-E DE-LU (optional alt path — requires ENTSOE_API_KEY).
ENTSOE_TRAIN_START = _utc(2024, 12, 1)
ENTSOE_TRAIN_END = _utc(2024, 12, 21)      # 20 train days
ENTSOE_TEST_START = _utc(2024, 12, 21)
ENTSOE_TEST_END = _utc(2024, 12, 31)       # 10 test days
ENTSOE_AREA = "DE_LU"

# SMARD DE-LU — free, no key, 15-min native, primary EU price source.
SMARD_TRAIN_START = _utc(2026, 1, 12)
SMARD_TRAIN_END = _utc(2026, 4, 3)  #5       
SMARD_TEST_START = _utc(2026, 4, 3)
SMARD_TEST_END = _utc(2026, 4, 19)        

# Scenario — aligned with SMARD so the agent runs on real DE-LU prices.
SCENARIO_TRAIN_START = SMARD_TRAIN_START
SCENARIO_TRAIN_END = SMARD_TRAIN_END       # 83 training days for behavioral predictor
SCENARIO_TEST_START = SMARD_TEST_START
SCENARIO_TEST_END = SMARD_TEST_END         # 14 test days for streaming simulation


# --------------------------------------------------------------------------- #
# Appliances (simulator-driven; no dataset channel numbers)                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ApplianceSpec:
    name: str
    rated_kw: float
    cycle_slots: int             # number of 15-min slots a cycle occupies
    on_power_threshold_w: float
    max_power_w: float
    bufferable: bool             # True => MILP may shift this load in time
    deadline_hours: tuple[int, ...] = ()
    # UTC hours by which each cycle must have *finished*. Empty = no deadline.
    # For heater/AC: (7, 18) means pre-heat must be done before 07:00 and
    # 18:00 each day. The MILP enforces this as a hard start-before constraint.


APPLIANCES: dict[str, ApplianceSpec] = {
    "dishwasher": ApplianceSpec(
        name="dishwasher",
        rated_kw=2.5,
        cycle_slots=8,                 # ~2 hours
        on_power_threshold_w=20.0,
        max_power_w=2500.0,
        bufferable=True,
    ),
    "washing_machine": ApplianceSpec(
        name="washing_machine",
        rated_kw=2.4,
        cycle_slots=6,                 # ~1.5 hours
        on_power_threshold_w=20.0,
        max_power_w=2400.0,
        bufferable=True,
    ),
    "ev_charger": ApplianceSpec(
        name="ev_charger",
        rated_kw=7.0,
        cycle_slots=0,                 # continuous, not cycle-based
        on_power_threshold_w=50.0,
        max_power_w=7000.0,
        bufferable=True,
    ),
    # Heater/AC: 1-hour bufferable cycle with morning and evening comfort deadlines.
    # The MILP must schedule the cycle to FINISH by each deadline_hour so the
    # space is warm when the user needs it. Pre-conditioning (starting early) is
    # always allowed; starting after the deadline is forbidden by a hard constraint.
    "heater": ApplianceSpec(
        name="heater",
        rated_kw=2.0,
        cycle_slots=4,                 # ~1 hour
        on_power_threshold_w=30.0,
        max_power_w=2000.0,
        bufferable=True,
        deadline_hours=(7, 18),        # pre-heat must finish by 07:00 and 18:00 UTC
    ),
    # Fridge: always-on compressor cycling — not schedulable by the MILP.
    # Adds realistic MinMax background noise to the aggregate trace.
    "fridge": ApplianceSpec(
        name="fridge",
        rated_kw=0.15,
        cycle_slots=0,                 # continuous compressor, not shift-able
        on_power_threshold_w=10.0,
        max_power_w=150.0,
        bufferable=False,
    ),
}

# EV daily need (kWh by 07:00 local) — used as MILP comfort constraint.
EV_DAILY_NEED_KWH = 24.0
EV_DEADLINE_HOUR = 7             # 07:00 local clock


# --------------------------------------------------------------------------- #
# MPC horizons                                                                #
# --------------------------------------------------------------------------- #
# Short horizon drives the receding-horizon MILP; long horizon is used only
# for deadline-feasibility sanity checks.
SHORT_HORIZON_SLOTS = 8           # 2 h at 15-min resolution
LONG_HORIZON_SLOTS = SLOTS_PER_DAY   # 24 h, legacy / sanity check


# --------------------------------------------------------------------------- #
# Trigger thresholds (when does the outer loop fire?)                         #
# --------------------------------------------------------------------------- #
TRIGGER_COOLDOWN_S = 30.0         # min seconds between successive replans
TRIGGER_RESYNC_MINUTES = 15       # periodic replan regardless of events
# Price deviation (relative) above which to replan.
REPLAN_PRICE_DEVIATION = 0.25
# Approaching-commit: replan within N minutes of a tentative task's planned start.
TRIGGER_COMMIT_LOOKAHEAD_MIN = 5
# Deadline-guard: replan if required_rate / current_rate > this ratio.
TRIGGER_DEADLINE_SAFETY = 1.2


# --------------------------------------------------------------------------- #
# HITL policy tolerances (AUTO vs ASK)                                        #
# --------------------------------------------------------------------------- #
HITL_EV_TOLERANCE_KW = 1.5        # EV power change within this → AUTO
HITL_SHIFT_TOLERANCE_MIN = 15     # tentative-task shift within this → AUTO
HITL_ASK_SHIFT_MIN = 30           # shift beyond this → ASK
HITL_COST_BUMP_USD = 0.50         # cost-increasing deadline-guard → ASK
SLEEP_WINDOW_START = time(22, 0)  # any start crossing into 22–06 → ASK
SLEEP_WINDOW_END = time(6, 0)


# --------------------------------------------------------------------------- #
# Model selection                                                             #
# --------------------------------------------------------------------------- #
PRICE_SOURCE = "smard"            # "smard" | "nyiso" | "entsoe"
PRICE_ORACLE_IMPL = "naive"       # default flipped to 'naive' for zero-dep runs
BEHAVIORAL_PREDICTOR_IMPL = "hybrid"


# --------------------------------------------------------------------------- #
# LangGraph / HITL / logging                                                  #
# --------------------------------------------------------------------------- #
GRAPH_CHECKPOINT_DB = CACHE_DIR / "graph_state.sqlite"
RUN_LOG_PATH = CACHE_DIR / "run_log.jsonl"

# Risk-appetite multiplier on the ghost-reservation utility term in the MILP.
RESERVATION_LAMBDA = 0.5


def days_between(start: datetime, end: datetime) -> int:
    """Return the number of whole days between two UTC datetimes."""
    result = (end - start).days
    logger.debug("days_between(%s, %s) = %d", start.date(), end.date(), result)
    return result
