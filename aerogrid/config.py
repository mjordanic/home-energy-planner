"""Static configuration for AeroGrid.

Date windows, appliance specs, paths, and model/horizon/trigger/HITL tuning
all live here so scripts, notebooks, and the digital twin agree on the same
constants.

After the *intent-driven* refactor of April 2026, the optimizer treats two
loads as continuous variable-power and the rest as event-driven:

* **EV charger** — continuous power, hard energy deadline at 07:00 UTC,
  hard availability gate (no charging before :data:`EV_AVAILABLE_FROM_HOUR`).
* **Heater** — continuous power, multiple energy windows specified in
  :data:`HEATER_DEADLINES`. Each window has its own kWh target.
* **Dishwasher / washing machine** — *event-driven*. The user starts them;
  the agent only proposes a *small reschedule* (≤ :data:`HITL_RESCHEDULE_WINDOW_HOURS`
  hours forward) if it would save more than
  :data:`HITL_RESCHEDULE_MIN_SAVINGS_EUR`. Whether the simulated user
  accepts is per-appliance and lives in :data:`HITL_AUTO_RESPONSES`.
* **Base Load** — deterministic always-on inflexible demand (fridge + lights
  + standby + cooking) with an evening peak. Modelled as a per-hour kW
  profile (see :data:`BASE_LOAD_PROFILE_KW`) that both strategies pay for
  every slot. Not an ``ApplianceSpec`` — no decision variables, not
  shiftable (ADR 0003). Replaces the previously-unwired ``fridge`` entry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
NYISO_DIR = DATA_DIR / "nyiso"
ENTSOE_DIR = DATA_DIR / "entsoe"
SMARD_DIR = DATA_DIR / "smard"
CACHE_DIR = DATA_DIR / "cache"

for _d in (NYISO_DIR, ENTSOE_DIR, SMARD_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured data directory exists: %s", _d)

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
HOUSE_POWER_CAP_KW = 10.0


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
SMARD_TRAIN_END = _utc(2026, 4, 3)
SMARD_TEST_START = _utc(2026, 4, 3)
SMARD_TEST_END = _utc(2026, 4, 19)

# Simulation window — aligned with SMARD so the agent runs on real DE-LU prices.
SIM_TEST_START = SMARD_TEST_START
SIM_TEST_END = SMARD_TEST_END               # streaming-simulation days


# --------------------------------------------------------------------------- #
# Appliances (simulator-driven; no dataset channel numbers)                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ApplianceSpec:
    """Static description of one appliance.

    ``cycle_slots`` is non-zero only for *event-driven cycle appliances*
    (dishwasher, washing machine) — appliances the user starts and that
    run a fixed-shape cycle of known length and rated power. The continuous
    loads (EV, heater) use ``cycle_slots = 0``; their behaviour is modelled
    by their own dedicated MILP variables. Always-on background demand is
    modelled by the deterministic :data:`BASE_LOAD_PROFILE_KW`, not by an
    ``ApplianceSpec`` (ADR 0003).
    """
    name: str
    rated_kw: float
    cycle_slots: int             # 0 for continuous / always-on loads
    on_power_threshold_w: float
    max_power_w: float
    bufferable: bool             # True => MILP / HITL may shift this load


@dataclass(frozen=True)
class HeaterEnergyDeadline:
    """One energy delivery deadline for the heater.

    The heater must have delivered ``kwh_required`` kWh of heating energy
    by hour ``hour:00`` UTC. The window the energy is integrated over
    runs from the *previous* deadline (circular over 24 h) up to and
    including ``hour:00``. Example: with ``HEATER_DEADLINES`` = (7, 18),
    the 07:00 deadline integrates over 18:00 → 07:00 (overnight, 13 h)
    and the 18:00 deadline integrates over 07:00 → 18:00 (daytime, 11 h).
    """
    hour: int                    # UTC hour by which the energy must be delivered
    kwh_required: float          # kWh required in the window ending at this hour


APPLIANCES: dict[str, ApplianceSpec] = {
    "dishwasher": ApplianceSpec(
        name="dishwasher",
        rated_kw=2.5,
        cycle_slots=8,                 # ~2 hours fixed-shape cycle
        on_power_threshold_w=20.0,
        max_power_w=2500.0,
        bufferable=True,
    ),
    "washing_machine": ApplianceSpec(
        name="washing_machine",
        rated_kw=2.4,
        cycle_slots=6,                 # ~1.5 hours fixed-shape cycle
        on_power_threshold_w=20.0,
        max_power_w=2400.0,
        bufferable=True,
    ),
    "ev_charger": ApplianceSpec(
        name="ev_charger",
        rated_kw=7.0,
        cycle_slots=0,                 # continuous, MILP-controlled
        on_power_threshold_w=50.0,
        max_power_w=7000.0,
        bufferable=True,
    ),
    # Heater: continuous variable power, controlled by the MILP, with one
    # or more energy-delivery deadlines per day (see HEATER_DEADLINES).
    # The MILP can run the heater at any non-negative power up to rated_kw
    # at every slot; what matters is the integral over each deadline window.
    "heater": ApplianceSpec(
        name="heater",
        rated_kw=2.0,
        cycle_slots=0,                 # continuous, no fixed-shape cycle
        on_power_threshold_w=30.0,
        max_power_w=2000.0,
        bufferable=True,
    ),
    # NOTE: fridge removed — replaced by the deterministic Base Load profile
    # (BASE_LOAD_PROFILE_KW + get_base_load_kw). ADR 0003.
}


# --------------------------------------------------------------------------- #
# EV charging                                                                  #
# --------------------------------------------------------------------------- #
EV_DAILY_NEED_KWH = 24.0
EV_DEADLINE_HOUR = 7              # 07:00 UTC — kWh must be delivered by this time
# Earliest UTC hour at which the EV is plugged in and charging is allowed.
# Until this hour each day, p_ev[t] is forced to zero by the MILP and by the
# fallback policy. The scenario simulator schedules the EV plug-in event at
# exactly this hour to keep the streaming twin and the optimizer in sync.
EV_AVAILABLE_FROM_HOUR = 20


# --------------------------------------------------------------------------- #
# Heater                                                                       #
# --------------------------------------------------------------------------- #
# Energy comfort deadlines for the heater. Each entry's ``kwh_required`` must
# be delivered in the window ending at ``hour:00`` UTC; the window starts at
# the previous deadline (circular over 24 h). With the defaults below:
#   * 07:00 deadline → 4 kWh integrated over 18:00 → 07:00 (overnight)
#   * 18:00 deadline → 2 kWh integrated over 07:00 → 18:00 (daytime)
HEATER_DEADLINES: tuple[HeaterEnergyDeadline, ...] = (
    HeaterEnergyDeadline(hour=7, kwh_required=4.0),
    HeaterEnergyDeadline(hour=18, kwh_required=2.0),
)


# --------------------------------------------------------------------------- #
# MPC horizon                                                                 #
# --------------------------------------------------------------------------- #
# Length of the receding-horizon optimisation in hours. The MILP is re-solved
# every TRIGGER_RESYNC_MINUTES (or on event triggers), and only the first
# slot's setpoints are committed before the next replan, so this is a true
# rolling horizon. 24 h is the default because the heater energy windows can
# span up to 13 h and we want at least one full window in view at all times.
HORIZON_HOURS = 24
SHORT_HORIZON_SLOTS = HORIZON_HOURS * (60 // SLOT_MINUTES)   # 24 h × 4 slots/h = 96
# Forecast horizon must be at least the optimisation horizon. The seasonal-naive
# oracle pads to whatever length is requested; the Chronos / GridFM oracles
# fall back to seasonal if they can't produce 96 slots.
FORECAST_HORIZON_SLOTS = SHORT_HORIZON_SLOTS


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
# HITL policy tolerances (AUTO vs ASK) and reschedule offers                  #
# --------------------------------------------------------------------------- #
HITL_EV_TOLERANCE_KW = 1.5        # EV power change within this → AUTO
HITL_SHIFT_TOLERANCE_MIN = 15     # tentative-task shift within this → AUTO
HITL_ASK_SHIFT_MIN = 30           # shift beyond this → ASK
HITL_COST_BUMP_USD = 0.50         # cost-increasing deadline-guard → ASK
SLEEP_WINDOW_START = time(22, 0)  # any start crossing into 22–06 → ASK
SLEEP_WINDOW_END = time(6, 0)

# Reschedule proposals for event-driven cycle appliances (dishwasher,
# washing machine). When the user starts the appliance, the agent searches
# for a cheaper start time within the next HITL_RESCHEDULE_WINDOW_HOURS hours
# and offers the shift to the user iff the savings exceed
# HITL_RESCHEDULE_MIN_SAVINGS_EUR. In simulation, HITL_AUTO_RESPONSES decides
# the simulated user's reply per appliance ("accept" runs at proposed time,
# "decline" runs immediately at the original onset time).
HITL_RESCHEDULE_WINDOW_HOURS = 2.0
HITL_RESCHEDULE_MIN_SAVINGS_EUR = 0.10
HITL_AUTO_RESPONSES: dict[str, str] = {
    "dishwasher": "accept",
    "washing_machine": "decline",
}


# --------------------------------------------------------------------------- #
# Model selection                                                             #
# --------------------------------------------------------------------------- #
PRICE_SOURCE = "smard"            # "smard" | "nyiso" | "entsoe"
PRICE_ORACLE_IMPL = "naive"       # default flipped to 'naive' for zero-dep runs


# --------------------------------------------------------------------------- #
# LangGraph / HITL / logging                                                  #
# --------------------------------------------------------------------------- #
GRAPH_CHECKPOINT_DB = CACHE_DIR / "graph_state.sqlite"
RUN_LOG_PATH = CACHE_DIR / "run_log.jsonl"
# Per-15-min slot comparison between baseline and optimizer (strategies.py).
SLOT_LOG_PATH = CACHE_DIR / "slot_log.parquet"
# Per-event decision log at 1-second resolution for both strategies.
EVENT_LOG_PATH = CACHE_DIR / "event_log.parquet"


# --------------------------------------------------------------------------- #
# Simulation inputs (digital twin only)                                       #
# --------------------------------------------------------------------------- #
# APPLIANCE_ONSETS:
#   Sequence of (appliance_name, timestamp_utc) pairs.  Each pair schedules an
#   appliance onset on the streamer; both strategies see the same gated list.
#
# INJECTED_PRICE_SPIKES:
#   Sequence of (timestamp_utc, delta_eur_per_mwh) pairs. At each timestamp's
#   15-minute slot, the price server adds the delta on top of realized price.
#
# Leave either list empty to disable that input.
#
# Onset times are UTC.  The simulation window is Apr 3–18 2026, DE-LU (CEST =
# UTC+2), so e.g. German 19:30 → UTC 17:30.  Dishwasher onsets cluster after
# meals; washing machine onsets spread across mornings and afternoons.
def _onset(day: int, app: str, h: int, m: int) -> tuple[str, datetime]:
    return (app, datetime(2026, 4, day, h, m, tzinfo=timezone.utc))


_DW = "dishwasher"
_WM = "washing_machine"

APPLIANCE_ONSETS: tuple[tuple[str, datetime], ...] = (
    # --- Apr 3 (Mon) ---
    _onset(3, _DW,  9, 15),   # after breakfast
    _onset(3, _DW, 17, 30),   # after dinner
    _onset(3, _WM,  7, 30),   # morning load
    _onset(3, _WM, 13, 45),   # midday load
    # --- Apr 4 (Tue) ---
    _onset(4, _DW, 10, 45),   # after late breakfast
    _onset(4, _DW, 18, 00),   # after dinner
    _onset(4, _DW, 20, 15),   # late-evening top-up
    _onset(4, _WM,  8, 00),
    _onset(4, _WM, 14, 15),
    # --- Apr 5 (Wed) ---
    _onset(5, _DW, 11, 30),
    _onset(5, _DW, 19, 15),
    _onset(5, _WM,  6, 45),   # early load
    _onset(5, _WM, 15, 30),
    _onset(5, _WM, 17, 00),
    # --- Apr 6 (Thu) ---
    _onset(6, _DW, 10, 00),
    _onset(6, _DW, 17, 45),
    _onset(6, _DW, 20, 30),
    _onset(6, _WM,  7, 15),
    _onset(6, _WM, 14, 45),
    # --- Apr 7 (Fri) ---
    _onset(7, _DW, 12, 00),
    _onset(7, _DW, 18, 45),
    _onset(7, _WM,  8, 30),
    _onset(7, _WM, 16, 00),
    # --- Apr 8 (Sat) weekend: more loads ---
    _onset(8, _DW,  9, 30),
    _onset(8, _DW, 13, 00),
    _onset(8, _DW, 18, 30),
    _onset(8, _WM,  8, 00),
    _onset(8, _WM, 11, 45),
    _onset(8, _WM, 15, 15),
    # --- Apr 9 (Sun) weekend ---
    _onset(9, _DW, 11, 15),
    _onset(9, _DW, 19, 00),
    _onset(9, _WM,  9, 00),
    _onset(9, _WM, 13, 30),
    _onset(9, _WM, 16, 45),
    # --- Apr 10 (Mon) ---
    _onset(10, _DW, 10, 30),
    _onset(10, _DW, 18, 15),
    _onset(10, _WM,  7, 45),
    _onset(10, _WM, 14, 00),
    # --- Apr 11 (Tue) ---
    _onset(11, _DW, 12, 15),
    _onset(11, _DW, 17, 15),
    _onset(11, _DW, 20, 45),
    _onset(11, _WM,  8, 15),
    _onset(11, _WM, 15, 00),
    # --- Apr 12 (Wed) ---
    _onset(12, _DW,  9, 45),
    _onset(12, _DW, 19, 30),
    _onset(12, _WM,  6, 30),
    _onset(12, _WM, 11, 30),
    _onset(12, _WM, 16, 30),
    # --- Apr 13 (Thu) ---
    _onset(13, _DW, 11, 00),
    _onset(13, _DW, 18, 00),
    _onset(13, _WM,  7, 30),
    _onset(13, _WM, 13, 15),
    # --- Apr 14 (Fri) ---
    _onset(14, _DW, 10, 15),
    _onset(14, _DW, 17, 00),
    _onset(14, _DW, 20, 00),
    _onset(14, _WM,  8, 45),
    _onset(14, _WM, 14, 30),
    # --- Apr 15 (Sat) weekend ---
    _onset(15, _DW,  9, 00),
    _onset(15, _DW, 13, 45),
    _onset(15, _DW, 19, 45),
    _onset(15, _WM,  7, 00),
    _onset(15, _WM, 11, 15),
    _onset(15, _WM, 16, 15),
    # --- Apr 16 (Sun) weekend ---
    _onset(16, _DW, 12, 30),
    _onset(16, _DW, 18, 45),
    _onset(16, _WM,  9, 15),
    _onset(16, _WM, 14, 00),
    _onset(16, _WM, 17, 30),
    # --- Apr 17 (Mon) ---
    _onset(17, _DW, 10, 45),
    _onset(17, _DW, 18, 30),
    _onset(17, _WM,  7, 00),
    _onset(17, _WM, 13, 00),
    # --- Apr 18 (Tue) ---
    _onset(18, _DW, 11, 45),
    _onset(18, _DW, 17, 15),
    _onset(18, _WM,  8, 00),
    _onset(18, _WM, 15, 45),
)
INJECTED_PRICE_SPIKES: tuple[tuple[datetime, float], ...] = ()


# --------------------------------------------------------------------------- #
# Base Load profile (ADR 0003)                                                #
# --------------------------------------------------------------------------- #
# Deterministic per-hour always-on inflexible demand: fridge compressor +
# LED lighting + standby electronics + cooking.  One value per hour-of-day
# (UTC).  Evening peak 17–22h; overnight ~0.2 kW; morning spike 07–09h.
# Total ≈ 9.5 kWh/day.
#
# Hour:  0     1     2     3     4     5     6     7     8     9    10    11
#       12    13    14    15    16    17    18    19    20    21    22    23
# Total ≈ 9.95 kWh/day.
BASE_LOAD_PROFILE_KW: tuple[float, ...] = (
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,  # 00–06: overnight
    0.50, 0.50, 0.40, 0.40, 0.40,               # 07–11: morning
    0.40, 0.40, 0.40, 0.40, 0.40,               # 12–16: daytime
    0.75, 0.75, 0.75, 0.75, 0.75,               # 17–21: evening peak (~0.75 kW)
    0.40, 0.20,                                  # 22–23: wind-down
)
assert len(BASE_LOAD_PROFILE_KW) == 24, "BASE_LOAD_PROFILE_KW must have exactly 24 entries"


def get_base_load_kw(now: datetime, n_slots: int) -> list[float]:
    """Return a slot-aligned Base Load array of length ``n_slots``.

    Slot 0 corresponds to the 15-minute slot that contains ``now``, floored
    to the nearest slot boundary.  Each subsequent slot advances by
    ``SLOT_MINUTES`` minutes.  The hour of each slot's start time determines
    which entry of :data:`BASE_LOAD_PROFILE_KW` applies.

    Args:
        now: Current simulation time (UTC).  Used to anchor slot 0.
        n_slots: Number of 15-min slots to produce.

    Returns:
        A list of ``n_slots`` floats, each the Base Load kW for that slot.
    """
    # Floor now to the start of the current slot.
    slot0 = now.replace(
        minute=(now.minute // SLOT_MINUTES) * SLOT_MINUTES,
        second=0,
        microsecond=0,
    )
    result: list[float] = []
    for t in range(n_slots):
        slot_t = slot0 + timedelta(minutes=SLOT_MINUTES * t)
        result.append(BASE_LOAD_PROFILE_KW[slot_t.hour])
    return result


def days_between(start: datetime, end: datetime) -> int:
    """Return the number of whole days between two UTC datetimes."""
    result = (end - start).days
    logger.debug("days_between(%s, %s) = %d", start.date(), end.date(), result)
    return result
