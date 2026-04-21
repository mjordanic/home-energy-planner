"""Static configuration for AeroGrid.

All date windows, appliance specs, paths and model selection live here so that
scripts, notebooks and the digital twin all agree on the same split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
UKDALE_DIR = DATA_DIR / "ukdale"
NYISO_DIR = DATA_DIR / "nyiso"
ENTSOE_DIR = DATA_DIR / "entsoe"
CACHE_DIR = DATA_DIR / "cache"

for _d in (UKDALE_DIR, NYISO_DIR, ENTSOE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
HOUSE_POWER_CAP_KW = 10.0

# --------------------------------------------------------------------------- #
# Date windows                                                                #
# --------------------------------------------------------------------------- #
def _utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)

# UK-DALE House 1 — 60 days inside the well-instrumented 2014 window.
UKDALE_TRAIN_START = _utc(2014, 10, 1)
UKDALE_TRAIN_END = _utc(2014, 11, 16)      # 46 train days
UKDALE_TEST_START = _utc(2014, 11, 16)
UKDALE_TEST_END = _utc(2014, 11, 30)       # 14 test days
# 6-hour 16 kHz FLAC slice inside the test window (for DWT demo only).
# A full 3-day slice would be ~8 GB stereo at 16 kHz; 6 h gives enough onsets
# (~2–4 per appliance) to visualise without eating disk.
UKDALE_16KHZ_START = datetime(2014, 11, 16, 18, tzinfo=timezone.utc)
UKDALE_16KHZ_END = datetime(2014, 11, 17, 0, tzinfo=timezone.utc)

# NYISO — 90 days, last 14 reserved for simulation.
NYISO_TRAIN_START = _utc(2024, 10, 1)
NYISO_TRAIN_END = _utc(2024, 12, 16)       # 76 train days
NYISO_TEST_START = _utc(2024, 12, 16)
NYISO_TEST_END = _utc(2024, 12, 30)        # 14 test days
NYISO_ZONE = "N.Y.C."                      # matches NYISO CSV zone name

# ENTSO-E DE-LU (optional alt path).
ENTSOE_TRAIN_START = _utc(2024, 12, 1)
ENTSOE_TRAIN_END = _utc(2024, 12, 21)      # 20 train days
ENTSOE_TEST_START = _utc(2024, 12, 21)
ENTSOE_TEST_END = _utc(2024, 12, 31)       # 10 test days
ENTSOE_AREA = "DE_LU"

# --------------------------------------------------------------------------- #
# Appliances (UK-DALE House 1 channel map + modelled EV)                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ApplianceSpec:
    name: str
    ukdale_channel: int | None   # None => synthetic (EV)
    rated_kw: float
    cycle_slots: int             # number of 15-min slots a cycle occupies
    on_power_threshold_w: float
    max_power_w: float
    bufferable: bool             # True => MILP may shift this load in time

APPLIANCES: dict[str, ApplianceSpec] = {
    "dishwasher": ApplianceSpec(
        name="dishwasher",
        ukdale_channel=6,
        rated_kw=2.5,
        cycle_slots=8,                 # ~2 hours
        on_power_threshold_w=20.0,
        max_power_w=2500.0,
        bufferable=True,
    ),
    "washing_machine": ApplianceSpec(
        name="washing_machine",
        ukdale_channel=5,
        rated_kw=2.4,
        cycle_slots=6,                 # ~1.5 hours
        on_power_threshold_w=20.0,
        max_power_w=2400.0,
        bufferable=True,
    ),
    "ev_charger": ApplianceSpec(
        name="ev_charger",
        ukdale_channel=None,           # not present in UK-DALE; purely modelled
        rated_kw=7.0,
        cycle_slots=0,                 # continuous, not cycle-based
        on_power_threshold_w=50.0,
        max_power_w=7000.0,
        bufferable=True,
    ),
}

# EV daily need (kWh by 07:00 local) — used as MILP comfort constraint.
EV_DAILY_NEED_KWH = 24.0
EV_DEADLINE_SLOT = 28            # 07:00 in 15-min slots (07 * 4)

# --------------------------------------------------------------------------- #
# Model selection                                                             #
# --------------------------------------------------------------------------- #
# Which price oracle implementation to use. Options:
#   "gridfm"   — primary, physics-informed (asayghe1/GridFM, NYISO)
#   "chronos"  — Chronos-2 zero-shot (any market)
#   "naive"    — seasonal-naive fallback
PRICE_ORACLE_IMPL = "gridfm"

# Which behavioral predictor to use.
#   "hybrid"   — logistic(hour, dow) + Gaussian KDE (default)
#   "chronos"  — Chronos-2 over binned onset counts (alt)
#   "mamba"    — Mamba-3 stub (NotImplementedError)
BEHAVIORAL_PREDICTOR_IMPL = "hybrid"

# --------------------------------------------------------------------------- #
# Signature matching / NILM                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NILMConfig:
    wavelet: str = "db4"
    dwt_level: int = 4
    onset_energy_threshold: float = 3.0   # z-score of D1+D2 rolling energy
    min_onset_gap_s: float = 15.0         # debounce
    vi_trajectory_points: int = 64        # resample each V-I cycle to this grid
    signature_match_threshold: float = 0.75

NILM = NILMConfig()

# --------------------------------------------------------------------------- #
# LangGraph / HITL                                                            #
# --------------------------------------------------------------------------- #
GRAPH_CHECKPOINT_DB = CACHE_DIR / "graph_state.sqlite"
RUN_LOG_PATH = CACHE_DIR / "run_log.jsonl"

# Price deviation (relative) above which the monitor triggers a replan.
REPLAN_PRICE_DEVIATION = 0.25

# Risk-appetite multiplier on the ghost-reservation utility term in the MILP.
# Higher = we trust the behavioral predictor more and reserve harder.
RESERVATION_LAMBDA = 0.5

# --------------------------------------------------------------------------- #
# Sampling rates                                                              #
# --------------------------------------------------------------------------- #
UKDALE_MAINS_HZ = 1.0
UKDALE_SUBMETER_PERIOD_S = 6.0
UKDALE_HF_HZ = 16_000.0


def days_between(start: datetime, end: datetime) -> int:
    return (end - start).days
