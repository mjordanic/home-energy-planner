"""Programmatic NILM scenario simulator.

Builds a 1 Hz household aggregate + per-appliance ground truth from a
declarative schedule, so the agent has:

- a realistic baseline trace to disaggregate,
- a deterministic seed so interventions (e.g. "delay washing machine 2 h")
  produce byte-identical traces before the intervention point, and
- a parquet layout matching the one ``behavioral_predictor.load_onsets``
  already expects, so downstream components work without schema changes.

A ``ScenarioSpec`` is a declarative description: the start/end window, a
baseline draw, and per-appliance schedules with parametric models from
:mod:`aerogrid.sim.appliance_models`.

The five appliance model families are inspired by Klemen Jakšič's SmartSim
paper (on_off / decay / decay_grow / min_max / random_range) and reimplemented
from their parametric descriptions — no code ported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aerogrid.config import (
    APPLIANCES,
    EV_DAILY_NEED_KWH,
    SCENARIO_DIR,
    SCENARIO_TEST_START,
    SCENARIO_TRAIN_START,
    SLOT_MINUTES,
)
from aerogrid.sim.appliance_models import (
    ApplianceModel,
    DecayGrowModel,
    DecayModel,
    MinMaxModel,
    OnOffModel,
)
from aerogrid.types import Schedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplianceSchedule:
    """Schedule + power model for one appliance across the scenario window."""
    name: str                                   # must match an APPLIANCES key
    model: ApplianceModel
    cycle_starts: tuple[datetime, ...]          # explicit cycle start times
    cycle_duration_s: int                       # duration of one cycle at 1 Hz


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative scenario description. Pure data; regeneration is deterministic."""
    start: datetime
    end: datetime
    seed: int = 42
    base_load_w: float = 150.0                  # constant baseload (fridge, router)
    appliances: tuple[ApplianceSchedule, ...] = ()

    @property
    def total_seconds(self) -> int:
        """Total scenario duration in seconds (equals the number of 1 Hz samples)."""
        return int((self.end - self.start).total_seconds())


@dataclass
class ScenarioOutput:
    """Fully-expanded scenario ready for parquet export and downstream training.

    Attributes:
        spec: The declarative spec that produced this output.
        mains: 1 Hz aggregate trace — columns ``timestamp``, ``power_w``, ``split``.
        per_appliance: Per-appliance 1 Hz traces with the same column schema.
        onsets: Table of cycle-start events — columns ``timestamp``,
            ``appliance``, ``split``.
    """

    spec: ScenarioSpec
    mains: pd.DataFrame                         # timestamp, power_w, split
    per_appliance: dict[str, pd.DataFrame]      # per-name: timestamp, power_w, split
    onsets: pd.DataFrame                        # timestamp, appliance, split


# --------------------------------------------------------------------------- #
# Natural-time schedule helpers                                               #
# --------------------------------------------------------------------------- #
def natural_cycle_times(
    start: datetime,
    end: datetime,
    peak_hour: float,
    hour_spread: float,
    cycles_per_week: float,
    rng: np.random.Generator,
) -> tuple[datetime, ...]:
    """Draw cycle start times over [start, end) with hour-of-day bias.

    Poisson with rate ``cycles_per_week / 7`` per day; each cycle's hour drawn
    from N(peak_hour, hour_spread) clipped to [0, 24).
    """
    n_days = max(0, (end - start).days)
    per_day = cycles_per_week / 7.0
    out: list[datetime] = []
    for d in range(n_days):
        day = start + timedelta(days=d)
        n = int(rng.poisson(per_day))
        for _ in range(n):
            h = float(rng.normal(peak_hour, max(hour_spread, 0.1)))
            h = max(0.0, min(23.999, h))
            hh = int(h)
            mm = int(round((h - hh) * 60))
            if mm >= 60:
                hh += 1
                mm = 0
            hh = min(hh, 23)
            out.append(day.replace(hour=hh, minute=mm, second=0, microsecond=0))
    return tuple(sorted(out))


def daily_ev_cycle_times(
    start: datetime,
    end: datetime,
    plug_in_hour: int = 19,
) -> tuple[datetime, ...]:
    """One EV plug-in per day at ``plug_in_hour`` local (UTC)."""
    n_days = max(0, (end - start).days)
    out: list[datetime] = []
    for d in range(n_days):
        day = start + timedelta(days=d)
        out.append(day.replace(hour=plug_in_hour, minute=0, second=0, microsecond=0))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Factory: default "demo" scenario spec                                       #
# --------------------------------------------------------------------------- #
def default_scenario_spec(
    start: datetime,
    end: datetime,
    *,
    seed: int = 42,
) -> ScenarioSpec:
    """A demo-ready spec with three appliances running at habitually-expensive hours.

    The natural times are chosen to NOT line up with cheap overnight price
    slots, so the MILP has something meaningful to shift — otherwise the
    before/after intervention plots would be uninteresting.
    """
    rng = np.random.default_rng(seed)

    dish_spec    = APPLIANCES["dishwasher"]
    wash_spec    = APPLIANCES["washing_machine"]
    ev_spec      = APPLIANCES["ev_charger"]
    heater_spec  = APPLIANCES["heater"]
    fridge_spec  = APPLIANCES["fridge"]

    # ---- Power models ---- #

    # Dishwasher: DecayGrow captures the heater burst → rinse → heater profile.
    dish_model = DecayGrowModel(
        peak_w=dish_spec.max_power_w,
        trough_w=150.0,
        tau_decay_s=600.0,
        tau_grow_s=900.0,
        turn_frac=0.55,
        noise_std_w=20.0,
    )
    # Washer: Decay — high surge for heat/agitate, then tapers to spin baseline.
    wash_model = DecayModel(
        peak_w=wash_spec.max_power_w,
        baseline_w=300.0,
        tau_s=1200.0,
        noise_std_w=25.0,
    )
    # EV: OnOff at 7 kW while charging.
    ev_model = OnOffModel(power_w=ev_spec.rated_kw * 1000.0, noise_std_w=30.0)

    # Heater/AC: OnOff at rated power with moderate noise (thermostat kicks in).
    heater_model = OnOffModel(power_w=heater_spec.max_power_w, noise_std_w=50.0)

    # Fridge: MinMax compressor cycling — 40 % duty, ~20-min sub-cycle.
    fridge_model = MinMaxModel(
        min_w=fridge_spec.max_power_w * 0.05,   # idle draw (~7 W)
        max_w=fridge_spec.max_power_w,           # compressor on (~150 W)
        duty=0.4,
        sub_cycle_s=1200,                        # 20-min compressor period
        noise_std_w=3.0,
    )

    # ---- Cycle durations ---- #
    dish_cycle_s   = dish_spec.cycle_slots   * SLOT_MINUTES * 60   # 7 200 s
    wash_cycle_s   = wash_spec.cycle_slots   * SLOT_MINUTES * 60   # 5 400 s
    heater_cycle_s = heater_spec.cycle_slots * SLOT_MINUTES * 60   # 3 600 s
    ev_cycle_s     = int(EV_DAILY_NEED_KWH / ev_spec.rated_kw * 3600)  # ≈ 12 343 s

    # Fridge runs continuously — one very long "cycle" spanning the full window.
    fridge_cycle_s = int((end - start).total_seconds())

    # ---- Cycle start times ---- #

    # Dishwasher peaks in the evening (~20:00), ~5/week.
    dish_starts = natural_cycle_times(
        start, end,
        peak_hour=20.0, hour_spread=0.75, cycles_per_week=5.0, rng=rng,
    )
    # Washing machine in mid-morning (~10:30), now ~7/week (daily) for richer signal.
    wash_starts = natural_cycle_times(
        start, end,
        peak_hour=10.5, hour_spread=1.0, cycles_per_week=7.0, rng=rng,
    )
    # EV every day at 19:00 (user comes home from work).
    ev_starts = daily_ev_cycle_times(start, end, plug_in_hour=19)

    # Heater natural start: 1 h before each comfort deadline so the space is
    # warm at 07:00 and 18:00.  The MILP may shift these starts even earlier
    # when pre-deadline prices are cheaper, but never later (hard constraint).
    heater_starts = natural_cycle_times(
        start, end,
        peak_hour=6.0, hour_spread=0.5, cycles_per_week=5.0, rng=rng,
    ) + natural_cycle_times(
        start, end,
        peak_hour=17.0, hour_spread=0.5, cycles_per_week=5.0, rng=rng,
    )

    # Fridge: single "always-on" cycle starting at the scenario start.
    fridge_starts = (start,)

    return ScenarioSpec(
        start=start,
        end=end,
        seed=seed,
        base_load_w=150.0,
        appliances=(
            ApplianceSchedule("dishwasher",     dish_model,    dish_starts,    dish_cycle_s),
            ApplianceSchedule("washing_machine", wash_model,   wash_starts,    wash_cycle_s),
            ApplianceSchedule("ev_charger",      ev_model,     ev_starts,      ev_cycle_s),
            ApplianceSchedule("heater",          heater_model, heater_starts,  heater_cycle_s),
            ApplianceSchedule("fridge",          fridge_model, fridge_starts,  fridge_cycle_s),
        ),
    )


# --------------------------------------------------------------------------- #
# Generator                                                                   #
# --------------------------------------------------------------------------- #
def _tag_split(df: pd.DataFrame, test_start: datetime = SCENARIO_TEST_START) -> pd.DataFrame:
    """Add a ``split`` column (``"train"`` / ``"test"``) to ``df`` in-place and return it."""
    s = pd.Series("train", index=df.index, dtype="object")
    s.loc[df["timestamp"] >= pd.Timestamp(test_start)] = "test"
    df["split"] = s.astype("category")
    return df


class ScenarioGenerator:
    """Deterministic 1 Hz scenario expansion from a ScenarioSpec."""

    def generate(self, spec: ScenarioSpec) -> ScenarioOutput:
        """Expand a ``ScenarioSpec`` into full 1 Hz traces.

        For each appliance in the spec the method places each cycle's power
        trace (sampled from the parametric model) at its scheduled offset
        inside a zero-initialised per-appliance array, then sums all arrays
        plus the base load to produce the aggregate mains trace.

        The random generator is re-seeded from ``spec.seed`` on every call, so
        the output is byte-identical for the same spec regardless of how many
        times ``generate`` is called.

        Args:
            spec: Declarative scenario description.

        Returns:
            :class:`ScenarioOutput` with mains, per-appliance, and onset tables.

        Raises:
            ValueError: if ``spec.end <= spec.start``.
        """
        n_samples = spec.total_seconds
        if n_samples <= 0:
            logger.error(
                "ScenarioGenerator.generate: spec.end=%s must be after spec.start=%s",
                spec.end.isoformat(), spec.start.isoformat(),
            )
            raise ValueError("ScenarioSpec.end must be strictly after .start")
        logger.info(
            "ScenarioGenerator.generate: start=%s end=%s n_samples=%d appliances=%d seed=%d",
            spec.start.isoformat(), spec.end.isoformat(),
            n_samples, len(spec.appliances), spec.seed,
        )
        rng = np.random.default_rng(spec.seed)

        # Build aggregate + per-appliance traces in a single numpy pass.
        mains = np.full(n_samples, spec.base_load_w, dtype=np.float32)
        per_appliance_arr: dict[str, np.ndarray] = {}
        onset_rows: list[dict] = []

        for ap in spec.appliances:
            trace = np.zeros(n_samples, dtype=np.float32)
            n_skipped = 0
            for cycle_start in ap.cycle_starts:
                s0 = int((cycle_start - spec.start).total_seconds())
                if s0 < 0 or s0 >= n_samples:
                    n_skipped += 1
                    logger.debug(
                        "ScenarioGenerator: %s cycle at %s skipped (s0=%d out of range [0,%d))",
                        ap.name, cycle_start.isoformat(), s0, n_samples,
                    )
                    continue
                s1 = min(s0 + ap.cycle_duration_s, n_samples)
                cycle_len = s1 - s0
                if cycle_len <= 0:
                    n_skipped += 1
                    continue
                cycle_trace = ap.model.sample_cycle(cycle_len, rng)
                trace[s0:s1] = trace[s0:s1] + cycle_trace
                onset_rows.append(
                    {"timestamp": cycle_start, "appliance": ap.name}
                )
            per_appliance_arr[ap.name] = trace
            mains = mains + trace
            logger.debug(
                "ScenarioGenerator: %s placed=%d skipped=%d cycles "
                "trace_max=%.1fW trace_mean=%.1fW",
                ap.name, len(ap.cycle_starts) - n_skipped, n_skipped,
                float(trace.max()), float(trace.mean()),
            )

        ts = pd.date_range(spec.start, periods=n_samples, freq="1s", tz="UTC")

        mains_df = _tag_split(pd.DataFrame({"timestamp": ts, "power_w": mains}))
        per_appliance_df: dict[str, pd.DataFrame] = {
            name: _tag_split(pd.DataFrame({"timestamp": ts, "power_w": arr}))
            for name, arr in per_appliance_arr.items()
        }

        if onset_rows:
            onsets = pd.DataFrame(onset_rows)
            onsets["timestamp"] = pd.to_datetime(onsets["timestamp"], utc=True)
            onsets = onsets.sort_values("timestamp").reset_index(drop=True)
            onsets = _tag_split(onsets)
        else:
            onsets = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime([], utc=True),
                    "appliance": pd.Series([], dtype="object"),
                    "split": pd.Series([], dtype="category"),
                }
            )

        result = ScenarioOutput(
            spec=spec,
            mains=mains_df,
            per_appliance=per_appliance_df,
            onsets=onsets,
        )
        logger.info(
            "ScenarioGenerator.generate: done — mains_rows=%d onset_events=%d appliances=%s",
            len(mains_df), len(onsets), list(per_appliance_df.keys()),
        )
        return result

    # ------------------------------------------------------------------ #
    # Intervention API                                                   #
    # ------------------------------------------------------------------ #
    def apply_intervention_delay(
        self,
        spec: ScenarioSpec,
        appliance: str,
        delay: timedelta,
        *,
        from_time: datetime | None = None,
    ) -> ScenarioSpec:
        """Shift every cycle of ``appliance`` by ``delay``.

        If ``from_time`` is given, only cycles starting at or after that
        moment are shifted — useful when the test window is the only segment
        the agent can act on.
        """
        new_apps: list[ApplianceSchedule] = []
        for ap in spec.appliances:
            if ap.name != appliance:
                new_apps.append(ap)
                continue
            new_cycles = tuple(
                (t + delay) if (from_time is None or t >= from_time) else t
                for t in ap.cycle_starts
            )
            new_apps.append(replace(ap, cycle_starts=new_cycles))
        return replace(spec, appliances=tuple(new_apps))

    def apply_intervention_from_schedule(
        self,
        spec: ScenarioSpec,
        schedule: Schedule,
    ) -> ScenarioSpec:
        """Rewrite cycle starts so each scheduled task begins at the MILP slot.

        Only cycles already scheduled in the slot_start..slot_start+horizon
        window are replaced; cycles outside the horizon remain untouched.
        This lets the demo say: "given the MILP produced this plan, re-run
        the scenario with exactly those start times and show the resulting
        aggregate."
        """
        horizon_end = schedule.slot_start + timedelta(
            minutes=SLOT_MINUTES * schedule.horizon_slots
        )
        # Build per-appliance proposed starts from the schedule.
        proposed: dict[str, datetime] = {}
        for task in schedule.tasks:
            t0 = schedule.slot_start + timedelta(minutes=SLOT_MINUTES * task.start_slot)
            proposed[task.appliance] = t0

        new_apps: list[ApplianceSchedule] = []
        for ap in spec.appliances:
            if ap.name not in proposed:
                new_apps.append(ap)
                continue
            target = proposed[ap.name]
            remaining = tuple(
                t if not (schedule.slot_start <= t < horizon_end) else target
                for t in ap.cycle_starts
            )
            new_apps.append(replace(ap, cycle_starts=remaining))
        return replace(spec, appliances=tuple(new_apps))


# --------------------------------------------------------------------------- #
# Parquet I/O (schema-compatible with the rest of the repo)                   #
# --------------------------------------------------------------------------- #
def write_scenario_parquet(out: ScenarioOutput, out_dir: Path) -> dict[str, Path]:
    """Write mains, onsets, and per-appliance parquets; return the path map."""
    logger.info("write_scenario_parquet: writing scenario to %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    mains_path = out_dir / "mains_1hz.parquet"
    out.mains.to_parquet(mains_path, index=False)
    paths["mains"] = mains_path
    logger.debug("write_scenario_parquet: wrote mains %d rows → %s", len(out.mains), mains_path)

    onsets_path = out_dir / "onsets.parquet"
    out.onsets.to_parquet(onsets_path, index=False)
    paths["onsets"] = onsets_path
    logger.debug("write_scenario_parquet: wrote onsets %d rows → %s", len(out.onsets), onsets_path)

    for name, df in out.per_appliance.items():
        p = out_dir / f"{name}_1hz.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
        logger.debug("write_scenario_parquet: wrote %s %d rows → %s", name, len(df), p)

    logger.info("write_scenario_parquet: wrote %d files to %s", len(paths), out_dir)
    return paths


def load_scenario_mains(path: Path | None = None) -> pd.DataFrame:
    """Load the mains 1 Hz trace (timestamp, power_w, split)."""
    p = path or (SCENARIO_DIR / "mains_1hz.parquet")
    df = pd.read_parquet(p)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def load_scenario_appliance(appliance: str, path: Path | None = None) -> pd.DataFrame:
    """Load one appliance's 1 Hz ground-truth trace."""
    p = path or (SCENARIO_DIR / f"{appliance}_1hz.parquet")
    df = pd.read_parquet(p)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


__all__ = [
    "ApplianceSchedule",
    "ScenarioSpec",
    "ScenarioOutput",
    "ScenarioGenerator",
    "default_scenario_spec",
    "natural_cycle_times",
    "daily_ev_cycle_times",
    "write_scenario_parquet",
    "load_scenario_mains",
    "load_scenario_appliance",
]
