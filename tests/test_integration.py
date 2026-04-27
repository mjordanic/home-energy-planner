"""End-to-end smoke test for the streaming agent loop.

Builds a small scenario in a temp dir, loads ground-truth traces into the
perfect disaggregator, and runs 30 simulated minutes of the inner loop
against an in-memory price feed. Asserts the outer loop (MPC) fires at
least one replan and produces a non-empty plan.

This is a smoke test, not a quality benchmark — it verifies the plumbing
(scenario → disagg → onset_detector → trigger → graph → commit).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aerogrid.behavioral_predictor import HybridBehavioralPredictor
from aerogrid.commit import CommitTracker
from aerogrid.config import APPLIANCES, EV_DAILY_NEED_KWH
from aerogrid.graph import build_graph, make_thread_id
from aerogrid.nilm import Disaggregator, OnsetDetector, RollingDisaggregator
from aerogrid.price_oracle import SeasonalNaiveOracle
from aerogrid.sim.appliance_models import DecayGrowModel, DecayModel, OnOffModel
from aerogrid.sim.scenario import (
    ApplianceSchedule,
    ScenarioGenerator,
    ScenarioSpec,
    write_scenario_parquet,
)
from aerogrid.sim.streamer import ScenarioStreamer
from aerogrid.triggers import TriggerManager


@pytest.fixture
def tiny_scenario(tmp_path: Path):
    start = datetime(2024, 12, 14, tzinfo=timezone.utc)
    test_start = datetime(2024, 12, 15, tzinfo=timezone.utc)
    end = datetime(2024, 12, 16, tzinfo=timezone.utc)

    dish_model = DecayGrowModel(
        peak_w=2500.0, trough_w=200.0,
        tau_decay_s=600.0, tau_grow_s=900.0,
        turn_frac=0.5, noise_std_w=5.0,
    )
    wash_model = DecayModel(peak_w=2400.0, baseline_w=300.0, tau_s=1200.0, noise_std_w=5.0)
    ev_model = OnOffModel(power_w=7000.0, noise_std_w=15.0)
    spec = ScenarioSpec(
        start=start, end=end, seed=1, base_load_w=150.0,
        appliances=(
            ApplianceSchedule(
                "dishwasher", dish_model,
                cycle_starts=(
                    start + timedelta(hours=20),
                    test_start + timedelta(hours=20, minutes=10),
                ),
                cycle_duration_s=3600,
            ),
            ApplianceSchedule(
                "washing_machine", wash_model,
                cycle_starts=(
                    start + timedelta(hours=10),
                    test_start + timedelta(hours=10, minutes=5),
                ),
                cycle_duration_s=2700,
            ),
            ApplianceSchedule(
                "ev_charger", ev_model,
                cycle_starts=(
                    start + timedelta(hours=19),
                    test_start + timedelta(hours=19),
                ),
                cycle_duration_s=3600 * 3,
            ),
        ),
    )
    gen = ScenarioGenerator()
    out = gen.generate(spec)
    for df in [out.mains, *out.per_appliance.values(), out.onsets]:
        if df.empty:
            continue
        df["split"] = np.where(
            df["timestamp"] >= pd.Timestamp(test_start), "test", "train"
        )
        df["split"] = df["split"].astype("category")

    write_scenario_parquet(out, tmp_path)
    return {
        "start": start,
        "test_start": test_start,
        "end": end,
        "dir": tmp_path,
        "onsets": out.onsets,
        "mains": out.mains,
    }


@pytest.fixture
def tiny_disagg(tiny_scenario):
    """Load ground-truth traces for test split — no training needed."""
    return Disaggregator.from_scenario(
        scenario_dir=tiny_scenario["dir"],
        appliances=["dishwasher", "washing_machine"],
        split="test",
    )


class _InMemoryPriceFeed:
    """Replaces PriceServer with a synthetic cheap-night / expensive-day curve."""
    def __init__(self, start: datetime, end: datetime):
        idx = pd.date_range(start - timedelta(days=7), end, freq="15min", tz="UTC")
        hod = idx.hour + idx.minute / 60.0
        lbmp = 30 + 30 * np.sin(2 * np.pi * (hod - 7) / 24)
        self.df = pd.DataFrame({"timestamp": idx, "lbmp": lbmp.astype(np.float32)})

    def history(self, now):
        return self.df[self.df["timestamp"] < now]

    def realized(self, now):
        slot_min = (now.minute // 15) * 15
        slot = now.replace(minute=slot_min, second=0, microsecond=0)
        row = self.df[self.df["timestamp"] == slot]
        return float(row["lbmp"].iloc[0]) if not row.empty else None


def test_inner_loop_fires_replan_and_produces_plan(tiny_scenario, tiny_disagg):
    """Run 30 simulated minutes and verify at least one replan produces a plan."""
    onsets_df = pd.read_parquet(tiny_scenario["dir"] / "onsets.parquet")
    extra_rows = []
    base = tiny_scenario["start"]
    for d in range(6):
        extra_rows.extend([
            {"appliance": "dishwasher",
             "timestamp": base - timedelta(days=d + 1, hours=0) + timedelta(hours=20),
             "split": "train"},
            {"appliance": "washing_machine",
             "timestamp": base - timedelta(days=d + 1, hours=0) + timedelta(hours=10),
             "split": "train"},
        ])
    extra_df = pd.DataFrame(extra_rows)
    extra_df["timestamp"] = pd.to_datetime(extra_df["timestamp"], utc=True)
    onsets_df = pd.concat([extra_df, onsets_df], ignore_index=True)
    onsets_df["split"] = onsets_df["split"].astype("category")

    predictor = HybridBehavioralPredictor().fit(onsets_df)
    oracle = SeasonalNaiveOracle()
    feed = _InMemoryPriceFeed(tiny_scenario["start"], tiny_scenario["end"])

    builder, checkpointer = build_graph(
        price_oracle=oracle,
        predictor=predictor,
        price_history_provider=feed.history,
        auto_confirm=True,
        horizon_slots=8,
    )
    graph = builder.compile(checkpointer=checkpointer)

    rolling = RollingDisaggregator(tiny_disagg)
    detectors = {
        name: OnsetDetector(name, threshold_w=APPLIANCES[name].on_power_threshold_w)
        for name in tiny_disagg.appliances()
    }
    commit = CommitTracker(remaining_ev_kwh=EV_DAILY_NEED_KWH)
    trig = TriggerManager(cooldown_s=30.0)

    streamer = ScenarioStreamer(
        mains_path=tiny_scenario["dir"] / "mains_1hz.parquet",
        realized_price_provider=feed.realized,
    )
    start_sim = tiny_scenario["test_start"]
    end_sim = start_sim + timedelta(minutes=30)

    n_replans = 0
    n_plans = 0
    previous_plan = None
    for sample in streamer.iter_samples(start=start_sim, end=end_sim):
        rolling.append(sample.p_mains_w, sample.t)
        commit.tick(sample.t)
        per_appliance = rolling.infer_latest(sample.t)
        new_onsets = []
        for name, p in per_appliance.items():
            o = detectors[name].update(p, sample.t)
            if o is not None:
                new_onsets.append(o)

        trigger = trig.evaluate(
            now=sample.t,
            latest_sample=sample,
            new_onsets=new_onsets,
            committed_tasks=commit.committed_tasks,
            remaining_ev_kwh=commit.remaining_ev_kwh,
            ev_power_setpoint_kw=commit.ev_power_setpoint_kw,
        )
        if trigger is None:
            continue

        state_in = {
            "now": sample.t,
            "latest_sample": sample,
            "committed_tasks": list(commit.committed_tasks),
            "remaining_ev_kwh": commit.remaining_ev_kwh,
            "ev_power_setpoint_kw": commit.ev_power_setpoint_kw,
            "previous_plan": previous_plan,
            "replan_trigger": trigger,
            "event_log": [],
        }
        cfg = {"configurable": {"thread_id": make_thread_id(sample.t)}}
        result = graph.invoke(state_in, config=cfg)
        n_replans += 1
        plan = result.get("current_plan")
        if plan is not None:
            n_plans += 1
            previous_plan = plan
            commit.adopt_plan(plan, sample.t)
        trig.notify_replanned(sample.t)

    assert n_replans >= 1, "TriggerManager should have fired at least once (periodic)"
    assert n_plans >= 1, "Graph should have produced at least one plan"
