"""Streaming digital-twin driver.

Topology (per scenario run):

  1 Hz sample loop:
      for each Sample from ScenarioStreamer:
          RollingDisaggregator.append(p_mains)  (ground-truth lookup)
          CommitTracker.tick(now)
          OnsetDetector.update(appliance, p_est, now)
          TriggerManager.evaluate(state)
          if trigger: invoke the graph (slow path) → commit plan

  Slow path (LangGraph, invoked on trigger only):
      forecast_price → predict_behavior → optimize → hitl_gate → commit_plan

All simulated work is logged to ``data/cache/run_log.jsonl`` and a final
summary line is printed with total cost, savings, HITL prompts, and replans.

Run:
    python -m aerogrid.sim.digital_twin --days 2
    python -m aerogrid.sim.digital_twin --inject-spike --price-impl naive
    python -m aerogrid.sim.digital_twin --no-auto-confirm
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from langgraph.types import Command

from aerogrid.behavioral_predictor import load_onsets, make_predictor
from aerogrid.commit import CommitTracker
from aerogrid.logging_config import setup_logging
from aerogrid.config import (
    APPLIANCES,
    EV_DAILY_NEED_KWH,
    RUN_LOG_PATH,
    SCENARIO_DIR,
    SCENARIO_TEST_END,
    SCENARIO_TEST_START,
    SLOT_MINUTES,
)
from aerogrid.graph import build_graph, make_thread_id
from aerogrid.nilm import Disaggregator, OnsetDetector, RollingDisaggregator
from aerogrid.price_oracle import make_oracle
from aerogrid.sim.price_server import PriceServer
from aerogrid.sim.streamer import ScenarioStreamer
from aerogrid.triggers import TriggerManager

logger = logging.getLogger(__name__)


def _jsonable(x: Any) -> Any:
    """Recursively convert ``x`` to a JSON-serialisable Python primitive.

    Handles ``None``, scalars, ``datetime`` (→ ISO string), objects with
    ``as_dict()``, lists/tuples, dicts, and numpy arrays.  Falls back to
    ``str(x)`` for any unknown type so the log line is never lost.
    """
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


def main() -> int:
    """Run the 1 Hz streaming simulation over the configured test window.

    Sets up all components (``PriceServer``, ``ScenarioStreamer``,
    ``RollingDisaggregator``, ``OnsetDetector`` instances, ``TriggerManager``,
    ``CommitTracker``, LangGraph outer loop), then iterates every simulated
    second in the test window.

    On each sample:
    - The rolling disaggregator ingests the mains reading.
    - The commit tracker ticks (decrements EV kWh, expires committed tasks).
    - Per-appliance power estimates feed into onset detectors.
    - Realised cost accumulates whenever a 15-min slot boundary is crossed.
    - If ``TriggerManager`` fires, the outer-loop graph is invoked and the
      resulting plan is committed.

    All replan events are written to ``data/cache/run_log.jsonl`` (one JSON
    object per line).  A periodic status line is printed every simulated day
    and a summary is printed at the end.

    Returns:
        0 on success (for use with ``raise SystemExit(main())``).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=None,
                    help="cap the simulated days (default: full scenario test window).")
    ap.add_argument("--hours", type=float, default=None,
                    help="cap simulated hours (takes precedence over --days).")
    ap.add_argument("--auto-confirm", dest="auto_confirm", action="store_true",
                    default=True, help="auto-accept HITL prompts (default).")
    ap.add_argument("--no-auto-confirm", dest="auto_confirm", action="store_false")
    ap.add_argument("--inject-spike", action="store_true",
                    help="inject a surprise price spike on day 2 @ 18:00.")
    ap.add_argument("--price-impl", default="naive",
                    choices=["gridfm", "chronos", "naive"])
    ap.add_argument("--log", type=Path, default=RUN_LOG_PATH)
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO). Use DEBUG for very verbose output.",
    )
    ap.add_argument("--no-log-file", action="store_true",
                    help="disable file logging (console only).")
    args = ap.parse_args()

    # Configure logging first so all subsequent imports and operations are logged.
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    py_log_path = setup_logging(
        level=log_level,
        auto_file=not args.no_log_file,
        console=False, # only log to file, flip it to True to log to console as well
    )
    logger.info(
        "digital_twin: logging configured level=%s file=%s",
        args.log_level, py_log_path or "disabled",
    )

    # ---- Components ---- #
    logger.info("digital_twin: initialising components")
    server = PriceServer()
    if args.inject_spike:
        server.spike_at = SCENARIO_TEST_START + timedelta(days=2, hours=18)
        logger.info("digital_twin: price spike scheduled at %s", server.spike_at.isoformat())
    streamer = ScenarioStreamer(realized_price_provider=server.realized)
    disagg = Disaggregator.from_scenario(SCENARIO_DIR, split="test")
    rolling = RollingDisaggregator(disagg)
    detectors = {
        name: OnsetDetector(
            appliance=name,
            threshold_w=APPLIANCES[name].on_power_threshold_w,
        )
        for name in disagg.appliances()
    }

    logger.info("digital_twin: loading onsets and fitting behavioral predictor")
    predictor = make_predictor().fit(load_onsets()) # behavioral predictor
    logger.info("digital_twin: instantiating price oracle impl=%s", args.price_impl)
    oracle = make_oracle(args.price_impl)

    builder, checkpointer = build_graph(
        price_oracle=oracle,
        predictor=predictor,
        price_history_provider=server.history,
        auto_confirm=args.auto_confirm,
    )
    graph = builder.compile(checkpointer=checkpointer)

    commit = CommitTracker(remaining_ev_kwh=EV_DAILY_NEED_KWH)
    trig = TriggerManager()

    # ---- Run window ---- #
    start = SCENARIO_TEST_START
    end = SCENARIO_TEST_END
    if args.hours is not None:
        end = min(end, start + timedelta(hours=args.hours))
    elif args.days is not None:
        end = min(end, start + timedelta(days=args.days))

    args.log.parent.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    n_replans = 0
    n_hitl = 0
    last_replan_reason: str | None = None
    cumulative_cost = 0.0
    cumulative_baseline_cost = 0.0
    last_realized_price: float | None = None
    previous_plan = None

    duration_h = (end - start).total_seconds() / 3600.0
    logger.info(
        "digital_twin: starting simulation %s → %s (%.1fh) auto_confirm=%s price_impl=%s jsonl_log=%s",
        start.isoformat(), end.isoformat(), duration_h, args.auto_confirm, args.price_impl, args.log,
    )
    with args.log.open("w") as log_fh:
        for sample in streamer.iter_samples(start=start, end=end):
            n_samples += 1
            # --- Inner loop ---
            rolling.append(sample.p_mains_w, sample.t)
            commit.tick(sample.t, dt_s=1.0)

            per_appliance = rolling.infer_latest(sample.t)
            new_onsets = []
            for name, p in per_appliance.items():
                o = detectors[name].update(p, sample.t)
                if o is not None:
                    new_onsets.append(o)

            # Accrue realized cost when a slot-boundary price is delivered.
            if sample.realized_price is not None:
                last_realized_price = sample.realized_price
                total_load = commit.ev_power_setpoint_kw + sum(
                    APPLIANCES[t.appliance].rated_kw for t in commit.committed_tasks
                )
                slot_cost = (
                    total_load * (SLOT_MINUTES / 60.0) * (sample.realized_price / 1000.0)
                )
                cumulative_cost += slot_cost
                naive_ev = APPLIANCES["ev_charger"].rated_kw #TODO aren't we missing other appliances?
                cumulative_baseline_cost += (
                    naive_ev * (SLOT_MINUTES / 60.0) * (sample.realized_price / 1000.0)
                )
                logger.debug(
                    "digital_twin: slot boundary at=%s price=%.2f total_load=%.2fkW "
                    "slot_cost=%.4f cumulative_cost=%.4f",
                    sample.t.isoformat(), sample.realized_price, total_load,
                    slot_cost, cumulative_cost,
                )

            # --- Trigger check ---
            trigger = trig.evaluate(
                now=sample.t,
                latest_sample=sample,
                new_onsets=new_onsets,
                committed_tasks=commit.committed_tasks,
                price_forecast=None,    # forecast is produced inside the graph
                remaining_ev_kwh=commit.remaining_ev_kwh,
                ev_power_setpoint_kw=commit.ev_power_setpoint_kw,
            )

            if trigger is not None:
                logger.info(
                    "digital_twin: TRIGGER kind=%s detail=%r at=%s (replan #%d)",
                    trigger.kind, trigger.detail, sample.t.isoformat(), n_replans + 1,
                )
                state_in = {
                    "now": sample.t,
                    "latest_sample": sample,
                    "per_appliance_power_w": per_appliance,
                    "new_onsets": new_onsets,
                    "committed_tasks": list(commit.committed_tasks),
                    "remaining_ev_kwh": commit.remaining_ev_kwh,
                    "ev_power_setpoint_kw": commit.ev_power_setpoint_kw,
                    "previous_plan": previous_plan,
                    "replan_trigger": trigger,
                    "event_log": [],
                    "cumulative_cost": cumulative_cost,
                    "cumulative_baseline_cost": cumulative_baseline_cost,
                }
                cfg = {"configurable": {"thread_id": make_thread_id(sample.t)}}
                try:
                    result = graph.invoke(state_in, config=cfg)
                except Exception as e:                  # noqa: BLE001
                    logger.error(
                        "digital_twin: graph error at=%s: %r", sample.t.isoformat(), e,
                    )
                    continue

                if "__interrupt__" in result:
                    n_hitl += 1
                    logger.info(
                        "digital_twin: HITL interrupt #%d at=%s — auto-resuming with 'yes'",
                        n_hitl, sample.t.isoformat(),
                    )
                    result = graph.invoke(Command(resume="yes"), config=cfg)

                n_replans += 1
                last_replan_reason = trigger.detail or trigger.kind
                trig.notify_replanned(sample.t)
                new_plan = result.get("current_plan")
                if new_plan is not None:
                    logger.info(
                        "digital_twin: adopting new plan at=%s solver=%s expected_cost=%.4f",
                        sample.t.isoformat(), new_plan.solver_status, new_plan.expected_cost,
                    )
                    commit.adopt_plan(new_plan, sample.t)
                    previous_plan = new_plan

                entry = {
                    "sample": n_samples,
                    "now": sample.t.isoformat(),
                    "p_mains_w": sample.p_mains_w,
                    "trigger": trigger.as_dict(),
                    "hitl": _jsonable(result.get("hitl_decision")),
                    "plan": _jsonable(new_plan),
                    "commit": commit.snapshot(),
                    "realized_price": last_realized_price,
                    "cumulative_cost": cumulative_cost,
                    "cumulative_baseline_cost": cumulative_baseline_cost,
                }
                log_fh.write(json.dumps(entry) + "\n")

            # Periodic status
            if n_samples % (SLOT_MINUTES * 60 * 4 * 6) == 0:       # every simulated day
                saved = (
                    (1 - cumulative_cost / cumulative_baseline_cost) * 100
                    if cumulative_baseline_cost > 0 else 0.0
                )
                status_msg = (
                    f"[{sample.t.isoformat()}] samples={n_samples:,} "
                    f"replans={n_replans} hitl={n_hitl} "
                    f"cost=${cumulative_cost:.2f} baseline=${cumulative_baseline_cost:.2f} "
                    f"saved={saved:+.1f}%"
                )
                logger.info("digital_twin periodic status: %s", status_msg)

    # ---- Summary ---- #
    saved = (
        (1 - cumulative_cost / cumulative_baseline_cost) * 100
        if cumulative_baseline_cost > 0 else 0.0
    )
    logger.info(
        "digital_twin SUMMARY: samples=%d replans=%d hitl=%d "
        "cost=%.2f baseline=%.2f savings=%.1f%% last_trigger=%s",
        n_samples, n_replans, n_hitl,
        cumulative_cost, cumulative_baseline_cost, saved,
        last_replan_reason or "N/A",
    )
    print("\n=== simulation summary ===")
    print(f"samples:            {n_samples:,}")
    print(f"replans:            {n_replans}")
    print(f"hitl prompts:       {n_hitl}")
    if last_replan_reason:
        print(f"last trigger:       {last_replan_reason}")
    print(f"cumulative cost:    ${cumulative_cost:.2f}")
    print(f"baseline cost:      ${cumulative_baseline_cost:.2f}")
    print(f"savings:            {saved:+.1f}%")
    print(f"log:                {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
