"""Digital twin driver: glues the streamer + price server into the LangGraph.

Run:
    python -m aerogrid.sim.digital_twin --use-test-window
    python -m aerogrid.sim.digital_twin --use-test-window --hours 48
    python -m aerogrid.sim.digital_twin --use-test-window --inject-spike

The driver writes every node emission to data/cache/run_log.jsonl and prints a
final summary with cost, savings, HITL prompt count, replan count.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from langgraph.types import Command

from aerogrid.behavioral_predictor import load_onsets, make_predictor
from aerogrid.config import (
    NYISO_TEST_START,
    NYISO_TEST_END,
    RUN_LOG_PATH,
    SLOT_MINUTES,
    UKDALE_TEST_END,
    UKDALE_TEST_START,
)
from aerogrid.graph import build_graph, make_thread_id
from aerogrid.price_oracle import make_oracle
from aerogrid.signal_watcher import SignalWatcher
from aerogrid.sim.price_server import PriceServer
from aerogrid.sim.streamer import Streamer


def _to_jsonable(x: Any) -> Any:
    """Coerce arbitrary state values into JSON-loggable primitives."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if hasattr(x, "as_dict"):
        return x.as_dict()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-test-window", action="store_true",
                    help="replay the UK-DALE + NYISO test slices (14 days).")
    ap.add_argument("--hours", type=int, default=None,
                    help="cap the simulated hours (default: full test window).")
    ap.add_argument("--auto-confirm", action="store_true", default=True,
                    help="auto-accept HITL prompts (default).")
    ap.add_argument("--no-auto-confirm", dest="auto_confirm", action="store_false")
    ap.add_argument("--inject-spike", action="store_true",
                    help="inject a surprise price spike on test day 2 at 18:00.")
    ap.add_argument("--price-impl", default="naive",
                    choices=["gridfm", "chronos", "naive"])
    ap.add_argument("--log", type=Path, default=RUN_LOG_PATH)
    args = ap.parse_args()

    # --- set up components ------------------------------------------------ #
    streamer = Streamer()
    server = PriceServer()
    if args.inject_spike:
        server.spike_at = NYISO_TEST_START + timedelta(days=2, hours=18)
        print(f"will inject spike at {server.spike_at.isoformat()}")

    predictor = make_predictor().fit(load_onsets())
    watcher = SignalWatcher.from_cache()
    oracle = make_oracle(args.price_impl)

    # NYISO test window and UK-DALE test window are different calendar years.
    # The twin clock runs on the NYISO calendar; the streamer's ground-truth
    # onsets carry UK-DALE timestamps, so we translate them into the NYISO
    # calendar by offsetting the date by the fixed delta between the two
    # windows.
    tz_offset = NYISO_TEST_START - UKDALE_TEST_START

    builder, checkpointer = build_graph(
        watcher=watcher,
        price_oracle=oracle,
        predictor=predictor,
        price_history_provider=server.history,
        realized_price_provider=server.realized,
        auto_confirm=args.auto_confirm,
    )
    graph = builder.compile(checkpointer=checkpointer)

    # --- run loop --------------------------------------------------------- #
    start = UKDALE_TEST_START
    end = UKDALE_TEST_END
    if args.hours:
        end = min(end, start + timedelta(hours=args.hours))

    args.log.parent.mkdir(parents=True, exist_ok=True)
    n_ticks = 0
    n_hitl = 0
    n_replan = 0
    last_replan_reason: str | None = None
    last_state: dict = {}

    with args.log.open("w") as log_fh:
        for tick in streamer.iter_ticks(start=start, end=end):
            nyiso_now = tick.now + tz_offset
            translated_onsets = [
                type(o)(
                    appliance=o.appliance,
                    timestamp=o.timestamp + tz_offset,
                    confidence=o.confidence,
                    source=o.source,
                )
                for o in tick.new_onsets
            ]
            thread_id = make_thread_id(nyiso_now)

            state = {
                "now": nyiso_now,
                "chunk_start": (tick.chunk_start + tz_offset) if tick.chunk_start else None,
                "mains_chunk": tick.mains_chunk,
                "new_onsets": translated_onsets,
                "recent_onsets": last_state.get("recent_onsets", []),
                "realized_prices": last_state.get("realized_prices", []),
                "event_log": [],
                "cumulative_cost": last_state.get("cumulative_cost", 0.0),
                "cumulative_baseline_cost": last_state.get("cumulative_baseline_cost", 0.0),
                "iteration": last_state.get("iteration", 0),
                "replan_reason": None,
            }

            cfg = {"configurable": {"thread_id": thread_id}}
            try:
                result = graph.invoke(state, config=cfg)
            except Exception as e:  # noqa: BLE001
                print(f"[{nyiso_now.isoformat()}] graph error: {e!r}")
                continue

            inter = result.get("__interrupt__")
            if inter is not None:
                n_hitl += 1
                # Demo: auto-yes.
                result = graph.invoke(Command(resume="yes"), config=cfg)

            if result.get("replan_reason"):
                n_replan += 1
                last_replan_reason = result["replan_reason"]

            last_state = result
            n_ticks += 1

            entry = {
                "tick": n_ticks,
                "now": nyiso_now.isoformat(),
                "ukdale_now": tick.now.isoformat(),
                "new_onsets": [_to_jsonable(o) for o in (result.get("new_onsets") or [])],
                "schedule": _to_jsonable(result.get("schedule")),
                "realized_price": (result.get("realized_prices") or [None])[-1],
                "cumulative_cost": result.get("cumulative_cost"),
                "cumulative_baseline_cost": result.get("cumulative_baseline_cost"),
                "replan_reason": result.get("replan_reason"),
                "user_confirmation": result.get("user_confirmation"),
            }
            log_fh.write(json.dumps(entry) + "\n")

            if n_ticks % 96 == 0:  # once per simulated day
                cum = result.get("cumulative_cost", 0.0)
                base = result.get("cumulative_baseline_cost", 0.0)
                saved = (1 - cum / base) * 100 if base > 0 else 0
                print(
                    f"[{nyiso_now.isoformat()}] day {n_ticks // 96}: "
                    f"cost=${cum:.2f} baseline=${base:.2f} "
                    f"saved={saved:+.1f}%  hitl={n_hitl}  replans={n_replan}"
                )

    # --- summary ---------------------------------------------------------- #
    cum = last_state.get("cumulative_cost", 0.0)
    base = last_state.get("cumulative_baseline_cost", 0.0)
    saved = (1 - cum / base) * 100 if base > 0 else 0
    print("\n=== simulation summary ===")
    print(f"ticks:             {n_ticks}")
    print(f"hitl prompts:      {n_hitl}")
    print(f"replan events:     {n_replan}")
    if last_replan_reason:
        print(f"last replan:       {last_replan_reason}")
    print(f"cumulative cost:   ${cum:.2f}")
    print(f"baseline cost:     ${base:.2f}")
    print(f"savings:           {saved:+.1f}%")
    print(f"log:               {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
