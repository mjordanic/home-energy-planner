"""Digital-twin orchestrator: streams data and ticks N independent strategies.

Architecture
------------
The digital twin owns ONLY what is genuinely shared by the simulation
environment:

* :class:`ScenarioStreamer` — yields one mains sample per simulated second,
* :class:`PriceServer` — provides the realized electricity price every
  strategy is evaluated against (this is physical reality, not a forecast),
* the queue of synthetic injected onsets used for stress tests,
* cross-strategy onset gating.

It does NOT own any policy logic, NILM disaggregator, behavioural predictor,
price oracle, or LangGraph.  Each strategy carries its own copy of whatever
it needs and builds it inside its own constructor.

Per simulation tick the orchestrator:

1. pulls injected onsets due at this timestamp,
2. **gates** them across strategies — an injected onset is suppressed if
   any strategy still has the appliance pending or running (prevents
   unrealistic double-loading).  Natural onsets perceived by each strategy's
   own machinery are NOT gated,
3. ticks every strategy with the same ``(sample, gated_injected_onsets)`` pair,
4. at every 15-min slot boundary, asks each strategy for a SlotRecord and
   merges them into a wide row (column-prefixed by ``strategy.name``),
5. flushes events from each strategy into a single per-decision event log.

Outputs
-------
``slot_log.parquet``
    One row per 15-min slot. Columns: timestamp, price, suppressed/permitted
    INJECTED onsets, plus ``<strategy>_*`` columns for every strategy's
    power profile and cumulative cost.

``event_log.parquet``
    One row per decision at 1-second resolution. Schema is uniform across
    strategies. ``strategy="stream"`` rows are gating events emitted by the
    digital twin itself; everything else is strategy-emitted.

``optimizer_replans.jsonl``
    OptimizerStrategy's per-replan log with full plan detail. Plan-level
    inspection only — for cost/power analysis use the parquet files.

CLI
---
::

    python -m aerogrid.sim.digital_twin --days 2
    python -m aerogrid.sim.digital_twin --hours 12
    python -m aerogrid.sim.digital_twin --no-auto-confirm
    python -m aerogrid.sim.digital_twin --price-impl chronos
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aerogrid.config import (
    EVENT_LOG_PATH,
    HITL_AUTO_RESPONSES,
    HORIZON_HOURS,
    INJECTED_APPLIANCE_ONSETS,
    INJECTED_PRICE_SPIKES,
    RUN_LOG_PATH,
    SCENARIO_TEST_END,
    SCENARIO_TEST_START,
    SHORT_HORIZON_SLOTS,
    SLOT_LOG_PATH,
    SLOT_MINUTES,
)
from aerogrid.logging_config import setup_logging
from aerogrid.sim.price_server import PriceServer
from aerogrid.sim.strategies import (
    BaselineStrategy,
    OptimizerStrategy,
    Strategy,
    make_event,
)
from aerogrid.sim.streamer import ScenarioStreamer
from aerogrid.types import ApplianceOnset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Onset gating across strategies                                              #
# --------------------------------------------------------------------------- #

def _gate_onsets(
    candidate_onsets: list[ApplianceOnset],
    strategies: list[Strategy],
    now: datetime,
) -> tuple[list[ApplianceOnset], list[tuple[ApplianceOnset, list[str]]]]:
    """Split *candidate_onsets* (INJECTED) into permitted vs suppressed.

    An injected onset is suppressed if **any** strategy still has the same
    appliance pending or running.  Natural onsets perceived inside each
    strategy do NOT pass through this function.
    """
    permitted: list[ApplianceOnset] = []
    suppressed: list[tuple[ApplianceOnset, list[str]]] = []
    for onset in candidate_onsets:
        blockers = [
            s.name for s in strategies
            if s.has_pending_appliance(onset.appliance)
        ]
        if blockers:
            suppressed.append((onset, blockers))
            logger.info(
                "digital_twin: SUPPRESSED %s onset at %s (blocked by %s)",
                onset.appliance, now.isoformat(), ",".join(blockers),
            )
        else:
            permitted.append(onset)
    return permitted, suppressed


# --------------------------------------------------------------------------- #
# Main run loop                                                                #
# --------------------------------------------------------------------------- #

def run(
    *,
    strategies: list[Strategy],
    streamer: ScenarioStreamer,
    start: datetime,
    end: datetime,
    slot_log_path: Path,
    event_log_path: Path,
) -> dict:
    """Stream samples, drive every strategy, and write parquet outputs.

    Policy-agnostic: works for any list of objects implementing
    :class:`Strategy`.  Adding a new strategy class requires no change here.
    """
    slot_rows: list[dict] = []
    event_rows: list[dict] = []
    last_realized_price: float | None = None
    n_samples = 0
    n_suppressed_total = 0

    duration_h = (end - start).total_seconds() / 3600.0
    logger.info(
        "digital_twin: streaming %s → %s (%.1fh) with %d strategies: %s",
        start.isoformat(), end.isoformat(), duration_h,
        len(strategies), [s.name for s in strategies],
    )

    for sample in streamer.iter_samples(start=start, end=end):
        n_samples += 1

        # 1. Pull synthetic injected onsets due at this timestamp.
        injected = streamer.consume_injected_onsets(sample.t)

        # 2. Cross-strategy gating.
        permitted, suppressed = _gate_onsets(injected, strategies, sample.t)
        n_suppressed_total += len(suppressed)

        # 3. Stream-level events for the unified event log.
        for onset in permitted:
            event_rows.append(make_event(
                timestamp=onset.timestamp,
                strategy="stream",
                event_type="onset_permitted",
                appliance=onset.appliance,
                price_eur_mwh=last_realized_price,
                detail=f"source={onset.source} confidence={onset.confidence:.2f}",
            ))
        for onset, blockers in suppressed:
            event_rows.append(make_event(
                timestamp=onset.timestamp,
                strategy="stream",
                event_type="onset_suppressed",
                appliance=onset.appliance,
                price_eur_mwh=last_realized_price,
                detail=(
                    f"blocked_by=[{','.join(blockers)}] "
                    f"source={onset.source}"
                ),
            ))

        # 4. Tick every strategy with the same (sample, gated injected onsets).
        for s in strategies:
            s.tick(sample, permitted, dt_s=1.0)

        # 5. Slot boundary: collect a SlotRecord from every strategy.
        if sample.realized_price is not None:
            last_realized_price = sample.realized_price
            row: dict = {
                "timestamp": sample.t,
                "price_eur_mwh": sample.realized_price,
                "permitted_onsets": ",".join(sorted({o.appliance for o in permitted})),
                "suppressed_onsets": ",".join(
                    sorted({o.appliance for o, _ in suppressed})
                ),
            }
            for s in strategies:
                rec = s.get_slot_record(sample.t, sample.realized_price)
                row.update(rec.to_flat_dict(prefix=s.name))
            slot_rows.append(row)

        # 6. Flush each strategy's pending events into the shared event log.
        for s in strategies:
            event_rows.extend(s.flush_events(price_eur_mwh=last_realized_price))

        # Periodic status log every simulated 6 hours.
        if n_samples % (3600 * 6) == 0:
            cost_str = " | ".join(
                f"{s.name}=€{s.cumulative_cost:.2f}" for s in strategies
            )
            logger.info(
                "digital_twin status [%s] samples=%d suppressed=%d  %s",
                sample.t.isoformat(), n_samples, n_suppressed_total, cost_str,
            )

    # Cleanup hooks (e.g. close OptimizerStrategy's JSONL).
    for s in strategies:
        s.close()

    slot_log_path.parent.mkdir(parents=True, exist_ok=True)
    if slot_rows:
        pd.DataFrame(slot_rows).to_parquet(slot_log_path, index=False)
        logger.info(
            "digital_twin: slot log → %s (%d rows)", slot_log_path, len(slot_rows),
        )
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    if event_rows:
        pd.DataFrame(event_rows).to_parquet(event_log_path, index=False)
        logger.info(
            "digital_twin: event log → %s (%d rows)", event_log_path, len(event_rows),
        )

    return {
        "n_samples": n_samples,
        "n_suppressed_total": n_suppressed_total,
        "strategies": [s.summary() for s in strategies],
        "slot_log_path": slot_log_path,
        "event_log_path": event_log_path,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    """Run the 1 Hz streaming simulation over the configured test window."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=None,
                    help="cap the simulated days (default: full scenario test window)")
    ap.add_argument("--hours", type=float, default=None,
                    help="cap simulated hours (takes precedence over --days)")
    ap.add_argument("--auto-confirm", dest="auto_confirm", action="store_true",
                    default=True, help="auto-resolve HITL prompts (default)")
    ap.add_argument("--no-auto-confirm", dest="auto_confirm", action="store_false")
    ap.add_argument("--price-impl", default="naive",
                    choices=["gridfm", "chronos", "naive"],
                    help="price oracle for the OptimizerStrategy")
    ap.add_argument("--horizon-hours", type=float, default=HORIZON_HOURS,
                    help="receding-horizon length in hours (default from config)")
    ap.add_argument("--replan-jsonl", type=Path, default=RUN_LOG_PATH,
                    help="path for OptimizerStrategy's per-replan JSONL log")
    ap.add_argument("--slot-log", type=Path, default=SLOT_LOG_PATH,
                    help="path for the per-slot multi-strategy parquet")
    ap.add_argument("--event-log", type=Path, default=EVENT_LOG_PATH,
                    help="path for the per-event decision parquet")
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO)",
    )
    ap.add_argument("--no-log-file", action="store_true",
                    help="disable file logging (console only)")
    args = ap.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    py_log_path = setup_logging(
        level=log_level, auto_file=not args.no_log_file, console=False,
    )
    logger.info(
        "digital_twin: logging level=%s file=%s",
        args.log_level, py_log_path or "disabled",
    )

    horizon_slots = max(1, int(round(args.horizon_hours * 60.0 / SLOT_MINUTES)))
    if horizon_slots != SHORT_HORIZON_SLOTS:
        logger.info(
            "digital_twin: horizon_slots %d → %d (%.1fh)",
            SHORT_HORIZON_SLOTS, horizon_slots, args.horizon_hours,
        )

    # ---- Shared infrastructure (truly shared by every strategy) -----------
    server = PriceServer()
    server.spike_events = [(at, float(mag)) for at, mag in INJECTED_PRICE_SPIKES]
    streamer = ScenarioStreamer(realized_price_provider=server.realized)
    for appliance, at in INJECTED_APPLIANCE_ONSETS:
        streamer.add_onset(appliance=appliance, timestamp=at)

    # ---- Strategies (each fully encapsulates its own private machinery) ---
    # To add another strategy, instantiate it here and append to the list.
    baseline = BaselineStrategy()
    optimizer = OptimizerStrategy(
        price_history_provider=server.history,
        price_oracle_impl=args.price_impl,
        horizon_slots=horizon_slots,
        auto_confirm=args.auto_confirm,
        auto_responses=HITL_AUTO_RESPONSES,
        replan_jsonl_path=args.replan_jsonl,
    )
    strategies: list[Strategy] = [baseline, optimizer]

    # ---- Window selection ------------------------------------------------
    start = SCENARIO_TEST_START
    end = SCENARIO_TEST_END
    if args.hours is not None:
        end = min(end, start + timedelta(hours=args.hours))
    elif args.days is not None:
        end = min(end, start + timedelta(days=args.days))

    summary = run(
        strategies=strategies,
        streamer=streamer,
        start=start,
        end=end,
        slot_log_path=args.slot_log,
        event_log_path=args.event_log,
    )

    # ---- Final summary ---------------------------------------------------
    print("\n=== simulation summary ===")
    print(f"samples:                    {summary['n_samples']:,}")
    print(f"injected onsets suppressed: {summary['n_suppressed_total']}")
    print(f"slot log:                   {summary['slot_log_path']}")
    print(f"event log:                  {summary['event_log_path']}")
    print()
    print(f"{'strategy':<14} {'cost (€)':>10}   detail")
    for s in summary["strategies"]:
        detail = ", ".join(
            f"{k}={v}" for k, v in s.items()
            if k not in ("name", "cumulative_cost_eur") and v is not None
        )
        print(f"{s['name']:<14} {s['cumulative_cost_eur']:>10.3f}   {detail}")
    print()
    baseline_summary = next(
        (s for s in summary["strategies"] if s["name"] == "baseline"), None,
    )
    if baseline_summary is not None and baseline_summary["cumulative_cost_eur"] > 1e-9:
        b = baseline_summary["cumulative_cost_eur"]
        for s in summary["strategies"]:
            if s["name"] == "baseline":
                continue
            saved = (1.0 - s["cumulative_cost_eur"] / b) * 100.0
            print(f"  {s['name']} vs baseline savings: {saved:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
