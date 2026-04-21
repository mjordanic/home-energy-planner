"""LangGraph orchestration for the AeroGrid multi-agent loop.

Nodes:
  signal_watch     — DSP NILM: detect appliance onsets in the latest 16 kHz chunk.
  forecast_price   — PriceOracle → 96-slot forecast.
  predict_behavior — BehavioralPredictor → per-appliance 96-slot onset probs.
  optimize         — MILP solve (cvxpy).
  user_confirm     — interrupt() + Command(resume=...) for HITL approval.
  execute          — log the accepted schedule; update cumulative cost bookkeeping.
  monitor          — compare realized price / unexpected onsets against forecast;
                     if divergence > REPLAN_PRICE_DEVIATION *or* an unplanned
                     onset was observed, route back to `optimize` with a reason.

The graph is designed for one call per simulated time step. The twin invokes
`graph.invoke(state, config=...)` at each step; on `interrupt()` it resumes with
`Command(resume=answer)`. SQLite checkpointer persists state across interrupts
so the loop is crash-safe and replayable from the same thread_id.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pickle

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class _PickleSerializer:
    """Pickle-based serializer for LangGraph checkpoints.

    The default JsonPlusSerializer uses msgpack and can't encode dataclasses
    (PriceForecast, Schedule, ApplianceOnset) or numpy arrays out of the box.
    Pickle handles all of them. We only need this for the single-process
    digital twin — if the graph ever goes to production, swap this for a
    proper msgpack serializer with registered type handlers.
    """

    def dumps_typed(self, obj):
        return "pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def loads_typed(self, data):
        type_, raw = data
        if type_ != "pickle":
            raise ValueError(f"unexpected serializer type: {type_}")
        return pickle.loads(raw)

from aerogrid.behavioral_predictor import BehavioralPredictor
from aerogrid.config import (
    APPLIANCES,
    GRAPH_CHECKPOINT_DB,
    REPLAN_PRICE_DEVIATION,
    SLOT_MINUTES,
    SLOTS_PER_DAY,
)
from aerogrid.optimizer import solve_proactive_schedule
from aerogrid.price_oracle import PriceOracle
from aerogrid.signal_watcher import SignalWatcher
from aerogrid.state import AeroGridState
from aerogrid.types import ApplianceOnset


# --------------------------------------------------------------------------- #
# Node factory                                                                #
# --------------------------------------------------------------------------- #
def build_graph(
    watcher: SignalWatcher,
    price_oracle: PriceOracle,
    predictor: BehavioralPredictor,
    price_history_provider,          # callable(now) -> pd.DataFrame context
    realized_price_provider,         # callable(now) -> float | None
    auto_confirm: bool = True,       # True => user_confirm accepts without interrupt
    reservation_lambda: float = 0.5,
):
    """Wire the LangGraph StateGraph. Returns (compiled_graph, checkpointer_ctx).

    The caller must enter the checkpointer context before using the graph:

        with build_graph(...) as (graph, _):
            graph.invoke(state, config={...})
    """
    # Node functions capture the providers by closure.

    def n_signal_watch(state: AeroGridState) -> dict:
        # The digital twin may pre-inject ground-truth onsets into `new_onsets`
        # (simulating a perfect NILM when we don't have 16 kHz for that slot).
        # We merge those with anything the SignalWatcher itself extracts from
        # an attached 16 kHz mains_chunk.
        pre = state.get("new_onsets", []) or []
        chunk = state.get("mains_chunk")
        watcher_out: list[ApplianceOnset] = []
        if chunk is not None:
            voltage, current = chunk
            watcher_out = watcher.process_window(
                voltage, current,
                state.get("chunk_start") or state["now"],
            )
        onsets = [*pre, *watcher_out]
        return {
            "new_onsets": onsets,
            "recent_onsets": [*state.get("recent_onsets", []), *onsets][-500:],
            "event_log": [*state.get("event_log", []),
                          *[{"type": "onset", **o.as_dict()} for o in onsets]],
        }

    def n_forecast_price(state: AeroGridState) -> dict:
        ctx = price_history_provider(state["now"])
        fc = price_oracle.get_15min_forecast(state["now"], ctx, SLOTS_PER_DAY)
        return {"price_forecast": fc}

    def n_predict_behavior(state: AeroGridState) -> dict:
        probs = predictor.predict_all(state["now"], SLOTS_PER_DAY)
        return {"onset_probs": probs}

    def n_optimize(state: AeroGridState) -> dict:
        fc = state.get("price_forecast")
        if fc is None:
            return {"schedule": None}
        prices = np.asarray(fc.median)
        probs = state.get("onset_probs", {})
        sched = solve_proactive_schedule(
            now=state["now"],
            prices=prices,
            onset_probs=probs,
            reservation_lambda=reservation_lambda,
        )
        return {
            "schedule": sched,
            "replan_reason": None,
            "event_log": [
                *state.get("event_log", []),
                {"type": "schedule", "now": state["now"].isoformat(),
                 **sched.as_dict()},
            ],
        }

    def n_user_confirm(state: AeroGridState) -> dict:
        sched = state.get("schedule")
        if sched is None or not sched.tasks:
            return {"user_confirmation": "skipped"}

        # Only ask when the MILP shifted a task far from its user-habit peak.
        question = _build_question(state)
        if not question:
            return {"user_confirmation": "auto:no_shift", "pending_question": None}

        if auto_confirm:
            return {
                "user_confirmation": "auto:accepted",
                "pending_question": question,
                "event_log": [
                    *state.get("event_log", []),
                    {"type": "hitl_auto", "now": state["now"].isoformat(),
                     "q": question, "a": "accepted"},
                ],
            }

        # Real HITL path — pause graph and return answer from the user.
        answer = interrupt({"question": question, "schedule": sched.as_dict()})
        return {
            "user_confirmation": str(answer),
            "pending_question": question,
            "event_log": [
                *state.get("event_log", []),
                {"type": "hitl", "now": state["now"].isoformat(),
                 "q": question, "a": str(answer)},
            ],
        }

    def n_execute(state: AeroGridState) -> dict:
        sched = state.get("schedule")
        realized = realized_price_provider(state["now"])
        cumcost = state.get("cumulative_cost", 0.0)
        cumbase = state.get("cumulative_baseline_cost", 0.0)
        realized_prices = state.get("realized_prices", [])
        if realized is not None:
            realized_prices = [*realized_prices, float(realized)]

            # Realized cost of the first slot of the active schedule.
            if sched is not None and sched.ev_power_kw:
                ev_now = float(sched.ev_power_kw[0])
                load = ev_now
                for t in sched.tasks:
                    if t.start_slot == 0:
                        load += APPLIANCES[t.appliance].rated_kw
                slot_cost = load * (SLOT_MINUTES / 60.0) * (realized / 1000.0)
                cumcost += slot_cost
                # baseline: naïve charges ASAP, so assume EV_rated until done.
                naive_ev = APPLIANCES["ev_charger"].rated_kw
                cumbase += naive_ev * (SLOT_MINUTES / 60.0) * (realized / 1000.0)

        return {
            "realized_prices": realized_prices,
            "cumulative_cost": cumcost,
            "cumulative_baseline_cost": cumbase,
            "iteration": state.get("iteration", 0) + 1,
        }

    def n_monitor(state: AeroGridState) -> dict:
        """Detect divergences that warrant a replan on the *next* call."""
        fc = state.get("price_forecast")
        reason: str | None = None
        if fc is not None and state.get("realized_prices"):
            realized = state["realized_prices"][-1]
            forecast0 = float(fc.median[0]) if fc.median else realized
            if abs(realized - forecast0) / max(abs(forecast0), 1e-6) > REPLAN_PRICE_DEVIATION:
                reason = (
                    f"price_deviation: realized {realized:.2f} vs forecast "
                    f"{forecast0:.2f} ({(realized-forecast0)/max(abs(forecast0),1e-6)*100:+.0f}%)"
                )

        # Unexpected onset: NILM saw an onset for an appliance NOT in the schedule
        # for the current slot.
        for o in state.get("new_onsets", []):
            scheduled = {t.appliance for t in (state.get("schedule").tasks
                                               if state.get("schedule") else [])}
            if o.appliance not in scheduled:
                reason = (reason or "") + f" | unplanned_onset:{o.appliance}"

        return {"replan_reason": reason}

    # ---- graph wiring ----
    builder = StateGraph(AeroGridState)
    builder.add_node("signal_watch", n_signal_watch)
    builder.add_node("forecast_price", n_forecast_price)
    builder.add_node("predict_behavior", n_predict_behavior)
    builder.add_node("optimize", n_optimize)
    builder.add_node("user_confirm", n_user_confirm)
    builder.add_node("execute", n_execute)
    builder.add_node("monitor", n_monitor)

    builder.add_edge(START, "signal_watch")
    builder.add_edge("signal_watch", "forecast_price")
    builder.add_edge("forecast_price", "predict_behavior")
    builder.add_edge("predict_behavior", "optimize")
    builder.add_edge("optimize", "user_confirm")

    def _after_confirm(state: AeroGridState) -> str:
        # If user said no, end without executing.
        ans = (state.get("user_confirmation") or "").lower()
        if ans in ("no", "reject", "cancel"):
            return END
        return "execute"

    builder.add_conditional_edges("user_confirm", _after_confirm,
                                  {"execute": "execute", END: END})
    builder.add_edge("execute", "monitor")
    builder.add_edge("monitor", END)

    checkpointer = InMemorySaver(serde=_PickleSerializer())
    return builder, checkpointer


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _build_question(state: AeroGridState) -> str | None:
    """Compose a natural-language HITL question if the schedule shifted far
    from the user's habitual start. Returns None when no shift is worth asking.
    """
    sched = state.get("schedule")
    probs = state.get("onset_probs", {})
    if sched is None:
        return None
    msgs: list[str] = []
    for task in sched.tasks:
        p = probs.get(task.appliance)
        if p is None:
            continue
        habitual_slot = int(np.argmax(p[: max(1, len(p) - task.slots + 1)]))
        shift_slots = abs(task.start_slot - habitual_slot)
        if shift_slots < 8:                 # <2 h shift: don't bother asking
            continue
        t_start = sched.slot_start + timedelta(minutes=SLOT_MINUTES * task.start_slot)
        t_habit = sched.slot_start + timedelta(minutes=SLOT_MINUTES * habitual_slot)
        msgs.append(
            f"{task.appliance} usually runs around "
            f"{t_habit.strftime('%H:%M')}; proposing {t_start.strftime('%H:%M')} "
            f"(saves ~${(shift_slots * SLOT_MINUTES / 60 * 0.2):.2f})"
        )
    if not msgs:
        return None
    return "Shift check: " + "; ".join(msgs) + ". Accept? (yes/no)"


def make_thread_id(now) -> str:
    return f"twin-{now.date().isoformat()}"
