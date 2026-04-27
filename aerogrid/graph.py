"""LangGraph orchestration for the outer (MPC) loop.

The inner 1 Hz sample loop lives in :mod:`aerogrid.sim.digital_twin` — it
runs the disaggregator, onset detector, and commit tracker every sample,
and only invokes this graph when :class:`aerogrid.triggers.TriggerManager`
fires. Keeping the graph to the "slow path" keeps it small, testable, and
fast enough to run many times per simulated hour.

Nodes (in order):
  forecast_price    — short-horizon price quantile forecast
  predict_behavior  — per-appliance onset probabilities over the horizon
  optimize          — receding-horizon MILP (with committed-task pinning
                      and deadline guard)
  hitl_gate         — decides AUTO vs ASK via :func:`aerogrid.hitl_policy.decide`.
                      On ASK, uses ``interrupt()`` so the caller can resume
                      with a user answer via ``Command(resume=...)``.
  commit_plan       — persists the confirmed plan into state; the caller
                      adopts it into the ``CommitTracker``.
"""
from __future__ import annotations

import logging
import pickle
from datetime import timedelta

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from aerogrid.behavioral_predictor import BehavioralPredictor
from aerogrid.config import SHORT_HORIZON_SLOTS
from aerogrid.hitl_policy import decide as hitl_decide
from aerogrid.optimizer import solve_receding_horizon
from aerogrid.price_oracle import PriceOracle
from aerogrid.state import AeroGridState
from aerogrid.triggers import time_to_deadline_hours
from aerogrid.types import HITLDecision

logger = logging.getLogger(__name__)


class _PickleSerializer:
    """Pickle-based serializer for LangGraph checkpoints.

    The default msgpack-based serializer can't handle our frozen dataclasses
    + numpy arrays out of the box. Pickle does; this is safe because the
    checkpointer only ever sees trusted, in-process state.
    """
    def dumps_typed(self, obj):
        """Serialize ``obj`` to a ``(type_tag, bytes)`` tuple using pickle."""
        return "pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def loads_typed(self, data):
        """Deserialize a ``(type_tag, bytes)`` tuple produced by ``dumps_typed``."""
        type_, raw = data
        if type_ != "pickle":
            raise ValueError(f"unexpected serializer type: {type_}")
        return pickle.loads(raw)


def build_graph(
    price_oracle: PriceOracle,
    predictor: BehavioralPredictor,
    price_history_provider,                  # callable(now) -> pd.DataFrame
    *,
    horizon_slots: int = SHORT_HORIZON_SLOTS,
    auto_confirm: bool = True,
):
    """Wire the five-node outer-loop StateGraph and return ``(builder, checkpointer)``.

    The compiled graph runs: ``forecast_price → predict_behavior → optimize →
    hitl_gate → (commit_plan | END)``.

    Args:
        price_oracle: Forecaster implementing :class:`PriceOracle`.
        predictor: Onset predictor implementing :class:`BehavioralPredictor`.
        price_history_provider: Callable ``(now: datetime) -> pd.DataFrame``
            returning the price context for the oracle (past rows only).
        horizon_slots: Number of 15-min slots in the receding horizon.
        auto_confirm: When ``True`` the HITL gate auto-accepts all plans,
            bypassing the LangGraph ``interrupt()`` mechanism.  Set to
            ``False`` in production to allow real user confirmation.

    Returns:
        A ``(StateGraph builder, InMemorySaver checkpointer)`` tuple.  Call
        ``builder.compile(checkpointer=checkpointer)`` to get the runnable
        graph.
    """

    def n_forecast_price(state: AeroGridState) -> dict:
        """Produce a short-horizon price quantile forecast from the price oracle."""
        now = state["now"]
        logger.info("graph.n_forecast_price: now=%s oracle=%s", now.isoformat(), type(price_oracle).__name__)
        ctx = price_history_provider(now)
        logger.debug("graph.n_forecast_price: price context rows=%d", len(ctx))
        fc = price_oracle.get_15min_forecast(now, ctx, horizon_slots)
        logger.info(
            "graph.n_forecast_price: source=%s median[0]=%.2f horizon=%d",
            fc.source, fc.median[0] if fc.median else float("nan"), horizon_slots,
        )
        return {"price_forecast": fc}

    def n_predict_behavior(state: AeroGridState) -> dict:
        """Predict per-appliance onset probabilities for the upcoming horizon."""
        now = state["now"]
        logger.info(
            "graph.n_predict_behavior: now=%s predictor=%s horizon=%d",
            now.isoformat(), type(predictor).__name__, horizon_slots,
        )
        probs = predictor.predict_all(now, horizon_slots)
        for app, p in probs.items():
            logger.debug(
                "graph.n_predict_behavior: %s max_prob=%.4f mean_prob=%.4f",
                app, float(p.max()), float(p.mean()),
            )
        return {"onset_probs": probs}

    def n_optimize(state: AeroGridState) -> dict:
        """Solve the receding-horizon MILP and append the result to the event log."""
        now = state["now"]
        fc = state.get("price_forecast")
        if fc is None:
            logger.warning("graph.n_optimize: no price_forecast in state — skipping optimization")
            return {"current_plan": None}
        prices = np.asarray(fc.median, dtype=float)
        probs = state.get("onset_probs", {})
        remaining_ev_kwh = float(state.get("remaining_ev_kwh", 0.0))
        committed = state.get("committed_tasks") or []
        logger.info(
            "graph.n_optimize: now=%s remaining_ev=%.2fkWh committed=%s",
            now.isoformat(), remaining_ev_kwh, [t.appliance for t in committed],
        )
        hours = time_to_deadline_hours(now)
        sched = solve_receding_horizon(
            now=now,
            prices=prices,
            onset_probs=probs,
            remaining_ev_kwh=remaining_ev_kwh,
            time_to_deadline_h=hours,
            committed_tasks=committed,
            horizon_slots=horizon_slots,
        )
        logger.info(
            "graph.n_optimize: plan solver=%s expected_cost=%.4f tasks=%s",
            sched.solver_status,
            sched.expected_cost,
            [t.appliance for t in sched.tasks if not t.committed],
        )
        return {
            "current_plan": sched,
            "event_log": [
                *state.get("event_log", []),
                {
                    "type": "optimize",
                    "now": now.isoformat(),
                    **sched.as_dict(),
                },
            ],
        }

    def n_hitl_gate(state: AeroGridState) -> dict:
        """Run the HITL policy; interrupt the graph if user confirmation is required."""
        old_plan = state.get("previous_plan")
        new_plan = state.get("current_plan")
        logger.info(
            "graph.n_hitl_gate: now=%s auto_confirm=%s",
            state["now"].isoformat(), auto_confirm,
        )
        decision = hitl_decide(old_plan, new_plan)
        logger.info(
            "graph.n_hitl_gate: decision action=%s reason=%r",
            decision.action, decision.reason,
        )

        updates: dict = {
            "hitl_decision": decision,
            "pending_question": decision.question if decision.action == "ask" else None,
        }
        updates["event_log"] = [
            *state.get("event_log", []),
            {
                "type": "hitl",
                "now": state["now"].isoformat(),
                **decision.as_dict(),
            },
        ]

        if decision.action == "auto":
            updates["user_confirmation"] = f"auto:{decision.reason}"
            return updates

        if auto_confirm:
            logger.info("graph.n_hitl_gate: ASK suppressed by auto_confirm=True")
            updates["user_confirmation"] = "auto:bypass"
            return updates

        # Real HITL path — pause graph, resume with a user answer.
        logger.info("graph.n_hitl_gate: interrupting graph for user input — question=%r", decision.question)
        answer = interrupt(
            {
                "question": decision.question,
                "reason": decision.reason,
                "new_plan": new_plan.as_dict() if new_plan else None,
            }
        )
        logger.info("graph.n_hitl_gate: user answered=%r", answer)
        updates["user_confirmation"] = str(answer)
        return updates

    def n_commit_plan(state: AeroGridState) -> dict:
        """Persist the new plan into state so the caller can adopt it."""
        new_plan = state.get("current_plan")
        logger.info(
            "graph.n_commit_plan: now=%s committing plan tasks=%s",
            state["now"].isoformat(),
            [t.appliance for t in new_plan.tasks] if new_plan else [],
        )
        return {
            "current_plan": new_plan,
            "previous_plan": new_plan,
            "last_replan_at": state["now"],
        }

    def _after_hitl(state: AeroGridState) -> str:
        """Conditional edge: route to ``commit_plan`` unless the user rejected."""
        ans = (state.get("user_confirmation") or "").lower()
        if ans in ("no", "reject", "cancel"):
            logger.info("graph._after_hitl: user rejected plan — routing to END")
            return END
        logger.debug("graph._after_hitl: user_confirmation=%r → commit_plan", ans)
        return "commit_plan"

    builder = StateGraph(AeroGridState)
    builder.add_node("forecast_price", n_forecast_price)
    builder.add_node("predict_behavior", n_predict_behavior)
    builder.add_node("optimize", n_optimize)
    builder.add_node("hitl_gate", n_hitl_gate)
    builder.add_node("commit_plan", n_commit_plan)

    builder.add_edge(START, "forecast_price")
    builder.add_edge("forecast_price", "predict_behavior")
    builder.add_edge("predict_behavior", "optimize")
    builder.add_edge("optimize", "hitl_gate")
    builder.add_conditional_edges(
        "hitl_gate", _after_hitl, {"commit_plan": "commit_plan", END: END}
    )
    builder.add_edge("commit_plan", END)

    checkpointer = InMemorySaver(serde=_PickleSerializer())
    logger.info(
        "build_graph: graph built horizon=%d auto_confirm=%s oracle=%s predictor=%s",
        horizon_slots, auto_confirm,
        type(price_oracle).__name__, type(predictor).__name__,
    )
    return builder, checkpointer


def make_thread_id(now) -> str:
    """Generate a deterministic LangGraph thread ID for the given simulation timestamp.

    The ID encodes the date and hour so each simulated hour gets its own
    checkpoint namespace, preventing cross-hour state bleed during long runs.
    """
    return f"twin-{now.date().isoformat()}-{now.hour:02d}"


__all__ = ["build_graph", "make_thread_id"]
