"""LangGraph orchestration for the outer (MPC) loop.

The inner 1 Hz sample loop lives in :mod:`aerogrid.sim.digital_twin` — it
runs the commit tracker every sample, and only invokes this graph when
:class:`aerogrid.triggers.TriggerManager` fires. Keeping the graph to the
"slow path" keeps it small, testable, and fast enough to run many times
per simulated hour.

Nodes (in order):
  forecast_price       — short-horizon price quantile forecast
  optimize             — receding-horizon LP: continuous EV + heater plus
                         per-window heater energy deadlines, with
                         committed-task pinning and soft-slack fallback.
  propose_reschedule   — when the trigger was a ``new_onset`` for an
                         event-driven appliance (dishwasher / washing
                         machine), search forward up to
                         ``HITL_RESCHEDULE_WINDOW_HOURS`` for a cheaper
                         start. Emits a ``RescheduleProposal`` if savings
                         exceed ``HITL_RESCHEDULE_MIN_SAVINGS_EUR``.
  hitl_gate            — decides AUTO vs ASK using both the plan-diff
                         (``hitl_policy.decide``) and the reschedule
                         proposal (``hitl_policy.decide_reschedule``).
                         On ASK, uses ``interrupt()`` so the caller can
                         resume with a user answer via
                         ``Command(resume=...)``.
  commit_plan          — persists the confirmed plan into state; the caller
                         adopts it into the ``CommitTracker`` (and pins
                         any accepted reschedule).
"""
from __future__ import annotations

import logging
import pickle
from datetime import timedelta

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from aerogrid.config import (
    APPLIANCES,
    HITL_RESCHEDULE_MIN_SAVINGS_EUR,
    HITL_RESCHEDULE_WINDOW_HOURS,
    SHORT_HORIZON_SLOTS,
    SLOT_MINUTES,
)
from aerogrid.hitl_policy import decide as hitl_decide
from aerogrid.hitl_policy import decide_reschedule as hitl_decide_reschedule
from aerogrid.optimizer import solve_receding_horizon
from aerogrid.price_oracle import PriceOracle
from aerogrid.state import AeroGridState
from aerogrid.triggers import time_to_deadline_hours
from aerogrid.types import (
    ApplianceOnset,
    HITLDecision,
    PendingCycle,
    RescheduleProposal,
)

logger = logging.getLogger(__name__)


_SLOT_HOURS = SLOT_MINUTES / 60.0
_PER_SLOT = _SLOT_HOURS / 1000.0


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


def _propose_for_onset(
    appliance: str,
    onset_at,
    prices: np.ndarray,
    cycle_slots: int,
    rated_kw: float,
    horizon_slots: int,
    window_hours: float = HITL_RESCHEDULE_WINDOW_HOURS,
) -> RescheduleProposal | None:
    """Price-only reschedule scoring (legacy, retained for benchmarking).

    Computes the cycle's *isolated* cost ``Σ_t π[t] · P_a · κ`` for each
    candidate start slot in ``[0, window_slots]`` and returns the cheapest.
    This helper does **not** know about the EV/heater plan, the house power
    cap, or any other committed task — it is purely the price-vs-shift
    analysis. The runtime path (``n_optimize`` → ``n_propose_reschedule``)
    no longer uses it because the joint MIP in
    :func:`aerogrid.optimizer.solve_receding_horizon` always produces a
    cap-feasible, plan-aware shift.

    Kept for the optimizer notebook (Scenario L) where we juxtapose this
    naive shift with the joint MIP's choice to make the trade-off visible.

    Args:
        appliance: Cycle appliance name.
        onset_at: Wall-clock UTC datetime of the user-detected onset.
        prices: Length-``horizon_slots`` price array (currency/MWh) starting
            at the slot containing ``onset_at``.
        cycle_slots: Number of 15-min slots the cycle occupies.
        rated_kw: Cycle's rated power (kW).
        horizon_slots: Number of 15-min slots in the optimiser's horizon.
        window_hours: Maximum forward shift considered (default 2 h).

    Returns:
        ``RescheduleProposal`` (which will collapse to an "AUTO run-now"
        decision in :func:`hitl_policy.decide_reschedule` when the savings
        are below threshold), or ``None`` if no candidate fits.
    """
    if cycle_slots <= 0 or cycle_slots > horizon_slots:
        return None
    window_slots = int(round(window_hours * 60.0 / SLOT_MINUTES))
    last_start = min(window_slots, horizon_slots - cycle_slots)
    if last_start < 0:
        return None

    cost_at: dict[int, float] = {}
    for s in range(0, last_start + 1):
        c = float(prices[s : s + cycle_slots].sum() * rated_kw * _PER_SLOT)
        cost_at[s] = c

    cost_now = cost_at.get(0, 0.0)
    best_slot = min(cost_at, key=cost_at.get)
    best_cost = cost_at[best_slot]
    proposed = onset_at + timedelta(minutes=SLOT_MINUTES * best_slot)
    return RescheduleProposal(
        appliance=appliance,
        onset_at=onset_at,
        proposed_start_at=proposed,
        cycle_slots=cycle_slots,
        rated_kw=rated_kw,
        cost_now_eur=cost_now,
        cost_proposed_eur=best_cost,
    )


def _pending_cycles_from_onsets(
    new_onsets: list[ApplianceOnset] | None,
    committed_apps: set[str],
    horizon_slots: int,
    window_hours: float = HITL_RESCHEDULE_WINDOW_HOURS,
) -> list[PendingCycle]:
    """Convert a list of fresh user-onset events into ``PendingCycle`` inputs.

    Each onset is filtered (must be a known event-driven cycle appliance,
    not already committed, and the cycle must fit in the horizon) and then
    wrapped in a :class:`~aerogrid.types.PendingCycle` whose allowed start
    window is ``[0, window_slots]`` slots from the onset.

    Args:
        new_onsets: User onsets emitted this trigger-evaluation tick.
        committed_apps: Names of appliances already pinned by the
            commit tracker — they are skipped to avoid double-counting.
        horizon_slots: Optimiser horizon (used to clip ``latest_start``).
        window_hours: User-allowed forward shift (default
            ``HITL_RESCHEDULE_WINDOW_HOURS``).

    Returns:
        Zero or more pending cycles, deduplicated by appliance name (the
        first onset wins if the same appliance appears twice).
    """
    out: list[PendingCycle] = []
    seen: set[str] = set()
    if not new_onsets:
        return out
    window_slots = int(round(window_hours * 60.0 / SLOT_MINUTES))
    for onset in new_onsets:
        if onset.appliance in committed_apps or onset.appliance in seen:
            continue
        spec = APPLIANCES.get(onset.appliance)
        if spec is None or spec.cycle_slots <= 0:
            continue
        if spec.cycle_slots > horizon_slots:
            continue
        last_start = min(window_slots, horizon_slots - spec.cycle_slots)
        if last_start < 0:
            continue
        out.append(PendingCycle(
            appliance=onset.appliance,
            cycle_slots=int(spec.cycle_slots),
            rated_kw=float(spec.rated_kw),
            earliest_start_slot=0,
            latest_start_slot=int(last_start),
        ))
        seen.add(onset.appliance)
    return out


def build_graph(
    price_oracle: PriceOracle,
    price_history_provider,                  # callable(now) -> pd.DataFrame
    *,
    horizon_slots: int = SHORT_HORIZON_SLOTS,
    auto_confirm: bool = True,
    auto_responses: dict[str, str] | None = None,
):
    """Wire the outer-loop StateGraph and return ``(builder, checkpointer)``.

    The compiled graph runs::

        forecast_price → optimize → propose_reschedule
                       → hitl_gate → (commit_plan | END)

    Args:
        price_oracle: Forecaster implementing :class:`PriceOracle`.
        price_history_provider: Callable ``(now: datetime) -> pd.DataFrame``
            returning the price context for the oracle (past rows only).
        horizon_slots: Number of 15-min slots in the receding horizon.
        auto_confirm: When ``True`` the HITL gate auto-resolves all questions.
            For reschedule proposals, the resolution is per-appliance via
            ``auto_responses`` (defaults to
            :data:`aerogrid.config.HITL_AUTO_RESPONSES`). For other plan-level
            asks, the resolution is a blanket "yes". Set ``False`` in
            production for real ``interrupt()`` semantics.
        auto_responses: Per-appliance auto-reply override
            (``{"dishwasher": "accept", "washing_machine": "decline"}``).
            Used only when ``auto_confirm=True``.

    Returns:
        A ``(StateGraph builder, InMemorySaver checkpointer)`` tuple.  Call
        ``builder.compile(checkpointer=checkpointer)`` to get the runnable
        graph.
    """
    from aerogrid.config import HITL_AUTO_RESPONSES
    auto_responses = auto_responses if auto_responses is not None else HITL_AUTO_RESPONSES

    def n_forecast_price(state: AeroGridState) -> dict:
        """Produce a short-horizon price quantile forecast from the price oracle."""
        now = state["now"]
        logger.info(
            "graph.n_forecast_price: now=%s oracle=%s",
            now.isoformat(), type(price_oracle).__name__,
        )
        ctx = price_history_provider(now)
        logger.debug("graph.n_forecast_price: price context rows=%d", len(ctx))
        fc = price_oracle.get_15min_forecast(now, ctx, horizon_slots)
        logger.info(
            "graph.n_forecast_price: source=%s median[0]=%.2f horizon=%d",
            fc.source,
            fc.median[0] if fc.median else float("nan"),
            horizon_slots,
        )
        return {"price_forecast": fc}

    def n_optimize(state: AeroGridState) -> dict:
        """Solve the receding-horizon program and append the result to the event log.

        When ``state["new_onsets"]`` contains user-triggered cycle starts
        (dishwasher / washing machine), they are converted to
        :class:`~aerogrid.types.PendingCycle` entries and included in the
        joint solve. The optimiser becomes a small MIP that picks every
        cycle's start slot together with the EV/heater plan, naturally
        respecting the house power cap and any deadline pressure. When
        there are no fresh onsets the program stays a pure LP.
        """
        now = state["now"]
        fc = state.get("price_forecast")
        if fc is None:
            logger.warning(
                "graph.n_optimize: no price_forecast in state — skipping optimization",
            )
            return {"current_plan": None}
        prices = np.asarray(fc.median, dtype=float)
        remaining_ev_kwh = float(state.get("remaining_ev_kwh", 0.0))
        remaining_heater = state.get("remaining_heater_kwh_by_window") or None
        committed = state.get("committed_tasks") or []
        committed_apps = {t.appliance for t in committed}
        pending_cycles = _pending_cycles_from_onsets(
            state.get("new_onsets"),
            committed_apps=committed_apps,
            horizon_slots=horizon_slots,
        )
        logger.info(
            "graph.n_optimize: now=%s remaining_ev=%.2fkWh remaining_heater=%s "
            "committed=%s pending=%s",
            now.isoformat(), remaining_ev_kwh, remaining_heater,
            [t.appliance for t in committed],
            [pc.appliance for pc in pending_cycles],
        )
        hours = time_to_deadline_hours(now)
        sched = solve_receding_horizon(
            now=now,
            prices=prices,
            remaining_ev_kwh=remaining_ev_kwh,
            remaining_heater_kwh_by_window=remaining_heater,
            time_to_deadline_h=hours,
            committed_tasks=committed,
            pending_cycles=pending_cycles,
            horizon_slots=horizon_slots,
        )
        logger.info(
            "graph.n_optimize: plan solver=%s expected_cost=%.4f tasks=%s cycle_starts=%s",
            sched.solver_status,
            sched.expected_cost,
            [t.appliance for t in sched.tasks if not t.committed],
            sched.cycle_starts,
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

    def n_propose_reschedule(state: AeroGridState) -> dict:
        """Build a reschedule proposal from the joint optimiser's chosen slot.

        For each user-triggered onset, the optimiser placed the cycle at
        ``schedule.cycle_starts[appliance]`` jointly with the EV / heater
        plan and under the house cap. Here we:

        1. Pick the appliance named in the trigger (``trigger.detail``).
        2. Read its chosen start slot from the plan.
        3. If the chosen slot is the onset slot itself, no proposal is
           generated (running now is already optimal).
        4. Otherwise, re-solve the MIP with that cycle pinned at slot 0
           — that gives the *true plan-level* cost of "run now",
           accounting for whatever EV/heater rebalancing it would force.
           ``savings = cost_now − cost_proposed`` is therefore plan-level
           honest, including cap interactions.

        The returned proposal is then evaluated by the HITL policy.
        """
        trigger = state.get("replan_trigger")
        if trigger is None or trigger.kind != "new_onset":
            return {"pending_reschedule": None}
        appliance = (trigger.detail or "").strip()
        spec = APPLIANCES.get(appliance)
        if spec is None or spec.cycle_slots <= 0:
            return {"pending_reschedule": None}
        committed_apps = {t.appliance for t in (state.get("committed_tasks") or [])}
        if appliance in committed_apps:
            return {"pending_reschedule": None}

        plan = state.get("current_plan")
        if plan is None or appliance not in plan.cycle_starts:
            return {"pending_reschedule": None}
        proposed_slot = int(plan.cycle_starts[appliance])
        if proposed_slot <= 0:
            # The MIP already chose to run now — nothing to ask about.
            logger.info(
                "graph.n_propose_reschedule: %s placed at slot 0 by joint MIP "
                "→ no proposal", appliance,
            )
            return {"pending_reschedule": None}

        # Plan-level cost for "run now": pin the cycle at slot 0 and
        # re-solve. The EV/heater plan may differ from the optimal
        # (free) solve because the cap constraint changes — that's the
        # whole point: this gives the true cost of declining the shift.
        fc = state.get("price_forecast")
        if fc is None or not fc.median:
            return {"pending_reschedule": None}
        prices = np.asarray(fc.median, dtype=float)
        remaining_ev_kwh = float(state.get("remaining_ev_kwh", 0.0))
        remaining_heater = state.get("remaining_heater_kwh_by_window") or None
        committed = state.get("committed_tasks") or []
        # Pin the cycle at slot 0 by passing earliest=latest=0.
        pinned_now = PendingCycle(
            appliance=appliance,
            cycle_slots=int(spec.cycle_slots),
            rated_kw=float(spec.rated_kw),
            earliest_start_slot=0,
            latest_start_slot=0,
        )
        # Other pending cycles from this trigger keep their full window —
        # we only constrain the appliance under question.
        other_pending = [
            pc for pc in _pending_cycles_from_onsets(
                state.get("new_onsets"),
                committed_apps=committed_apps,
                horizon_slots=horizon_slots,
            )
            if pc.appliance != appliance
        ]
        try:
            sched_now = solve_receding_horizon(
                now=state["now"],
                prices=prices,
                remaining_ev_kwh=remaining_ev_kwh,
                remaining_heater_kwh_by_window=remaining_heater,
                time_to_deadline_h=time_to_deadline_hours(state["now"]),
                committed_tasks=committed,
                pending_cycles=[pinned_now, *other_pending],
                horizon_slots=horizon_slots,
            )
        except Exception as e:                       # noqa: BLE001
            logger.warning(
                "graph.n_propose_reschedule: pinned-now solve failed (%r) — "
                "falling back to price-only cost", e,
            )
            sched_now = None

        decline_plan: object | None = None
        if sched_now is not None and sched_now.solver_status in (
            "optimal", "optimal_inaccurate"
        ):
            cost_now_eur = float(sched_now.expected_cost)
            decline_plan = sched_now
        else:
            # Conservative fallback: use the price-only run-now cost. There
            # is no decline plan to swap in — the next replan will fix any
            # cap excess once the cycle is actually committed.
            cost_now_eur = float(
                prices[: spec.cycle_slots].sum() * spec.rated_kw * _PER_SLOT
            )

        cost_proposed_eur = float(plan.expected_cost)
        proposed_at = state["now"] + timedelta(minutes=SLOT_MINUTES * proposed_slot)
        proposal = RescheduleProposal(
            appliance=appliance,
            onset_at=state["now"],
            proposed_start_at=proposed_at,
            cycle_slots=int(spec.cycle_slots),
            rated_kw=float(spec.rated_kw),
            cost_now_eur=cost_now_eur,
            cost_proposed_eur=cost_proposed_eur,
        )
        logger.info(
            "graph.n_propose_reschedule: %s shift %.0fmin saves €%.3f "
            "(plan-level cost_now=%.4f cost_best=%.4f)",
            appliance, proposal.shift_minutes, proposal.savings_eur,
            proposal.cost_now_eur, proposal.cost_proposed_eur,
        )
        return {"pending_reschedule": proposal, "decline_plan": decline_plan}

    def n_hitl_gate(state: AeroGridState) -> dict:
        """Run the HITL policy.

        Decision priority:

        1. If a reschedule proposal exists, evaluate that first. It can
           short-circuit straight to ASK (with the savings phrasing) or
           AUTO (when savings are too small to bother the user).
        2. Otherwise, fall back to the plan-level diff between
           ``previous_plan`` and ``current_plan``.

        On ASK, ``interrupt()`` is called and the caller (digital twin or
        notebook) resumes with the user's answer. ``auto_confirm`` short-
        circuits this for simulation.
        """
        old_plan = state.get("previous_plan")
        new_plan = state.get("current_plan")
        proposal = state.get("pending_reschedule")
        logger.info(
            "graph.n_hitl_gate: now=%s auto_confirm=%s reschedule=%s",
            state["now"].isoformat(), auto_confirm,
            proposal.appliance if proposal else "None",
        )

        if proposal is not None:
            decision = hitl_decide_reschedule(proposal)
        else:
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
                "reschedule": proposal.as_dict() if proposal else None,
            },
        ]

        if decision.action == "auto":
            updates["user_confirmation"] = _auto_resolution(decision, proposal)
            return updates

        if auto_confirm:
            answer = _simulated_answer(proposal, auto_responses)
            logger.info(
                "graph.n_hitl_gate: ASK suppressed by auto_confirm=True → answer=%r",
                answer,
            )
            updates["user_confirmation"] = answer
            return updates

        # Real HITL path — pause graph, resume with a user answer.
        logger.info(
            "graph.n_hitl_gate: interrupting graph for user input — question=%r",
            decision.question,
        )
        answer = interrupt(
            {
                "question": decision.question,
                "reason": decision.reason,
                "new_plan": new_plan.as_dict() if new_plan else None,
                "reschedule": proposal.as_dict() if proposal else None,
            }
        )
        logger.info("graph.n_hitl_gate: user answered=%r", answer)
        updates["user_confirmation"] = str(answer)
        return updates

    def n_commit_plan(state: AeroGridState) -> dict:
        """Persist the new plan into state so the caller can adopt it.

        When the HITL gate produced a reschedule proposal and the user
        ended up *declining* it, the EV/heater plan from ``current_plan``
        was optimised assuming the cycle would shift — running the cycle
        at slot 0 instead can therefore breach the house cap. To stay
        cap-feasible we swap in ``decline_plan``, which is the same
        joint MIP solve but with the cycle pinned at slot 0 (computed
        in ``n_propose_reschedule``).

        Also echoes the reschedule proposal into the committed-plan
        state so the digital twin can call
        ``CommitTracker.adopt_cycle_start`` for the chosen start time.
        """
        new_plan = state.get("current_plan")
        proposal = state.get("pending_reschedule")
        decline_plan = state.get("decline_plan")
        ans = (state.get("user_confirmation") or "").lower()
        is_decline = ans in ("decline", "no", "reject", "cancel")

        if proposal is not None and is_decline and decline_plan is not None:
            logger.info(
                "graph.n_commit_plan: %s declined → swapping current_plan to "
                "decline_plan (cycle pinned at slot 0)",
                proposal.appliance,
            )
            new_plan = decline_plan

        logger.info(
            "graph.n_commit_plan: now=%s committing plan tasks=%s reschedule=%s "
            "user=%r",
            state["now"].isoformat(),
            [t.appliance for t in new_plan.tasks] if new_plan else [],
            proposal.appliance if proposal else "None",
            ans or None,
        )
        return {
            "current_plan": new_plan,
            "previous_plan": new_plan,
            "last_replan_at": state["now"],
            "pending_reschedule": proposal,
            # Clear decline_plan so it doesn't leak into the next replan.
            "decline_plan": None,
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
    builder.add_node("optimize", n_optimize)
    builder.add_node("propose_reschedule", n_propose_reschedule)
    builder.add_node("hitl_gate", n_hitl_gate)
    builder.add_node("commit_plan", n_commit_plan)

    builder.add_edge(START, "forecast_price")
    builder.add_edge("forecast_price", "optimize")
    builder.add_edge("optimize", "propose_reschedule")
    builder.add_edge("propose_reschedule", "hitl_gate")
    builder.add_conditional_edges(
        "hitl_gate", _after_hitl, {"commit_plan": "commit_plan", END: END}
    )
    builder.add_edge("commit_plan", END)

    checkpointer = InMemorySaver(serde=_PickleSerializer())
    logger.info(
        "build_graph: graph built horizon=%d auto_confirm=%s oracle=%s",
        horizon_slots, auto_confirm, type(price_oracle).__name__,
    )
    return builder, checkpointer


def _auto_resolution(decision: HITLDecision, proposal: RescheduleProposal | None) -> str:
    """Map an AUTO decision to the same answer string vocabulary as a user reply.

    For reschedule auto-decisions the answer is ``"decline"`` (run-now is
    the natural meaning of an auto-AUTO outcome on a reschedule proposal).
    For plan-level auto-decisions we fall back to the existing ``"auto:..."``
    convention, which the simple yes/no router treats as approval.
    """
    if proposal is not None and decision.action == "auto":
        return "decline"
    return f"auto:{decision.reason}"


def _simulated_answer(
    proposal: RescheduleProposal | None,
    auto_responses: dict[str, str],
) -> str:
    """Return the simulated user's answer when ``auto_confirm=True`` is set.

    For a reschedule proposal, look the appliance up in ``auto_responses``
    (default ``HITL_AUTO_RESPONSES``: dishwasher → ``"accept"``, washing
    machine → ``"decline"``). For any other ASK we approve with ``"yes"``,
    matching the previous default behaviour.
    """
    if proposal is None:
        return "yes"
    return auto_responses.get(proposal.appliance, "yes")


def make_thread_id(now) -> str:
    """Generate a deterministic LangGraph thread ID for the given simulation timestamp.

    The ID encodes the date and hour so each simulated hour gets its own
    checkpoint namespace, preventing cross-hour state bleed during long runs.
    """
    return f"twin-{now.date().isoformat()}-{now.hour:02d}"


__all__ = ["build_graph", "make_thread_id"]
