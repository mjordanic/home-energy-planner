# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

AeroGrid is a research **demo** (not production): a 1 Hz streaming agent that shifts deferrable household loads (EV charger, hot-water heater, dishwasher, washing machine) into cheap price slots, using real SMARD DE-LU wholesale prices and a receding-horizon LP/MIP solver inside a LangGraph loop. `README.md` is the canonical deep reference — read it for the optimizer math, output schemas, and the full list of known gaps.

## Commands

All Python runs through `uv` (see AGENTS.md — never `pip`/`python -m`/`poetry`):

```bash
uv sync --extra dev                      # install (add --extra forecast for Chronos, --extra eu for ENTSO-E)
uv run python scripts/fetch_smard_prices.py   # fetch real DE-LU prices (no API key; hard-fails, no synthetic fallback)
uv run pytest -q                         # full test suite
uv run pytest tests/test_optimizer.py -q # one test file
uv run pytest -k power_cap -q            # one test by name pattern
uv run python -m aerogrid.sim.digital_twin              # full ~16-day streaming simulation
uv run python -m aerogrid.sim.digital_twin --hours 8 --horizon-hours 6 --no-log-file  # fast smoke run
uv run jupyter lab notebooks/            # demo notebooks (03 price oracle, 05 optimizer, 06 end-to-end)
```

`pytest` is configured with `asyncio_mode = "auto"` (graph nodes are async — no need to mark coroutine tests). Simulation outputs land in `data/cache/` (`slot_log.parquet`, `event_log.parquet`, `run_log.jsonl`).

## Architecture

The pieces below only make sense together — tracing a single replan touches `digital_twin → strategies → triggers → graph → optimizer → commit`.

**Digital twin owns the environment, not the policy.** `sim/digital_twin.py` owns the simulation world only — the price feed (`sim/price_server.py`), the 1 Hz clock + onset injection (`sim/streamer.py`), and the `APPLIANCE_ONSETS` list in `config.py`. It feeds the *same* inputs to N independent strategies running in parallel and collects their outputs, so strategies can be compared fairly against identical realized prices. **All scheduling logic lives inside the strategies**, never in the twin.

**Two strategies, deliberately asymmetric** (`sim/strategies.py`): `BaselineStrategy` is parameter-free, charges ASAP, owns no oracle/graph/tracker. `OptimizerStrategy` owns its *own* price oracle, LangGraph, `CommitTracker`, and `TriggerManager` — so two optimizer instances with different oracles can run in one simulation. This **per-strategy isolation** is the core design constraint: don't introduce shared mutable scheduling state across strategies.

**Cross-strategy onset gating** is the one thing the twin does coordinate: an onset is suppressed if *any* strategy still has that appliance running, preventing a phantom second cycle when strategies disagree on timing.

**The optimizer agent loop** is a LangGraph outer loop (`graph.py`, schema in `state.py`): `forecast_price → optimize → propose_reschedule → hitl_gate → commit_plan`. It is **event-driven, not clock-driven** (`triggers.py`): replans fire on new onset, ≥25% price surprise, EV deadline slip, or a 15-min periodic resync safety net — with a 30 s cooldown to prevent thrashing. HITL gating (`hitl_policy.py`) is pure AUTO/ASK decision functions; the commit step (`commit.py`) tracks remaining EV/heater kWh and adopts HITL outcomes.

**The optimizer itself** (`optimizer.py` → `solve_receding_horizon()`) is a pure LP in the common case, collapsing to a small MIP only when `pending_cycles` await a HITL decision (HiGHS via CVXPY). Energy constraints are **soft slacks** (penalty 1000×) so the LP never goes infeasible — missed energy shows up as non-zero slack, not a solver error. If every solve fails it returns a deterministic ASAP fallback plan.

## Gotchas

- **Forecast vs realized cost.** `Schedule.expected_cost` and `baseline_cost` are both computed from the price *forecast* (two hypothetical plans). Realized cost is accumulated slot-by-slot in `cumulative_cost` from loads that physically ran. Don't conflate them.
- **No synthetic price fallback.** The SMARD fetcher raises `FetchError` on any failure by design (see the [[feedback_real_data]] memory). The one exception is the optional ENTSO-E EU path, which has a synthetic fallback when `ENTSOE_API_KEY` is unset.
- **Deliberately out of scope** (removed, do not re-add without discussion): NILM disaggregation, synthetic household traces, behavioural onset prediction. Onsets are listed manually in `config.APPLIANCE_ONSETS`.
- **Domain docs.** Per AGENTS.md this is single-context: canonical glossary in `CONTEXT.md` and decisions in `docs/adr/` at the repo root (created lazily — may not exist yet).
