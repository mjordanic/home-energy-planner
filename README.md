# AeroGrid — Streaming Home Energy Optimiser (Demo)

> **This is a research demo, not production software.**  
> It shows one end-to-end approach: real market prices → LP/MIP optimiser → LangGraph agent → human-in-the-loop HITL. Every component works, but most have known gaps listed at the bottom of this file.

AeroGrid is a 1 Hz streaming agent that shifts deferrable household loads — EV charger, hot-water heater, dishwasher, washing machine — into cheap price slots, using real SMARD DE-LU wholesale prices and a receding-horizon LP/MIP solver inside a LangGraph loop.

---

## Architecture

![LangGraph node structure](docs/langgraph_structure.png)

The **digital twin** is a thin orchestrator. It streams real DE-LU prices and a manually-listed sequence of appliance onsets into **N independent strategy agents** that run in parallel against the same inputs.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Digital Twin — owns ONLY the simulation environment                  │
│   Streamer.iter_samples()  ─▶  Sample(t, realized_price)             │
│   PriceServer.realized()   ─▶  realized €/MWh at slot boundaries     │
│   APPLIANCE_ONSETS list    ─▶  user-driven cycle starts              │
│                                                                      │
│   Per 1 Hz sample:                                                   │
│     1. pull onsets due now                                           │
│     2. cross-strategy gating (suppress if appliance still running)   │
│     3. tick every strategy with the same (sample, gated_onsets)      │
│     4. at slot boundaries: collect SlotRecord → wide parquet row     │
│     5. flush per-strategy events into shared event log               │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ same inputs → both strategies
          ┌────────────┴────────────┐
          ▼                         ▼
  BaselineStrategy          OptimizerStrategy
  ─────────────────         ──────────────────
  ASAP, no oracle           own price oracle
  no graph                  own LangGraph
  no CommitTracker          own CommitTracker
                            own TriggerManager
```

The **OptimizerStrategy** runs a full agent loop:

```
TriggerManager fires  →  forecast_price  →  optimize (LP/MIP)
                      →  propose_reschedule  →  hitl_gate  →  commit_plan
```

Triggers: new appliance onset, ≥ 25 % price surprise, EV deadline slip, or 15-min periodic resync. A 30 s cooldown prevents thrashing.

---

## Optimizer

![Scenario E — 8 kW cap forces EV and heater to share peak slots](docs/scenario_e_power_cap.png)

*Scenario E from `notebooks/05_optimizer.ipynb`: 8 kW house cap with EV (7 kW rated) + heater (2 kW rated). The LP throttles both loads to stay under the cap while still meeting the 07:00 EV deadline and the overnight 4 kWh heater window. Orange = EV, red = heater, dashed = cap.*

`aerogrid/optimizer.py` → `solve_receding_horizon()`

Pure **LP** in the common case; collapses to a small **MIP** when `pending_cycles` are passed (one cycle appliance onset awaiting a HITL decision). Solved by HiGHS via CVXPY, typically in milliseconds.

**Decision variables** over `T` slots of 15 min each:

| variable | description |
|---|---|
| `p_ev[t]` | EV charging power (kW), zero outside availability window |
| `p_heat[t]` | heater power (kW) |
| `s_a[t] ∈ {0,1}` | binary start indicator per pending cycle, per allowed slot |
| `σ_ev, σ_heat[k]` | soft slack for EV and heater energy constraints |

**Constraints:** charger rating + EV availability gate (C1) · EV energy deadline (C2, hard inside horizon / proportional outside) · heater per-window energy (C3) · heater rating (C4) · house power cap (C5) · pending cycle placement exactly once (C6).

**Objective:** minimise forecast electricity cost + 1000 × slack penalties.

**Fallback:** if every solver fails, the function returns a deterministic ASAP plan (EV charges from first open slot, heater runs from start of each window, cycles placed at `earliest_start_slot`).

---

## Demo Notebooks

Three notebooks in `notebooks/` walk through the system from data to full end-to-end run:

| notebook | what it shows |
|---|---|
| `03_price_oracle.ipynb` | SMARD price EDA, seasonal-naive vs Chronos oracle comparison |
| `05_optimizer.ipynb` | 13 LP/MIP scenarios (EV gate, deadline regimes, heater windows, power-cap coupling, HITL stress test, horizon sensitivity, joint MIP vs naive reschedule) |
| `06_end_to_end.ipynb` | Full streaming simulation — baseline vs optimizer side by side, cumulative cost chart, per-appliance power breakdown, event log |

Run them with:

```bash
uv run jupyter lab notebooks/
```

---

## Quickstart

```bash
# 1. Python — pyenv reads .python-version (3.12.13)
pyenv install

# 2. Dependencies
uv sync --extra dev

# 3. Fetch real DE-LU prices (no API key needed)
uv run python scripts/fetch_smard_prices.py

# 4. Tests
uv run pytest -q

# 5. Full 16-day streaming simulation
uv run python -m aerogrid.sim.digital_twin

# Shorter smoke runs
uv run python -m aerogrid.sim.digital_twin --hours 24
uv run python -m aerogrid.sim.digital_twin --hours 8 --horizon-hours 6 --no-log-file

# Optional: Chronos / GridFM price oracle (requires torch)
uv sync --extra forecast
uv run python -m aerogrid.sim.digital_twin --hours 24 --price-impl chronos
```

Outputs land in `data/cache/`:

| file | resolution | contents |
|---|---|---|
| `slot_log.parquet` | 15 min | one row per slot; `<strategy>_*` columns per strategy |
| `event_log.parquet` | 1 s | one row per decision; uniform schema across strategies |
| `run_log.jsonl` | per replan | full OptimizerStrategy plan detail |

---

## Key Design Choices

- **Per-strategy isolation.** Each strategy owns its oracle, LangGraph, CommitTracker, and TriggerManager. The digital twin shares nothing except the onset stream and the realized price. Two `OptimizerStrategy` instances can run with different oracles in the same simulation.
- **Cross-strategy onset gating.** An onset is suppressed if *any* strategy still has the same appliance running, preventing a phantom second cycle in the comparison when strategies disagree on timing.
- **Event-driven triggers.** Replans fire on state changes, not on a fixed clock — with a 30 s cooldown. The periodic 15-min resync is a safety net, not the primary trigger.
- **Soft slacks.** EV and heater energy constraints are soft (penalty = 1000 × slack). The LP never goes infeasible; missed energy shows up as a non-zero slack in the solution.
- **Forecast vs realized costs.** `Schedule.expected_cost` and `baseline_cost` are both computed from the *price forecast*, not realized prices. They compare two hypothetical plans. Realized cost is accumulated slot-by-slot in `cumulative_cost` (notebook 06) using only loads that physically ran.

---

## Repo Layout

```
aerogrid/
  config.py          paths, date windows, horizons, HITL tolerances,
                     EV / heater specs, APPLIANCE_ONSETS
  types.py           Sample, ApplianceOnset, Schedule,
                     RescheduleProposal, PendingCycle, …
  state.py           LangGraph TypedDict schema
  graph.py           outer-loop nodes:
                       forecast_price → optimize → propose_reschedule
                                      → hitl_gate → commit_plan
  optimizer.py       receding-horizon LP/MIP (HiGHS via CVXPY)
  price_oracle.py    GridFM / Chronos / SeasonalNaive
  triggers.py        TriggerManager (new_onset / price_surprise /
                     deadline_slip / periodic + cooldown)
  commit.py          CommitTracker — remaining EV/heater kWh,
                     committed cycle tasks, HITL outcome adoption
  hitl_policy.py     pure AUTO/ASK decision functions
  sim/
    streamer.py      1 Hz tick iterator + onset injection
    price_server.py  SMARD parquet feed + optional spike injection
    strategies.py    Strategy ABC + BaselineStrategy + OptimizerStrategy
    digital_twin.py  orchestrator: streamer + price server +
                     cross-strategy gating + parquet writers

scripts/             one-shot data jobs
  fetch_smard_prices.py    SMARD DE-LU, no key, hard-fails on error
  fetch_entsoe_prices.py   ENTSO-E alt (requires ENTSOE_API_KEY)
  _gen_readme_images.py    regenerate docs/ images

notebooks/           EDA + demos (03 price oracle, 05 optimizer, 06 e2e)
tests/               pytest suite
docs/                static images for this README
```

---

## Data

| source | window | path |
|---|---|---|
| SMARD DE-LU day-ahead 15 min (primary) | Jan 12 – Apr 18 2026 | `data/smard/de_lu_15min.parquet` |
| ENTSO-E DE-LU (optional, requires key) | Dec 2024 | `data/entsoe/de_lu_15min.parquet` |

The SMARD fetcher raises `FetchError` on any network or HTTP failure — there is no synthetic price fallback.

---

## What Is Deliberately Out of Scope

- **NILM disaggregation** — removed because cost was always computed from commanded setpoints, never from a disaggregated mains trace. Onsets are listed manually in `APPLIANCE_ONSETS`.
- **Synthetic household traces** — the earlier scenario generator has been removed; the simulator runs on real prices + a manually-configured onset list.
- **Behavioural onset prediction** — the previous `BehavioralPredictor` produced output nothing downstream consumed; removed.
- Sub-second replanning, RL / learned policies, live smart-meter integration.

---

## Known Limitations and Room for Improvement

This is a demo. The following are real gaps worth addressing before any production use:

**Appliance model**
- Cycle shapes are rectangular: dishwasher always runs at exactly 2.5 kW for 2 h. Real appliances have variable power profiles (wash / heat / spin).
- Only two cycle appliances are modelled. No dryer, air conditioning, water heater (resistive), EV with V2G.
- The EV is a single fixed-need daily demand (24 kWh). No state-of-charge model, no V2H/V2G, no variable departure time.

**Onset detection**
- Onsets are manually listed in `config.APPLIANCE_ONSETS`. In a real deployment you need NILM or smart plugs to detect when an appliance starts.

**Price forecasting**
- The default oracle (`naive`) is a seasonal baseline with no predictive power beyond yesterday's profile. Chronos and GridFM are optional but not tuned for DE-LU 15-min prices.
- The optimizer uses point forecasts — no uncertainty quantification, no scenario trees, no robust or stochastic MPC.

**Savings metric**
- `Schedule.savings()` compares forecast-optimised vs forecast-naive costs. It is *not* a realized counterfactual saving — you cannot observe what the naive strategy would have cost on the same realized prices in the same run.
- The "naive baseline" (`_baseline_cost`) is price-unaware but still deadline-aware. A truly naive household would sometimes miss the EV deadline entirely, which would make the savings look larger than they are.

**Simulation fidelity**
- The simulation ticks at 1 Hz but processes 15-min price slots. There is no intra-slot price variation.
- Both strategies see the same price oracle, so the optimizer does not have an information advantage from better forecasting — it only benefits from smarter load placement.
- No battery storage, no solar PV, no grid export.

**Engineering gaps**
- The `OptimizerStrategy` re-solves the full LP/MIP on every trigger. With a longer horizon or many pending cycles this becomes slow (a 48 h run takes ~6 min wall time on a laptop).
- No backtesting harness — comparing different oracle configurations requires running the full simulation each time.
- Test coverage stops at the optimizer and core types; `strategies.py` and `digital_twin.py` have no unit tests.
- The HITL `interrupt()` path works in simulation with `auto_confirm=True` but has not been tested with a real human-in-the-loop UI.
