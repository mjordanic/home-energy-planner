# AeroGrid — Streaming Multi-Strategy Home Energy Planner

AeroGrid is a 1 Hz streaming agent that runs a price forecaster and a
receding-horizon LP scheduler inside a LangGraph loop, against real SMARD
DE-LU wholesale prices. Every intervention the agent makes (e.g. *postpone
the dishwasher by 1 h to save €0.60*) is visible as a before/after waveform
and a concrete euro delta.

The agent controls only **continuous loads** (EV charger, heater) directly.
Cycle appliances (dishwasher, washing machine) are **user-triggered**: the
user starts them, and the agent reacts by proposing a small forward shift
if it would save enough money — the user accepts or declines via a HITL
prompt.

## Architecture

The digital twin is a thin orchestrator. It streams real DE-LU prices and
a manually-listed sequence of appliance onsets into **N independent
strategy agents**.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Digital twin — owns ONLY the simulation environment                    │
│   Streamer.iter_samples()  ─▶ Sample(t, realized_price)                │
│   PriceServer.realized()   ─▶ realized €/MWh at slot boundaries        │
│   APPLIANCE_ONSETS list    ─▶ user-driven cycle starts                 │
│                                                                        │
│   For every 1 Hz sample:                                               │
│     1. pull onsets due now                                             │
│     2. cross-strategy gating: drop any onset whose appliance is still  │
│        pending in ANY strategy (prevents a phantom second cycle when   │
│        strategies disagree on cycle length)                            │
│     3. for every strategy s:                                           │
│            s.tick(sample, gated_onsets, dt_s=1.0)                      │
│     4. at slot boundaries: s.get_slot_record(now, price) → wide row    │
│     5. flush events from every strategy into the shared event log      │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ same (sample, gated_onsets) tuple → both strategies
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Each Strategy — autonomous agent                                       │
│                                                                        │
│   BaselineStrategy            OptimizerStrategy                        │
│   ─────────────────           ──────────────────                       │
│   no oracle                   own price oracle                         │
│   no graph                    own compiled LangGraph                   │
│   no CommitTracker            own CommitTracker                        │
│   no TriggerManager           own TriggerManager                       │
│                                                                        │
│   parameter-free              constructor builds everything from       │
│   constructor                 (price_history_provider, oracle_impl,    │
│                                horizon_slots, …)                       │
│                                                                        │
│   ASAP policy:                MPC slow path, fires on TriggerManager:  │
│   - cycle onset → run now     forecast_price ─▶ optimize               │
│   - EV: rated power until                  ─▶ propose_reschedule       │
│         full                               ─▶ hitl_gate ─▶ commit_plan │
│   - heater: rated power                                                │
│         until kWh met                                                  │
└────────────────────────────────────────────────────────────────────────┘
```

Both strategies see exactly the same gated onset stream — the comparison
is symmetric. Two `OptimizerStrategy` instances in the same run can use
different price oracles; they're fully independent agents that happen to
be evaluated against the same realized prices.

Outputs (written to `data/cache/`):

| file | resolution | contents |
|---|---|---|
| `slot_log.parquet` | 15 min | one row per slot; `<strategy>_*` columns for every strategy's power profile + cumulative cost; stream-level columns for permitted/suppressed onsets |
| `event_log.parquet` | 1 second | one row per decision; uniform schema; `strategy="stream"` for digital-twin-level events (`onset_permitted` / `onset_suppressed`) |
| `run_log.jsonl` | per replan | OptimizerStrategy's full plan detail (HITL decisions, reschedule proposals, full power profiles) |

Key design choices:

- **Per-strategy ownership of planning machinery.** The price oracle and
  MPC graph are not shared across strategies — each strategy builds its
  own copies inside its own constructor. `BaselineStrategy` is
  parameter-free; `OptimizerStrategy` builds the oracle and LangGraph
  from a few high-level constructor args. This lets you run multiple
  optimizer instances side by side with different oracles.
- **One unified onset stream.** Both strategies see the same gated list
  every tick. There is no NILM disaggregator, no behavioural predictor,
  and no synthetic mains trace — onsets are listed directly in
  `APPLIANCE_ONSETS` (or queued via `Streamer.add_onset(...)`).
- **Cross-strategy onset gating.** When two strategies disagree (e.g. the
  optimizer deferred a wash 2 h forward but the baseline ran it
  immediately and finished), an onset for the same appliance arriving
  while *any* strategy still has it pending is suppressed by the digital
  twin.
- **Streaming input.** The agent sees every 1 Hz sample; replans are
  event-driven, not clock-driven (with a 30 s cooldown).
- **Receding-horizon MPC.** The optimizer is a *linear program* in the
  common case (no binary variables) over a configurable horizon
  (`HORIZON_HOURS`, default 24 h × 4 slots/h = 96 slots of 15 min). The
  EV-by-07:00 deadline is hard when inside the horizon and proportional
  × `γ = 1.2` when outside. The heater satisfies a tuple of
  *energy-between-deadlines* requirements (default 4 kWh by 07:00, 2 kWh
  by 18:00). When `pending_cycles` are passed in, the LP collapses to a
  small MIP that picks each cycle's start slot jointly with the EV /
  heater plan. See [Optimization method](#optimization-method) for the
  full formulation.
- **EV availability gate.** EV charging is locked to zero outside
  `EV_AVAILABLE_FROM_HOUR` (default 20:00 UTC) → `EV_DEADLINE_HOUR`
  (default 07:00 UTC). The gate is enforced as a per-slot upper bound on
  `p_ev`.
- **User-triggered cycle appliances.** Dishwasher and washing machine
  starts come from the onset stream, not the optimiser. When an onset
  arrives, the LangGraph runs `propose_reschedule` to find the cheapest
  start inside `HITL_RESCHEDULE_WINDOW_HOURS` (default 2 h) and asks the
  user *"Postpone X by 1.2 h to save €0.61?"* — the user accepts or
  declines.
- **Selective HITL.** Reschedule proposals below
  `HITL_RESCHEDULE_MIN_SAVINGS_EUR` auto-decline. For continuous-load
  plan changes, EV setpoint deltas < 1.5 kW and cost-neutral adjustments
  auto-commit; a cost increase ≥ €0.50 triggers a real prompt via
  LangGraph's `interrupt()`.
- **Simulated user.** `HITL_AUTO_RESPONSES` lets the digital twin reply
  programmatically — by default `dishwasher → "accept"` and
  `washing_machine → "decline"` so end-to-end runs exercise both paths.
- **Triggers.** `TriggerManager` fires on new onsets, ≥25 % price
  surprise, deadline slip (required EV rate > current × 1.2 *during the
  available charging window*), or periodic 15-min resync. A 30 s cooldown
  prevents thrashing.

## Optimization method

The optimizer (`aerogrid/optimizer.py`, function `solve_receding_horizon`)
is a **linear program** — pure LP when no `pending_cycles` are passed in,
or a small mixed-integer program when the caller passes pending cycles.
Solved by HiGHS via CVXPY. It is re-solved every time `TriggerManager`
fires, over a rolling horizon of `T = SHORT_HORIZON_SLOTS` slots of
`Δt = 15 min` each (default `HORIZON_HOURS = 24` ⇒ `T = 96`). The detailed
formulation is in the module docstring of `aerogrid/optimizer.py`; the
summary below is the minimum needed to reproduce the method.

### Decision variables

For slots `t = 0, …, T-1`:

- `p_ev[t]` — continuous EV charging power (kW), with `0 ≤ p_ev[t] ≤ B_ev[t]`
  where `B_ev[t] = P_ev_max` if slot `t` is inside the EV availability
  window and `0` otherwise.
- `p_heat[t]` — continuous heater power (kW), with `0 ≤ p_heat[t] ≤ P_heat_max`.
- `s_a[t] ∈ {0, 1}` — start indicators for each pending cycle `a`, one per
  allowed start slot. `Σ_t s_a[t] = 1`.
- `σ_ev, σ_heat[k] ≥ 0` — soft slacks on the EV energy and per-window
  heater energy constraints.

### Constraints

1. **Charger rating + availability** (C1): `0 ≤ p_ev[t] ≤ B_ev[t]`.
2. **EV energy / deadline** (C2) — state-dependent on the time `τ` to
   the next `EV_DEADLINE_HOUR`. Let `H = T·Δt` and `E` = remaining EV kWh:
   - If `τ ≤ H` (deadline inside horizon): `Δt · Σ_{t<t_d} p_ev[t] + σ_ev ≥ E`.
   - If `τ > H`: `Δt · Σ_t p_ev[t] + σ_ev ≥ E · (H/τ) · γ`, where
     `γ = TRIGGER_DEADLINE_SAFETY = 1.2`.
3. **Heater energy per deadline window** (C3) — for each entry
   `(hour_k, kWh_k)` in `HEATER_DEADLINES` whose window
   `(prev_hour_k, hour_k]` overlaps the horizon:
   `Δt · Σ_{t ∈ S_k} p_heat[t] + σ_heat[k] ≥ E_k`.
4. **Heater rating** (C4): `0 ≤ p_heat[t] ≤ P_heat_max`.
5. **House power cap** (C5): at every slot
   `p_ev[t] + p_heat[t] + Σ_{c ∈ committed} P_c · 1[t ∈ c.range]
                       + Σ_a P_a · z_a[t] ≤ P_max`,
   where `z_a[t]` is the running indicator built from `s_a`.
6. **Pending cycle placement** (C6): `s_a[t] ∈ {0,1}` for `t ∈ [earliest_a, latest_a]`,
   `Σ_t s_a[t] = 1`.

### Objective

```
minimize  C_actual  +  ρ_ev · σ_ev  +  ρ_h · Σ_k σ_heat[k]
```

with `C_actual = κ · Σ_t π[t] · (p_ev[t] + p_heat[t] + Σ_a P_a · z_a[t])`,
`κ = Δt / 1000` converting `kW × (€/MWh)` to € per slot, and
`ρ_ev, ρ_h = 1000` the slack penalties.

### Solver and fallback

For pure LP problems solvers are tried in the order **HiGHS → ECOS →
SciPy**; for MIP problems **HiGHS → GLPK_MI**. If the chain fails (or
returns a non-optimal status), the function returns a deterministic
fallback plan: charge the EV ASAP at rated power inside the availability
window, run the heater at rated power inside each deadline window, and
place each pending cycle at its earliest allowed start.

### Cycle reschedule flow

When a `new_onset` trigger fires for a cycle appliance, the LangGraph
node `propose_reschedule` reads the joint MIP's chosen start slot from
the plan and (if the cycle was placed at a future slot) constructs a
`RescheduleProposal` with `cost_now`, `cost_proposed`, `savings`, and a
human question: *"Postpone dishwasher by 1.2 h to 21:00? You can save
€0.61."*

The HITL gate either auto-declines (savings < `HITL_RESCHEDULE_MIN_SAVINGS_EUR`,
default €0.10) or asks the user. On accept the cycle is committed at
`proposed_start_at`; on decline a separately-prepared "decline plan" with
the cycle pinned at slot 0 is committed instead, so the EV / heater plan
stays cap-feasible.

### Reproducibility

Given fixed `(now, prices, remaining_ev_kwh, remaining_heater_kwh_by_window,
committed_tasks, pending_cycles)` and the configuration constants in
`aerogrid/config.py`, the LP/MIP is fully deterministic to within solver
tolerance.

## Data

| source | total | train | test (= simulation) | path |
|---|---|---|---|---|
| SMARD DE-LU day-ahead LBMP 15-min (primary) | 97 d | 83 d | **14 d** | `data/smard/de_lu_15min.parquet` |
| ENTSO-E DE-LU (optional alt, requires API key) | 30 d | 20 d | 10 d | `data/entsoe/de_lu_15min.parquet` |
| NYISO NYC RT LBMP (legacy, currently unavailable) | — | — | — | `data/nyiso/nyc_15min.parquet` |

The SMARD fetcher downloads real market data from Bundesnetzagentur's
public API (no key required) and raises `FetchError` on network/HTTP
failure — there is no synthetic price fallback.

## Quickstart

```bash
# 1) Python version (pyenv pins 3.12.13 via .python-version in the repo root)
pyenv install                             # reads .python-version

# 2) env
uv sync --extra dev

# 3) data — fetch real DE-LU prices (no key required)
.venv/bin/python scripts/fetch_smard_prices.py

# 4) tests
.venv/bin/python -m pytest -q

# 5) streaming digital-twin run over the 14-day test window
#    Writes data/cache/{slot_log.parquet, event_log.parquet, run_log.jsonl}.
.venv/bin/python -m aerogrid.sim.digital_twin

# 5b) shorter smoke run, 24 hours
.venv/bin/python -m aerogrid.sim.digital_twin --hours 24

# 5c) override the optimisation horizon (default 24 h) and silence the file log
.venv/bin/python -m aerogrid.sim.digital_twin --hours 8 --horizon-hours 6 --no-log-file

# 5d) try a different price oracle on the OptimizerStrategy
.venv/bin/python -m aerogrid.sim.digital_twin --hours 24 --price-impl chronos

# 6) notebooks (price oracle EDA, optimizer scenarios, end-to-end demo)
.venv/bin/python -m jupyter lab notebooks/
```

Optional: to run the Chronos / GridFM price oracles instead of the naive
seasonal baseline, install the forecast extras and flip
`PRICE_ORACLE_IMPL` in `aerogrid/config.py`:

```bash
uv sync --extra forecast        # adds chronos-forecasting + torch
```

The oracle chain always falls back to `SeasonalNaiveOracle`, and the
`source` field on every `PriceForecast` records which code path actually
produced each forecast.

### Configuring the onset stream

`APPLIANCE_ONSETS` in `aerogrid/config.py` defines the cycle starts the
digital twin will inject during a run. By default it's empty; for a demo
run, override it (e.g. inside a notebook) before calling
`twin.main()`:

```python
import aerogrid.sim.digital_twin as twin
from datetime import timedelta

start = twin.SIM_TEST_START
twin.APPLIANCE_ONSETS = (
    ("dishwasher",      start + timedelta(hours=1)),
    ("washing_machine", start + timedelta(hours=2, minutes=15)),
    ("dishwasher",      start + timedelta(hours=3)),  # may be suppressed
)
```

## Adding a new strategy

The digital twin's run loop is policy-agnostic — it just calls
`tick / has_pending_appliance / get_slot_record / flush_events` on each
agent. To add a new scheduling policy:

1. Subclass `aerogrid.sim.strategies.Strategy`, implement the abstract
   methods (`tick`, `has_pending_appliance`, `get_slot_record`, plus
   optional `close` / `summary`).
2. Construct an instance of your strategy in
   `aerogrid/sim/digital_twin.py`'s `main()` and append it to the
   `strategies` list. Each strategy's slot-log columns get auto-prefixed
   with `strategy.name`, and event-log rows are tagged with the same
   `strategy` field.

`BaselineStrategy` and `OptimizerStrategy` in `strategies.py` are
reference implementations — minimal naive ASAP and full MPC + LangGraph,
respectively.

## Repo layout

```
aerogrid/                     core library
  config.py                   paths, date windows, horizons, HITL tolerances,
                              EV availability window, heater deadlines,
                              auto-response defaults, APPLIANCE_ONSETS
  types.py                    Sample, ApplianceOnset, Schedule,
                              RescheduleProposal, ReplanTrigger, …
  state.py                    LangGraph TypedDict schema (streaming shape)
  graph.py                    outer-loop nodes:
                              forecast_price ─▶ optimize ─▶ propose_reschedule
                                              ─▶ hitl_gate ─▶ commit_plan
  optimizer.py                receding-horizon LP/MIP (HiGHS) — continuous EV
                              + heater, EV availability mask, heater per-window
                              energy deadlines, joint cycle placement,
                              soft-slack fallback
  price_oracle.py             GridFM / Chronos / Seasonal-naive
  triggers.py                 TriggerManager (new_onset / price_surprise /
                              deadline_slip / periodic + cooldown)
  commit.py                   CommitTracker — remaining_ev_kwh,
                              remaining_heater_kwh_by_window, committed cycle
                              tasks, adopt_cycle_start for HITL outcomes
  hitl_policy.py              pure AUTO/ASK decision functions for plans and
                              for cycle reschedule proposals
  sim/
    streamer.py               1 Hz tick iterator + add_onset() /
                              consume_injected_onsets() for the manually-
                              listed onset stream
    price_server.py           price parquet feed + optional spike injection
    strategies.py             Strategy ABC + BaselineStrategy +
                              OptimizerStrategy (each a self-contained
                              agent; OptimizerStrategy owns its oracle,
                              CommitTracker, TriggerManager and compiled
                              LangGraph)
    digital_twin.py           thin orchestrator: streamer + price server +
                              cross-strategy onset gating + slot/event-log
                              writers

scripts/                      one-shot data jobs
  fetch_smard_prices.py       real SMARD DE-LU LBMP, no key, hard fail on error
  fetch_entsoe_prices.py      ENTSO-E alt path (requires ENTSOE_API_KEY)
  fetch_nyiso_prices.py       legacy NYISO path (currently unavailable)

notebooks/                    EDA + demos (price oracle, optimizer, e2e)
tests/                        pytest suite (unit tests, integration smoke)
```

## What's deliberately out of scope

- **NILM disaggregation.** The earlier dummy-NILM machinery has been
  removed because it was not load-bearing (cost was always computed from
  each strategy's commanded setpoints, never from a disaggregated mains
  trace). Onsets are listed manually instead.
- **Synthetic household traces.** The earlier scenario generator and
  appliance-power models have been removed. The simulator runs on real
  prices + a manually-configured onset list.
- **Behavioural onset prediction.** The previous `BehavioralPredictor`
  produced output that nothing downstream consumed; it has been removed.
- Sub-second replanning — triggers cool down at 30 s to prevent thrashing.
- Reinforcement learning / learned scheduling policies.
- Live smart-meter / EV integration.
