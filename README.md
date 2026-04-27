# AeroGrid — Streaming Multi-Agent Home Energy Planner

AeroGrid is a 1 Hz streaming agent that runs a behavioral appliance-onset
predictor, a price forecaster, and a receding-horizon MILP scheduler inside
a LangGraph loop. It couples a programmatic household-load simulator to real
SMARD DE-LU wholesale prices, so every intervention the agent makes (e.g.
*delay the washing machine by 2 h*) is visible as a before/after waveform
and a concrete euro delta.

## NILM disaggregation — not in scope

**NILM (Non-Intrusive Load Monitoring) disaggregation is not the focus of
this project.** The project demonstrates the closed-loop MPC agent: price
forecast → behavioral prediction → MILP optimizer → HITL gate. Real-world
NILM accuracy (training on REDD / UK-DALE / REFIT, cross-dataset
generalization, etc.) is a separate research problem and is deliberately
left out of scope.

The codebase ships a **perfect (dummy) disaggregator** that returns the
simulator's own ground-truth per-appliance traces — giving a perfect score
by construction. This lets the entire pipeline (onset detection, triggers,
optimizer, HITL) be developed and tested without a trained disaggregator.

**To plug in a real NILM model**, subclass `DisaggregatorBase` in
`aerogrid/nilm/disaggregator.py` and/or `DisaggModel` in
`aerogrid/nilm/model.py`. The rest of the pipeline will work without
changes.

## Architecture

Three loops at three rates:

```
┌────────────────────────────────────────────────────────────────┐
│ Inner loop  (1 Hz, every sample)                               │
│   ScenarioStreamer.iter_samples() ─▶ Sample(t, p_mains, …)     │
│       ├─▶ RollingDisaggregator.append / infer_latest           │
│       │   (ground-truth lookup; swap for real NILM model)      │
│       ├─▶ OnsetDetector.update  (per appliance)                │
│       ├─▶ CommitTracker.tick    (decrement EV kWh, retire ...) │
│       └─▶ TriggerManager.evaluate → ReplanTrigger?             │
└────────────────────┬───────────────────────────────────────────┘
                     │ fires only when a trigger hits
                     ▼
┌────────────────────────────────────────────────────────────────┐
│ Outer loop (LangGraph, event-driven)                           │
│   forecast_price ─▶ predict_behavior ─▶ optimize ─▶ hitl_gate  │
│                                                    │           │
│                         auto ──▶ commit_plan ◀─────┤           │
│                                                    │           │
│                         ask  ──▶ interrupt → resume┘           │
└────────────────────────────────────────────────────────────────┘
```

Key design choices:

- **Streaming input.** The agent sees every 1 Hz meter reading; it is free to
  react at any moment. Replans are event-driven, not clock-driven.
- **Receding-horizon MPC.** The MILP optimizes only 2 h ahead (8 × 15-min
  slots). The full-day EV-by-07:00 deadline is tracked as a state-dependent
  constraint: hard when the deadline is inside the horizon, proportional
  when it's further out. This avoids over-committing to a 24 h price
  forecast we don't trust. See [Optimization method](#optimization-method)
  for the full formulation.
- **Selective HITL.** EV setpoint deltas < 1.5 kW, tentative cycle shifts <
  15 min, and cost-neutral adjustments auto-commit. A shift ≥ 30 min, a new
  appliance, a start crossing into 22:00–06:00, or a cost increase ≥ $0.50
  triggers a real user prompt via LangGraph's `interrupt()`.
- **Triggers.** `TriggerManager` fires on new unplanned onsets, ≥25 % price
  surprise, deadline slip (required EV rate > current × 1.2), or periodic
  15-min resync. A 30 s cooldown prevents MILP thrashing.
- **Disaggregation plug-in.** The `aerogrid/nilm/` package defines a
  `DisaggregatorBase` ABC. The shipped `Disaggregator` reads ground-truth
  traces (perfect score). Swap it for a real NILM model by subclassing
  `DisaggregatorBase`.
- **Deterministic interventions.** The scenario generator is a pure function
  of the spec + seed. `apply_intervention_delay` and
  `apply_intervention_from_schedule` rewrite only the target appliance's
  cycle starts, so the pre-intervention segment of the trace stays
  byte-identical — critical for honest before/after plots.

## Optimization method

The optimizer (`aerogrid/optimizer.py`, function `solve_receding_horizon`) is
a small mixed-integer linear program (MILP) solved by HiGHS via CVXPY. It is
re-solved every time `TriggerManager` fires, over a short rolling horizon of
`T = SHORT_HORIZON_SLOTS` (default 8) slots of `Δt = 15 min` each (so 2 h
total). The detailed formulation, including all edge cases, is documented in
the module docstring of `aerogrid/optimizer.py`; the summary below is the
minimum needed to reproduce the method.

### Decision variables

For slots `t = 0, …, T-1`:

- `p_ev[t] ∈ [0, P_ev_max]` — continuous EV charging power (kW).
- `s_a[t] ∈ {0, 1}` — binary start indicator for each *bufferable cycle
  appliance* `a` (dishwasher, washing machine, heater) with cycle length
  `L_a` slots and rated power `P_a`. At most one start per appliance per
  horizon: `Σ_t s_a[t] ≤ 1`.
- `σ_ev ≥ 0` — soft slack (kWh) on the EV energy constraint.

A derived "is-running" expression `z_a[t] = Σ_{k=0..L_a-1} s_a[t-k]`
(clipped at `t = 0`) describes whether cycle `a` is active in slot `t`. By
construction `z_a[t] ∈ {0, 1}` and is a linear function of `s_a` — no extra
integer variables are introduced.

### Constraints

1. **Charger rating**: `0 ≤ p_ev[t] ≤ P_ev_max`.
2. **EV energy / deadline** — state-dependent on the time `τ` to the next
   `EV_DEADLINE_HOUR` (07:00 UTC). Let `H = T·Δt` and `E` = remaining EV
   kWh:
   - If `τ ≤ H` (deadline inside horizon, slot `t_d = round(τ/Δt)`):
     `Δt · Σ_{t<t_d} p_ev[t] + σ_ev ≥ E`.
   - If `τ > H`: `Δt · Σ_t p_ev[t] + σ_ev ≥ E · (H/τ) · γ`, where
     `γ = TRIGGER_DEADLINE_SAFETY = 1.2` front-loads charging slightly to
     absorb forecast error.
   The slack `σ_ev` keeps the MILP feasible when the house cap blocks the
   required rate; a large penalty `ρ = 1000` in the objective drives it to
   zero whenever feasibility allows.
3. **Cycle must fit in the horizon**: `s_a[T - L_a + 1 :] = 0`.
4. **Comfort deadline** (optional, per appliance) — when `deadline_hours`
   is configured (e.g. heater pre-conditioning by 07:00 / 18:00), the
   cycle must *finish* by the next such deadline:
   `s_a[ℓ + 1 :] = 0` where `ℓ = (deadline-slot) − L_a`. If `ℓ ≤ 0` but
   the deadline is still ahead, the cycle is forced to start at slot 0
   (best effort).
5. **House power cap**, at every slot `t`:
   `p_ev[t] + Σ_a z_a[t]·P_a + Σ_{c ∈ committed} P_c · 1[t ∈ c.range] ≤ P_max`.
   Committed tasks (pinned by `CommitTracker`) appear as a constant load
   on the cap during the slots they still occupy — they are not
   re-decided.

### Objective

Three terms — true cost, soft incentive to *reserve* likely cycles in
cheap slots, and the slack penalty:

```
minimize  C_actual(p_ev, s)  −  λ · U_reservation(s)  +  ρ · σ_ev
```

with

- `C_actual = κ · Σ_t p_ev[t]·π[t] + Σ_a κ · P_a · Σ_t z_a[t]·π[t]`,
- `U_reservation = Σ_a Σ_t s_a[t] · P̂_a(t)` (alignment with the
  behavioural predictor's onset probabilities),
- `κ = Δt / 1000` converting `kW × (€/MWh)` to € per slot,
- `λ = RESERVATION_LAMBDA = 0.5` weighting the reservation utility,
- `ρ = 1000` the slack penalty.

Without `U_reservation` the optimizer would never start a *new*
(uncommitted) cycle, since every cycle strictly increases `C_actual`.
The reservation term gives a soft bonus to cycles the household is
statistically likely to want anyway, so the MILP places them in cheap
high-probability slots — turning predicted intent into an executable
plan.

### Solver and fallback

Solvers are tried in the order **HiGHS → GLPK_MI → SciPy**. If all three
fail (or return a non-optimal status), the function returns a
deterministic fallback plan: charge the EV ASAP at rated power until the
remaining kWh is satisfied, no new cycles. This guarantees the digital
twin always has an actionable setpoint to apply.

### Baseline cost and reported savings

`_baseline_cost` evaluates a *price-unaware* schedule: EV charges ASAP
from slot 0; each cycle appliance starts at `argmax_t P̂_a(t)` (truncated
to fit). The MILP's `expected_cost` is compared to this baseline to
produce the savings ratio `(baseline − expected) / baseline`, which is
the headline metric in the demonstration notebooks and in
`Schedule.savings()`.

### Reproducibility

Given fixed `(now, prices, onset_probs, remaining_ev_kwh,
committed_tasks)` and the configuration constants in `aerogrid/config.py`,
the MILP is fully deterministic to within solver tolerance. The price
oracle and behavioural predictor are the only stochastic upstreams; once
their outputs are recorded (e.g. in the `event_log`), the same plan is
recovered on a re-solve.

## Data

| source | total | train | test (= simulation) | path |
|---|---|---|---|---|
| Simulated household @ 1 Hz (mains + per-appliance) | 97 d | 83 d | **14 d** | `data/scenario/*_1hz.parquet` |
| Simulated onsets (for behavioral predictor fit) | 97 d | same | same | `data/scenario/onsets.parquet` |
| SMARD DE-LU day-ahead LBMP 15-min (primary) | 97 d | 83 d | **14 d** | `data/smard/de_lu_15min.parquet` |
| ENTSO-E DE-LU (optional alt, requires API key) | 30 d | 20 d | 10 d | `data/entsoe/de_lu_15min.parquet` |
| NYISO NYC RT LBMP (legacy, currently unavailable) | — | — | — | `data/nyiso/nyc_15min.parquet` |

Scenario files are written by `scripts/generate_scenario.py` and stamped with
`source="simulated"` in `MANIFEST.json`. The SMARD fetcher downloads real
market data from Bundesnetzagentur's public API (no key required) and raises
`FetchError` on network/HTTP failure — there is no synthetic price fallback.

## Quickstart

```bash
# 1) Python version (pyenv pins 3.12.13 via .python-version in the repo root)
pyenv install                             # reads .python-version

# 2) env
uv sync --extra dev

# 3) data
.venv/bin/python scripts/fetch_smard_prices.py           # real SMARD DE-LU LBMP (97 d, no key)
.venv/bin/python scripts/generate_scenario.py            # simulated household (97 d)

# 4) tests
.venv/bin/python -m pytest -q

# 5) streaming digital-twin run over the 14-day test window
.venv/bin/python -m aerogrid.sim.digital_twin

# 5b) shorter smoke run with a planted price spike
.venv/bin/python -m aerogrid.sim.digital_twin --hours 48 --inject-spike

# 6) notebooks (scenario EDA, optimizer, end-to-end, intervention demo)
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

## Plugging in a real NILM model

The `aerogrid/nilm/` package is designed as a plug-in point:

1. **Subclass `DisaggregatorBase`** in `aerogrid/nilm/disaggregator.py` —
   implement `appliances()` and `disaggregate(power_1hz)`.
2. **Optionally subclass `DisaggModel`** in `aerogrid/nilm/model.py` for the
   per-appliance model interface.
3. **Add training logic** in `aerogrid/nilm/train.py` (currently a
   placeholder).
4. Update `aerogrid/sim/digital_twin.py` to instantiate your disaggregator
   instead of the default `Disaggregator.from_scenario()`.

The `OnsetDetector` (threshold + debounce) and `power_to_onsets()` helper
are independent of the NILM model and will work with any disaggregator
that outputs per-appliance power traces.

## Repo layout

```
aerogrid/                     core library
  config.py                   paths, date windows, horizons, HITL tolerances
  types.py                    Sample, ApplianceOnset, Schedule, ReplanTrigger …
  state.py                    LangGraph TypedDict schema (streaming shape)
  graph.py                    outer-loop nodes: forecast → predict → optimize
                              → hitl_gate → commit_plan
  optimizer.py                receding-horizon MILP (HiGHS) with deadline guard
                              + committed-task pinning + soft-slack fallback
  price_oracle.py             GridFM / Chronos / Seasonal-naive
  behavioral_predictor.py     Hybrid KDE / Chronos / Mamba stub
  triggers.py                 TriggerManager (new_onset / price_surprise /
                              deadline_slip / periodic + cooldown)
  commit.py                   CommitTracker (remaining_ev_kwh, committed tasks)
  hitl_policy.py              pure AUTO/ASK decision function
  nilm/
    model.py                  DisaggModel ABC + PerfectDisaggModel (dummy)
    disaggregator.py          DisaggregatorBase ABC + perfect (ground-truth)
                              Disaggregator + RollingDisaggregator
    onset_detector.py         streaming threshold + debounce per appliance
    train.py                  placeholder — add real NILM training here
  sim/
    appliance_models.py       5 parametric power models, vectorized, seeded
    scenario.py               ScenarioSpec / Generator / intervention API
    streamer.py               1 Hz sample iterator over mains parquet
    price_server.py           price parquet feed + optional spike injection
    digital_twin.py           the 1 Hz sample loop + graph invocation

scripts/                      one-shot data jobs
  generate_scenario.py        scenario → parquet + MANIFEST.json
  fetch_smard_prices.py       real SMARD DE-LU LBMP, no key, hard fail on error
  fetch_entsoe_prices.py      ENTSO-E alt path (requires ENTSOE_API_KEY)
  fetch_nyiso_prices.py       legacy NYISO path (currently unavailable)

notebooks/                    EDA + demos
tests/                        pytest suite (unit tests, integration smoke)
```

## What's deliberately out of scope

- **Real-world NILM accuracy** (REDD / UK-DALE / REFIT cross-dataset eval).
  The shipped disaggregator is a perfect dummy backed by simulator ground
  truth. Plug in a real model by subclassing `DisaggregatorBase`.
- Sub-second replanning — triggers cooldown at 30 s to prevent thrashing.
- Stochastic scenario populations / Monte-Carlo intervention analysis.
- A 24 h MILP as the primary optimizer (the 2 h MPC plus deadline tracking
  replaces it).
- Reinforcement learning / learned scheduling policies.
- Live smart-meter / EV integration.

## Prior art

- Klemen Jakšič, *SmartSim: Simulator for Smart Meter Data* — the five
  parametric appliance-model families here are reimplemented from the
  documented behaviour described in the paper. No code from
  `klemenjak/smartsim` was ported; the repo has no explicit license at the
  root, so we reimplemented from the paper to keep this project cleanly
  licensable.
