# AeroGrid — Streaming Multi-Agent Home Energy Planner

AeroGrid is a 1 Hz streaming agent that runs a behavioral appliance-onset
predictor, a price forecaster, and a receding-horizon LP scheduler inside
a LangGraph loop. It couples a programmatic household-load simulator to real
SMARD DE-LU wholesale prices, so every intervention the agent makes (e.g.
*postpone the dishwasher by 1 h to save €0.60*) is visible as a before/after
waveform and a concrete euro delta.

The agent controls only **continuous loads** (EV charger, heater) directly.
Cycle appliances (dishwasher, washing machine) are **user-triggered**: the
user starts them, and the agent reacts by proposing a small forward shift if
it would save enough money — the user accepts or declines via a HITL prompt.

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
┌────────────────────────────────────────────────────────────────────┐
│ Outer loop (LangGraph, event-driven)                               │
│   forecast_price ─▶ predict_behavior ─▶ optimize                   │
│                                       │                            │
│                                       ▼                            │
│                            propose_reschedule  (cycle onsets only) │
│                                       │                            │
│                                       ▼                            │
│                              hitl_gate                             │
│                                       │                            │
│                       auto ──▶ commit_plan ◀── ask ── interrupt    │
└────────────────────────────────────────────────────────────────────┘
```

Key design choices:

- **Streaming input.** The agent sees every 1 Hz meter reading; it is free to
  react at any moment. Replans are event-driven, not clock-driven.
- **Receding-horizon MPC.** The optimizer is a *pure linear program* (no
  binary variables) over a configurable horizon (`HORIZON_HOURS`, default
  24 h × 4 slots/h = 96 slots of 15 min). Continuous EV charging and
  continuous heater power are the only decision variables. The
  full-day-ahead EV-by-07:00 deadline is hard when inside the horizon and
  proportional × `γ = 1.2` when outside. The heater satisfies a tuple of
  *energy-between-deadlines* requirements (default 4 kWh by 07:00, 2 kWh by
  18:00). See [Optimization method](#optimization-method) for the full
  formulation.
- **EV availability gate.** EV charging is locked to zero outside
  `EV_AVAILABLE_FROM_HOUR` (default 20:00 UTC) → `EV_DEADLINE_HOUR` (default
  07:00 UTC). The gate is enforced as a per-slot upper bound on `p_ev`.
- **User-triggered cycle appliances.** Dishwasher and washing machine starts
  come from the user, not the optimiser. When `OnsetDetector` flags a new
  onset, the LangGraph runs `propose_reschedule` to find the cheapest start
  inside `HITL_RESCHEDULE_WINDOW_HOURS` (default 2 h) and asks the user
  *"Postpone X by 1.2 h to save €0.61?"* — the user accepts or declines.
- **Selective HITL.** Reschedule proposals below
  `HITL_RESCHEDULE_MIN_SAVINGS_EUR` auto-decline (run now, don't bother the
  user). For continuous-load plan changes, EV setpoint deltas < 1.5 kW and
  cost-neutral adjustments auto-commit; a cost increase ≥ €0.50 triggers a
  real user prompt via LangGraph's `interrupt()`.
- **Simulated user.** `HITL_AUTO_RESPONSES` lets the digital twin reply
  programmatically — by default `dishwasher → "accept"` and
  `washing_machine → "decline"` so end-to-end runs exercise both paths.
- **Triggers.** `TriggerManager` fires on new unplanned onsets, ≥25 % price
  surprise, deadline slip (required EV rate > current × 1.2 *during the
  available charging window*), or periodic 15-min resync. A 30 s cooldown
  prevents thrashing.
- **Onset injection for testing.** `Streamer.add_onset(...)` queues
  `ApplianceOnset` events with `source="injected"` so test scripts and
  notebooks can fire arbitrary user behaviour without regenerating the
  scenario.
- **Disaggregation plug-in.** The `aerogrid/nilm/` package defines a
  `DisaggregatorBase` ABC. The shipped `Disaggregator` reads ground-truth
  traces (perfect score). Swap it for a real NILM model by subclassing
  `DisaggregatorBase`.
- **Deterministic interventions.** The scenario generator is a pure function
  of the spec + seed. The pre-intervention segment of the trace stays
  byte-identical across runs — critical for honest before/after plots.

## Optimization method

The optimizer (`aerogrid/optimizer.py`, function `solve_receding_horizon`)
is a small **linear program** — no binary variables — solved by HiGHS via
CVXPY. It is re-solved every time `TriggerManager` fires, over a rolling
horizon of `T = SHORT_HORIZON_SLOTS` slots of `Δt = 15 min` each (default
`HORIZON_HOURS = 24` ⇒ `T = 96`). Cycle appliances (dishwasher, washing
machine) are *not* decision variables — they are committed externally by
the HITL reschedule flow described in [Cycle reschedule flow](#cycle-reschedule-flow).
The detailed formulation is in the module docstring of
`aerogrid/optimizer.py`; the summary below is the minimum needed to
reproduce the method.

### Decision variables

For slots `t = 0, …, T-1`:

- `p_ev[t]` — continuous EV charging power (kW), with `0 ≤ p_ev[t] ≤ B_ev[t]`
  where `B_ev[t] = P_ev_max` if slot `t` is inside the EV availability
  window and `0` otherwise. The mask is computed from
  `EV_AVAILABLE_FROM_HOUR` (default 20:00) and `EV_DEADLINE_HOUR` (default
  07:00).
- `p_heat[t]` — continuous heater power (kW), with `0 ≤ p_heat[t] ≤ P_heat_max`.
- `σ_ev, σ_heat[k] ≥ 0` — soft slacks on the EV energy and per-window
  heater energy constraints.

### Constraints

1. **Charger rating + availability** (C1): `0 ≤ p_ev[t] ≤ B_ev[t]`.
2. **EV energy / deadline** (C2) — state-dependent on the time `τ` to the
   next `EV_DEADLINE_HOUR`. Let `H = T·Δt` and `E` = remaining EV kWh:
   - If `τ ≤ H` (deadline inside horizon): `Δt · Σ_{t<t_d} p_ev[t] + σ_ev ≥ E`.
   - If `τ > H`: `Δt · Σ_t p_ev[t] + σ_ev ≥ E · (H/τ) · γ`, where
     `γ = TRIGGER_DEADLINE_SAFETY = 1.2`.
3. **Heater energy per deadline window** (C3) — for each entry
   `(hour_k, kWh_k)` in `HEATER_DEADLINES` whose window
   `(prev_hour_k, hour_k]` overlaps the horizon, let `S_k` be the set of
   slots inside that overlap and `E_k` = remaining kWh in window `k`
   (tracked per-deadline by `CommitTracker`):
   `Δt · Σ_{t ∈ S_k} p_heat[t] + σ_heat[k] ≥ E_k`.
4. **Heater rating** (C4): `0 ≤ p_heat[t] ≤ P_heat_max`.
5. **House power cap** (C5), at every slot `t`:
   `p_ev[t] + p_heat[t] + Σ_{c ∈ committed} P_c · 1[t ∈ c.range] ≤ P_max`.
   Committed cycle tasks (pinned by `CommitTracker`) appear as a constant
   load on the cap during the slots they still occupy — they are not
   re-decided here.

### Objective

```
minimize  C_actual(p_ev, p_heat)  +  ρ_ev · σ_ev  +  ρ_h · Σ_k σ_heat[k]
```

with `C_actual = κ · Σ_t π[t] · (p_ev[t] + p_heat[t])`,
`κ = Δt / 1000` converting `kW × (€/MWh)` to € per slot, and
`ρ_ev, ρ_h = 1000` the slack penalties. There is no `U_reservation` term
any more — cycle appliances are not the optimiser's responsibility.

### Solver and fallback

Solvers are tried in the order **HiGHS → ECOS → SciPy**. If all three fail
(or return a non-optimal status), the function returns a deterministic
fallback plan: charge the EV ASAP at rated power inside the availability
window until the remaining kWh is satisfied; spread the heater's
remaining kWh evenly across each window.

### Cycle reschedule flow

When a `new_onset` trigger fires for a cycle appliance (dishwasher /
washing machine), the LangGraph node `propose_reschedule`:

1. Slices the price forecast starting at `onset_at`.
2. Evaluates the cycle cost at every shift `s = 0, 1, …, ⌊window / Δt⌋`
   (window = `HITL_RESCHEDULE_WINDOW_HOURS`, default 2 h).
3. Picks the cheapest shift `s★` and constructs a `RescheduleProposal`
   with `cost_now`, `cost_proposed`, `savings`, and a human question:
   *"Postpone dishwasher by 1.2 h to 21:00? You can save €0.61."*
4. The HITL gate either auto-declines (if savings <
   `HITL_RESCHEDULE_MIN_SAVINGS_EUR`, default €0.10) or asks the user.
5. On accept the cycle is committed at `proposed_start_at`; on decline it
   is committed at `onset_at` (run now). Either way `CommitTracker`
   pins the task and the next LP run sees it as a constant load on C5.

### Baseline cost and reported savings

`_baseline_cost` evaluates a *price-unaware* schedule: EV charges ASAP
from the start of its availability window; the heater spreads its
required kWh evenly across each window. The LP's `expected_cost` is
compared to this baseline to produce the savings ratio `(baseline −
expected) / baseline`, which is the headline metric in the demonstration
notebooks and in `Schedule.savings()`.

### Reproducibility

Given fixed `(now, prices, remaining_ev_kwh, remaining_heater_kwh_by_window,
committed_tasks)` and the configuration constants in `aerogrid/config.py`,
the LP is fully deterministic to within solver tolerance.

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

# 5c) override the optimisation horizon (default 24 h) and silence the file log
.venv/bin/python -m aerogrid.sim.digital_twin --hours 8 --horizon-hours 6 --no-log-file

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
  config.py                   paths, date windows, horizons, HITL tolerances,
                              EV availability window, heater deadlines,
                              auto-response defaults
  types.py                    Sample, ApplianceOnset, Schedule,
                              RescheduleProposal, ReplanTrigger …
  state.py                    LangGraph TypedDict schema (streaming shape)
  graph.py                    outer-loop nodes: forecast → predict → optimize
                              → propose_reschedule → hitl_gate → commit_plan
  optimizer.py                receding-horizon LP (HiGHS) — continuous EV +
                              heater, EV availability mask, heater per-window
                              energy deadlines, soft-slack fallback
  price_oracle.py             GridFM / Chronos / Seasonal-naive
  behavioral_predictor.py     Hybrid KDE / Chronos / Mamba stub
  triggers.py                 TriggerManager (new_onset / price_surprise /
                              deadline_slip / periodic + cooldown) +
                              ev_charging_window_hours helper
  commit.py                   CommitTracker — remaining_ev_kwh,
                              remaining_heater_kwh_by_window, committed cycle
                              tasks, adopt_cycle_start for HITL outcomes
  hitl_policy.py              pure AUTO/ASK decision functions for plans and
                              for cycle reschedule proposals
  nilm/
    model.py                  DisaggModel ABC + PerfectDisaggModel (dummy)
    disaggregator.py          DisaggregatorBase ABC + perfect (ground-truth)
                              Disaggregator + RollingDisaggregator
    onset_detector.py         streaming threshold + debounce per appliance
    train.py                  placeholder — add real NILM training here
  sim/
    appliance_models.py       5 parametric power models, vectorized, seeded
    scenario.py               ScenarioSpec / Generator / intervention API;
                              EV plug-in defaults to EV_AVAILABLE_FROM_HOUR;
                              heater natural draws live in the training split
                              (the test split is fully agent-controlled)
    streamer.py               1 Hz sample iterator over mains parquet +
                              add_onset() / consume_injected_onsets() for
                              programmatically injecting test onsets
    price_server.py           price parquet feed + optional spike injection
    digital_twin.py           the 1 Hz sample loop + graph invocation
                              (--horizon-hours overrides HORIZON_HOURS)

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
- A non-rolling 24 h LP solved once per day (the receding horizon LP plus
  deadline tracking replaces it; horizon length is configurable via
  `HORIZON_HOURS`).
- Reinforcement learning / learned scheduling policies.
- Live smart-meter / EV integration.

## Prior art

- Klemen Jakšič, *SmartSim: Simulator for Smart Meter Data* — the five
  parametric appliance-model families here are reimplemented from the
  documented behaviour described in the paper. No code from
  `klemenjak/smartsim` was ported; the repo has no explicit license at the
  root, so we reimplemented from the paper to keep this project cleanly
  licensable.
