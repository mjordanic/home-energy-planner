# PRD: Home Battery

Status: ready-for-agent

## Problem Statement

The AeroGrid demo can shift *deferrable* loads (EV, heater, dishwasher, washing
machine) into cheap price slots, but it cannot store cheap energy to spend
later. As a result, any household demand that is forced to occur during
expensive hours — an after-dinner dishwasher cycle, the heater's daytime
window, the always-on lights/standby load — is simply paid for at the peak
price. There is no way to buy energy when it is cheap and use it when it is
dear. A real home with a stationary battery (e.g. a Powerwall) can do exactly
that, and the demo has no device that demonstrates this fundamental
energy-buffering benefit.

## Solution

Add a **Home Battery**: a stationary household storage unit, distinct from the
EV battery, that the optimizer charges when energy is cheap and discharges to
cover the home's own load when energy is expensive. It is a buffer in the home
grid — it never exports to the grid, only offsets the household's own demand.
Capacity is a configurable parameter set in advance; the receding-horizon
optimizer decides when to charge and when to release, looking across the whole
horizon rather than greedily chasing the nearest price dip.

To make the battery's peak-shaving benefit real and visible, also add a
**Base Load**: a deterministic, inflexible always-on household demand profile
(fridge + lights + standby + cooking) with an evening peak, which both
strategies pay for every slot but only the battery-equipped optimizer can shave
at peaks. This replaces the previously-unwired `fridge` appliance.

The digital twin runs three strategies side by side so the battery's
contribution is cleanly attributable: the naive baseline (no battery), the
optimizer without a battery, and the optimizer with a battery. The notebooks
and READMEs are refreshed to document the new device, the optimizer math, and
plots that illustrate the savings.

## User Stories

1. As a homeowner, I want a Home Battery that charges when wholesale prices are
   low, so that I can store cheap energy for later.
2. As a homeowner, I want the battery to discharge to cover my household load
   during expensive hours, so that I pay less for the energy I must use then.
3. As a homeowner, I want to set the battery capacity in advance, so that I can
   model the device I actually own.
4. As a homeowner, I want the optimizer to decide charge/discharge timing
   automatically, so that I never have to manage the battery by hand.
5. As a homeowner, I want the optimizer to consider the whole price horizon
   (not just the next dip), so that it captures the global cheapest plan rather
   than a locally greedy one.
6. As a homeowner, I want the battery to never sell energy back to the grid, so
   that it behaves like a realistic behind-the-meter buffer and never produces
   an artificial trading profit.
7. As a homeowner, I want the battery to let my house briefly draw more total
   power than the grid connection allows (by supplying the excess itself), so
   that the battery also shaves my connection-limit peaks.
8. As a homeowner, I want the battery's round-trip losses respected, so that the
   optimizer only cycles it when the price spread genuinely beats the loss.
9. As a homeowner, I want the battery to start empty and warm up over the first
   night, so that the simulation makes no unrealistic free-energy assumption.
10. As a homeowner, I want the battery's stored energy to retain value at the
    end of the planning horizon, so that the optimizer does not myopically drain
    it or refuse to charge in a cheap overnight trough.
11. As a homeowner, I want an always-on Base Load that reflects my fridge,
    lights, and standby devices, so that the simulated bill reflects real
    inflexible demand.
12. As a homeowner, I want the Base Load to peak in the evening, so that the
    simulation reflects when my household actually consumes the most.
13. As a homeowner, I want the battery to cover my evening dishwasher cycle and
    daytime heater window from cheaply-stored energy, so that I see concrete
    savings on loads I cannot shift.
14. As an analyst, I want a strategy that runs the optimizer *without* a battery,
    so that I can separate the value of smart scheduling from the value of the
    battery.
15. As an analyst, I want a strategy that runs the optimizer *with* a battery,
    so that I can measure the battery's marginal contribution.
16. As an analyst, I want the naive baseline to remain battery-free, so that it
    stays the project's consistent no-optimization reference.
17. As an analyst, I want per-slot logging of charge power, discharge power, and
    state of charge, so that I can plot the battery's dispatch over time.
18. As an analyst, I want per-slot logging of the Base Load and the net grid
    draw, so that I can see what the battery is actually offsetting.
19. As an analyst, I want the cumulative-cost ordering (battery-optimizer ≤
    no-battery-optimizer ≤ baseline) to hold over a representative window, so
    that the demo's headline claim is verifiable.
20. As a notebook reader, I want a plot of the battery's state-of-charge
    trajectory against price, so that I can see it filling in troughs and
    emptying at peaks.
21. As a notebook reader, I want a charge/discharge-vs-price plot, so that I can
    confirm the optimizer buys low and spends high.
22. As a notebook reader, I want a cost waterfall / three-strategy cumulative
    cost comparison, so that I can quantify the battery's savings.
23. As a notebook reader, I want a representative-day dispatch plot showing the
    battery covering the evening peak, so that I understand the mechanism.
24. As a developer, I want the optimizer to remain a pure LP after adding the
    battery, so that solve times and determinism are preserved.
25. As a developer, I want the battery and Base Load to be optional inputs to
    the optimizer that default to "off", so that all existing optimizer tests
    pass unchanged.
26. As a developer, I want the battery to require no new replan triggers and no
    HITL prompts, so that the existing event-driven loop is unchanged.
27. As a developer, I want the Home Battery clearly named apart from the EV
    battery throughout code and docs, so that the two are never conflated.
28. As a developer, I want the dead `fridge` spec removed when Base Load lands,
    so that there is a single background-load concept.
29. As a maintainer, I want the README and CONTEXT glossary refreshed, so that
    the canonical references describe the new device accurately.

## Implementation Decisions

**Naming.** The device is `home_battery` (config prefix `HOME_BATTERY_*`),
canonically "Home Battery", deliberately distinct from the EV battery. See
`CONTEXT.md`.

**Grid model (ADR 0001).** No export: `net_grid[t] ≥ 0`. The battery only
offsets the household's own loads. The 10 kW `HOUSE_POWER_CAP_KW` is
reinterpreted as the **net grid-import** limit (`net_grid = loads + base_load +
charge − discharge`, `0 ≤ net_grid ≤ cap`). Charging counts against the cap;
discharging relaxes it (the battery may supply internal load that exceeds the
connection limit). With no battery present this is identical to today's
gross-load cap.

**Physical spec (config constants).** Capacity 13.5 kWh; max charge = max
discharge = 5 kW; charge efficiency = discharge efficiency = 0.95 each
(~90% round-trip); usable SoC range 0–capacity; initial SoC 0 kWh; no
degradation/cycling cost. Round-trip losses alone deter pointless churn and
guarantee the optimizer never charges and discharges in the same slot, so **no
new binary variables are introduced — the program stays a pure LP**.

**Optimizer (`solve_receding_horizon`).** New optional inputs: a battery spec
(default `None` ⇒ no battery, existing behavior) and `initial_soc_kwh`
(default 0); plus a `base_load_kw` per-slot array (default zeros ⇒ existing
behavior). Decision variables `p_chg[t] ≥ 0`, `p_dis[t] ≥ 0`, `soc[t]`.
Constraints: SoC dynamics `soc[t+1] = soc[t] + η_c·p_chg[t]·Δt −
p_dis[t]·Δt/η_d` with `soc[0] = initial_soc_kwh`; `0 ≤ soc ≤ capacity`;
`0 ≤ p_chg ≤ P_max`, `0 ≤ p_dis ≤ P_max`; net-draw `≥ 0`; net-draw `≤` cap
(replacing the gross-load cap, with `base_load_kw` added to the per-slot load).

**Value of stored energy (ADR 0002).** Add a linear objective reward
`+ soc[T] · λ`, where `λ = (min(forecast price over horizon) / 1000) · η_d`.
This values leftover energy at roughly its cheapest acquisition cost — removing
both edge-dumping and myopic under-charging, while pricing at the minimum
ensures no profit from hoarding. Linear in `soc[T]`, so the LP stays an LP. The
ASAP solver-failure fallback keeps the battery idle (`p_chg = p_dis = 0`),
which is always feasible.

**Base Load (ADR 0003).** A deterministic per-hour kW profile in config
(~0.2 kW overnight, ~0.4 kW daytime, ~0.5 kW morning 07–09h, ~0.9 kW evening
peak 17–22h; ~9–10 kWh/day), expanded to 15-min slots. It is exogenous demand,
not an `ApplianceSpec`: no decision variables, no deadline, not shiftable. The
graph node derives the slot-aligned array from `now` and passes it to the
optimizer; both streaming strategies add it to per-slot load and cost. The
`fridge` entry in `APPLIANCES` is deleted.

**Graph wiring.** `n_optimize` builds the slot-aligned `base_load_kw` array and,
when the strategy's battery is enabled, the battery spec + current SoC, and
passes them to `solve_receding_horizon`. No new nodes; no new HITL path; the
battery never produces a `RescheduleProposal`. `Schedule` gains
`battery_charge_kw`, `battery_discharge_kw`, and `soc_kwh` (per-slot) plus is
echoed through `as_dict()`.

**CommitTracker.** Tracks `soc_kwh` as new persistent state. `tick()` updates
SoC from the committed charge/discharge setpoints at 1 Hz (charge adds
`η_c·p·dt`, discharge removes `p·dt/η_d`), clamped to `[0, capacity]`. No daily
reset (unlike the EV/heater deadlines). `adopt_plan()` copies `p_chg[0]` and
`p_dis[0]` into charge/discharge setpoints alongside the existing EV/heater
setpoints.

**Strategies.** `OptimizerStrategy` gains a `battery_enabled` flag; when off it
behaves exactly as today (and its CommitTracker holds no battery). `SlotRecord`
gains `base_load_kw`, `battery_charge_kw`, `battery_discharge_kw`, `soc_kwh`,
and `net_grid_kw`. Cost accrues on **net grid draw** (`net_grid_kw × Δt ×
price/1000`), not gross load. `BaselineStrategy` adds the Base Load to its cost
but has no battery. The digital twin instantiates three strategies: `baseline`,
`optimizer_nobatt` (battery off), `optimizer_batt` (battery on).

**Savings semantics.** The optimizer's internal `baseline_cost` stays the
no-optimization, no-battery reference, so `Schedule.savings()` for the
battery-equipped optimizer reflects scheduling + battery combined; for the
no-battery optimizer it reflects scheduling only. Forecast-vs-realized cost
distinction is unchanged (see README gotchas).

## Testing Decisions

Good tests assert **external, observable behavior** — kWh delivered, SoC
trajectory, net grid draw, cap satisfaction, cost ordering — never solver
internals, variable names, or CVXPY structure. They prefer the highest existing
seam. No new seams are introduced.

**Optimizer — `tests/test_optimizer.py` (seam: `solve_receding_horizon`).**
Prior art: existing tests build price arrays (`_flat_prices`, `_cheap_overnight`)
and assert on `Schedule.ev_power_kw` / `heater_power_kw` / slack. New tests, in
the same style:
- With the battery off (default), output is identical to today (regression
  guard for the optional-param design).
- On a cheap-overnight/expensive-day curve, the battery charges in cheap slots
  and discharges in expensive ones.
- SoC follows the recursion exactly given charge/discharge powers and η.
- SoC never exceeds capacity nor drops below 0; `p_chg`/`p_dis` never exceed
  `P_max`.
- Round-trip losses: with a price spread below the loss threshold the battery
  does not cycle; above it, it does.
- Value-of-stored-energy: on a horizon whose only usable peak is near the edge,
  the battery still charges in the trough (does not drain to empty at `T`).
- Net grid draw is `≥ 0` (no export) and `≤ cap`; a high-load slot shows the
  battery discharging to keep net draw under the cap (discharge relaxes the
  gross-load limit).

**CommitTracker — `tests/test_commit.py` (seam: direct construct + `tick`/
`adopt_plan`).** Prior art: existing tests construct a `CommitTracker` and call
`adopt_cycle_start` / `tick` / `running_committed_tasks`. New tests:
- SoC rises by `η_c·p·dt` under a charge setpoint over a tick span.
- SoC falls by `p·dt/η_d` under a discharge setpoint.
- SoC clamps at `capacity` (over-charge) and at `0` (over-discharge).
- `adopt_plan` copies the plan's first charge/discharge setpoints.
- SoC is not reset at the EV/heater deadline hours.

**Integration — `tests/test_integration.py` (seam: `digital_twin.run`).** Prior
art: `test_digital_twin_runs_both_strategies_end_to_end` runs the twin against
`_InMemoryPriceFeed` and asserts on the parquet slot/event logs. New test:
- Run three strategies over a multi-day cheap-night/expensive-day window with an
  evening cycle onset.
- Assert `optimizer_batt_*` battery + SoC + `net_grid_kw` columns exist in the
  slot log.
- Assert cumulative-cost ordering `optimizer_batt ≤ optimizer_nobatt ≤
  baseline`.

## Out of Scope

- **Grid export / feed-in tariff.** The battery never sells back; no second
  price series (ADR 0001).
- **Solar / PV generation.** No on-site generation; the battery charges from the
  grid only.
- **Battery degradation / cycling cost.** Not modelled (efficiency losses
  suffice for this demo); the LP has a hook for it only conceptually.
- **SoC reserve floor for backup power.** "Risk minimization" is handled by
  round-trip losses + min-price terminal value + EV deadline safety, not an
  explicit reserve.
- **Stochastic / noisy base load.** The Base Load is deterministic; no random
  background noise (the old fridge's unfulfilled promise is dropped, not
  reinstated).
- **Multiple batteries or vehicle-to-home (EV discharge).** The EV remains a
  pure load.
- **New replan triggers or HITL prompts for the battery.** The battery rides the
  existing periodic + price-surprise replans and is fully automated.
- **Re-tuning the EV/heater/cycle parameters or the price source.**

## Further Notes

- Canonical references updated as part of this work: `README.md` (device list +
  optimizer math + output schemas), `CONTEXT.md` glossary (already seeded:
  Home Battery, EV Battery, Base Load, State of Charge, Net Grid Draw, Value of
  Stored Energy), and notebooks `05_optimizer` (battery LP scenarios + plots)
  and `06_end_to_end` (three-strategy cumulative cost + representative-day
  dispatch). `03_price_oracle` is unaffected.
- ADRs recorded under `docs/adr/`: `0001` (no-export + cap-as-net-import),
  `0002` (value-of-stored-energy terminal reward), `0003` (Base Load replaces
  fridge).
- Why the Base Load is load-bearing for the feature: with all costed load being
  overnight-EV-dominated, a no-export battery would have almost nothing to
  discharge into during peak hours. The Base Load (plus the existing evening
  dishwasher onsets and the daytime heater window) is what gives the battery
  genuine peak-shaving work.
- Suggested tracer-bullet build order: config constants + Base Load profile →
  optimizer LP variables/constraints/objective + value-of-stored-energy →
  CommitTracker SoC → `Schedule`/`SlotRecord`/strategies + `battery_enabled` →
  digital-twin third strategy → tests → notebooks → README.
