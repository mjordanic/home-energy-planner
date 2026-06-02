# Base Load: always-on inflexible demand (replaces fridge)

Status: ready-for-agent

## Parent

`.scratch/home-battery/PRD.md` — Home Battery

## What to build

Introduce a **Base Load**: the household's always-on, inflexible demand
(fridge + lights + standby + cooking) modelled as a deterministic per-hour kW
profile with an evening peak (~0.2 kW overnight, ~0.4 kW daytime, ~0.5 kW
morning 07–09h, ~0.9 kW evening peak 17–22h; ~9–10 kWh/day), expanded to
15-min slots. It is exogenous demand — no decision variables, no deadline, not
shiftable, **not** an `ApplianceSpec` (ADR 0003).

End-to-end, this slice threads the Base Load through every layer it touches
*without* introducing the battery:

- The optimizer (`solve_receding_horizon`) gains a new optional `base_load_kw`
  per-slot array, defaulting to zeros so existing behavior and tests are
  unchanged. The base load is added to the per-slot load the optimizer plans
  against.
- The optimize graph node derives the slot-aligned `base_load_kw` array from
  `now` and passes it to the optimizer.
- **Both** streaming strategies (baseline and optimizer) add the Base Load to
  their per-slot load and to accrued cost — it is demand every strategy must
  pay for.
- The per-slot log record (`SlotRecord`) gains a `base_load_kw` column.
- The dead `fridge` entry in `config.APPLIANCES` is deleted (it was referenced
  nowhere, never costed, never tested — ADR 0003).

This slice is the load-bearing prerequisite for the battery: without realistic
inflexible evening-peak demand, a no-export battery would have almost nothing
to discharge into.

## Acceptance criteria

- [ ] A deterministic per-hour Base Load profile lives in config and expands to
      a slot-aligned per-slot kW array, with an evening peak and ~9–10 kWh/day total.
- [ ] `solve_receding_horizon` accepts an optional `base_load_kw` array that
      defaults to zeros; with the default, optimizer output is byte-for-byte
      unchanged and all existing optimizer tests pass.
- [ ] The optimize graph node builds the slot-aligned base-load array from the
      current time and passes it to the optimizer.
- [ ] Both the baseline and optimizer strategies add the Base Load to per-slot
      load and to accrued cost.
- [ ] `SlotRecord` includes a `base_load_kw` field and it is written to the slot log.
- [ ] The `fridge` entry is removed from `config.APPLIANCES` and no longer
      referenced anywhere.
- [ ] Tests assert the base load contributes to cost for both strategies and that
      the off-by-default optimizer path is a regression-clean no-op.

## Blocked by

None — can start immediately.
