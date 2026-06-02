# PRD: Home Battery review fixes — realized no-export & bill-level savings

Status: ready-for-agent

## Problem Statement

The Home Battery feature (PR #2, feature `home-battery`) works, but a review of
the branch surfaced two correctness gaps that both flatter the battery's
headline number, plus a couple of cleanups.

1. **The realized simulation can illegally export to the grid.** ADR 0001 and
   the `CONTEXT.md` definition of **Net Grid Draw** say net grid draw is
   constrained `≥ 0` — the battery may only offset the household's own load,
   never sell energy back. The forecast honours this (an LP constraint), but
   the realized 1 Hz loop bills an *unclamped* net grid draw. The discharge
   setpoint is frozen from the last plan and replayed until the next replan
   (≤ 15 min); if realized load drops below forecast — the EV finishes, a cycle
   ends, or the Base-Load hour rolls over while the setpoint is stale — the
   fixed discharge can exceed the load it offsets, drive net grid draw negative,
   and book a *negative* slot cost. That is phantom export revenue, and it
   overstates the realized savings the demo reports as its headline.

2. **The forecast Savings overstate the battery's benefit.** `expected_cost`
   and `baseline_cost` both exclude the exogenous **Base Load** cost, yet the
   battery's discharge *credit* (`−Σ p_dis·price`) — which offsets Base Load,
   exactly as ADR 0003 intended — is included in `expected_cost`. A slot where
   the battery fully covers the Base-Load peak therefore reads as *negative*
   forecast cost: the forecast-space twin of the illegal export above. The
   battery's savings are systematically overstated.

3. **Cleanups.** The terminal-reward coefficient `λ` (and its `min(price)`)
   is computed twice in the optimizer, risking drift; and a stale Base-Load
   comment claims two different daily-energy totals.

## Solution

Enforce the no-export invariant on the realized path and measure Savings at the
whole-bill level, so the reported numbers are honest.

- **Throttle realized discharge.** When the battery discharges more than the
  household load it can offset, cap the realized discharge to that load and
  drain **State of Charge** by the energy *actually delivered*, not by the
  setpoint. A real inverter never dumps energy into nothing, and SoC stays
  honest for the next replan. The throttle lives where SoC lives — in
  `CommitTracker.tick()` — fed the household's offsettable load by the strategy
  each tick. The billed discharge is read back from the tracker so it can never
  drift from the discharge that drained the battery. (ADR 0001, "Realized-path
  enforcement".)

- **Measure Savings at the bill level.** Add the Base-Load cost (`Σ base·price`
  over the horizon) to both `expected_cost` and `baseline_cost`. The battery's
  discharge credit then nets against a real Base-Load cost: a fully-shaved slot
  reads ~0 (not negative), and the headline ratio reflects money off the
  *total* bill, which is the only thing a Home Battery exists to reduce. The
  ratio shrinks — you genuinely cannot save on inflexible load — but it is now
  honest, and it lines up with realized `cumulative_cost`, which already bills
  Base Load. (ADR 0004; `CONTEXT.md` **Savings**.)

- **Tidy the optimizer.** Compute `λ`/`min(price)` once and reuse it for both
  the objective and the `expected_cost` unpacking; drop the stale Base-Load
  daily-energy comment.

The simultaneous charge/discharge concern raised in review is explicitly **not**
actioned: round-trip losses already make it strictly suboptimal, the terminal
reward cannot game it, and hardening it would break the pure-LP design (see Out
of Scope).

## User Stories

1. As a demo viewer, I want the battery-equipped strategy's realized bill to
   never go negative, so that the savings I see are real and not an artefact of
   illegal grid export.
2. As a demo viewer, I want a slot where the battery fully covers the evening
   Base-Load peak to cost about zero, not less than zero, so that peak-shaving
   is shown honestly.
3. As a demo viewer, I want the reported Savings percentage to be a fraction of
   my *total* electricity bill, so that it reflects money I would actually save.
4. As a demo viewer, I want the forecast Savings and the realized cumulative
   cost to be measured on the same basis (both including Base Load), so that the
   two numbers are comparable.
5. As the optimizer, I want the realized discharge in any slot to never exceed
   the household load it can offset plus any concurrent charge, so that net grid
   draw stays `≥ 0` exactly as my forecast promised.
6. As the State of Charge, I want to fall only by the energy the battery
   actually delivered into household load, so that I am not over-drained by a
   stale discharge setpoint.
7. As the next replan, I want to start from a State of Charge that reflects real
   delivered energy, so that I do not plan against a falsely-low battery.
8. As `CommitTracker`, I want to be told the household's offsettable load for the
   current tick, so that I can throttle discharge without needing to know about
   the Base-Load profile myself.
9. As `OptimizerStrategy.get_slot_record`, I want to bill the discharge the
   tracker actually applied, so that the billed discharge and the SoC-draining
   discharge can never disagree.
10. As a non-battery strategy, I want my behaviour, costs, and SoC to be exactly
    unchanged, so that the regression-clean guarantee from the home-battery work
    still holds.
11. As the optimizer in no-battery mode (no `base_load_kw` supplied), I want my
    `expected_cost` and `baseline_cost` to be byte-for-byte unchanged, so that
    the existing no-op regression contract is preserved.
12. As a maintainer reading the optimizer, I want `λ` and `min(price)` computed
    in one place, so that the objective and the `expected_cost` accounting can
    never silently diverge.
13. As a maintainer reading `config.py`, I want one consistent statement of the
    Base-Load daily energy, so that I am not misled by contradictory comments.
14. As a reviewer, I want the change documented in ADR 0001 (realized
    enforcement) and ADR 0004 (bill-level savings), so that a future reader
    understands why these numbers are computed the way they are.

## Implementation Decisions

- **`CommitTracker.tick()` gains an optional `offsettable_load_kw` parameter
  (keyword-only, defaulting to "no throttle").** When supplied and the battery
  is discharging, the applied discharge is `min(setpoint, offsettable_load_kw +
  charge_setpoint)`; SoC drains by that *applied* amount (`applied · dt / η_d`),
  not by the raw setpoint. When the parameter is omitted, behaviour is exactly
  as today — this preserves every existing `tick()` caller and unit test.
  Charging is unchanged. (Chosen over computing the offsettable load inside the
  tracker, to keep the Base-Load profile a strategy-layer concern and keep
  `commit.py` free of a time-anchored profile dependency.)
- **`CommitTracker` exposes the applied discharge.** A new field
  `battery_discharge_applied_kw` records the throttled discharge from the most
  recent tick, and is included in `snapshot()`. It is the single source of truth
  for "how much the battery actually delivered this slot".
- **`OptimizerStrategy` assembles offsettable load and threads it down.** On each
  tick the strategy computes the household's offsettable load (EV + heater +
  running cycle + Base Load setpoints/profile) and passes it to
  `CommitTracker.tick(..., offsettable_load_kw=...)`. In `get_slot_record`, the
  billed discharge and the reported `net_grid_kw` use the tracker's
  `battery_discharge_applied_kw`, so net grid draw is `≥ 0` by construction.
- **Bill-level Savings in the optimizer.** `solve_receding_horizon` adds the
  Base-Load cost over the horizon (`Σ base_load_kw[t]·price[t]·per_slot_factor`)
  to `expected_cost`, and `_baseline_cost` adds the same term. When
  `base_load_kw` is `None`/zeros the added term is zero, so the no-battery /
  no-base-load output is unchanged. The discharge credit and terminal-reward
  accounting are otherwise untouched; `expected ≤ baseline` continues to hold
  because the same non-negative term is added to both.
- **Single `λ` computation.** `min_price` and `λ = (min_price/1000)·η_d` are
  computed once in `solve_receding_horizon` and reused for both the objective's
  terminal-reward term and the `expected_cost` back-out, removing the duplicate.
- **No optimizer model change for export.** The forecast LP already constrains
  `net_grid ≥ 0`; the realized throttle is the only new enforcement. No new
  decision variables, no binaries — the program stays a pure LP in the common
  case.
- **Docs already updated** in this branch: `CONTEXT.md` gains a **Savings** term;
  ADR 0001 gains a "Realized-path enforcement" section; ADR 0004 records
  bill-level savings.

## Testing Decisions

Good tests here assert *external behaviour* — net grid draw never negative, SoC
reflects delivered energy, Savings measured against the whole bill, no-op when
the battery/Base Load are absent — and never the throttle's internal arithmetic
or private fields beyond the one documented `battery_discharge_applied_kw`
accessor. All three seams already exist; prefer them over new ones.

- **`CommitTracker.tick()` — unit (`tests/test_commit.py`).** Prior art: the
  existing `test_battery_soc_falls_on_discharge_tick`,
  `test_battery_soc_clamps_at_zero`, `test_battery_no_soc_tracking_without_battery_spec`.
  New cases: discharge throttled to offsettable load when the setpoint exceeds
  it; SoC drains by the *throttled* (delivered) amount, not the setpoint;
  `battery_discharge_applied_kw` equals the throttled discharge; when
  `offsettable_load_kw` is omitted, SoC drains by the full setpoint exactly as
  today (backward-compatible).
- **`solve_receding_horizon()` — unit (`tests/test_optimizer.py`).** Prior art:
  `test_baseline_cost_uses_naive_window_charging` (asserts
  `expected_cost ≤ baseline_cost`), `test_battery_off_is_regression_clean`,
  `test_base_load_kw_default_is_regression_clean`. New/extended cases: with a
  Base Load supplied, both `expected_cost` and `baseline_cost` increase by the
  Base-Load energy cost; a battery slot that fully covers the Base-Load peak no
  longer yields negative forecast cost; with `base_load_kw` absent, both costs
  are unchanged (regression-clean); `expected ≤ baseline` still holds.
- **`OptimizerStrategy` end-to-end — integration (`tests/test_integration.py`).**
  Prior art: `test_three_strategies_battery_columns_and_cost_ordering`. New
  assertions on a battery-enabled run: every slot's `net_grid_kw ≥ 0`; no slot
  cost is negative; the billed discharge in a slot equals the tracker's applied
  discharge. Use the existing day/night price feed fixture to force a discharge
  into a shrinking load.

`pytest` is configured with `asyncio_mode = "auto"`; run with `uv run pytest -q`.

## Out of Scope

- **Simultaneous charge/discharge hardening.** Deliberately not done. Round-trip
  losses (`η < 1`) make charging and discharging in the same slot strictly
  wasteful, and the terminal reward (priced at `min` price) cannot make hoarding
  profitable, so the cost-minimising LP never does it. A binary would turn the
  common-case LP into a MIP (against the documented pure-LP design) and an `ε`
  tie-breaker adds a magic constant for a scenario (`η = 1.0`) that is never
  configured. The realized throttle covers any residual edge.
- **Slot-flooring helper extraction.** `config.get_base_load_kw` and
  `optimizer._floor_slot` contain identical flooring logic; they are left as two
  copies (trivially correct, no desync today) rather than extracted into a
  shared helper, to keep the diff small and avoid touching the cross-module
  import surface.
- **House power cap on the realized path.** Realized load exceeding
  `HOUSE_POWER_CAP_KW` is a pre-existing condition unrelated to the no-export
  invariant and is not addressed here.
- **Any new device, price series, feed-in tariff, or export model.** ADR 0001
  already rejected grid export; this PRD only enforces that existing decision.

## Further Notes

- This work lands on the existing PR branch for `home-battery`
  (`claude/compassionate-wozniak-HoX2L`, PR #2), updating it in place. The
  `CONTEXT.md`/ADR edits are already present in the working tree.
- The two correctness fixes are mirror images of one another — realized illegal
  export (#1) and forecast negative-cost peak-shaving (#2) are the same coherence
  bug in two spaces. Implement and review them together so the framing stays
  intact.
- Expect the headline Savings percentage in the notebooks to *drop* after the
  bill-level change; that is the intended, honest outcome, not a regression.
