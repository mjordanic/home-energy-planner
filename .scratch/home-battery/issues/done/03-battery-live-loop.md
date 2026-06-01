# Battery in the live loop + third digital-twin strategy

Status: ready-for-agent

## Parent

`.scratch/home-battery/PRD.md` — Home Battery

## What to build

Wire the Home Battery into the running 1 Hz streaming simulation so the demo
actually dispatches it, and add a third strategy so the battery's marginal
contribution is cleanly attributable.

End-to-end through the live loop:

- **CommitTracker** tracks `soc_kwh` as new persistent state. `tick()` updates
  SoC from the committed charge/discharge setpoints at 1 Hz (charge adds
  `η_c·p·dt`, discharge removes `p·dt/η_d`), clamped to `[0, capacity]`. There
  is **no daily reset** (unlike the EV/heater deadlines). `adopt_plan()` copies
  the plan's first charge/discharge setpoints (`p_chg[0]`, `p_dis[0]`) alongside
  the existing EV/heater setpoints. Battery starts empty (initial SoC 0) and
  warms up over the first night.
- **OptimizerStrategy** gains a `battery_enabled` flag. When off it behaves
  exactly as today and its CommitTracker holds no battery state. When on, the
  optimize graph node passes the battery spec + current SoC to the optimizer.
- **SlotRecord** gains `battery_charge_kw`, `battery_discharge_kw`, `soc_kwh`,
  and `net_grid_kw`. Cost accrues on **net grid draw**
  (`net_grid_kw × Δt × price/1000`), not gross load. The baseline strategy keeps
  paying for Base Load but has no battery.
- The **digital twin** instantiates three strategies side by side, fed identical
  realized prices: `baseline` (no battery), `optimizer_nobatt` (battery off),
  `optimizer_batt` (battery on). Cross-strategy onset gating is unchanged.

No new replan triggers and no HITL prompts — the battery rides the existing
event-driven loop. Integration test in `tests/test_integration.py`, extending
the existing end-to-end twin test against the in-memory price feed.

## Acceptance criteria

- [ ] `CommitTracker` integrates SoC at 1 Hz from committed setpoints (charge
      `+η_c·p·dt`, discharge `−p·dt/η_d`), clamps to `[0, capacity]`, and never
      resets SoC at the EV/heater deadline hours.
- [ ] `adopt_plan()` copies the plan's first charge and discharge setpoints.
- [ ] `OptimizerStrategy` honors a `battery_enabled` flag; with it off, behavior
      and CommitTracker state are unchanged from today.
- [ ] `SlotRecord` includes `battery_charge_kw`, `battery_discharge_kw`,
      `soc_kwh`, and `net_grid_kw`, all written to the slot log.
- [ ] Optimizer-strategy cost accrues on net grid draw, not gross load.
- [ ] The digital twin runs the three strategies (`baseline`,
      `optimizer_nobatt`, `optimizer_batt`) over identical realized prices.
- [ ] No new replan triggers and no HITL prompts are introduced for the battery.
- [ ] An integration test over a multi-day cheap-night/expensive-day window with
      an evening cycle onset asserts the `optimizer_batt_*` battery + SoC +
      `net_grid_kw` columns exist and that cumulative cost satisfies
      `optimizer_batt ≤ optimizer_nobatt ≤ baseline`.

## Blocked by

- `.scratch/home-battery/issues/02-battery-optimizer-lp.md` (needs the working
  battery LP and the `Schedule` battery fields).
