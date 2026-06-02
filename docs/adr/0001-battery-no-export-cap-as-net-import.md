# Home battery: no grid export, cap is net import

The home battery may **not** export to the grid: net grid draw is constrained
`≥ 0` at every slot, so the battery can only offset the household's own loads
(EV, heater, cycles, base load), never sell energy back. This matches a real
German grid-charged battery (no feed-in for grid-sourced energy) and keeps it a
genuine "buffer", not an arbitrage trader that could drive the bill negative and
break the savings ratio.

We also reinterpret the 10 kW `HOUSE_POWER_CAP_KW` as the **net grid-import**
limit (`net_grid = loads + charge − discharge ≤ cap`) rather than a gross-load
limit. Charging therefore competes with the EV for the cap overnight, and
discharging *relaxes* it — letting internal load briefly exceed 10 kW while the
battery supplies the excess (connection peak-shaving, a second battery benefit).
With no battery present this is identical to the previous gross-load cap.

Considered and rejected: symmetric export at wholesale price (unrealistic for a
home, makes the benefit an artificial trading profit) and a reduced feed-in
tariff (adds a second price series for little demo value).

## Realized-path enforcement

The forecast enforces `net_grid ≥ 0` as an LP constraint, but the LP only sees
the *forecast* load for each slot. In the realized 1 Hz simulation the discharge
setpoint is frozen from the last plan and replayed until the next replan
(≤ 15 min), so if realized load drops below forecast — the EV finishes, a cycle
ends, or the Base-Load hour rolls over while the setpoint is stale — a fixed
discharge can exceed the load it offsets and drive realized net grid draw
negative. That is an illegal export the forecast forbade.

The realized path therefore **throttles** discharge to the load it can offset
(`p_dis ≤ EV + heater + cycle + Base Load + charge`) and drains State of Charge
by the energy *actually delivered*, not by the setpoint. The throttle lives in
`CommitTracker.tick()` (which owns SoC), fed the household's offsettable load by
the strategy each tick; the billed discharge is read back from the tracker so it
cannot drift from the discharge that drained the battery.

Considered and rejected: **clamping the bill only** (`max(0, net_grid)` for
cost, leave SoC draining by the full setpoint) — stops negative cost but lets
SoC over-drain, so energy "discharged" into nothing vanishes from the books and
the next replan starts from a falsely-low SoC.
