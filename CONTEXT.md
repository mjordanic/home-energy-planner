# AeroGrid

Domain glossary for the AeroGrid home-energy-planning demo. Terms here are the
canonical vocabulary used across code, README, and notebooks. This file is a
glossary only — no implementation detail.

## Devices

**Home Battery**:
A stationary household storage unit that buffers grid energy: it can charge
(consume) when energy is cheap and discharge (give energy back to the home)
when energy is expensive. It has no comfort deadline — its only purpose is to
lower the bill and absorb price risk.
_Avoid_: Battery (ambiguous with the EV battery), BESS, accumulator.

**EV Battery**:
The electric vehicle's battery, modelled purely as a deadline-driven *load*:
it must receive a fixed amount of energy by a daily deadline and can only draw,
never give back. Distinct from the Home Battery.
_Avoid_: "the battery" (unqualified).

**Base Load**:
The household's always-on, inflexible demand (fridge, lights, standby,
cooking) modelled as a fixed per-hour power profile with an evening peak. The
optimizer cannot shift, decline, or reschedule it — it is exogenous demand
that the grid or the Home Battery must serve every slot. Replaces the former
(unwired) fridge appliance.
_Avoid_: Baseline (that is the no-optimization strategy), background load,
fridge.

## Concepts

**State of Charge (SoC)**:
The energy currently stored in the Home Battery, in kWh, between 0 and its
capacity. The single piece of state that couples one slot to the next.
_Avoid_: Charge level, fill, battery level.

**Net Grid Draw**:
The power actually imported from the grid at a slot: household load plus
battery charging minus battery discharging. Constrained to be ≥ 0 (no export)
and ≤ the house connection cap. It is the quantity the household pays for.
_Avoid_: Net load, grid power, consumption (gross load is different).

**Value of Stored Energy**:
The worth the optimizer assigns to energy still in the battery at the end of
the receding horizon, so it neither hoards nor dumps energy at the horizon
edge. Priced at the cheapest energy it could have been bought for.
_Avoid_: Terminal value, salvage value, end-of-horizon bonus.

**Savings**:
The fraction by which a strategy lowers the household's electricity bill
relative to a price-unaware reference that serves the same loads without
optimization. Measured against the **whole bill** — including the inflexible
Base Load that no strategy can shift or reduce — so it reflects money off the
total cost, not just the controllable portion. Because the Home Battery's only
purpose is to lower the bill, its benefit must be measured the same way: the
energy it discharges offsets real Base Load cost, never a phantom-free load.
_Avoid_: Efficiency, gain, cost reduction (unqualified), controllable savings.
