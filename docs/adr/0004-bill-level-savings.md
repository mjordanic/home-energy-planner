# Savings measured at the bill level

The reported **Savings** ratio compares the optimizer's forecast cost against a
price-unaware baseline. Both numbers — `expected_cost` and `baseline_cost` —
now include the exogenous **Base Load** cost (`Σ base·price` over the horizon),
not just the controllable EV/heater/cycle loads.

The driver is the Home Battery, not bookkeeping taste. The optimizer's
no-export constraint lets the battery discharge to offset *any* household load,
Base Load included (that evening peak is the whole reason ADR 0003 added a Base
Load to discharge into). The discharge **credit** (`−Σ p_dis·price`) therefore
already sits in `expected_cost`. If the Base Load **cost** it offsets is absent
from the cost, a slot where the battery fully covers the Base-Load peak shows up
as *negative* cost — the forecast-space twin of an illegal grid export. The
battery's savings are then systematically overstated. Adding the Base Load cost
to both sides makes the credit net against a real cost: a fully-shaved slot
reads ~0, not negative, and the headline ratio reflects money off the *total*
bill — which is the only thing a Home Battery exists to reduce.

A welcome side effect: realized `cumulative_cost` already bills Base Load every
slot, so including it in the forecast makes forecast and realized directly
comparable. The ratio shrinks (you genuinely cannot save on inflexible load),
but it is now honest.

Considered and rejected: **controllable-only savings** (exclude Base Load from
both, as before) — keeps a larger, flashier percentage but leaves the battery's
discharge credit offsetting a phantom zero-cost load, overstating its benefit
and making peak-shaving register as negative forecast cost.
