# Value-of-stored-energy terminal reward

The receding-horizon LP cannot see past its horizon, so energy left in the
battery at the final slot looks worthless to it. The concern was that this
would make the optimizer drain the battery to empty at the horizon edge and
refuse to charge in a cheap overnight trough whenever the next usable peak sits
near or beyond the 24 h edge. The EV and heater avoid this because hard
deadlines pull energy in; the battery has no deadline.

We add a linear reward to the objective: `+ soc[T] · λ`, where
`λ = (max(min forecast price over the horizon, 0) / 1000) · discharge_efficiency`.
Stored energy is thus valued at roughly the cheapest price it could have been
bought for and then discharged, and never below zero: with a negative cheapest
price an unclipped λ would penalise stored energy and push the plan to empty
the pack. Pricing at the *minimum* (not mean) ensures the battery can never
profit by hoarding energy it bought — so no speculative gaming. The term is
linear in `soc[T]`, so the program stays a pure LP.

## What it does in practice

Re-measured on real SMARD DE-LU prices (8 Nov 2026), the reward changes nothing
that is executed:

- Across 169 single 24 h plans (2–16 Mar 2026, one every 2 h, battery starting
  at 6 kWh), the first slot — the only part of a plan that is ever executed —
  is identical with λ = 0 and with the shipped λ, both before and after the
  clip at zero. With the clip, the planned charge left at the horizon end is
  identical too (1.28 kWh on average).
- Two 14-day closed-loop replays (re-solving every 15 min, committing the first
  slot, perfect price foresight, no cycle appliances) give bills that match
  to the cent with λ = 0 and the shipped, clipped λ: €37.44 for 2–16 Mar and
  €25.15 for 3–17 Apr, with the same end charge.

A likely explanation, not tested separately: valuing stored energy at the
cheapest price in view makes holding it and discharging it at that price a
tie, so the reward only settles ties at the horizon edge, and the receding
horizon re-solves long before the edge is reached. The feared edge-dumping and
under-charging do not show up in these runs. The reward is kept as a cheap
guard, not as a mechanism that changes results.

Considered and rejected: a hard terminal SoC floor (the carryover target is
arbitrary and forces uneconomic charging when prices are flat), pricing at the
mean price (it leaves about 11 kWh in the pack at the horizon end and changes
the first slot in 5 of the 169 plans above — speculative hoarding), and doing
nothing (equivalent in these runs, but leaves stored energy explicitly
worthless to the planner).
