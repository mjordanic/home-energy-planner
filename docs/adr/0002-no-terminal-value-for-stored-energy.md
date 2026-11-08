# No terminal value for stored energy

The receding-horizon LP cannot see past its horizon, so energy left in the
battery at the final slot looks worthless to it. The concern was that this
would make the optimizer drain the battery to empty at the horizon edge and
refuse to charge in a cheap overnight trough whenever the next usable peak sits
near or beyond the 24 h edge. The EV and heater avoid this because hard
deadlines pull energy in; the battery has no deadline.

We first added a linear reward to the objective, `+ soc[T] · λ` with
`λ = (max(min forecast price over the horizon, 0) / 1000) · discharge_efficiency`,
valuing stored energy at roughly the cheapest price it could have been bought
for. We have since removed it: the objective gives energy left at the horizon
end no value, and an always-on throughput tie-break of 0.01 €/MWh (0.001 ct/kWh)
on every kWh charged or discharged keeps the plan from charging and
discharging in the same slot when the round-trip loss alone would leave that
a tie.

## Why the reward went

Re-measured on real SMARD DE-LU prices (8 Nov 2026), the reward changed nothing
that is executed:

- Across 169 single 24 h plans (2–16 Mar 2026, one every 2 h, battery starting
  at 6 kWh), the first slot — the only part of a plan that is ever executed —
  is identical with λ = 0 and with the reward, both before and after it was
  clipped at zero.
- Two 14-day closed-loop replays (re-solving every 15 min, committing the first
  slot, perfect price foresight, no cycle appliances) give bills that match
  to the cent with λ = 0 and with the clipped reward: €37.44 for 2–16 Mar and
  €25.15 for 3–17 Apr.

A likely explanation, not tested separately: valuing stored energy at the
cheapest price in view makes holding it and discharging it at that price a
tie, so the reward only settled ties at the horizon edge, and the receding
horizon re-solves long before the edge is reached. The feared edge-dumping and
under-charging do not show up in these runs.

Keeping it also had costs. Unclipped, λ goes negative with a negative cheapest
price and penalises stored energy, pushing the plan to empty the pack; it
needed a clip to stay safe. And because it made holding and discharging a tie,
it interfered with the same-slot charge/discharge tie-break: with both terms
active the tie-break flipped hold-versus-discharge choices, so the tie-break
could only be applied when λ = 0. Without the reward the tie-break applies in
every slot and the program has one fewer term to back out of `expected_cost`.

Considered and rejected: a hard terminal SoC floor (the carryover target is
arbitrary and forces uneconomic charging when prices are flat), pricing at the
mean price (it leaves about 11 kWh in the pack at the horizon end and changes
the first slot in 5 of the 169 plans above — speculative hoarding), and keeping
the clipped minimum-price reward (equivalent in these runs, with the costs
above).
