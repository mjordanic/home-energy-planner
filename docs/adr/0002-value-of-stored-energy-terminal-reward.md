# Value-of-stored-energy terminal reward

The receding-horizon LP cannot see past its horizon, so energy left in the
battery at the final slot looks worthless to it. Unaddressed, this makes the
optimizer drain the battery to empty at the horizon edge and — worse — refuse to
charge in a cheap overnight trough whenever the next usable peak sits near or
beyond the 24 h edge, then miss that trough entirely at the next re-solve. The
EV and heater avoid this because hard deadlines pull energy in; the battery has
no deadline.

We add a linear reward to the objective: `+ soc[T] · λ`, where
`λ = (min forecast price over the horizon / 1000) · discharge_efficiency`.
Stored energy is thus valued at roughly the cheapest price it could have been
bought for and then discharged. This removes both the edge-dumping and the
under-charging, while pricing at the *minimum* (not mean) ensures the battery
can never profit by hoarding energy it bought — so no speculative gaming. The
term is linear in `soc[T]`, so the program stays a pure LP.

Considered and rejected: a hard terminal SoC floor (the carryover target is
arbitrary and forces uneconomic charging when prices are flat) and doing
nothing (accepts the myopic edge behaviour and a weaker demo).
