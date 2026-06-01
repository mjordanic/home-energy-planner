# Base Load replaces the fridge

The `fridge` entry in `config.APPLIANCES` was dead config: referenced nowhere
outside its own definition, never added to any strategy's power or cost, never
tested. Its docstring claimed it added "background noise" to the aggregate
trace, but that was never wired in.

We delete it and introduce a **Base Load**: the household's always-on,
inflexible demand (fridge + lights + standby + cooking), modelled as a
deterministic per-hour kW profile with an evening peak (~9–10 kWh/day). Unlike
an appliance it has no deadline, no cycle, and cannot be shifted or declined, so
it is *not* an `ApplianceSpec` — it is exogenous demand added to net grid draw
and to cost for every strategy. Its purpose is to give the home battery
realistic inflexible load at the evening price peak to shave; without it,
"discharge during peak hours" would have almost nothing to discharge into.
