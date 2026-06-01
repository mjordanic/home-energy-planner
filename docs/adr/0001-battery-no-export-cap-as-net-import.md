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
