# Home Battery in the optimizer (pure-LP buffering)

Status: ready-for-agent

## Parent

`.scratch/home-battery/PRD.md` — Home Battery

## What to build

Teach the receding-horizon optimizer to charge a **Home Battery** when energy
is cheap and discharge it to cover household load when energy is expensive,
across the whole horizon — staying a **pure LP** (no new binary variables;
round-trip losses alone prevent simultaneous charge/discharge).

End-to-end through the optimizer seam (`solve_receding_horizon`) and the
`Schedule` output:

- New `HOME_BATTERY_*` config constants: capacity 13.5 kWh; max charge = max
  discharge = 5 kW; charge efficiency = discharge efficiency = 0.95 each
  (~90% round-trip); usable SoC 0–capacity; no degradation cost.
- `solve_receding_horizon` gains two optional inputs: a battery spec
  (default `None` ⇒ no battery, today's behavior exactly) and `initial_soc_kwh`
  (default 0). Decision variables `p_chg[t] ≥ 0`, `p_dis[t] ≥ 0`, `soc[t]`.
- SoC dynamics: `soc[t+1] = soc[t] + η_c·p_chg[t]·Δt − p_dis[t]·Δt/η_d`,
  `soc[0] = initial_soc_kwh`, `0 ≤ soc ≤ capacity`, `0 ≤ p_chg,p_dis ≤ P_max`.
- **Grid model (ADR 0001):** the 10 kW `HOUSE_POWER_CAP_KW` is reinterpreted as
  the **net grid-import** limit. With `net_grid = loads + base_load + charge −
  discharge`, enforce `0 ≤ net_grid ≤ cap` at every slot (no export; discharge
  *relaxes* the cap so the battery can shave a connection-limit peak). With no
  battery present this is identical to today's gross-load cap.
- **Value of stored energy (ADR 0002):** add the linear objective reward
  `+ soc[T] · λ`, with `λ = (min(forecast price over horizon) / 1000) · η_d`.
  Keeps the program linear; removes both horizon-edge dumping and myopic
  under-charging; pricing at the minimum prevents any hoarding profit.
- The ASAP solver-failure fallback keeps the battery idle
  (`p_chg = p_dis = 0`), which is always feasible.
- `Schedule` gains per-slot `battery_charge_kw`, `battery_discharge_kw`, and
  `soc_kwh`, echoed through `as_dict()`.

Tests live in `tests/test_optimizer.py`, in the existing style (build price
arrays, assert on `Schedule` fields / slack — never solver internals).

## Acceptance criteria

- [ ] With the battery off (default `None`), `solve_receding_horizon` output is
      identical to today and all existing optimizer tests pass.
- [ ] On a cheap-overnight / expensive-day curve, the battery charges in cheap
      slots and discharges in expensive ones.
- [ ] SoC follows the recursion exactly given charge/discharge powers and η.
- [ ] SoC never exceeds capacity nor drops below 0; `p_chg`/`p_dis` never exceed `P_max`.
- [ ] Round-trip losses: with a price spread below the loss threshold the
      battery does not cycle; above it, it does.
- [ ] Value-of-stored-energy: on a horizon whose only usable peak is near the
      edge, the battery still charges in the trough rather than draining to empty at `T`.
- [ ] Net grid draw is `≥ 0` (no export) and `≤ cap`; a high-load slot shows the
      battery discharging to keep net draw under the cap.
- [ ] The optimizer remains a pure LP (no new binary variables); the ASAP
      fallback returns an idle-battery plan.
- [ ] The Home Battery is named distinctly from the EV battery throughout
      (config prefix `HOME_BATTERY_*`).

## Blocked by

- `.scratch/home-battery/issues/01-base-load.md` (battery `net_grid` sums the
  Base Load; relies on the `base_load_kw` optimizer input from slice 01).
