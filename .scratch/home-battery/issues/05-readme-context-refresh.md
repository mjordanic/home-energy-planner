# README + CONTEXT glossary refresh for the Home Battery

Status: ready-for-agent

## Parent

`.scratch/home-battery/PRD.md` — Home Battery

## What to build

Update the canonical references so they describe the Home Battery, the Base
Load, and the revised optimizer math accurately.

- **`README.md`:** add the Home Battery and Base Load to the device list; update
  the optimizer math section with the SoC dynamics, the net-grid (cap-as-net-
  import, no-export) model, and the value-of-stored-energy terminal reward;
  update the output schemas to include the new `Schedule` and slot-log fields
  (`battery_charge_kw`, `battery_discharge_kw`, `soc_kwh`, `base_load_kw`,
  `net_grid_kw`) and the three-strategy run.
- **`CONTEXT.md`:** verify the glossary entries (Home Battery, EV Battery, Base
  Load, State of Charge, Net Grid Draw, Value of Stored Energy) are accurate and
  consistent with what shipped; adjust if the implementation diverged.

Keep the Home Battery clearly named apart from the EV battery throughout. Do not
re-introduce out-of-scope features (grid export, solar, degradation cost, SoC
reserve floor, stochastic base load, V2H).

## Acceptance criteria

- [ ] README device list includes the Home Battery and the Base Load, named
      distinctly from the EV battery.
- [ ] README optimizer-math section documents SoC dynamics, the net-grid
      cap-as-net-import no-export model, and the value-of-stored-energy reward.
- [ ] README output schemas list the new `Schedule` / slot-log fields and the
      three-strategy run.
- [ ] `CONTEXT.md` glossary entries are verified accurate against the shipped
      implementation.
- [ ] No false claims or out-of-scope features are documented.

## Blocked by

- `.scratch/home-battery/issues/03-battery-live-loop.md` (output schemas and the
  three-strategy story depend on the shipped fields and live loop).
