# Realized no-export throttle: net grid draw ≥ 0 on the 1 Hz path

Status: ready-for-agent

## Parent

`.scratch/battery-review-fixes/PRD.md` — Home Battery review fixes

## What to build

Enforce the no-export invariant (ADR 0001, `CONTEXT.md` **Net Grid Draw** `≥ 0`)
on the *realized* simulation path, where it is currently violated. The forecast
LP already constrains `net_grid ≥ 0`, but the realized 1 Hz loop replays a frozen
discharge setpoint between replans (≤ 15 min); when realized load drops below
forecast, the fixed discharge can exceed the load it offsets, drive net grid draw
negative, and book a *negative* slot cost — phantom export revenue that overstates
the headline realized savings.

End-to-end, this slice throttles realized discharge to the load it can actually
offset and keeps **State of Charge** honest:

- `CommitTracker.tick()` gains a **keyword-only** `offsettable_load_kw` parameter
  that defaults to "no throttle". When supplied and the battery is discharging,
  the applied discharge is `min(setpoint, offsettable_load_kw + charge_setpoint)`,
  and SoC drains by that *applied* amount (`applied · dt / η_d`), not by the raw
  setpoint. Charging is unchanged. When the parameter is omitted, behaviour is
  exactly as today — every existing `tick()` caller and unit test is preserved.
- `CommitTracker` exposes the applied discharge via a new
  `battery_discharge_applied_kw` field, included in `snapshot()`. This is the
  single source of truth for "how much the battery actually delivered this slot".
- `OptimizerStrategy` computes the household's offsettable load each tick
  (EV + heater + running cycle + Base-Load setpoints/profile) and passes it to
  `CommitTracker.tick(..., offsettable_load_kw=...)`. The Base-Load profile stays
  a strategy-layer concern — `commit.py` gains no time-anchored profile dependency.
- In `OptimizerStrategy.get_slot_record`, the billed discharge and the reported
  `net_grid_kw` read back the tracker's `battery_discharge_applied_kw`, so the
  billed discharge can never drift from the discharge that drained the battery,
  and net grid draw is `≥ 0` by construction.

No optimizer model change and no new decision variables — this is realized-path
enforcement of an invariant the forecast already promised.

## Acceptance criteria

- [ ] `CommitTracker.tick()` accepts a keyword-only `offsettable_load_kw`; when
      omitted, SoC drains by the full setpoint exactly as today (backward-compatible,
      existing `test_commit.py` battery cases still pass).
- [ ] When `offsettable_load_kw` is supplied and the setpoint exceeds it, the
      applied discharge is throttled to `min(setpoint, offsettable_load_kw + charge)`
      and SoC drains by the throttled (delivered) amount, not the setpoint.
- [ ] `battery_discharge_applied_kw` records the throttled discharge from the most
      recent tick and is included in `snapshot()`.
- [ ] `OptimizerStrategy` assembles the offsettable load each tick and threads it
      into `tick()`; `get_slot_record` bills and reports the tracker's applied
      discharge.
- [ ] Integration test on a battery-enabled run (existing day/night price fixture,
      discharge forced into a shrinking load): every slot's `net_grid_kw ≥ 0`, no
      slot cost is negative, and the billed discharge equals the tracker's applied
      discharge.
- [ ] A non-battery strategy's behaviour, costs, and SoC are exactly unchanged.

## Blocked by

None — can start immediately.
