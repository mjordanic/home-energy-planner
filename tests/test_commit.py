from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aerogrid.commit import CommitTracker


def test_deferred_cycle_is_replannable_before_start():
    now = datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc)
    c = CommitTracker()
    c.adopt_cycle_start(
        appliance="dishwasher",
        slots=8,
        expected_kwh=5.0,
        start_at=now + timedelta(hours=1),
        now=now,
    )
    assert len(c.replannable_onsets(now)) == 1
    assert c.replannable_onsets(now)[0].appliance == "dishwasher"
    assert c.running_committed_tasks(now) == []


def test_deferred_cycle_can_be_nudged_again_before_start():
    now = datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc)
    c = CommitTracker()
    c.adopt_cycle_start(
        appliance="dishwasher",
        slots=8,
        expected_kwh=5.0,
        start_at=now + timedelta(hours=1),
        now=now,
    )
    # Nudge again before start.
    c.adopt_cycle_start(
        appliance="dishwasher",
        slots=8,
        expected_kwh=5.0,
        start_at=now + timedelta(hours=2),
        now=now + timedelta(minutes=15),
    )
    # At +75 min it is still deferred (second nudge took effect).
    t = now + timedelta(minutes=75)
    assert len(c.replannable_onsets(t)) == 1
    assert c.running_committed_tasks(t) == []


def test_cycle_becomes_running_once_start_time_is_reached():
    now = datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc)
    c = CommitTracker()
    start_at = now + timedelta(minutes=30)
    c.adopt_cycle_start(
        appliance="dishwasher",
        slots=8,
        expected_kwh=5.0,
        start_at=start_at,
        now=now,
    )
    t = start_at + timedelta(minutes=1)
    running = c.running_committed_tasks(t)
    assert len(running) == 1
    assert running[0].appliance == "dishwasher"


# --------------------------------------------------------------------------- #
# Home Battery SoC tracking                                                    #
# --------------------------------------------------------------------------- #
from aerogrid.config import BatterySpec


def test_battery_soc_rises_on_charge_tick():
    """SoC rises by η_c·p_chg·dt under a charge setpoint."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=0.0)
    c.battery_charge_setpoint_kw = 2.0
    # Tick 900 s (15 min = 0.25 h).
    c.tick(datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc), dt_s=900.0)
    expected_soc = batt.eta_charge * 2.0 * (900.0 / 3600.0)  # 0.95 * 2 * 0.25 = 0.475 kWh
    assert c.soc_kwh == pytest.approx(expected_soc, abs=1e-6)


def test_battery_soc_falls_on_discharge_tick():
    """SoC falls by p_dis·dt/η_d under a discharge setpoint."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=5.0)
    c.battery_discharge_setpoint_kw = 1.0
    c.tick(datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc), dt_s=900.0)
    # Energy removed from SoC = 1.0 * 0.25 / 0.95 = 0.26316 kWh
    expected_removed = 1.0 * (900.0 / 3600.0) / batt.eta_discharge
    assert c.soc_kwh == pytest.approx(5.0 - expected_removed, abs=1e-6)


def test_battery_soc_clamps_at_capacity():
    """SoC does not exceed capacity_kwh even when charge setpoint would overshoot."""
    batt = BatterySpec(capacity_kwh=1.0, max_charge_kw=10.0, max_discharge_kw=5.0)
    c = CommitTracker(battery_spec=batt, soc_kwh=0.99)
    c.battery_charge_setpoint_kw = 10.0
    c.tick(datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc), dt_s=900.0)
    assert c.soc_kwh <= batt.capacity_kwh + 1e-9


def test_battery_soc_clamps_at_zero():
    """SoC does not go below 0 even when discharge setpoint would overshoot."""
    batt = BatterySpec(capacity_kwh=5.0, max_charge_kw=5.0, max_discharge_kw=5.0)
    c = CommitTracker(battery_spec=batt, soc_kwh=0.01)
    c.battery_discharge_setpoint_kw = 5.0
    c.tick(datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc), dt_s=900.0)
    assert c.soc_kwh >= 0.0


def test_battery_adopt_plan_copies_first_slot_setpoints():
    """adopt_plan copies battery_charge_kw[0] and battery_discharge_kw[0] as setpoints."""
    from aerogrid.types import Schedule
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=0.0)
    now = datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc)
    plan = Schedule(
        slot_start=now,
        horizon_slots=4,
        ev_power_kw=[0.0, 0.0, 0.0, 0.0],
        heater_power_kw=[0.0, 0.0, 0.0, 0.0],
        battery_charge_kw=[3.5, 0.0, 0.0, 0.0],
        battery_discharge_kw=[0.0, 2.0, 0.0, 0.0],
        soc_kwh=[0.0, 0.875, 0.875, 0.875],
    )
    c.adopt_plan(plan, now)
    assert c.battery_charge_setpoint_kw == pytest.approx(3.5)
    assert c.battery_discharge_setpoint_kw == pytest.approx(0.0)


def test_battery_soc_not_reset_at_ev_deadline():
    """SoC is NOT reset when the EV daily deadline passes (unlike remaining_ev_kwh)."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=7.0)
    # Tick at 07:00:00 — the EV deadline hour.
    c.tick(datetime(2026, 4, 15, 7, 0, 0, tzinfo=timezone.utc), dt_s=1.0)
    # EV remaining should reset to EV_DAILY_NEED_KWH; SoC should NOT change.
    from aerogrid.config import EV_DAILY_NEED_KWH
    assert c.remaining_ev_kwh == pytest.approx(EV_DAILY_NEED_KWH)
    assert c.soc_kwh == pytest.approx(7.0, abs=1e-3)   # unchanged


def test_battery_no_soc_tracking_without_battery_spec():
    """Without battery_spec, tick does not update soc_kwh (stays at 0)."""
    c = CommitTracker()  # no battery_spec
    c.battery_charge_setpoint_kw = 5.0  # set a setpoint that should be ignored
    c.tick(datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc), dt_s=900.0)
    assert c.soc_kwh == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# No-export throttle: offsettable_load_kw                                      #
# --------------------------------------------------------------------------- #

def test_tick_without_offsettable_load_is_backward_compatible():
    """tick() with no offsettable_load_kw behaves exactly as before (full setpoint applied)."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=5.0)
    c.battery_discharge_setpoint_kw = 3.0
    c.tick(datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc), dt_s=900.0)
    expected_removed = 3.0 * (900.0 / 3600.0) / batt.eta_discharge
    assert c.soc_kwh == pytest.approx(5.0 - expected_removed, abs=1e-6)
    assert c.battery_discharge_applied_kw == pytest.approx(3.0)


def test_tick_throttles_discharge_when_setpoint_exceeds_offsettable_load():
    """When setpoint > offsettable_load + charge, discharge is throttled and SoC reflects it."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=5.0)
    c.battery_discharge_setpoint_kw = 4.0  # setpoint 4 kW
    c.battery_charge_setpoint_kw = 0.0
    # Offsettable load is only 1.5 kW, so applied discharge = min(4, 1.5) = 1.5
    c.tick(
        datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc),
        dt_s=900.0,
        offsettable_load_kw=1.5,
    )
    assert c.battery_discharge_applied_kw == pytest.approx(1.5)
    expected_removed = 1.5 * (900.0 / 3600.0) / batt.eta_discharge
    assert c.soc_kwh == pytest.approx(5.0 - expected_removed, abs=1e-6)


def test_tick_no_throttle_when_setpoint_within_offsettable_load():
    """When setpoint ≤ offsettable_load, applied discharge equals the setpoint unchanged."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=5.0)
    c.battery_discharge_setpoint_kw = 2.0
    c.battery_charge_setpoint_kw = 0.0
    c.tick(
        datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc),
        dt_s=900.0,
        offsettable_load_kw=3.0,  # load (3 kW) ≥ setpoint (2 kW) — no throttle
    )
    assert c.battery_discharge_applied_kw == pytest.approx(2.0)
    expected_removed = 2.0 * (900.0 / 3600.0) / batt.eta_discharge
    assert c.soc_kwh == pytest.approx(5.0 - expected_removed, abs=1e-6)


def test_battery_discharge_applied_kw_in_snapshot():
    """battery_discharge_applied_kw is present in snapshot()."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=5.0)
    c.battery_discharge_setpoint_kw = 4.0
    c.tick(
        datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc),
        dt_s=900.0,
        offsettable_load_kw=1.5,
    )
    snap = c.snapshot()
    assert "battery_discharge_applied_kw" in snap
    assert snap["battery_discharge_applied_kw"] == pytest.approx(1.5)


def test_tick_throttle_zero_when_no_load():
    """When offsettable_load_kw=0.0, discharge is fully throttled (net grid draw = 0)."""
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=5.0)
    c.battery_discharge_setpoint_kw = 3.0
    c.battery_charge_setpoint_kw = 0.0
    c.tick(
        datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc),
        dt_s=1.0,
        offsettable_load_kw=0.0,
    )
    assert c.battery_discharge_applied_kw == pytest.approx(0.0)
    # SoC unchanged when applied is 0.
    assert c.soc_kwh == pytest.approx(5.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Integration: no-export invariant over a multi-slot battery-enabled run       #
# --------------------------------------------------------------------------- #

def test_no_export_over_multi_slot_run_with_shrinking_load():
    """net_grid_kw ≥ 0 in every slot when discharge is forced into a shrinking load.

    Scenario: battery is discharging at 3 kW setpoint. Household load
    decreases slot by slot from 2.5 kW to 0.5 kW, so the raw setpoint would
    drive net_grid negative in every slot. The throttle must cap applied
    discharge to the available load.
    """
    batt = BatterySpec()
    c = CommitTracker(battery_spec=batt, soc_kwh=10.0)
    c.battery_discharge_setpoint_kw = 3.0
    c.battery_charge_setpoint_kw = 0.0

    loads_kw = [2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
    base_time = datetime(2026, 4, 15, 20, 0, tzinfo=timezone.utc)

    for i, load in enumerate(loads_kw):
        c.tick(base_time + timedelta(seconds=i), dt_s=1.0, offsettable_load_kw=load)
        net_grid = load + c.battery_charge_setpoint_kw - c.battery_discharge_applied_kw
        assert net_grid >= -1e-9, (
            f"slot {i}: net_grid={net_grid:.6f} (load={load}, "
            f"applied_dis={c.battery_discharge_applied_kw:.4f})"
        )
        # Billed discharge can never exceed the available offsettable load.
        assert c.battery_discharge_applied_kw <= load + 1e-9
        # Slot cost at any positive price is non-negative.
        price = 100.0  # EUR/MWh
        slot_cost = net_grid * (15 / 60) * (price / 1000.0)
        assert slot_cost >= -1e-9, f"slot {i}: negative cost={slot_cost:.6f}"
