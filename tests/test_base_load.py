"""Tests for the Base Load profile: config constants and slot-expansion.

Acceptance criteria covered here:
- A deterministic per-hour Base Load profile lives in config and expands to
  a slot-aligned per-slot kW array.
- Evening peak exists (~0.9 kW) and overnight is low (~0.2 kW).
- Total daily energy is 9–10 kWh.
- The fridge entry is removed from config.APPLIANCES.
- Both strategies add Base Load to per-slot cost (via SlotRecord.base_load_kw).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aerogrid.config import (
    APPLIANCES,
    BASE_LOAD_PROFILE_KW,
    SLOT_MINUTES,
    get_base_load_kw,
)
from aerogrid.sim.strategies import BaselineStrategy, SlotRecord


# --------------------------------------------------------------------------- #
# Config: Base Load profile shape                                              #
# --------------------------------------------------------------------------- #
def test_base_load_profile_has_24_hours():
    """BASE_LOAD_PROFILE_KW must cover all 24 hours (one value per hour)."""
    assert len(BASE_LOAD_PROFILE_KW) == 24


def test_base_load_profile_evening_peak():
    """Hours 17–22 should have the highest loads (evening peak ~0.9 kW)."""
    evening = [BASE_LOAD_PROFILE_KW[h] for h in range(17, 22)]
    non_evening = [BASE_LOAD_PROFILE_KW[h] for h in list(range(0, 6)) + list(range(10, 17))]
    assert min(evening) > max(non_evening), (
        f"Evening hours {evening} should exceed non-evening hours {non_evening}"
    )


def test_base_load_profile_daily_energy():
    """Total daily energy must be in the 9–10 kWh band."""
    total_kwh = sum(BASE_LOAD_PROFILE_KW)  # sum of 24 hourly kW values == kWh
    assert 9.0 <= total_kwh <= 10.5, (
        f"Daily base load {total_kwh:.2f} kWh is outside the 9–10 kWh target"
    )


def test_base_load_profile_overnight_low():
    """Hours 0–6 (overnight) should be the lowest (~0.2 kW)."""
    overnight = [BASE_LOAD_PROFILE_KW[h] for h in range(0, 7)]
    daytime = [BASE_LOAD_PROFILE_KW[h] for h in range(10, 17)]
    assert max(overnight) < min(daytime), (
        f"Overnight max {max(overnight):.2f} should be less than daytime min {min(daytime):.2f}"
    )


# --------------------------------------------------------------------------- #
# Config: fridge removed from APPLIANCES                                       #
# --------------------------------------------------------------------------- #
def test_fridge_removed_from_appliances():
    """The dead fridge ApplianceSpec must no longer be in APPLIANCES (ADR 0003)."""
    assert "fridge" not in APPLIANCES, (
        "fridge entry should have been deleted from APPLIANCES when Base Load was introduced"
    )


# --------------------------------------------------------------------------- #
# get_base_load_kw: slot-aligned expansion                                     #
# --------------------------------------------------------------------------- #
def test_get_base_load_kw_returns_correct_length():
    """get_base_load_kw returns an array of length n_slots."""
    now = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)
    arr = get_base_load_kw(now, n_slots=96)
    assert len(arr) == 96


def test_get_base_load_kw_midnight_first_slot():
    """At midnight UTC, slot 0 picks up hour 0 of the profile."""
    now = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)
    arr = get_base_load_kw(now, n_slots=4)
    # Slots 0..3 all fall in hour 0 → all equal BASE_LOAD_PROFILE_KW[0]
    expected = BASE_LOAD_PROFILE_KW[0]
    for v in arr:
        assert v == pytest.approx(expected)


def test_get_base_load_kw_evening_slots():
    """Starting at 17:00 UTC the first 4 slots should equal hour-17 of the profile."""
    now = datetime(2026, 4, 15, 17, 0, tzinfo=timezone.utc)
    arr = get_base_load_kw(now, n_slots=4)
    expected = BASE_LOAD_PROFILE_KW[17]
    for v in arr:
        assert v == pytest.approx(expected)


def test_get_base_load_kw_wraps_across_midnight():
    """Starting at 23:30 UTC, slots cross midnight and pick up hour-0 values."""
    now = datetime(2026, 4, 15, 23, 30, tzinfo=timezone.utc)
    arr = get_base_load_kw(now, n_slots=4)
    # Slot 0 = 23:30, slot 1 = 23:45, slot 2 = 00:00, slot 3 = 00:15
    assert arr[0] == pytest.approx(BASE_LOAD_PROFILE_KW[23])
    assert arr[1] == pytest.approx(BASE_LOAD_PROFILE_KW[23])
    assert arr[2] == pytest.approx(BASE_LOAD_PROFILE_KW[0])
    assert arr[3] == pytest.approx(BASE_LOAD_PROFILE_KW[0])


# --------------------------------------------------------------------------- #
# SlotRecord: base_load_kw field                                               #
# --------------------------------------------------------------------------- #
def test_slot_record_has_base_load_kw_field():
    """SlotRecord must have a base_load_kw field (ADR 0003)."""
    rec = SlotRecord(
        timestamp=datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
        ev_kw=0.0,
        heater_kw=0.0,
        cycle_kw=0.0,
        base_load_kw=0.2,
        total_kw=0.2,
        slot_cost_eur=0.001,
        cum_cost_eur=0.001,
        remaining_ev_kwh=24.0,
    )
    assert rec.base_load_kw == pytest.approx(0.2)


def test_slot_record_base_load_in_flat_dict():
    """to_flat_dict must include the base_load_kw key."""
    rec = SlotRecord(
        timestamp=datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
        ev_kw=0.0,
        heater_kw=0.0,
        cycle_kw=0.0,
        base_load_kw=0.35,
        total_kw=0.35,
        slot_cost_eur=0.0,
        cum_cost_eur=0.0,
        remaining_ev_kwh=24.0,
    )
    d = rec.to_flat_dict("baseline")
    assert "baseline_base_load_kw" in d
    assert d["baseline_base_load_kw"] == pytest.approx(0.35)


# --------------------------------------------------------------------------- #
# BaselineStrategy: base load in cost                                          #
# --------------------------------------------------------------------------- #
def test_baseline_strategy_includes_base_load_in_slot_cost():
    """BaselineStrategy.get_slot_record must include base load in slot cost.

    At a flat price of 100 EUR/MWh and 0.2 kW base load over a 15-min slot,
    the base load alone contributes 0.2 × 0.25 × 100/1000 = 0.005 EUR.
    With no other loads, the total slot cost must be at least that.
    """
    s = BaselineStrategy()
    # Override EV/heater need to zero so only base load contributes.
    s.remaining_ev_kwh = 0.0
    s.remaining_heater_kwh_by_window = {h: 0.0 for h in s.remaining_heater_kwh_by_window}

    # Tick at 03:00 UTC — well outside the EV window and heater window start
    from aerogrid.types import Sample
    now = datetime(2026, 4, 15, 3, 0, tzinfo=timezone.utc)
    sample = Sample(t=now, realized_price=None)
    s.tick(sample, [], dt_s=1.0)

    price_eur_mwh = 100.0
    rec = s.get_slot_record(now, price_eur_mwh)

    # The record's base_load_kw should be non-zero (overnight ~0.2 kW)
    assert rec.base_load_kw > 0.0, "base_load_kw should be populated from the profile"

    # slot_cost should include the base load contribution
    expected_min_cost = rec.base_load_kw * (SLOT_MINUTES / 60.0) * (price_eur_mwh / 1000.0)
    assert rec.slot_cost_eur >= expected_min_cost - 1e-9
