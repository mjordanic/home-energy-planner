from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
