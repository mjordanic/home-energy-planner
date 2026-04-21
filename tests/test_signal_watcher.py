"""Tests for the SignalWatcher NILM front-end."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from aerogrid.signal_watcher import SignalWatcher
from aerogrid.vi_features import (
    SAMPLES_PER_CYCLE,
    average_signature,
    cosine_similarity,
    extract_single_cycle,
    vi_trajectory_descriptor,
)


FS = 16_000
MAINS = 50.0


def _synthesize(duration_s: float, onset_times_s: list[tuple[float, str]]
                ) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic (voltage, current) stereo signal.

    onset_times_s: list of (t_seconds, 'resistive'|'inductive'). Each onset
    adds a 4-second exponentially-decaying burst.
    """
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(12345)
    voltage = 230 * np.sqrt(2) * np.sin(2 * np.pi * MAINS * t)
    current = 0.8 * np.sin(2 * np.pi * MAINS * t) + rng.normal(0, 0.03, n)
    for ts, kind in onset_times_s:
        s0 = int(ts * FS)
        nburst = FS * 4
        s1 = min(s0 + nburst, n)
        tb = np.arange(s1 - s0) / FS
        if kind == "resistive":
            amp = 9.5 * np.exp(-tb / 2.0) + 2.5
            phase = 0.0
        else:
            amp = 7.0 * np.exp(-tb / 1.5) + 1.5
            phase = np.pi / 4
        current[s0:s1] += amp * np.sin(2 * np.pi * MAINS * tb - phase)
    return voltage.astype(np.float32), current.astype(np.float32)


def _build_signatures() -> dict[str, np.ndarray]:
    """Extract signatures from clean planted bursts."""
    v, c = _synthesize(120.0, [(10.0, "resistive"), (50.0, "inductive"),
                               (80.0, "resistive"), (100.0, "inductive")])
    sigs: dict[str, list] = {"dishwasher": [], "washing_machine": []}
    # known onsets: 10, 50, 80, 100
    for ts, kind in [(10, "dishwasher"), (50, "washing_machine"),
                     (80, "dishwasher"), (100, "washing_machine")]:
        base = int((ts + 0.5) * FS)
        for k in range(6):
            s0 = base + k * SAMPLES_PER_CYCLE
            s1 = s0 + SAMPLES_PER_CYCLE
            if s1 > len(v):
                break
            vi = extract_single_cycle(v[s0:s1], c[s0:s1], start_idx=0)
            sigs[kind].append(vi)
    return {k: average_signature(list(map(tuple, zip(*[(p[0], p[1]) for p in pairs]))
                                        if pairs else [])) if False
            else average_signature([(vv, cc) for vv, cc in pairs])
            for k, pairs in sigs.items()}


def test_known_transient_is_detected_and_classified():
    sigs = _build_signatures()
    # quick sanity: signatures should be distinguishable
    assert cosine_similarity(sigs["dishwasher"], sigs["washing_machine"]) < 0.99

    v, c = _synthesize(60.0, [(15.0, "resistive"), (40.0, "inductive")])
    sw = SignalWatcher(signatures=sigs)
    events = sw.process_window(v, c, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert len(events) == 2, f"expected 2 events, got {len(events)}"
    times = sorted((e.timestamp.second + e.timestamp.minute * 60) for e in events)
    assert abs(times[0] - 15) < 2, f"first onset drift: {times[0]}"
    assert abs(times[1] - 40) < 2, f"second onset drift: {times[1]}"

    by_time = sorted(events, key=lambda e: e.timestamp)
    assert by_time[0].appliance == "dishwasher"
    assert by_time[1].appliance == "washing_machine"
    for e in events:
        assert e.confidence > 0.8


def test_unknown_transient_is_rejected():
    # Build signatures but throw a CAPACITIVE (phase +π/3) transient at the
    # watcher — it should match neither and be labelled unknown.
    sigs = _build_signatures()
    n = FS * 20
    t = np.arange(n) / FS
    rng = np.random.default_rng(0)
    voltage = (230 * np.sqrt(2) * np.sin(2 * np.pi * MAINS * t)).astype(np.float32)
    current = (0.8 * np.sin(2 * np.pi * MAINS * t) + rng.normal(0, 0.03, n)).astype(np.float32)
    s0 = FS * 5
    s1 = s0 + FS * 4
    tb = np.arange(s1 - s0) / FS
    current[s0:s1] += (6.0 * np.exp(-tb / 1.5) + 1.0) * np.sin(
        2 * np.pi * MAINS * tb + np.pi / 3
    )

    # Raise the match threshold so neither resistive nor inductive signature
    # claims a capacitive transient.
    sw = SignalWatcher(signatures=sigs, match_threshold=0.99)
    events = sw.process_window(voltage, current,
                               datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert events == [], f"expected no events, got {events}"


def test_vi_descriptor_is_l2_per_half():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, SAMPLES_PER_CYCLE)
    c = rng.normal(0, 1, SAMPLES_PER_CYCLE)
    d = vi_trajectory_descriptor(v, c)
    half = len(d) // 2
    assert abs(np.linalg.norm(d[:half]) - 1.0) < 1e-6
    assert abs(np.linalg.norm(d[half:]) - 1.0) < 1e-6
