"""Unit tests for the parametric appliance models."""
from __future__ import annotations

import numpy as np
import pytest

from aerogrid.sim.appliance_models import (
    ApplianceModel,
    DecayGrowModel,
    DecayModel,
    MinMaxModel,
    OnOffModel,
    RandomRangeModel,
)


ALL_MODELS: list[ApplianceModel] = [
    OnOffModel(power_w=2400.0, noise_std_w=10.0),
    DecayModel(peak_w=2500.0, baseline_w=300.0, tau_s=180.0, noise_std_w=10.0),
    DecayGrowModel(
        peak_w=2200.0, trough_w=150.0, tau_decay_s=120.0, tau_grow_s=150.0,
        turn_frac=0.5, noise_std_w=10.0,
    ),
    MinMaxModel(min_w=80.0, max_w=800.0, duty=0.3, sub_cycle_s=60.0, noise_std_w=2.0),
    RandomRangeModel(min_w=50.0, max_w=200.0),
]


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: type(m).__name__)
def test_sample_cycle_shape_and_dtype(model: ApplianceModel):
    rng = np.random.default_rng(0)
    trace = model.sample_cycle(1800, rng)
    assert trace.shape == (1800,)
    assert trace.dtype == np.float32
    assert np.all(trace >= 0.0), "power must be non-negative"


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: type(m).__name__)
def test_same_seed_same_output(model: ApplianceModel):
    a = model.sample_cycle(900, np.random.default_rng(42))
    b = model.sample_cycle(900, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: type(m).__name__)
def test_different_seeds_differ_for_stochastic_models(model: ApplianceModel):
    a = model.sample_cycle(900, np.random.default_rng(1))
    b = model.sample_cycle(900, np.random.default_rng(2))
    # OnOff with noise, MinMax with noise, Decay, DecayGrow, RandomRange — all have RNG input
    # The only risk: if noise_std is 0 everywhere.  None of ALL_MODELS has that.
    assert not np.array_equal(a, b)


def test_on_off_hugs_power_w_on_average():
    m = OnOffModel(power_w=2400.0, noise_std_w=5.0)
    trace = m.sample_cycle(10_000, np.random.default_rng(0))
    assert abs(trace.mean() - 2400.0) < 5.0


def test_decay_starts_near_peak_ends_near_baseline():
    m = DecayModel(peak_w=2500.0, baseline_w=300.0, tau_s=60.0, noise_std_w=0.0)
    trace = m.sample_cycle(600, np.random.default_rng(0))  # 10 time constants
    assert trace[0] == pytest.approx(2500.0, abs=1.0)
    assert trace[-1] == pytest.approx(300.0, abs=5.0)
    # Monotone-decreasing in absence of noise
    assert np.all(np.diff(trace) <= 1e-3)


def test_decay_grow_turns_at_turn_frac():
    m = DecayGrowModel(
        peak_w=2000.0, trough_w=200.0,
        tau_decay_s=60.0, tau_grow_s=60.0,
        turn_frac=0.5, noise_std_w=0.0,
    )
    trace = m.sample_cycle(1200, np.random.default_rng(0))
    turn_idx = 600
    # Minimum is at the turn
    assert np.argmin(trace) == pytest.approx(turn_idx, abs=5)


def test_min_max_has_correct_duty_cycle():
    m = MinMaxModel(min_w=100.0, max_w=1000.0, duty=0.25, sub_cycle_s=40.0, noise_std_w=0.0)
    trace = m.sample_cycle(4000, np.random.default_rng(0))
    frac_high = (trace > 550.0).mean()
    assert abs(frac_high - 0.25) < 0.01


def test_random_range_honors_bounds():
    m = RandomRangeModel(min_w=50.0, max_w=200.0)
    trace = m.sample_cycle(5000, np.random.default_rng(0))
    assert trace.min() >= 50.0 - 1e-6
    assert trace.max() <= 200.0 + 1e-6
    # Approx uniform mean
    assert abs(trace.mean() - 125.0) < 5.0
