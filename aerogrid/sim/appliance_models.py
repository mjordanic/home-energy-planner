"""Parametric appliance power models for the scenario simulator.

Five families inspired by Klemen Jakšič's SmartSim paper
("SmartSim: Simulator for Smart Meter Data"): on_off, decay, decay_grow,
min_max, random_range. Reimplemented from the parametric descriptions,
vectorized with numpy, and threading a ``np.random.Generator`` through every
call so scenarios are deterministic and safe under parallel use.

A model emits a power trace of ``n_samples`` at 1 Hz for a single cycle. The
scenario generator places the trace at the cycle's start time and sums across
all appliances to produce the household aggregate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class ApplianceModel(Protocol):
    """Protocol satisfied by all parametric appliance power models.

    Each model generates a single power-draw cycle as a ``float32`` numpy
    array of length ``n_samples`` at 1 Hz.  The caller places the returned
    array at the cycle's start offset inside the household aggregate trace.
    """

    def sample_cycle(
        self, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:  # pragma: no cover - protocol
        """Generate one power cycle of ``n_samples`` seconds at 1 Hz.

        Args:
            n_samples: Desired length of the output array (seconds).
            rng: Seeded random number generator; used for noise draws so
                the scenario is deterministic given the same seed.

        Returns:
            ``(n_samples,)`` float32 array of instantaneous power in watts,
            clipped to ≥ 0.
        """
        ...


@dataclass(frozen=True)
class OnOffModel:
    """Constant power while running, with optional Gaussian noise.

    Suitable for resistive loads whose draw is set by a thermostat or timer
    (kettles, toasters, incandescent bulbs once warm).
    """
    power_w: float
    noise_std_w: float = 5.0

    def sample_cycle(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return a flat power trace at ``power_w`` with optional Gaussian noise."""
        logger.debug("OnOffModel.sample_cycle: n=%d power_w=%.1f noise=%.1f", n_samples, self.power_w, self.noise_std_w)
        base = np.full(n_samples, self.power_w, dtype=np.float32)
        if self.noise_std_w > 0.0:
            base = base + rng.normal(0.0, self.noise_std_w, size=n_samples).astype(
                np.float32
            )
        return np.maximum(base, 0.0)


@dataclass(frozen=True)
class DecayModel:
    """Exponential decay from peak_w to baseline_w with time constant tau_s.

    Typical of motors that start high and settle, or resistive heaters whose
    element cools and draws less as the cycle progresses.
    """
    peak_w: float
    baseline_w: float
    tau_s: float
    noise_std_w: float = 5.0

    def sample_cycle(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return an exponentially decaying power curve from ``peak_w`` to ``baseline_w``."""
        logger.debug(
            "DecayModel.sample_cycle: n=%d peak_w=%.1f baseline_w=%.1f tau_s=%.1f",
            n_samples, self.peak_w, self.baseline_w, self.tau_s,
        )
        t = np.arange(n_samples, dtype=np.float32)
        tau = max(float(self.tau_s), 1.0)
        base = self.baseline_w + (self.peak_w - self.baseline_w) * np.exp(-t / tau)
        if self.noise_std_w > 0.0:
            base = base + rng.normal(0.0, self.noise_std_w, size=n_samples).astype(
                np.float32
            )
        return np.maximum(base.astype(np.float32), 0.0)


@dataclass(frozen=True)
class DecayGrowModel:
    """Decay from peak_w to trough_w, then regrow toward peak_w.

    Models appliances that draw a surge, settle, then ramp back up — e.g.
    a washing-machine fill/heat/spin profile in coarse shape, or a dishwasher
    that pulls for the heater, pauses for a rinse, and pulls again for dry.
    """
    peak_w: float
    trough_w: float
    tau_decay_s: float
    tau_grow_s: float
    turn_frac: float = 0.5       # fraction of the cycle at which decay → grow
    noise_std_w: float = 5.0

    def sample_cycle(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return a bi-phase power curve: decay until ``turn_frac``, then regrow."""
        logger.debug(
            "DecayGrowModel.sample_cycle: n=%d peak_w=%.1f trough_w=%.1f turn_frac=%.2f",
            n_samples, self.peak_w, self.trough_w, self.turn_frac,
        )
        turn_idx = int(max(0.0, min(1.0, self.turn_frac)) * n_samples)
        t = np.arange(n_samples, dtype=np.float32)
        tau_d = max(float(self.tau_decay_s), 1.0)
        tau_g = max(float(self.tau_grow_s), 1.0)
        decay = self.trough_w + (self.peak_w - self.trough_w) * np.exp(-t / tau_d)
        t_grow = np.maximum(t - turn_idx, 0.0)
        grow = self.peak_w - (self.peak_w - self.trough_w) * np.exp(-t_grow / tau_g)
        base = np.where(t < turn_idx, decay, grow).astype(np.float32)
        if self.noise_std_w > 0.0:
            base = base + rng.normal(0.0, self.noise_std_w, size=n_samples).astype(
                np.float32
            )
        return np.maximum(base, 0.0)


@dataclass(frozen=True)
class MinMaxModel:
    """Two-state PWM oscillation between min_w and max_w.

    Thermostatic appliances (fridge compressor, electric hob on low) approximate
    this: the controller pulses power to hold a setpoint. ``duty`` is the
    fraction of each sub-cycle spent at ``max_w``; ``sub_cycle_s`` is the pulse
    period in seconds.
    """
    min_w: float
    max_w: float
    duty: float
    sub_cycle_s: float
    noise_std_w: float = 2.0

    def sample_cycle(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return a PWM oscillation alternating between ``min_w`` and ``max_w``."""
        logger.debug(
            "MinMaxModel.sample_cycle: n=%d min_w=%.1f max_w=%.1f duty=%.2f sub_cycle_s=%.1f",
            n_samples, self.min_w, self.max_w, self.duty, self.sub_cycle_s,
        )
        sub = max(int(round(self.sub_cycle_s)), 1)
        phase = np.arange(n_samples) % sub
        threshold = int(max(0.0, min(1.0, self.duty)) * sub)
        base = np.where(phase < threshold, self.max_w, self.min_w).astype(np.float32)
        if self.noise_std_w > 0.0:
            base = base + rng.normal(0.0, self.noise_std_w, size=n_samples).astype(
                np.float32
            )
        return np.maximum(base, 0.0)


@dataclass(frozen=True)
class RandomRangeModel:
    """Uniform random power within [min_w, max_w].

    Suitable for electronics with highly variable load (TVs, desktops, chargers
    on an irregular workload).
    """
    min_w: float
    max_w: float

    def sample_cycle(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return uniformly distributed random power in ``[min_w, max_w]``."""
        logger.debug(
            "RandomRangeModel.sample_cycle: n=%d min_w=%.1f max_w=%.1f",
            n_samples, self.min_w, self.max_w,
        )
        lo, hi = float(self.min_w), float(self.max_w)
        if hi < lo:
            lo, hi = hi, lo
        return rng.uniform(lo, hi, size=n_samples).astype(np.float32)


__all__ = [
    "ApplianceModel",
    "OnOffModel",
    "DecayModel",
    "DecayGrowModel",
    "MinMaxModel",
    "RandomRangeModel",
]
