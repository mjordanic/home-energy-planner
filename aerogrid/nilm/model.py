"""NILM model interface and perfect (dummy) implementation.

NILM disaggregation is **not** the focus of this project — the project
demonstrates the closed-loop MPC agent (price forecast → behavioral
prediction → MILP optimizer → HITL gate).  We provide:

- :class:`DisaggModel` — abstract base class defining the per-appliance
  disaggregation contract.  Subclass this to plug in a real NILM model
  (e.g. one trained on REDD / UK-DALE / REFIT).
- :class:`PerfectDisaggModel` — returns ground-truth power from the
  simulator's own traces.  Perfect score by construction.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class DisaggModel(ABC):
    """Per-appliance NILM model interface.

    To integrate a real NILM model, subclass ``DisaggModel`` and implement
    :meth:`predict`.  The rest of the pipeline (onset detection, triggers,
    optimizer) will work without changes.
    """

    @abstractmethod
    def predict(self, mains_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict appliance state from aggregate mains windows.

        Args:
            mains_window: ``(N, W)`` array of mains power windows (watts).

        Returns:
            ``(active, power_w)`` where ``active`` is ``(N,)`` binary
            (0 = off, 1 = on) and ``power_w`` is ``(N,)`` estimated
            appliance power in watts.
        """


class PerfectDisaggModel(DisaggModel):
    """Dummy model returning ground-truth appliance power.

    Not a real NILM model — uses the simulator's own per-appliance trace
    so the rest of the pipeline can be developed and tested without a
    trained disaggregator.  Replace with a real :class:`DisaggModel`
    subclass for production use.
    """

    def __init__(self, ground_truth_w: np.ndarray, threshold_w: float = 10.0):
        self.ground_truth_w = np.asarray(ground_truth_w, dtype=np.float32)
        self.threshold_w = threshold_w

    def predict(self, mains_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Not used in streaming mode — see :class:`RollingDisaggregator`."""
        n = mains_window.shape[0]
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    def at(self, idx: int) -> tuple[float, float]:
        """Return ``(active, power_w)`` at sample index *idx*.

        Args:
            idx: Zero-based sample index into the ground-truth trace.

        Returns:
            ``(1.0, power)`` if the appliance power exceeds
            ``threshold_w``, else ``(0.0, power)``.
        """
        if idx < 0 or idx >= len(self.ground_truth_w):
            logger.debug(
                "PerfectDisaggModel.at: idx=%d out of range [0, %d) — returning (0.0, 0.0)",
                idx, len(self.ground_truth_w),
            )
            return 0.0, 0.0
        power = float(self.ground_truth_w[idx])
        active = 1.0 if power > self.threshold_w else 0.0
        return active, power
