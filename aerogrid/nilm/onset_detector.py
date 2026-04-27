"""Streaming onset detector over per-appliance power estimates.

One instance per appliance. The logic mirrors the offline threshold-crossing
detector that used to live in ``scripts/fetch_ukdale_subset.py``:

  - sample power enters via ``.update(p_w, t)``
  - an onset event fires when the signal crosses ``on_power_threshold_w``
    upward, guarded by a minimum-gap debounce (default 10 min)

In the streaming loop the disaggregator feeds per-appliance estimates in, and
the detector emits ``ApplianceOnset`` events that the graph consumes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from aerogrid.types import ApplianceOnset

logger = logging.getLogger(__name__)


@dataclass
class OnsetDetector:
    appliance: str
    threshold_w: float
    debounce: timedelta = timedelta(minutes=10)

    _was_on: bool = False
    _last_onset_at: datetime | None = None

    def update(self, p_w: float, t: datetime, confidence: float = 1.0) -> ApplianceOnset | None:
        """Process one power sample and return an onset event if one fires.

        An onset fires when the signal transitions from below to above
        ``threshold_w`` **and** at least ``debounce`` time has elapsed since
        the last onset.  This prevents spurious re-fires during a single
        sustained cycle.

        Args:
            p_w: Estimated appliance power for this sample (watts).
            t: Wall-clock timestamp of the sample (UTC).
            confidence: Confidence score in [0, 1] attached to the onset event.

        Returns:
            :class:`~aerogrid.types.ApplianceOnset` if an onset fires, else ``None``.
        """
        is_on = p_w > self.threshold_w
        onset = None
        if is_on and not self._was_on:
            if (
                self._last_onset_at is None
                or (t - self._last_onset_at) >= self.debounce
            ):
                onset = ApplianceOnset(
                    appliance=self.appliance,
                    timestamp=t,
                    confidence=float(min(max(confidence, 0.0), 1.0)),
                    source="disaggregator",
                )
                self._last_onset_at = t
                logger.info(
                    "OnsetDetector: onset fired appliance=%s at=%s p_w=%.1fW confidence=%.2f",
                    self.appliance, t.isoformat(), p_w, onset.confidence,
                )
            else:
                gap = (t - self._last_onset_at).total_seconds()
                logger.debug(
                    "OnsetDetector: %s rising edge suppressed by debounce (gap=%.1fs < %.1fs)",
                    self.appliance, gap, self.debounce.total_seconds(),
                )
        elif not is_on and self._was_on:
            logger.debug(
                "OnsetDetector: %s fell below threshold at=%s p_w=%.1fW threshold=%.1fW",
                self.appliance, t.isoformat(), p_w, self.threshold_w,
            )
        self._was_on = is_on
        return onset

    def reset(self) -> None:
        """Reset internal state so the detector treats the next sample as a cold start."""
        logger.debug("OnsetDetector.reset: %s", self.appliance)
        self._was_on = False
        self._last_onset_at = None
