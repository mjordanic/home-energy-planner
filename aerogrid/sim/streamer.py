"""Tick streamer — emits one ``Sample`` per simulated second with prices on slot boundaries.

The streamer no longer reads any parquet trace.  Each call to
:meth:`iter_samples` walks ``[start, end)`` at 1 Hz and yields a
:class:`~aerogrid.types.Sample` whose ``realized_price`` is populated only on
15-min slot boundaries (the price comes from the
:class:`~aerogrid.sim.price_server.PriceServer` provider).

Appliance onsets are *injected* manually via :meth:`add_onset` and consumed by
the digital twin once per tick via :meth:`consume_injected_onsets`.  The
streamer is the only entry-point for onsets — there is no "natural" onset
detection because the simulator no longer carries a synthetic mains trace.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterator

from aerogrid.config import SLOT_MINUTES
from aerogrid.types import ApplianceOnset, Sample

logger = logging.getLogger(__name__)


def _slot_floor(t: datetime) -> datetime:
    """Round ``t`` down to the start of its 15-min slot (zeroes seconds/μs)."""
    return t.replace(
        minute=(t.minute // SLOT_MINUTES) * SLOT_MINUTES,
        second=0,
        microsecond=0,
    )


@dataclass
class Streamer:
    """Generate 1 Hz simulation ticks and queue manually injected onsets."""
    realized_price_provider: Callable[[datetime], float | None] | None = None
    _injected_onsets: list[ApplianceOnset] = field(default_factory=list)

    def add_onset(
        self,
        appliance: str,
        timestamp: datetime,
        confidence: float = 1.0,
    ) -> None:
        """Queue an :class:`ApplianceOnset` for emission at ``timestamp``."""
        onset = ApplianceOnset(
            appliance=appliance,
            timestamp=timestamp,
            confidence=float(confidence),
            source="injected",
        )
        self._injected_onsets.append(onset)
        self._injected_onsets.sort(key=lambda o: o.timestamp)
        logger.info(
            "Streamer.add_onset: queued appliance=%s at=%s",
            appliance, timestamp.isoformat(),
        )

    def consume_injected_onsets(self, now: datetime) -> list[ApplianceOnset]:
        """Pop and return all injected onsets whose timestamp is ≤ ``now``."""
        ready: list[ApplianceOnset] = []
        while self._injected_onsets and self._injected_onsets[0].timestamp <= now:
            ready.append(self._injected_onsets.pop(0))
        if ready:
            logger.debug(
                "Streamer.consume_injected_onsets: emitting %d onset(s) at=%s",
                len(ready), now.isoformat(),
            )
        return ready

    def iter_samples(
        self,
        start: datetime,
        end: datetime,
    ) -> Iterator[Sample]:
        """Yield one :class:`~aerogrid.types.Sample` per simulated second.

        The first sample of each 15-min slot carries a non-``None``
        ``realized_price`` from ``realized_price_provider``; all other
        samples in the slot have ``realized_price=None``.
        """
        logger.info(
            "Streamer.iter_samples: window %s → %s",
            start.isoformat(), end.isoformat(),
        )
        provider = self.realized_price_provider
        last_slot: datetime | None = None
        n_slots = 0
        t = start
        step = timedelta(seconds=1)
        while t < end:
            slot = _slot_floor(t)
            realized: float | None = None
            if slot != last_slot:
                realized = provider(slot) if provider is not None else None
                last_slot = slot
                n_slots += 1
                logger.debug(
                    "Streamer: new slot boundary slot=%s realized_price=%s",
                    slot.isoformat(),
                    f"{realized:.2f}" if realized is not None else "None",
                )
            yield Sample(t=t, realized_price=realized)
            t = t + step
        logger.info("Streamer.iter_samples: finished streaming %d slots", n_slots)


__all__ = ["Streamer"]
