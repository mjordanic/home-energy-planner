"""1 Hz sample streamer backed by scenario parquet.

Reads ``mains_1hz.parquet`` from the scenario directory and yields one
``Sample`` per simulated second, attaching the 15-min realized LBMP (from the
:class:`aerogrid.sim.price_server.PriceServer`) when a slot boundary crosses.

The streamer is deliberately thin — it contains no disaggregation logic, no
onset detection, and no scheduling. The digital twin's inner loop is
responsible for feeding each sample into the disaggregator, commit tracker,
and trigger manager.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from aerogrid.config import (
    SCENARIO_DIR,
    SCENARIO_TEST_END,
    SCENARIO_TEST_START,
    SLOT_MINUTES,
)
from aerogrid.types import Sample

logger = logging.getLogger(__name__)


def _slot_floor(t: datetime) -> datetime:
    """Round ``t`` down to the start of its 15-min slot (zeroes seconds/μs)."""
    return t.replace(
        minute=(t.minute // SLOT_MINUTES) * SLOT_MINUTES,
        second=0,
        microsecond=0,
    )


@dataclass
class ScenarioStreamer:
    """Iterate a pre-generated scenario's mains at 1 Hz."""
    mains_path: Path | None = None
    realized_price_provider: Callable[[datetime], float | None] | None = None

    def _load(self) -> pd.DataFrame:
        """Load, tz-localise, and sort the mains parquet into a DataFrame."""
        path = self.mains_path or (SCENARIO_DIR / "mains_1hz.parquet")
        logger.info("ScenarioStreamer._load: reading mains from %s", path)
        if not path.exists():
            logger.error(
                "ScenarioStreamer._load: mains parquet not found at %s — run generate_scenario.py", path,
            )
            raise FileNotFoundError(
                f"no scenario mains at {path}. Run scripts/generate_scenario.py first."
            )
        df = pd.read_parquet(path)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("ScenarioStreamer._load: loaded %d samples", len(df))
        return df

    def iter_samples(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[Sample]:
        """Yield one :class:`~aerogrid.types.Sample` per simulated second.

        The first sample at each 15-min slot boundary carries a non-``None``
        ``realized_price`` (fetched from ``realized_price_provider``); all
        other samples in the slot have ``realized_price=None``.

        Args:
            start: First timestamp to include (defaults to ``SCENARIO_TEST_START``).
            end:   Exclusive upper bound (defaults to ``SCENARIO_TEST_END``).

        Yields:
            :class:`~aerogrid.types.Sample` instances in chronological order.
        """
        df = self._load()
        start = start or SCENARIO_TEST_START
        end = end or SCENARIO_TEST_END
        mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
        window = df.loc[mask]
        logger.info(
            "ScenarioStreamer.iter_samples: window %s → %s (%d samples)",
            start.isoformat(), end.isoformat(), len(window),
        )

        last_slot: datetime | None = None
        provider = self.realized_price_provider
        n_slots = 0
        for row in window.itertuples(index=False):
            t: datetime = row.timestamp.to_pydatetime()
            slot = _slot_floor(t)
            realized: float | None = None
            if slot != last_slot:
                # Fresh 15-min boundary — pull realized price once.
                realized = provider(slot) if provider is not None else None
                last_slot = slot
                n_slots += 1
                logger.debug(
                    "ScenarioStreamer: new slot boundary slot=%s realized_price=%s",
                    slot.isoformat(), f"{realized:.2f}" if realized is not None else "None",
                )
            yield Sample(t=t, p_mains_w=float(row.power_w), realized_price=realized)
        logger.info("ScenarioStreamer.iter_samples: finished streaming %d slots", n_slots)
