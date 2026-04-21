"""Mock 15-min price feed backed by the NYISO test parquet.

Exposes two callables that the LangGraph nodes can bind as providers:
  history_provider(now) -> pd.DataFrame  (timestamp, lbmp) strictly < now
  realized_provider(now) -> float        # realized LBMP at the current 15-min slot

A "surprise spike" can be injected at a configured timestamp so we can exercise
the replan path without waiting for a naturally-volatile slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from aerogrid.config import SLOT_MINUTES
from aerogrid.price_oracle import load_price_history


@dataclass
class PriceServer:
    prices: pd.DataFrame = field(default_factory=load_price_history)
    spike_at: datetime | None = None
    spike_magnitude: float = 150.0          # $/MWh added

    def __post_init__(self):
        self.prices = self.prices.sort_values("timestamp").reset_index(drop=True)

    def _slot_for(self, now: datetime) -> datetime:
        floor = (now.minute // SLOT_MINUTES) * SLOT_MINUTES
        return now.replace(minute=floor, second=0, microsecond=0)

    def history(self, now: datetime) -> pd.DataFrame:
        """Past prices only — strictly less than `now`'s slot."""
        slot = self._slot_for(now)
        return self.prices[self.prices["timestamp"] < slot]

    def realized(self, now: datetime) -> float | None:
        slot = self._slot_for(now)
        row = self.prices[self.prices["timestamp"] == slot]
        if row.empty:
            return None
        price = float(row["lbmp"].iloc[0])
        if self.spike_at is not None and self._slot_for(self.spike_at) == slot:
            price += self.spike_magnitude
        return price
