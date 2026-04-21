"""Price forecasters.

Three concrete implementations share the PriceOracle interface:
- GridFMPriceOracle: physics-informed foundation model (NYISO, asayghe1/GridFM).
  If torch or the weights are unavailable, falls back to ChronosPriceOracle,
  then to SeasonalNaiveOracle.
- ChronosPriceOracle: Amazon Chronos-2 zero-shot, works on any 15-min series.
- SeasonalNaiveOracle: median-by-(hour, day-of-week) over training history.

All three expose `get_15min_forecast(now, horizon_slots=96) -> PriceForecast`.

The oracles are stateless with respect to time — the caller passes a context
DataFrame (timestamp, price) covering at least 7 days up to `now`. This keeps
the oracles compatible with the rolling-window evaluation used by the digital
twin during the 14-day test slice.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from aerogrid.config import (
    NYISO_DIR,
    NYISO_ZONE,
    PRICE_ORACLE_IMPL,
    SLOT_MINUTES,
    SLOTS_PER_DAY,
)
from aerogrid.types import PriceForecast

CONTEXT_SLOTS_DEFAULT = 7 * SLOTS_PER_DAY     # 7 days of 15-min history


def _default_prices_parquet() -> Path:
    """Primary price file produced by fetch_nyiso_prices.py."""
    slug = NYISO_ZONE.lower().replace(".", "").replace(" ", "_")
    return NYISO_DIR / f"{slug}_15min.parquet"


def load_price_history(path: Path | None = None) -> pd.DataFrame:
    """Load the configured 15-min price series, sorted by timestamp."""
    path = path or _default_prices_parquet()
    if not path.exists():
        raise FileNotFoundError(
            f"no price parquet at {path}; run fetch_nyiso_prices.py first"
        )
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


class PriceOracle(ABC):
    name: ClassVar[str] = "base"

    @abstractmethod
    def get_15min_forecast(
        self,
        now: datetime,
        context: pd.DataFrame,
        horizon_slots: int = SLOTS_PER_DAY,
    ) -> PriceForecast:
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _slot_start(now: datetime) -> datetime:
        """Round down to the start of the current 15-min slot (UTC)."""
        minute_floor = (now.minute // SLOT_MINUTES) * SLOT_MINUTES
        return now.replace(minute=minute_floor, second=0, microsecond=0)

    @staticmethod
    def _take_context(context: pd.DataFrame, now: datetime,
                      slots: int = CONTEXT_SLOTS_DEFAULT) -> pd.DataFrame:
        ctx = context[context["timestamp"] < now].tail(slots).copy()
        if ctx.empty:
            raise ValueError("no context rows before `now` — load more history")
        return ctx


# --------------------------------------------------------------------------- #
# Seasonal-naive                                                              #
# --------------------------------------------------------------------------- #
class SeasonalNaiveOracle(PriceOracle):
    """Median price by (hour-of-day, day-of-week) over the provided context."""

    name: ClassVar[str] = "naive"

    def get_15min_forecast(self, now, context, horizon_slots=SLOTS_PER_DAY):
        ctx = self._take_context(context, now, slots=28 * SLOTS_PER_DAY)
        ctx = ctx.assign(
            hod=ctx["timestamp"].dt.hour,
            dow=ctx["timestamp"].dt.dayofweek,
            slot_in_hour=(ctx["timestamp"].dt.minute // SLOT_MINUTES),
        )
        tbl = (
            ctx.groupby(["dow", "hod", "slot_in_hour"])["lbmp"]
            .median()
            .reset_index()
        )
        slot0 = self._slot_start(now)
        medians: list[float] = []
        for k in range(horizon_slots):
            t = slot0 + timedelta(minutes=SLOT_MINUTES * k)
            row = tbl[
                (tbl["dow"] == t.weekday())
                & (tbl["hod"] == t.hour)
                & (tbl["slot_in_hour"] == t.minute // SLOT_MINUTES)
            ]
            if row.empty:
                medians.append(float(ctx["lbmp"].median()))
            else:
                medians.append(float(row["lbmp"].iloc[0]))
        med = np.asarray(medians)
        return PriceForecast(
            slot_start=slot0,
            median=list(med),
            q10=list(med - 5.0),
            q90=list(med + 5.0),
            source=self.name,
        )


# --------------------------------------------------------------------------- #
# Chronos-2 (optional dep — falls back to naive)                              #
# --------------------------------------------------------------------------- #
class ChronosPriceOracle(PriceOracle):
    """Zero-shot Chronos-2 over the last 7 days of 15-min history."""

    name: ClassVar[str] = "chronos"

    def __init__(self, model_name: str = "amazon/chronos-t5-tiny"):
        self.model_name = model_name
        self._pipeline = None
        self._fallback = SeasonalNaiveOracle()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch  # type: ignore
            from chronos import ChronosPipeline  # type: ignore
        except ImportError:
            self._pipeline = False
            return False
        try:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        except Exception as e:  # noqa: BLE001
            print(f"ChronosPriceOracle: failed to load {self.model_name}: {e!r}")
            self._pipeline = False
        return self._pipeline

    def get_15min_forecast(self, now, context, horizon_slots=SLOTS_PER_DAY):
        pipeline = self._ensure_pipeline()
        if not pipeline:
            out = self._fallback.get_15min_forecast(now, context, horizon_slots)
            return PriceForecast(
                slot_start=out.slot_start,
                median=out.median,
                q10=out.q10,
                q90=out.q90,
                source=f"{self.name}/fallback_naive",
            )

        import torch  # type: ignore
        ctx = self._take_context(context, now, slots=CONTEXT_SLOTS_DEFAULT)
        series = torch.tensor(ctx["lbmp"].to_numpy(dtype=np.float32))
        quantiles, mean = pipeline.predict_quantiles(
            context=series,
            prediction_length=horizon_slots,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        q = quantiles[0].numpy()              # (horizon, 3)
        q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
        return PriceForecast(
            slot_start=self._slot_start(now),
            median=list(q50.astype(float)),
            q10=list(q10.astype(float)),
            q90=list(q90.astype(float)),
            source=self.name,
        )


# --------------------------------------------------------------------------- #
# GridFM (physics-informed; primary)                                          #
# --------------------------------------------------------------------------- #
class GridFMPriceOracle(PriceOracle):
    """Wrapper around asayghe1/GridFM.

    GridFM expects NYISO-shaped inputs (per-zone LBMP at 5-min, load, emissions).
    Installing the GridFM package is non-trivial (heavy torch + GCN deps), so
    this wrapper is best-effort: if import or weight-load fails, we fall back
    to ChronosPriceOracle and from there to SeasonalNaiveOracle. The `source`
    field on the returned PriceForecast reflects which implementation actually
    produced the numbers so downstream code / notebooks can attribute
    performance correctly.
    """

    name: ClassVar[str] = "gridfm"

    def __init__(self):
        self._model = None
        self._chronos = ChronosPriceOracle()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # type: ignore
            from gridfm import GridFM  # type: ignore
        except ImportError:
            self._model = False
            return False
        try:
            self._model = GridFM.from_pretrained("asayghe1/GridFM")
            self._model.eval()
        except Exception as e:  # noqa: BLE001
            print(f"GridFMPriceOracle: failed to load weights: {e!r}")
            self._model = False
        return self._model

    def get_15min_forecast(self, now, context, horizon_slots=SLOTS_PER_DAY):
        model = self._ensure_model()
        if not model:
            out = self._chronos.get_15min_forecast(now, context, horizon_slots)
            return PriceForecast(
                slot_start=out.slot_start,
                median=out.median,
                q10=out.q10,
                q90=out.q90,
                source=f"{self.name}/fallback_{out.source}",
            )

        import torch  # type: ignore
        ctx = self._take_context(context, now, slots=CONTEXT_SLOTS_DEFAULT)
        # GridFM's public tutorial expects a (T, F) tensor — at minimum LBMP.
        x = torch.tensor(ctx["lbmp"].to_numpy(dtype=np.float32)).unsqueeze(-1)
        with torch.no_grad():
            y = model(x.unsqueeze(0), horizon=horizon_slots)   # (1, H, 3)
        q = y.squeeze(0).cpu().numpy()
        q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
        return PriceForecast(
            slot_start=self._slot_start(now),
            median=list(q50.astype(float)),
            q10=list(q10.astype(float)),
            q90=list(q90.astype(float)),
            source=self.name,
        )


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def make_oracle(impl: str | None = None) -> PriceOracle:
    impl = (impl or PRICE_ORACLE_IMPL).lower()
    if impl == "gridfm":
        return GridFMPriceOracle()
    if impl == "chronos":
        return ChronosPriceOracle()
    if impl == "naive":
        return SeasonalNaiveOracle()
    raise ValueError(f"unknown price oracle impl: {impl!r}")
