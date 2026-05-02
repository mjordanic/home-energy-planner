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

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from aerogrid.config import (
    NYISO_DIR,
    NYISO_ZONE,
    PRICE_ORACLE_IMPL,
    PRICE_SOURCE,
    SLOT_MINUTES,
    SLOTS_PER_DAY,
    SMARD_DIR,
)
from aerogrid.types import PriceForecast

logger = logging.getLogger(__name__)

CONTEXT_SLOTS_DEFAULT = 7 * SLOTS_PER_DAY     # 7 days of 15-min history


def _default_prices_parquet() -> Path:
    """Primary price file; source controlled by PRICE_SOURCE in config."""
    if PRICE_SOURCE == "smard":
        return SMARD_DIR / "de_lu_15min.parquet"
    if PRICE_SOURCE == "entsoe":
        return SMARD_DIR.parent / "entsoe" / "de_lu_15min.parquet"
    # nyiso (legacy)
    slug = NYISO_ZONE.lower().replace(".", "").replace(" ", "_")
    return NYISO_DIR / f"{slug}_15min.parquet"


_FETCH_HINT = {
    "smard": "fetch_smard_prices.py",
    "entsoe": "fetch_entsoe_prices.py",
    "nyiso": "fetch_nyiso_prices.py",
}


def load_price_history(path: Path | None = None) -> pd.DataFrame:
    """Load the configured 15-min price series, sorted by timestamp."""
    path = path or _default_prices_parquet()
    logger.info("load_price_history: loading price data from %s", path)
    if not path.exists():
        hint = _FETCH_HINT.get(PRICE_SOURCE, "the appropriate fetch script")
        logger.error(
            "load_price_history: price parquet not found at %s — run scripts/%s first",
            path, hint,
        )
        raise FileNotFoundError(
            f"no price parquet at {path}; run scripts/{hint} first"
        )
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    logger.info(
        "load_price_history: loaded %d rows spanning %s → %s",
        len(df),
        df["timestamp"].iloc[0].isoformat() if len(df) else "N/A",
        df["timestamp"].iloc[-1].isoformat() if len(df) else "N/A",
    )
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
        """Return the last ``slots`` rows of ``context`` that are strictly before ``now``.

        Raises:
            ValueError: if no rows exist before ``now`` (insufficient history).
        """
        ctx = context[context["timestamp"] < now].tail(slots).copy()
        if ctx.empty:
            logger.error(
                "_take_context: no rows before now=%s in context of %d rows",
                now.isoformat(), len(context),
            )
            raise ValueError("no context rows before `now` — load more history")
        logger.debug(
            "_take_context: now=%s context_rows=%d requested=%d returned=%d",
            now.isoformat(), len(context), slots, len(ctx),
        )
        return ctx


# --------------------------------------------------------------------------- #
# Seasonal-naive                                                              #
# --------------------------------------------------------------------------- #
class SeasonalNaiveOracle(PriceOracle):
    """Median price by (hour-of-day, day-of-week) over the provided context."""

    name: ClassVar[str] = "naive"

    def get_15min_forecast(self, now, context, horizon_slots=SLOTS_PER_DAY):
        """Return seasonal-median forecast with empirical q10/q90 quantiles.

        Uses up to 28 days of context to build a lookup table of
        median/q10/q90 by ``(day-of-week, hour-of-day, 15-min slot)`` and
        then joins that table onto the forecast horizon timestamps.  Any
        slot combination with no history falls back to the global median/quantiles.
        """
        logger.debug(
            "SeasonalNaiveOracle.get_15min_forecast: now=%s horizon=%d",
            now.isoformat(), horizon_slots,
        )
        ctx = self._take_context(context, now, slots=28 * SLOTS_PER_DAY)
        ctx = ctx.assign(
            hod=ctx["timestamp"].dt.hour,
            dow=ctx["timestamp"].dt.dayofweek,
            slot_in_hour=(ctx["timestamp"].dt.minute // SLOT_MINUTES),
        )
        slot0 = self._slot_start(now)
        season_stats = (
            ctx.groupby(["dow", "hod", "slot_in_hour"])["lbmp"]
            .agg(
                median="median",
                q10=lambda s: s.quantile(0.1),
                q90=lambda s: s.quantile(0.9),
            )
            .reset_index()
        )

        horizon = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start=slot0,
                    periods=horizon_slots,
                    freq=f"{SLOT_MINUTES}min",
                )
            }
        ).assign(
            dow=lambda d: d["timestamp"].dt.dayofweek,
            hod=lambda d: d["timestamp"].dt.hour,
            slot_in_hour=lambda d: d["timestamp"].dt.minute // SLOT_MINUTES,
        )

        forecast = horizon.merge(
            season_stats,
            on=["dow", "hod", "slot_in_hour"],
            how="left",
        )
        fallback = ctx["lbmp"]
        forecast["median"] = forecast["median"].fillna(float(fallback.median()))
        forecast["q10"] = forecast["q10"].fillna(float(fallback.quantile(0.1)))
        forecast["q90"] = forecast["q90"].fillna(float(fallback.quantile(0.9)))

        fc = PriceForecast(
            slot_start=slot0,
            median=forecast["median"].astype(float).tolist(),
            q10=forecast["q10"].astype(float).tolist(),
            q90=forecast["q90"].astype(float).tolist(),
            source=self.name,
        )
        logger.info(
            "SeasonalNaiveOracle: forecast source=%s horizon=%d median[0]=%.2f",
            self.name, horizon_slots, fc.median[0] if fc.median else float("nan"),
        )
        return fc


# --------------------------------------------------------------------------- #
# Chronos-2 (optional dep — falls back to naive)                              #
# --------------------------------------------------------------------------- #
class ChronosPriceOracle(PriceOracle):
    """Zero-shot Chronos-2 over the last 7 days of 15-min history."""

    name: ClassVar[str] = "chronos"

    def __init__(self, model_name: str = "amazon/chronos-t5-tiny"):
        """
        Args:
            model_name: HuggingFace model ID for the Chronos checkpoint.
                Defaults to ``chronos-t5-tiny`` (≈8 M params, CPU-friendly).
        """
        self.model_name = model_name
        self._pipeline = None
        self._fallback = SeasonalNaiveOracle()

    def _ensure_pipeline(self):
        """Lazily load Chronos; returns ``False`` on any import or load failure."""
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch  # type: ignore
            from chronos import ChronosPipeline  # type: ignore
        except ImportError:
            logger.warning(
                "ChronosPriceOracle: torch or chronos not available — will use SeasonalNaive fallback"
            )
            self._pipeline = False
            return False
        try:
            logger.info("ChronosPriceOracle: loading model %s", self.model_name)
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info("ChronosPriceOracle: model %s loaded successfully", self.model_name)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "ChronosPriceOracle: failed to load %s: %r — falling back to SeasonalNaive",
                self.model_name, e,
            )
            self._pipeline = False
        return self._pipeline

    def get_15min_forecast(self, now, context, horizon_slots=SLOTS_PER_DAY):
        """Forecast via Chronos-2, falling back to ``SeasonalNaiveOracle`` if unavailable.

        Uses the last ``CONTEXT_SLOTS_DEFAULT`` (7 days × 96 slots) rows of
        ``context`` as the autoregressive input.  The ``source`` field on the
        returned ``PriceForecast`` is ``"chronos"`` on success and
        ``"chronos/fallback_naive"`` when the fallback fires.
        """
        logger.debug(
            "ChronosPriceOracle.get_15min_forecast: now=%s horizon=%d",
            now.isoformat(), horizon_slots,
        )
        pipeline = self._ensure_pipeline()
        if not pipeline:
            logger.info("ChronosPriceOracle: pipeline unavailable — delegating to SeasonalNaive fallback")
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
        logger.debug("ChronosPriceOracle: running inference on %d context rows", len(ctx))
        series = torch.tensor(ctx["lbmp"].to_numpy(dtype=np.float32))
        quantiles, mean = pipeline.predict_quantiles(
            context=series,
            prediction_length=horizon_slots,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        q = quantiles[0].numpy()              # (horizon, 3)
        q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
        fc = PriceForecast(
            slot_start=self._slot_start(now),
            median=list(q50.astype(float)),
            q10=list(q10.astype(float)),
            q90=list(q90.astype(float)),
            source=self.name,
        )
        logger.info(
            "ChronosPriceOracle: forecast source=%s horizon=%d median[0]=%.2f",
            self.name, horizon_slots, fc.median[0] if fc.median else float("nan"),
        )
        return fc


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
        """Initialise with a lazy GridFM handle and a Chronos fallback chain."""
        self._model = None
        self._chronos = ChronosPriceOracle()

    def _ensure_model(self):
        """Lazily load GridFM weights; returns ``False`` if import or load fails."""
        if self._model is not None:
            return self._model
        try:
            import torch  # type: ignore
            from gridfm import GridFM  # type: ignore
        except ImportError:
            logger.warning(
                "GridFMPriceOracle: gridfm or torch not installed — will cascade to Chronos fallback"
            )
            self._model = False
            return False
        try:
            logger.info("GridFMPriceOracle: loading weights from asayghe1/GridFM")
            self._model = GridFM.from_pretrained("asayghe1/GridFM")
            self._model.eval()
            logger.info("GridFMPriceOracle: GridFM loaded successfully")
        except Exception as e:  # noqa: BLE001
            logger.error(
                "GridFMPriceOracle: failed to load weights: %r — cascading to Chronos", e,
            )
            self._model = False
        return self._model

    def get_15min_forecast(self, now, context, horizon_slots=SLOTS_PER_DAY):
        """Forecast via GridFM, cascading to Chronos then SeasonalNaive on failure.

        The ``source`` field on the returned ``PriceForecast`` records which
        code path actually produced the numbers, e.g. ``"gridfm"``,
        ``"gridfm/fallback_chronos"``, or ``"gridfm/fallback_chronos/fallback_naive"``.
        """
        logger.debug(
            "GridFMPriceOracle.get_15min_forecast: now=%s horizon=%d",
            now.isoformat(), horizon_slots,
        )
        model = self._ensure_model()
        if not model:
            logger.info("GridFMPriceOracle: model unavailable — delegating to Chronos fallback")
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
        logger.debug("GridFMPriceOracle: running inference on %d context rows", len(ctx))
        # GridFM's public tutorial expects a (T, F) tensor — at minimum LBMP.
        x = torch.tensor(ctx["lbmp"].to_numpy(dtype=np.float32)).unsqueeze(-1)
        with torch.no_grad():
            y = model(x.unsqueeze(0), horizon=horizon_slots)   # (1, H, 3)
        q = y.squeeze(0).cpu().numpy()
        q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
        fc = PriceForecast(
            slot_start=self._slot_start(now),
            median=list(q50.astype(float)),
            q10=list(q10.astype(float)),
            q90=list(q90.astype(float)),
            source=self.name,
        )
        logger.info(
            "GridFMPriceOracle: forecast source=%s horizon=%d median[0]=%.2f",
            self.name, horizon_slots, fc.median[0] if fc.median else float("nan"),
        )
        return fc


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def make_oracle(impl: str | None = None) -> PriceOracle:
    """Instantiate a ``PriceOracle`` by implementation name.

    Args:
        impl: One of ``"gridfm"``, ``"chronos"``, or ``"naive"``.
            Defaults to ``PRICE_ORACLE_IMPL`` from ``config.py``.

    Raises:
        ValueError: if ``impl`` is not a recognised oracle name.
    """
    impl = (impl or PRICE_ORACLE_IMPL).lower()
    logger.info("make_oracle: instantiating price oracle impl=%s", impl)
    if impl == "gridfm":
        return GridFMPriceOracle()
    if impl == "chronos":
        return ChronosPriceOracle()
    if impl == "naive":
        return SeasonalNaiveOracle()
    logger.error("make_oracle: unknown impl=%r", impl)
    raise ValueError(f"unknown price oracle impl: {impl!r}")
