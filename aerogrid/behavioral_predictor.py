"""Per-appliance onset-probability predictor.

Default: HybridBehavioralPredictor — per-appliance Gaussian KDE over hour-of-day
plus a weekend/weekday multiplier, scaled by the empirical daily onset rate.
This is a deliberate "honest baseline": real UK-DALE appliance onset logs are
sparse (a few events per day at most) and dominated by strong hour-of-day /
day-of-week periodicity, so a 1.5B SSM would be overkill. The shipped default
is fast, interpretable, and fits on CPU in milliseconds.

Alternatives are provided but gated:
- ChronosBehavioralPredictor: feeds a Chronos-2 model with binned onset counts.
  Falls back to hybrid if chronos-forecasting/torch aren't installed.
- MambaBehavioralPredictor: raises NotImplementedError — requires mamba-ssm +
  (a) GPU or (b) a mamba.cpp analog which doesn't exist as of April 2026.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from aerogrid.config import (
    APPLIANCES,
    BEHAVIORAL_PREDICTOR_IMPL,
    SLOT_MINUTES,
    SLOTS_PER_DAY,
    UKDALE_DIR,
)


class BehavioralPredictor(ABC):
    name: ClassVar[str] = "base"

    @abstractmethod
    def fit(self, onsets_df: pd.DataFrame) -> "BehavioralPredictor":
        ...

    @abstractmethod
    def predict_onsets(
        self,
        appliance: str,
        start_time: datetime,
        horizon_slots: int = SLOTS_PER_DAY,
    ) -> np.ndarray:
        """Return a (horizon_slots,) array of P(onset in 15-min slot)."""

    def predict_all(
        self, start_time: datetime, horizon_slots: int = SLOTS_PER_DAY
    ) -> dict[str, np.ndarray]:
        return {
            name: self.predict_onsets(name, start_time, horizon_slots)
            for name in APPLIANCES
            if APPLIANCES[name].bufferable and APPLIANCES[name].cycle_slots > 0
        }


# --------------------------------------------------------------------------- #
# Hybrid (default)                                                            #
# --------------------------------------------------------------------------- #
class HybridBehavioralPredictor(BehavioralPredictor):
    """Per-appliance: Gaussian KDE over hour-of-day × weekend/weekday multiplier.

    predict_onsets returns P(onset in 15-min slot) for each slot in the horizon.
    The scaling is set so that the per-slot probabilities sum (over 24h) to the
    empirical daily onset rate — meaning the predictor is calibrated to the
    observed mean number of onsets per day.
    """

    name: ClassVar[str] = "hybrid"

    def __init__(self, kde_bw: float = 0.6):
        self.kde_bw = kde_bw
        self._density_24: dict[str, np.ndarray] = {}       # per-appliance (96,) over 24h
        self._weekend_mult: dict[str, float] = {}
        self._daily_rate: dict[str, float] = {}
        self._fit = False

    def fit(self, onsets_df: pd.DataFrame):
        from scipy.stats import gaussian_kde

        appliances = [a for a in APPLIANCES if APPLIANCES[a].cycle_slots > 0]
        # training window: explicitly the train split
        train = onsets_df[onsets_df["split"] == "train"].copy()
        if train.empty:
            raise ValueError("onsets_df has no rows with split=='train'")

        n_days_total = max(
            1,
            int(
                (train["timestamp"].max() - train["timestamp"].min()).total_seconds()
                // 86400
            ) + 1,
        )
        n_weekend = int((train["timestamp"].dt.dayofweek >= 5).sum())
        n_weekday = int((train["timestamp"].dt.dayofweek < 5).sum())
        n_days_weekend = 2 * (n_days_total / 7.0) or 1.0
        n_days_weekday = 5 * (n_days_total / 7.0) or 1.0

        slot_centers = np.arange(SLOTS_PER_DAY) * (SLOT_MINUTES / 60.0)  # 0, 0.25, ...

        for app in appliances:
            sub = train[train["appliance"] == app]
            n = len(sub)
            if n < 3:
                # not enough data — uniform prior
                self._density_24[app] = np.full(SLOTS_PER_DAY, 1.0 / SLOTS_PER_DAY)
                self._daily_rate[app] = n / n_days_total
                self._weekend_mult[app] = 1.0
                continue
            hod = (sub["timestamp"].dt.hour + sub["timestamp"].dt.minute / 60.0).to_numpy()
            # wrap ±24 h to avoid boundary bias
            wrapped = np.concatenate([hod - 24, hod, hod + 24])
            kde = gaussian_kde(wrapped, bw_method=self.kde_bw)
            raw = kde.evaluate(slot_centers)
            # Normalize so the slot density sums to 1 over 24h (discrete).
            density = raw / (raw.sum() + 1e-12)

            self._density_24[app] = density
            self._daily_rate[app] = n / n_days_total

            # Weekend / weekday mean multiplier.
            wk_rate = (n_weekend / n_days_weekend) if n_days_weekend else 0.0
            wd_rate = (n_weekday / n_days_weekday) if n_days_weekday else 0.0
            base = (n / n_days_total) if n else 1.0
            self._weekend_mult[app] = (wk_rate / base) if base > 0 else 1.0

        self._fit = True
        return self

    def predict_onsets(self, appliance, start_time, horizon_slots=SLOTS_PER_DAY):
        if not self._fit:
            raise RuntimeError("call .fit(onsets_df) first")
        if appliance not in self._density_24:
            return np.zeros(horizon_slots, dtype=np.float64)

        density = self._density_24[appliance]
        daily_rate = self._daily_rate[appliance]
        we_mult = self._weekend_mult[appliance]

        probs = np.empty(horizon_slots, dtype=np.float64)
        t = start_time
        for k in range(horizon_slots):
            slot = (t.hour * 60 + t.minute) // SLOT_MINUTES
            rate_today = daily_rate * (we_mult if t.weekday() >= 5 else 1.0)
            # density is probability that, given an onset happens today, it
            # falls in this 15-min slot. Multiplying by expected onsets/day
            # gives expected onsets in this slot, which for small values ≈ P(onset).
            probs[k] = min(density[slot] * rate_today, 1.0)
            t = t + timedelta(minutes=SLOT_MINUTES)
        return probs


# --------------------------------------------------------------------------- #
# Chronos (alt)                                                               #
# --------------------------------------------------------------------------- #
class ChronosBehavioralPredictor(BehavioralPredictor):
    """Chronos-2 over per-15-min onset counts. Graceful fallback to hybrid."""

    name: ClassVar[str] = "chronos"

    def __init__(self, model_name: str = "amazon/chronos-t5-tiny"):
        self.model_name = model_name
        self._pipeline = None
        self._fallback = HybridBehavioralPredictor()
        self._series: dict[str, np.ndarray] = {}

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch  # type: ignore # noqa: F401
            from chronos import ChronosPipeline  # type: ignore
        except ImportError:
            self._pipeline = False
            return False
        try:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name, device_map="cpu"
            )
        except Exception as e:  # noqa: BLE001
            print(f"ChronosBehavioralPredictor: load failed {e!r}")
            self._pipeline = False
        return self._pipeline

    def fit(self, onsets_df: pd.DataFrame):
        self._fallback.fit(onsets_df)
        train = onsets_df[onsets_df["split"] == "train"].copy()
        if train.empty:
            return self
        start = train["timestamp"].min().floor("15min")
        end = train["timestamp"].max().ceil("15min")
        idx = pd.date_range(start, end, freq="15min", tz="UTC")
        for app, sub in train.groupby("appliance", observed=True):
            counts = pd.Series(0, index=idx, dtype=float)
            bins = sub["timestamp"].dt.floor("15min")
            counts.loc[bins] += 1.0
            self._series[str(app)] = counts.to_numpy()
        return self

    def predict_onsets(self, appliance, start_time, horizon_slots=SLOTS_PER_DAY):
        pipeline = self._ensure_pipeline()
        if not pipeline or appliance not in self._series:
            return self._fallback.predict_onsets(appliance, start_time, horizon_slots)

        import torch  # type: ignore
        ctx = torch.tensor(self._series[appliance][-7 * SLOTS_PER_DAY:], dtype=torch.float32)
        q = pipeline.predict_quantiles(ctx, horizon_slots, quantile_levels=[0.5])[0][0].numpy()
        return np.clip(q.squeeze(), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Mamba stub                                                                  #
# --------------------------------------------------------------------------- #
class MambaBehavioralPredictor(BehavioralPredictor):
    name: ClassVar[str] = "mamba"

    def fit(self, onsets_df):
        raise NotImplementedError(
            "Mamba-3 1.5B inference requires mamba-ssm + GPU (no mamba.cpp exists "
            "as of April 2026). Use BEHAVIORAL_PREDICTOR_IMPL='hybrid' or 'chronos'."
        )

    def predict_onsets(self, appliance, start_time, horizon_slots=SLOTS_PER_DAY):
        raise NotImplementedError("see fit() docstring")


# --------------------------------------------------------------------------- #
# Factory + convenience loader                                                #
# --------------------------------------------------------------------------- #
def make_predictor(impl: str | None = None) -> BehavioralPredictor:
    impl = (impl or BEHAVIORAL_PREDICTOR_IMPL).lower()
    if impl == "hybrid":
        return HybridBehavioralPredictor()
    if impl == "chronos":
        return ChronosBehavioralPredictor()
    if impl == "mamba":
        return MambaBehavioralPredictor()
    raise ValueError(f"unknown behavioral predictor impl: {impl!r}")


def load_onsets(path: Path | None = None) -> pd.DataFrame:
    path = path or UKDALE_DIR / "house_1" / "onsets.parquet"
    df = pd.read_parquet(path)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df
