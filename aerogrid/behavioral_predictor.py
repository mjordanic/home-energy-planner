"""Per-appliance onset-probability predictor.

Default: HybridBehavioralPredictor — per-appliance 15-min slot histogram
with circular smoothing plus a weekend/weekday multiplier.
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

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from aerogrid.config import (
    APPLIANCES,
    BEHAVIORAL_PREDICTOR_IMPL,
    SCENARIO_DIR,
    SLOT_MINUTES,
    SLOTS_PER_DAY,
)

logger = logging.getLogger(__name__)


class BehavioralPredictor(ABC):
    """Abstract base for per-appliance onset-probability predictors.

    Subclasses implement ``fit`` and ``predict_onsets``; the shared
    ``predict_all`` wraps ``predict_onsets`` for every bufferable,
    cycle-based appliance in the global ``APPLIANCES`` registry.
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def fit(self, onsets_df: pd.DataFrame) -> "BehavioralPredictor":
        """Train the predictor on historical onset records.

        Args:
            onsets_df: DataFrame with columns ``timestamp`` (UTC tz-aware
                datetime), ``appliance`` (str), and ``split``
                (``"train"`` | ``"test"``).  Only the ``"train"`` rows
                are used; the ``"test"`` split is held out.

        Returns:
            ``self`` (for chaining).
        """
        ...

    @abstractmethod
    def predict_onsets(
        self,
        appliance: str,
        start_time: datetime,
        horizon_slots: int = SLOTS_PER_DAY,
    ) -> np.ndarray:
        """Return a (horizon_slots,) float64 array of P(onset in each 15-min slot).

        Values are in [0, 1]; they represent the probability that the
        named appliance starts a new cycle during each successive 15-min
        window beginning at ``start_time``.
        """

    def predict_all(
        self, start_time: datetime, horizon_slots: int = SLOTS_PER_DAY
    ) -> dict[str, np.ndarray]:
        """Call ``predict_onsets`` for every bufferable, cycle-based appliance.

        Returns a dict keyed by appliance name; excludes the EV charger
        (``cycle_slots == 0``) and any non-bufferable loads.
        """
        return {
            name: self.predict_onsets(name, start_time, horizon_slots)
            for name in APPLIANCES
            if APPLIANCES[name].bufferable and APPLIANCES[name].cycle_slots > 0
        }


# --------------------------------------------------------------------------- #
# Hybrid (default)                                                            #
# --------------------------------------------------------------------------- #
class HybridBehavioralPredictor(BehavioralPredictor):
    """Per-appliance slot histogram + weekend multiplier baseline.

    The model learns three interpretable pieces per appliance:
    1) ``slot_pmf``: a 96-slot onset-time distribution (smoothed histogram).
    2) ``daily_rate``: empirical mean onsets/day from training data.
    3) ``weekend_mult``: weekend daily-rate multiplier relative to all days.

    ``predict_onsets`` then applies:

        P(onset in slot k) ~= slot_pmf[slot_of_day_k] * daily_rate * day_multiplier

    where ``day_multiplier`` is ``weekend_mult`` on Sat/Sun and ``1.0`` otherwise.
    """

    name: ClassVar[str] = "hybrid"

    def __init__(self, smoothing_window: int = 5, min_onsets_for_profile: int = 3):
        """
        Args:
            smoothing_window: Circular moving-average window (in 15-min slots)
                applied to each appliance slot histogram. Must be >= 1.
            min_onsets_for_profile: Minimum number of training onsets required
                to fit a slot profile for an appliance. Appliances below this
                threshold are treated as unsupported and return zero onsets.
        """
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if min_onsets_for_profile < 1:
            raise ValueError("min_onsets_for_profile must be >= 1")
        self.smoothing_window = int(smoothing_window)
        self.min_onsets_for_profile = int(min_onsets_for_profile)
        self._density_24: dict[str, np.ndarray] = {}       # per-appliance (96,) over 24h
        self._weekend_mult: dict[str, float] = {}
        self._daily_rate: dict[str, float] = {}
        self._fit = False

    def fit(self, onsets_df: pd.DataFrame):
        """Fit a pandas-first histogram model over training onsets.

        Steps:
        1) Build per-appliance/day onset counts -> ``daily_rate``.
        2) Build per-appliance/weekend-day onset counts -> ``weekend_mult``.
        3) Build per-appliance 96-slot histogram, smooth circularly, normalize.

        Appliances with fewer than ``min_onsets_for_profile`` onsets are left
        unsupported; ``predict_onsets`` then returns all zeros for them.
        """
        # Keep only training rows; test rows are intentionally excluded from fit.
        train = onsets_df[onsets_df["split"] == "train"].copy()
        if train.empty:
            logger.error(
                "HybridBehavioralPredictor.fit: onsets_df has no 'train' rows — cannot fit"
            )
            raise ValueError("onsets_df has no rows with split=='train'")

        logger.info(
            "HybridBehavioralPredictor.fit: fitting on %d training onset events", len(train),
        )

        # Reset learned state so repeated fit() calls do not mix old/new models.
        self._density_24.clear()
        self._weekend_mult.clear()
        self._daily_rate.clear()

        # Candidate appliances come from whatever is present in train data.
        # A later threshold check decides whether each appliance is modelable.
        appliances = sorted(train["appliance"].astype(str).unique().tolist())
        logger.debug(
            "HybridBehavioralPredictor.fit: candidate appliances=%s", appliances,
        )

        # Calendar/date features used by both daily-rate and slot-profile steps.
        train["date"] = train["timestamp"].dt.floor("D")
        train["slot"] = (
            (train["timestamp"].dt.hour * 60 + train["timestamp"].dt.minute) // SLOT_MINUTES
        ).astype(int)
        train["is_weekend"] = train["timestamp"].dt.dayofweek >= 5

        # Build an explicit full day index over the train span.
        # This denominator avoids bias from only counting "days with events".
        dates = (
            pd.DataFrame({"date": pd.date_range(train["date"].min(), train["date"].max(), freq="D")})
            .assign(is_weekend=lambda x: x["date"].dt.dayofweek >= 5)
        )
        n_days_total = len(dates)
        n_days_weekend = int(dates["is_weekend"].sum())

        # Per-appliance per-day event counts.
        # Columns: appliance, date, onsets.
        per_day = (
            train.groupby(["appliance", "date"], observed=True)
            .size()
            .rename("onsets")
            .reset_index()
        )

        # Baseline expected onsets/day per appliance over all calendar days.
        daily_rate = (
            per_day.groupby("appliance", observed=True)["onsets"]
            .sum()
            .div(max(1, n_days_total))
            .to_dict()
        )

        # Total weekend/weekday events by appliance.
        # Table shape: index=appliance, columns={False, True}, values=event totals.
        weekend_day_counts = (
            per_day.merge(dates, on="date", how="left")
            .groupby(["appliance", "is_weekend"], observed=True)["onsets"]
            .sum()
            .unstack(fill_value=0)
            .astype(np.float64)
        )

        for app in appliances:
            # Slice event rows for this appliance.
            sub = train[train["appliance"] == app]
            if len(sub) < self.min_onsets_for_profile:
                # Not enough signal to infer a reliable time-of-day profile.
                # Keep appliance unsupported so predict_onsets returns zeros.
                logger.warning(
                    "HybridBehavioralPredictor.fit: %s has only %d onsets "
                    "(< min_onsets_for_profile=%d) — will return zero probs",
                    app, len(sub), self.min_onsets_for_profile,
                )
                continue

            # 96-bin histogram over slots; missing slots default to zero.
            slot_counts = (
                sub.groupby("slot", observed=True)
                .size()
                .reindex(np.arange(SLOTS_PER_DAY), fill_value=0)
                .to_numpy(dtype=np.float64)
            )

            if self.smoothing_window > 1:
                # Circular moving average:
                # wrap both ends so midnight-adjacent slots smooth into each other.
                pad = self.smoothing_window // 2
                wrapped = np.pad(slot_counts, (pad, pad), mode="wrap")
                kernel = np.ones(self.smoothing_window, dtype=np.float64) / self.smoothing_window
                slot_counts = np.convolve(wrapped, kernel, mode="valid")
                if slot_counts.shape[0] != SLOTS_PER_DAY:
                    slot_counts = slot_counts[:SLOTS_PER_DAY]

            # Normalize histogram -> PMF over day slots (shape component).
            density = slot_counts / (slot_counts.sum() + 1e-12)
            self._density_24[app] = density

            # Store baseline expected onsets/day (scale component).
            base = float(daily_rate.get(app, 0.0))
            self._daily_rate[app] = base

            # Weekend multiplier:
            # weekend_rate / baseline_daily_rate.
            # If no weekend days exist, weekend rate defaults to zero.
            wk_events = float(
                weekend_day_counts.loc[app, True]
                if app in weekend_day_counts.index and True in weekend_day_counts.columns
                else 0.0
            )
            wk_rate = wk_events / n_days_weekend if n_days_weekend else 0.0
            # If baseline is zero, default to neutral multiplier (=1.0).
            self._weekend_mult[app] = (wk_rate / base) if base > 0 else 1.0

        # Mark model as fitted (enables predict_onsets()).
        self._fit = True
        logger.info(
            "HybridBehavioralPredictor.fit: fitted %d appliances (%s)",
            len(self._density_24), sorted(self._density_24.keys()),
        )
        for app in self._density_24:
            logger.debug(
                "HybridBehavioralPredictor.fit: %s daily_rate=%.4f weekend_mult=%.3f",
                app, self._daily_rate[app], self._weekend_mult[app],
            )
        return self

    def predict_onsets(self, appliance, start_time, horizon_slots=SLOTS_PER_DAY):
        """Return per-slot onset probabilities for ``appliance`` starting at ``start_time``.

        For each slot k the probability is:

            P(onset in slot k) ≈ density[slot_of_day] × daily_rate × weekend_mult

        where ``density`` sums to 1 over 24 h, ``daily_rate`` is the observed
        mean onsets/day, and ``weekend_mult`` adjusts for Saturday/Sunday
        patterns.  Values are clipped to [0, 1].

        This implementation is intentionally equivalent to the explicit
        step-by-step prediction cell in ``notebooks/08_behavioral_predictor_low_level.ipynb``.

        Raises:
            RuntimeError: if ``fit`` has not been called yet.
        """
        # Guard 1: prediction is only valid after fit() has populated learned
        # per-appliance parameters.
        if not self._fit:
            logger.error("HybridBehavioralPredictor.predict_onsets: called before fit()")
            raise RuntimeError("call .fit(onsets_df) first")

        # Guard 2: appliances that were not modeled during fit()
        # (e.g., insufficient onsets) return an all-zero probability vector.
        # This makes "unsupported appliance" behavior explicit and safe.
        if appliance not in self._density_24:
            logger.debug(
                "HybridBehavioralPredictor.predict_onsets: %s not in model — returning zeros",
                appliance,
            )
            return np.zeros(horizon_slots, dtype=np.float64)

        # Learned per-appliance components:
        # - density: 96-slot PMF over a day (shape only, sums ~1)
        # - daily_rate: expected onsets/day (scale)
        # - we_mult: weekend multiplier relative to baseline daily_rate
        density = self._density_24[appliance]
        daily_rate = self._daily_rate[appliance]
        we_mult = self._weekend_mult[appliance]

        # Allocate output vector for horizon probabilities.
        probs = np.empty(horizon_slots, dtype=np.float64)

        # Rolling timestamp cursor; moved forward by SLOT_MINUTES each step.
        t = start_time
        for k in range(horizon_slots):
            # Map current timestamp to slot-of-day index in [0, SLOTS_PER_DAY-1].
            # Example at 15-min resolution: 00:00 -> 0, 00:15 -> 1, ..., 23:45 -> 95.
            slot = (t.hour * 60 + t.minute) // SLOT_MINUTES

            # Weekend-aware daily rate:
            # on Sat/Sun, scale baseline daily_rate by learned weekend multiplier.
            rate_today = daily_rate * (we_mult if t.weekday() >= 5 else 1.0)

            # density is probability that, given an onset happens today, it
            # falls in this 15-min slot. Multiplying by expected onsets/day
            # gives expected onsets in this slot, which for small values ≈ P(onset).
            #
            # The min(..., 1.0) clamp enforces probabilistic bounds in edge cases
            # where rate_today is high.
            probs[k] = min(density[slot] * rate_today, 1.0)

            # Advance to next horizon slot.
            t = t + timedelta(minutes=SLOT_MINUTES)
        logger.debug(
            "HybridBehavioralPredictor.predict_onsets: %s horizon=%d "
            "max_prob=%.4f mean_prob=%.4f",
            appliance, horizon_slots, probs.max(), probs.mean(),
        )
        return probs


# --------------------------------------------------------------------------- #
# Chronos (alt)                                                               #
# --------------------------------------------------------------------------- #
class ChronosBehavioralPredictor(BehavioralPredictor):
    """Chronos-2 over per-15-min onset counts. Graceful fallback to hybrid."""

    name: ClassVar[str] = "chronos"

    def __init__(self, model_name: str = "amazon/chronos-t5-tiny"):
        """
        Args:
            model_name: HuggingFace model ID for the Chronos checkpoint.
                Defaults to ``chronos-t5-tiny`` (≈8 M params, CPU-friendly).
                Larger variants (``small``, ``base``, ``large``) improve
                accuracy at the cost of inference latency.
        """
        self.model_name = model_name
        self._pipeline = None
        self._fallback = HybridBehavioralPredictor()
        self._series: dict[str, np.ndarray] = {}

    def _ensure_pipeline(self):
        """Lazily load the Chronos pipeline; returns ``False`` if unavailable.

        On first call the method attempts to import ``torch`` and
        ``chronos.ChronosPipeline`` and load the pretrained weights.  Any
        import or load failure sets ``self._pipeline = False`` so subsequent
        calls skip the expensive import attempt.
        """
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch  # type: ignore # noqa: F401
            from chronos import ChronosPipeline  # type: ignore
        except ImportError:
            logger.warning(
                "ChronosBehavioralPredictor: torch or chronos not available — using HybridBehavioralPredictor fallback"
            )
            self._pipeline = False
            return False
        try:
            logger.info("ChronosBehavioralPredictor: loading model %s", self.model_name)
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name, device_map="cpu"
            )
            logger.info("ChronosBehavioralPredictor: model %s loaded", self.model_name)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "ChronosBehavioralPredictor: load failed for %s: %r — falling back to Hybrid",
                self.model_name, e,
            )
            self._pipeline = False
        return self._pipeline

    def fit(self, onsets_df: pd.DataFrame):
        """Fit the hybrid fallback and build per-appliance 15-min onset count series.

        The count series (one float per 15-min slot over the training window)
        serves as the Chronos context when ``predict_onsets`` is called.
        """
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
        """Predict onset probabilities via Chronos, falling back to hybrid if unavailable.

        Uses the last 7 days of the fitted count series as context.  Falls
        back to ``HybridBehavioralPredictor`` if the Chronos pipeline failed
        to load or if no training series exists for ``appliance``.
        """
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
    """Placeholder for a Mamba-3 SSM predictor — not yet runnable on CPU.

    Raises ``NotImplementedError`` on both ``fit`` and ``predict_onsets``.
    The stub exists so the factory and config can reference the ``"mamba"``
    implementation name without a silent fallback hiding the gap.
    """

    name: ClassVar[str] = "mamba"

    def fit(self, onsets_df):
        """Not implemented — requires mamba-ssm + GPU."""
        raise NotImplementedError(
            "Mamba-3 1.5B inference requires mamba-ssm + GPU (no mamba.cpp exists "
            "as of April 2026). Use BEHAVIORAL_PREDICTOR_IMPL='hybrid' or 'chronos'."
        )

    def predict_onsets(self, appliance, start_time, horizon_slots=SLOTS_PER_DAY):
        """Not implemented — see ``fit`` docstring."""
        raise NotImplementedError("see fit() docstring")


# --------------------------------------------------------------------------- #
# Factory + convenience loader                                                #
# --------------------------------------------------------------------------- #
def make_predictor(impl: str | None = None) -> BehavioralPredictor:
    """Instantiate a ``BehavioralPredictor`` by implementation name.

    Args:
        impl: One of ``"hybrid"``, ``"chronos"``, or ``"mamba"``.  Defaults
            to ``BEHAVIORAL_PREDICTOR_IMPL`` from ``config.py``.

    Raises:
        ValueError: if ``impl`` is not a recognised predictor name.
    """
    impl = (impl or BEHAVIORAL_PREDICTOR_IMPL).lower()
    logger.info("make_predictor: instantiating behavioral predictor impl=%s", impl)
    if impl == "hybrid":
        return HybridBehavioralPredictor()
    if impl == "chronos":
        return ChronosBehavioralPredictor()
    if impl == "mamba":
        return MambaBehavioralPredictor()
    logger.error("make_predictor: unknown impl=%r", impl)
    raise ValueError(f"unknown behavioral predictor impl: {impl!r}")


def load_onsets(path: Path | None = None) -> pd.DataFrame:
    """Load the onset training table produced by ``generate_scenario.py``.

    Returns a DataFrame with columns ``timestamp`` (UTC tz-aware), ``appliance``
    (str), and ``split`` (``"train"`` | ``"test"``).

    Args:
        path: Override the default path ``data/scenario/onsets.parquet``.

    Raises:
        FileNotFoundError: propagated from ``pd.read_parquet`` if the file
            is absent — run ``scripts/generate_scenario.py`` first.
    """
    path = path or SCENARIO_DIR / "onsets.parquet"
    logger.info("load_onsets: loading onset data from %s", path)
    df = pd.read_parquet(path)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    logger.info(
        "load_onsets: loaded %d onset events across appliances=%s",
        len(df),
        sorted(df["appliance"].unique().tolist()) if len(df) else [],
    )
    return df
