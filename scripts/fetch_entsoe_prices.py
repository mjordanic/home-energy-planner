"""Optional: fetch 30 days of ENTSO-E DE-LU day-ahead prices, upsample to 15 min.

If ENTSOE_API_KEY is in the environment (or .env), calls the ENTSO-E API via
entsoe-py. Otherwise, writes a deterministic synthetic 15-min curve so the EU
alt demo path is still exercisable offline. The primary demo market is NYISO
via fetch_nyiso_prices.py; this script exists solely so a user can drive
ChronosPriceOracle against EU data if they want.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from aerogrid.config import (
    ENTSOE_AREA,
    ENTSOE_DIR,
    ENTSOE_TEST_END,
    ENTSOE_TEST_START,
    ENTSOE_TRAIN_END,
    ENTSOE_TRAIN_START,
)
from scripts._common import write_manifest

load_dotenv(REPO_ROOT / ".env")


def _try_fetch_real(area: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not key:
        return None
    try:
        from entsoe import EntsoePandasClient  # type: ignore
    except ImportError:
        print("entsoe-py not installed; `pip install entsoe-py` for EU path")
        return None
    try:
        cli = EntsoePandasClient(api_key=key)
        s = cli.query_day_ahead_prices(
            area,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
        )
        s = s.tz_convert("UTC")
        df = s.reset_index()
        df.columns = ["timestamp", "price_eur_mwh"]
        return df
    except Exception as e:  # noqa: BLE001
        print(f"ENTSO-E API call failed: {e!r}")
        return None


def _synthesize(start: datetime, end: datetime) -> pd.DataFrame:
    """EU-ish day-ahead curve: two daily peaks, weekly modulation, moderate noise."""
    rng = np.random.default_rng(20241221)
    n_hours = int((end - start).total_seconds() // 3600)
    t = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    hod = np.asarray(t.tz_convert("Europe/Berlin").hour, dtype=float)
    dow = np.asarray(t.tz_convert("Europe/Berlin").dayofweek)
    shape = (
        70
        + 30 * np.exp(-((hod - 8) ** 2) / 4)
        + 45 * np.exp(-((hod - 19) ** 2) / 4)
        - 25 * np.exp(-((hod - 13) ** 2) / 6)  # midday solar dip
    )
    weekend_mult = np.where(dow >= 5, 0.80, 1.0)
    noise = rng.normal(0, 6, n_hours)
    # AR(1)
    for i in range(1, n_hours):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 6)
    price = shape * weekend_mult + noise
    return pd.DataFrame({"timestamp": t, "price_eur_mwh": price.astype(np.float32)})


def _upsample_to_15min(df_hourly: pd.DataFrame) -> pd.DataFrame:
    idx15 = pd.date_range(
        df_hourly["timestamp"].iloc[0],
        df_hourly["timestamp"].iloc[-1] + pd.Timedelta("45min"),
        freq="15min",
        tz="UTC",
    )
    out = (
        df_hourly.set_index("timestamp")
        .reindex(idx15)
        .ffill()
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    return out


def _tag_split(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series("train", index=df.index, dtype="object")
    s.loc[df["timestamp"] >= ENTSOE_TEST_START] = "test"
    df["split"] = s.astype("category")
    return df[(df["timestamp"] >= ENTSOE_TRAIN_START) & (df["timestamp"] < ENTSOE_TEST_END)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="force synthetic even if API key is set")
    args = ap.parse_args()

    source = "synthetic"
    df: pd.DataFrame | None = None
    if not args.synthetic:
        df = _try_fetch_real(ENTSOE_AREA, ENTSOE_TRAIN_START, ENTSOE_TEST_END)
        if df is not None:
            source = "real"
            print(f"ENTSO-E: got {len(df)} hourly prices for {ENTSOE_AREA}")

    if df is None:
        df = _synthesize(ENTSOE_TRAIN_START, ENTSOE_TEST_END)
        print(f"ENTSO-E synthetic: {len(df)} hourly prices")

    df15 = _tag_split(_upsample_to_15min(df))
    out = ENTSOE_DIR / "de_lu_15min.parquet"
    df15.to_parquet(out, index=False)
    print(
        f"15-min rows: {len(df15):,} (train={(df15['split']=='train').sum():,} "
        f"test={(df15['split']=='test').sum():,})  "
        f"mean={df15['price_eur_mwh'].mean():.2f} EUR/MWh"
    )

    write_manifest(
        ENTSOE_DIR / "MANIFEST.json",
        source=source,
        url_base="https://web-api.tp.entsoe.eu" if source == "real" else None,
        windows={
            "train": (ENTSOE_TRAIN_START, ENTSOE_TRAIN_END),
            "test": (ENTSOE_TEST_START, ENTSOE_TEST_END),
        },
        files={"de_lu_15min_parquet": out},
        extras={"area": ENTSOE_AREA, "rows_15min": int(len(df15))},
    )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
