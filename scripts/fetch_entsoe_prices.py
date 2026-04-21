"""Fetch 30 days of ENTSO-E DE-LU day-ahead prices, upsample to 15 min.

Requires ENTSOE_API_KEY in the environment (or .env) and the entsoe-py package
(`uv sync --extra eu`). Raises FetchError if the key is missing or the API
call fails — no synthetic fallback. The primary demo market is NYISO via
fetch_nyiso_prices.py; this script is only needed for the EU alt demo path.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from scripts._common import FetchError, write_manifest

load_dotenv(REPO_ROOT / ".env")


def _try_fetch_real(area: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch real ENTSO-E prices. Raises FetchError if key is missing or call fails."""
    key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not key:
        raise FetchError(
            "ENTSOE_API_KEY is not set. Add it to .env or the environment and retry."
        )
    try:
        from entsoe import EntsoePandasClient  # type: ignore
    except ImportError as e:
        raise FetchError(
            "entsoe-py is not installed. Run: uv sync --extra eu"
        ) from e
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
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"ENTSO-E API call failed: {e}") from e


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
    df = _try_fetch_real(ENTSOE_AREA, ENTSOE_TRAIN_START, ENTSOE_TEST_END)
    print(f"ENTSO-E: got {len(df)} hourly prices for {ENTSOE_AREA}")

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
        source="real",
        url_base="https://web-api.tp.entsoe.eu",
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
