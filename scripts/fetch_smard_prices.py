"""Fetch SMARD DE-LU day-ahead prices (15-min native resolution).

Source: Bundesnetzagentur SMARD public API — no API key required.
  Index:  https://www.smard.de/app/chart_data/4169/DE-LU/index_quarterhour.json
  Data:   https://www.smard.de/app/chart_data/4169/DE-LU/4169_DE-LU_quarterhour_{ts}.json

Each data chunk covers roughly one week. The API returns UTC timestamps in
milliseconds and prices in EUR/MWh. The output column is named `lbmp` to keep
the rest of the codebase (price oracle, sim) source-agnostic.

Raises FetchError if any network call fails — no synthetic fallback.

Output:
  data/smard/de_lu_15min.parquet   — (timestamp, lbmp, split)
  data/smard/MANIFEST.json
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aerogrid.config import (
    SMARD_DIR,
    SMARD_TEST_END,
    SMARD_TEST_START,
    SMARD_TRAIN_END,
    SMARD_TRAIN_START,
)
from aerogrid.logging_config import setup_logging
from scripts._common import FetchError, http_get, write_manifest

logger = logging.getLogger(__name__)

_INDEX_URL = "https://www.smard.de/app/chart_data/4169/DE-LU/index_quarterhour.json"
_CHUNK_URL = "https://www.smard.de/app/chart_data/4169/DE-LU/4169_DE-LU_quarterhour_{ts}.json"


def _fetch() -> pd.DataFrame:
    """Download all chunks that overlap the configured window. Raises FetchError."""
    logger.info("_fetch: fetching SMARD index from %s", _INDEX_URL)
    try:
        r = http_get(_INDEX_URL, timeout=15)
    except Exception as e:
        logger.error("_fetch: SMARD index fetch failed: %s", e)
        raise FetchError(f"SMARD index fetch failed: {e}") from e
    if r.status_code != 200:
        logger.error("_fetch: SMARD index HTTP %d", r.status_code)
        raise FetchError(f"SMARD index HTTP {r.status_code}")

    timestamps = r.json()["timestamps"]
    logger.debug("_fetch: index contains %d chunk timestamps", len(timestamps))

    start_ms = int(SMARD_TRAIN_START.timestamp() * 1000)
    end_ms = int(SMARD_TEST_END.timestamp() * 1000)
    # A chunk at ts_ms covers roughly one week; include the chunk just before
    # start in case it overlaps.
    week_ms = 7 * 24 * 3600 * 1000
    relevant = [ts for ts in timestamps if ts >= start_ms - week_ms and ts < end_ms]

    if not relevant:
        logger.error("_fetch: no SMARD chunks for window %s → %s", SMARD_TRAIN_START, SMARD_TEST_END)
        raise FetchError("no SMARD chunks found for the configured date window")

    logger.info("_fetch: downloading %d SMARD chunks", len(relevant))
    frames: list[pd.DataFrame] = []
    for i, ts in enumerate(relevant):
        url = _CHUNK_URL.format(ts=ts)
        logger.debug("_fetch: chunk %d/%d ts=%d url=%s", i + 1, len(relevant), ts, url)
        try:
            r = http_get(url, timeout=15)
        except Exception as e:
            logger.error("_fetch: SMARD chunk %d fetch failed: %s", ts, e)
            raise FetchError(f"SMARD chunk {ts} fetch failed: {e}") from e
        if r.status_code != 200:
            logger.error("_fetch: SMARD chunk %d HTTP %d", ts, r.status_code)
            raise FetchError(f"SMARD chunk {ts} HTTP {r.status_code}")
        series = r.json()["series"]
        df = pd.DataFrame(series, columns=["ts_ms", "lbmp"])
        df = df[df["lbmp"].notna()]
        logger.debug("_fetch: chunk %d → %d rows (after dropping NaN)", ts, len(df))
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True).drop_duplicates("ts_ms")
    raw["timestamp"] = pd.to_datetime(raw["ts_ms"], unit="ms", utc=True)
    raw = raw[["timestamp", "lbmp"]].sort_values("timestamp").reset_index(drop=True)
    return raw


def _tag_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``split`` column and filter to the configured train/test window.

    Rows in ``[SMARD_TRAIN_START, SMARD_TEST_START)`` are tagged ``"train"``;
    rows in ``[SMARD_TEST_START, SMARD_TEST_END)`` are tagged ``"test"``.
    Rows outside the window are dropped.
    """
    s = pd.Series("train", index=df.index, dtype="object")
    s.loc[df["timestamp"] >= SMARD_TEST_START] = "test"
    df = df.copy()
    df["split"] = s.astype("category")
    return df[(df["timestamp"] >= SMARD_TRAIN_START) & (df["timestamp"] < SMARD_TEST_END)]


def main() -> int:
    """Download SMARD prices, tag the train/test split, write parquet and manifest."""
    setup_logging(level=logging.INFO, console=True)
    logger.info(
        "fetch_smard_prices: downloading DE-LU prices %s..%s",
        SMARD_TRAIN_START.date(), SMARD_TEST_END.date(),
    )
    raw = _fetch()
    logger.info("fetch_smard_prices: raw rows=%d", len(raw))
    df = _tag_split(raw)

    out = SMARD_DIR / "de_lu_15min.parquet"
    df.to_parquet(out, index=False)
    logger.info("fetch_smard_prices: wrote %d rows to %s", len(df), out)

    print(f"15-min rows: {len(df):,}  "
          f"(train={(df['split']=='train').sum():,}  "
          f"test={(df['split']=='test').sum():,})")
    print(f"price stats: mean={df['lbmp'].mean():.2f}  "
          f"std={df['lbmp'].std():.2f}  "
          f"min={df['lbmp'].min():.2f}  max={df['lbmp'].max():.2f}  EUR/MWh")

    write_manifest(
        SMARD_DIR / "MANIFEST.json",
        source="real",
        url_base="https://www.smard.de/app/chart_data/4169/DE-LU/",
        windows={
            "train": (SMARD_TRAIN_START, SMARD_TRAIN_END),
            "test": (SMARD_TEST_START, SMARD_TEST_END),
        },
        files={"de_lu_15min_parquet": out},
        extras={
            "area": "DE-LU",
            "currency": "EUR/MWh",
            "rows_15min": int(len(df)),
            "price_stats": {
                "mean": float(df["lbmp"].mean()),
                "std": float(df["lbmp"].std()),
                "min": float(df["lbmp"].min()),
                "max": float(df["lbmp"].max()),
            },
        },
    )
    logger.info("fetch_smard_prices: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
