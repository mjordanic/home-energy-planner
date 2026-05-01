"""Fetch 90 days of NYISO real-time + day-ahead LBMP prices.

Real source: NYISO public CSV archive — no API key required.
  RT (5-min):   https://mis.nyiso.com/public/csv/realtime/YYYYMMDDrealtime_zone.csv
  DAM (hourly): https://mis.nyiso.com/public/csv/damlbmp/YYYYMMDDdamlbmp_zone.csv

Raises FetchError if the server is unreachable or any day returns a non-200
status — no synthetic fallback.

Output:
  data/nyiso/<zone>_15min.parquet     — (timestamp, lbmp, split)
  data/nyiso/<zone>_dam.parquet       — (timestamp, lbmp_da, split)
  data/nyiso/MANIFEST.json
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aerogrid.config import (
    NYISO_DIR,
    NYISO_TEST_END,
    NYISO_TEST_START,
    NYISO_TRAIN_END,
    NYISO_TRAIN_START,
    NYISO_ZONE,
)
from aerogrid.logging_config import setup_logging
from scripts._common import FetchError, http_get, write_manifest

logger = logging.getLogger(__name__)

NYISO_RT_URL = "https://mis.nyiso.com/public/csv/realtime/{ymd}realtime_zone.csv"
NYISO_DAM_URL = "https://mis.nyiso.com/public/csv/damlbmp/{ymd}damlbmp_zone.csv"


# --------------------------------------------------------------------------- #
# Real fetch                                                                  #
# --------------------------------------------------------------------------- #
def _try_fetch_real(zone: str, start: datetime, end: datetime
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download per-day CSVs and concatenate. Raises FetchError on any failure."""
    n_days = (end - start).days
    logger.info("_try_fetch_real: fetching NYISO zone=%s for %d days (%s → %s)", zone, n_days, start.date(), end.date())
    rt_frames: list[pd.DataFrame] = []
    dam_frames: list[pd.DataFrame] = []
    day = start
    day_num = 0
    while day < end:
        ymd = day.strftime("%Y%m%d")
        day_num += 1
        logger.debug("_try_fetch_real: day %d/%d %s", day_num, n_days, ymd)
        try:
            r_rt = http_get(NYISO_RT_URL.format(ymd=ymd), timeout=15)
            r_dam = http_get(NYISO_DAM_URL.format(ymd=ymd), timeout=15)
        except Exception as e:
            logger.error("_try_fetch_real: network error on %s: %s", ymd, e)
            raise FetchError(f"NYISO network error on {ymd}: {e}") from e
        if r_rt.status_code != 200 or r_dam.status_code != 200:
            logger.error(
                "_try_fetch_real: HTTP error on %s RT=%d DAM=%d",
                ymd, r_rt.status_code, r_dam.status_code,
            )
            raise FetchError(f"NYISO HTTP {r_rt.status_code}/{r_dam.status_code} on {ymd}")
        rt_frames.append(pd.read_csv(io.StringIO(r_rt.text)))
        dam_frames.append(pd.read_csv(io.StringIO(r_dam.text)))
        day += timedelta(days=1)
    rt = pd.concat(rt_frames, ignore_index=True)
    dam = pd.concat(dam_frames, ignore_index=True)

    # NYISO CSV columns: 'Time Stamp', 'Name', 'PTID', 'LBMP ($/MWHr)', ...
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Name"].astype(str).str.strip() == zone].copy()
        df["timestamp"] = pd.to_datetime(df["Time Stamp"], utc=False).dt.tz_localize(
            "US/Eastern", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")
        df = df.dropna(subset=["timestamp"])
        df["lbmp"] = df["LBMP ($/MWHr)"].astype(float)
        return df[["timestamp", "lbmp"]].sort_values("timestamp").reset_index(drop=True)

    return _filter(rt), _filter(dam)


# --------------------------------------------------------------------------- #
# Post-processing                                                             #
# --------------------------------------------------------------------------- #
def _aggregate_15min(rt: pd.DataFrame) -> pd.DataFrame:
    """Mean-aggregate 5-min LBMP into 15-min slots aligned on quarter hours."""
    out = (
        rt.set_index("timestamp")["lbmp"]
        .resample("15min", origin="start_day", label="left")
        .mean()
        .reset_index()
    )
    out.columns = ["timestamp", "lbmp"]
    return out


def _tag_split(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series("train", index=df.index, dtype="object")
    s.loc[df["timestamp"] >= NYISO_TEST_START] = "test"
    df["split"] = s.astype("category")
    return df[(df["timestamp"] >= NYISO_TRAIN_START) & (df["timestamp"] < NYISO_TEST_END)]


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default=NYISO_ZONE)
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    setup_logging(level=getattr(logging, args.log_level, logging.INFO), console=True)
    logger.info(
        "fetch_nyiso_prices: zone=%s %s → %s",
        args.zone, NYISO_TRAIN_START.date(), NYISO_TEST_END.date(),
    )
    rt, dam = _try_fetch_real(args.zone, NYISO_TRAIN_START, NYISO_TEST_END)

    rt_15 = _tag_split(_aggregate_15min(rt))
    dam_tagged = _tag_split(dam.copy())

    zone_slug = args.zone.lower().replace(".", "").replace(" ", "_")
    rt_path = NYISO_DIR / f"{zone_slug}_15min.parquet"
    dam_path = NYISO_DIR / f"{zone_slug}_dam.parquet"
    rt_15.to_parquet(rt_path, index=False)
    dam_tagged.to_parquet(dam_path, index=False)

    print(f"15-min rows: {len(rt_15):,} (train={(rt_15['split']=='train').sum():,} "
          f"test={(rt_15['split']=='test').sum():,})")
    print(f"DAM hourly rows: {len(dam_tagged):,}")
    print(
        f"price stats 15min: mean={rt_15['lbmp'].mean():.2f}  "
        f"std={rt_15['lbmp'].std():.2f}  "
        f"min={rt_15['lbmp'].min():.2f}  max={rt_15['lbmp'].max():.2f}"
    )

    write_manifest(
        NYISO_DIR / "MANIFEST.json",
        source="real",
        url_base="https://mis.nyiso.com/public/csv/",
        windows={
            "train": (NYISO_TRAIN_START, NYISO_TRAIN_END),
            "test": (NYISO_TEST_START, NYISO_TEST_END),
        },
        files={
            "rt_15min_parquet": rt_path,
            "dam_hourly_parquet": dam_path,
        },
        extras={
            "zone": args.zone,
            "rows_15min": int(len(rt_15)),
            "rows_dam": int(len(dam_tagged)),
            "price_stats_15min": {
                "mean": float(rt_15["lbmp"].mean()),
                "std": float(rt_15["lbmp"].std()),
                "min": float(rt_15["lbmp"].min()),
                "max": float(rt_15["lbmp"].max()),
            },
        },
    )
    logger.info(
        "fetch_nyiso_prices: done rt_15=%d dam=%d", len(rt_15), len(dam_tagged),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
