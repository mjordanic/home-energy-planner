"""Fetch 90 days of NYISO real-time + day-ahead LBMP prices.

Real source: NYISO public CSV archive — no API key.
  RT (5-min):   https://mis.nyiso.com/public/csv/realtime/YYYYMMDDrealtime_zone.csv
  DAM (hourly): https://mis.nyiso.com/public/csv/damlbmp/YYYYMMDDdamlbmp_zone.csv

If the server is unreachable or --synthetic is passed, generates a plausible
15-min price curve with realistic daily/weekly/volatility shape.

Output:
  data/nyiso/<zone>_15min.parquet     — (timestamp, lbmp, split)
  data/nyiso/<zone>_dam.parquet       — (timestamp, lbmp_da, split)
  data/nyiso/MANIFEST.json
"""
from __future__ import annotations

import argparse
import io
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
from scripts._common import http_get, write_manifest

NYISO_RT_URL = "https://mis.nyiso.com/public/csv/realtime/{ymd}realtime_zone.csv"
NYISO_DAM_URL = "https://mis.nyiso.com/public/csv/damlbmp/{ymd}damlbmp_zone.csv"


# --------------------------------------------------------------------------- #
# Real fetch                                                                  #
# --------------------------------------------------------------------------- #
def _try_fetch_real(zone: str, start: datetime, end: datetime
                    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Download per-day CSVs and concatenate. Returns (rt, dam) or None."""
    rt_frames: list[pd.DataFrame] = []
    dam_frames: list[pd.DataFrame] = []
    day = start
    while day < end:
        ymd = day.strftime("%Y%m%d")
        try:
            r_rt = http_get(NYISO_RT_URL.format(ymd=ymd), timeout=15)
            r_dam = http_get(NYISO_DAM_URL.format(ymd=ymd), timeout=15)
        except Exception as e:  # noqa: BLE001
            print(f"  {ymd}: network error {e!r}")
            return None
        if r_rt.status_code != 200 or r_dam.status_code != 200:
            print(f"  {ymd}: HTTP {r_rt.status_code}/{r_dam.status_code}")
            return None
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
# Synthetic                                                                   #
# --------------------------------------------------------------------------- #
_RNG = np.random.default_rng(20241216)


def _synthesize(start: datetime, end: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realistic NYC-ish price curve at 5-min and hourly."""
    n_min = int((end - start).total_seconds() // 60)
    t_5min = pd.date_range(start, periods=n_min // 5, freq="5min", tz="UTC")
    t_hour = pd.date_range(start, periods=n_min // 60, freq="h", tz="UTC")

    # local ET hour-of-day (NYISO prices are set on ET clock).
    local = t_5min.tz_convert("US/Eastern")
    hod = np.asarray(local.hour + local.minute / 60.0, dtype=float)
    dow = np.asarray(local.dayofweek)

    # Double-peak daily shape: morning (~08) + evening (~18).
    shape = (
        25
        + 12 * np.sin(2 * np.pi * (hod - 7) / 24)
        + 18 * np.exp(-((hod - 8) ** 2) / 5)
        + 25 * np.exp(-((hod - 18) ** 2) / 6)
    )
    weekend_mult = np.where(dow >= 5, 0.75, 1.0)
    trend = 2.0 * np.sin(2 * np.pi * np.arange(len(t_5min)) / (12 * 24 * 7))  # weekly
    # AR(1) noise + occasional spikes.
    noise = np.zeros(len(t_5min))
    for i in range(1, len(t_5min)):
        noise[i] = 0.85 * noise[i - 1] + _RNG.normal(0, 3.5)
    spikes = _RNG.choice([0, 0, 0, 0, 0, 1], size=len(t_5min)) * _RNG.uniform(
        20, 90, size=len(t_5min)
    )
    lbmp_5 = (shape * weekend_mult + trend + noise + spikes).clip(min=-5)

    rt = pd.DataFrame({"timestamp": t_5min, "lbmp": lbmp_5.astype(np.float32)})

    # Day-ahead = smoothed 5-min aggregated to hourly, with small bias.
    dam_lbmp = (
        rt.set_index("timestamp")
        .resample("1h")["lbmp"]
        .mean()
        .reindex(t_hour)
        .ffill()
        .bfill()
        .to_numpy()
    )
    dam_lbmp = dam_lbmp + _RNG.normal(0, 1.5, len(t_hour))
    dam = pd.DataFrame({"timestamp": t_hour, "lbmp": dam_lbmp.astype(np.float32)})
    return rt, dam


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
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    rt: pd.DataFrame | None = None
    dam: pd.DataFrame | None = None
    source = "synthetic"
    if not args.synthetic:
        print(f"attempting real NYISO download for zone={args.zone}, "
              f"{NYISO_TRAIN_START.date()}..{NYISO_TEST_END.date()}")
        attempt = _try_fetch_real(args.zone, NYISO_TRAIN_START, NYISO_TEST_END)
        if attempt is not None:
            rt, dam = attempt
            source = "real"
        else:
            print("falling back to synthetic")

    if source == "synthetic":
        rt, dam = _synthesize(NYISO_TRAIN_START, NYISO_TEST_END)

    assert rt is not None and dam is not None

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
        source=source,
        url_base="https://mis.nyiso.com/public/csv/" if source == "real" else None,
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
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
