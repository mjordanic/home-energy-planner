"""Fetch a 60-day UK-DALE House 1 subset (mains 1 Hz + per-appliance 6 s).

Downloads from the UKERC EDC mirror. Raises FetchError if the server is
unreachable or any file returns a non-200 status — no synthetic fallback.

Usage:
    python scripts/fetch_ukdale_subset.py               # download real data
    python scripts/fetch_ukdale_subset.py --with-16khz  # also synthesize 3 days of 16 kHz FLAC
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aerogrid.config import (
    APPLIANCES,
    UKDALE_16KHZ_END,
    UKDALE_16KHZ_START,
    UKDALE_DIR,
    UKDALE_HF_HZ,
    UKDALE_MAINS_HZ,
    UKDALE_SUBMETER_PERIOD_S,
    UKDALE_TEST_END,
    UKDALE_TEST_START,
    UKDALE_TRAIN_END,
    UKDALE_TRAIN_START,
)
from scripts._common import FetchError, http_get, write_manifest

UKDALE_BASE = (
    "https://data.ukedc.rl.ac.uk/simplebrowse/edc/efficiency/residential/"
    "EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-disaggregated/house_1"
)
HOUSE_DIR = UKDALE_DIR / "house_1"
HOUSE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Real fetch attempt                                                          #
# --------------------------------------------------------------------------- #
def _try_fetch_real() -> None:
    """Download labels + 3 channels. Raises FetchError on any failure."""
    channels = {1: "mains", 5: "washing_machine", 6: "dishwasher"}
    try:
        r = http_get(f"{UKDALE_BASE}/labels.dat", timeout=15)
        if r.status_code != 200:
            raise FetchError(f"labels.dat returned {r.status_code}")
        (HOUSE_DIR / "labels.dat").write_bytes(r.content)
        for ch in channels:
            r = http_get(f"{UKDALE_BASE}/channel_{ch}.dat", timeout=60)
            if r.status_code != 200:
                raise FetchError(f"channel_{ch}.dat returned {r.status_code}")
            (HOUSE_DIR / f"channel_{ch}.dat").write_bytes(r.content)
            print(f"  channel_{ch}.dat ({len(r.content) / 1e6:.1f} MB)")
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"UKERC download failed: {e}") from e


# --------------------------------------------------------------------------- #
# 16 kHz FLAC helpers (onset sampling used only for the optional HF file)    #
# --------------------------------------------------------------------------- #
_RNG = np.random.default_rng(20261116)


def _sample_onsets(
    start: datetime, end: datetime, per_day_avg: float, hour_prefs: np.ndarray
) -> list[datetime]:
    """Sparse onsets between start and end weighted toward certain hours."""
    n_days = (end - start).days + 1
    onsets: list[datetime] = []
    for d in range(n_days):
        n = _RNG.poisson(per_day_avg)
        for _ in range(n):
            h = _RNG.choice(24, p=hour_prefs / hour_prefs.sum())
            m = _RNG.integers(0, 60)
            s = _RNG.integers(0, 60)
            onsets.append(start + timedelta(days=d, hours=int(h), minutes=int(m), seconds=int(s)))
    onsets.sort()
    return onsets


def _write_dat(path: Path, ts: np.ndarray, power: np.ndarray) -> None:
    """UK-DALE .dat: space-delimited `<unix_ts> <power_watts>` per line."""
    # Use numpy.savetxt for speed; power quantized to 2 decimals (file still ~3x smaller).
    stacked = np.column_stack([ts.astype(np.int64), np.round(power, 2)])
    np.savetxt(path, stacked, fmt=["%d", "%.2f"])
    print(f"  {path.name} -> {path.stat().st_size / 1e6:.1f} MB")


# --------------------------------------------------------------------------- #
# Post-processing: split + onsets parquet                                     #
# --------------------------------------------------------------------------- #
def _load_dat(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path, dtype=np.float64)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(arr[:, 0].astype(np.int64), unit="s", utc=True),
            "power_w": arr[:, 1].astype(np.float32),
        }
    )
    return df


def _tag_split(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series("train", index=df.index, dtype="object")
    s.loc[df["timestamp"] >= UKDALE_TEST_START] = "test"
    df["split"] = s.astype("category")
    return df[(df["timestamp"] >= UKDALE_TRAIN_START) & (df["timestamp"] < UKDALE_TEST_END)]


def _detect_onsets(df: pd.DataFrame, spec, name: str) -> pd.DataFrame:
    """Threshold-crossing onset detection with debounce (min 10-min gap)."""
    on = df["power_w"].values > spec.on_power_threshold_w
    prev = np.concatenate([[False], on[:-1]])
    rising = on & ~prev
    idx = np.where(rising)[0]
    ts = df["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    last_t = -(10 ** 12)
    kept: list[int] = []
    for i in idx:
        if ts[i] - last_t >= 600:
            kept.append(i)
            last_t = ts[i]
    kept_df = df.iloc[kept][["timestamp", "split"]].copy()
    kept_df["appliance"] = name
    return kept_df


def _post_process() -> dict[str, pd.DataFrame]:
    mains = _tag_split(_load_dat(HOUSE_DIR / "channel_1.dat"))
    dish = _tag_split(_load_dat(HOUSE_DIR / "channel_6.dat"))
    wash = _tag_split(_load_dat(HOUSE_DIR / "channel_5.dat"))

    mains.to_parquet(HOUSE_DIR / "mains_1hz.parquet", index=False)
    dish.to_parquet(HOUSE_DIR / "dishwasher_6s.parquet", index=False)
    wash.to_parquet(HOUSE_DIR / "washing_machine_6s.parquet", index=False)
    print(
        f"parquet: mains {len(mains):,} rows, dishwasher {len(dish):,}, washer {len(wash):,}"
    )

    onsets = pd.concat(
        [
            _detect_onsets(dish, APPLIANCES["dishwasher"], "dishwasher"),
            _detect_onsets(wash, APPLIANCES["washing_machine"], "washing_machine"),
        ],
        ignore_index=True,
    ).sort_values("timestamp")
    onsets.to_parquet(HOUSE_DIR / "onsets.parquet", index=False)

    for split in ("train", "test"):
        sub = onsets[onsets["split"] == split]
        print(f"  onsets[{split}]: dishwasher={(sub['appliance']=='dishwasher').sum()} "
              f"washer={(sub['appliance']=='washing_machine').sum()}")

    return {"mains": mains, "dishwasher": dish, "washing_machine": wash, "onsets": onsets}


# --------------------------------------------------------------------------- #
# 16 kHz FLAC (optional)                                                      #
# --------------------------------------------------------------------------- #
def _generate_synthetic_16khz() -> Path:
    """Synthesize a 3-day stereo 16 kHz mains (voltage + current) file.

    Channel 0 = voltage (≈230 V, 50 Hz sine + minor 3rd/5th harmonic).
    Channel 1 = current: baseload + appliance transients at known onset times.

    To keep the file manageable we emit one 3-day FLAC at 16 kHz (~900 MB).
    """
    import soundfile as sf

    out = HOUSE_DIR / "mains_16khz_3day.flac"
    fs = int(UKDALE_HF_HZ)
    dur_s = int((UKDALE_16KHZ_END - UKDALE_16KHZ_START).total_seconds())
    print(f"synthesizing {dur_s / 3600:.1f}h of 16 kHz (fs={fs} Hz) — this takes ~2 min")

    # Write in 1-hour blocks to keep memory reasonable.
    block_s = 3600
    block_n = fs * block_s
    t_block = np.arange(block_n) / fs

    # Seed onsets inside the 16 kHz window.
    dish_on = _sample_onsets(UKDALE_16KHZ_START, UKDALE_16KHZ_END, per_day_avg=1.2,
                             hour_prefs=np.ones(24))
    wash_on = _sample_onsets(UKDALE_16KHZ_START, UKDALE_16KHZ_END, per_day_avg=1.0,
                             hour_prefs=np.ones(24))
    print(f"  dishwasher onsets inside 16 kHz window: {len(dish_on)}")
    print(f"  washer     onsets inside 16 kHz window: {len(wash_on)}")

    with sf.SoundFile(out, mode="w", samplerate=fs, channels=2, format="FLAC",
                      subtype="PCM_16") as f:
        for b in range(dur_s // block_s):
            block_start = UKDALE_16KHZ_START + timedelta(seconds=b * block_s)
            voltage = 230 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t_block)
            voltage += 3 * np.sin(2 * np.pi * 150 * t_block)  # 3rd harmonic
            voltage += 1 * np.sin(2 * np.pi * 250 * t_block)  # 5th

            # Current baseline: in-phase ~0.8 A (baseload).
            current = 0.8 * np.sin(2 * np.pi * 50 * t_block)

            # Superimpose onsets inside this block.
            for on in dish_on + wash_on:
                t0 = (on - block_start).total_seconds()
                if 0 <= t0 < block_s:
                    s0 = int(t0 * fs)
                    # Dishwasher: resistive (in-phase) spike lasting 5 s.
                    # Washer: inductive (phase-shifted by ~45°) spike.
                    is_dish = on in dish_on
                    dur_samples = fs * 5
                    end_samples = min(s0 + dur_samples, block_n)
                    t_on = np.arange(end_samples - s0) / fs
                    if is_dish:
                        amp = 9.5 * np.exp(-t_on / 2.0) + 2.5
                        phase = 0.0  # resistive
                    else:
                        amp = 7.0 * np.exp(-t_on / 1.5) + 1.5
                        phase = np.pi / 4  # inductive
                    extra = amp * np.sin(2 * np.pi * 50 * t_on - phase)
                    current[s0:end_samples] += extra

            current += _RNG.normal(0, 0.05, block_n)
            # Scale to int16 range (/300 V headroom, /15 A headroom).
            stereo = np.column_stack([voltage / 300.0, current / 15.0])
            f.write(stereo.astype(np.float32))
            if b % 6 == 5:
                print(f"  wrote {b+1}h of 16 kHz")

    print(f"  16 kHz file -> {out.stat().st_size / 1e6:.1f} MB")
    return out


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-16khz", action="store_true",
                    help="also synthesize 3 days of 16 kHz FLAC (real HF download not available)")
    args = ap.parse_args()

    print(f"downloading UK-DALE House 1 from {UKDALE_BASE}")
    _try_fetch_real()

    dfs = _post_process()

    files = {
        "labels": HOUSE_DIR / "labels.dat",
        "channel_1": HOUSE_DIR / "channel_1.dat",
        "channel_5": HOUSE_DIR / "channel_5.dat",
        "channel_6": HOUSE_DIR / "channel_6.dat",
        "mains_1hz_parquet": HOUSE_DIR / "mains_1hz.parquet",
        "dishwasher_6s_parquet": HOUSE_DIR / "dishwasher_6s.parquet",
        "washing_machine_6s_parquet": HOUSE_DIR / "washing_machine_6s.parquet",
        "onsets_parquet": HOUSE_DIR / "onsets.parquet",
    }
    extras = {
        "row_counts": {k: int(len(v)) for k, v in dfs.items()},
        "onset_counts": {
            "dishwasher": int((dfs["onsets"]["appliance"] == "dishwasher").sum()),
            "washing_machine": int((dfs["onsets"]["appliance"] == "washing_machine").sum()),
        },
    }

    if args.with_16khz:
        path16 = _generate_synthetic_16khz()
        files["mains_16khz_flac"] = path16
        extras["hf_source"] = "synthetic"

    write_manifest(
        UKDALE_DIR / "MANIFEST.json",
        source="real",
        url_base=UKDALE_BASE,
        windows={
            "train": (UKDALE_TRAIN_START, UKDALE_TRAIN_END),
            "test": (UKDALE_TEST_START, UKDALE_TEST_END),
            "hf": (UKDALE_16KHZ_START, UKDALE_16KHZ_END),
        },
        files=files,
        extras=extras,
    )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
