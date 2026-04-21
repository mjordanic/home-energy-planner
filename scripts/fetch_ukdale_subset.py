"""Fetch a 60-day UK-DALE House 1 subset (mains 1 Hz + per-appliance 6 s).

Attempts to download from the UKERC EDC mirror. If that server is unreachable
(which it frequently is) or --synthetic is passed, generates a realistic
UK-DALE-shaped dataset so the rest of the pipeline can be exercised.

Usage:
    python scripts/fetch_ukdale_subset.py               # try real, fall back
    python scripts/fetch_ukdale_subset.py --synthetic   # force synthetic
    python scripts/fetch_ukdale_subset.py --with-16khz  # also 3 days of 16 kHz FLAC
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
def _try_fetch_real() -> bool:
    """Best-effort download of labels + 3 channels. Returns True on success."""
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
    except Exception as e:  # noqa: BLE001
        print(f"UKERC unreachable or download failed: {e!r}")
        return False
    return True


# --------------------------------------------------------------------------- #
# Synthetic generator                                                         #
# --------------------------------------------------------------------------- #
_RNG = np.random.default_rng(20261116)


def _dishwasher_cycle(n_sec: int) -> np.ndarray:
    """2h cycle: heat, wash, heat, rinse/dry."""
    sec = np.arange(n_sec, dtype=float)
    p = np.full(n_sec, 3.0)                         # standby
    # heat 1: 0..20 min
    p[0 : 20 * 60] = 2200 + _RNG.normal(0, 30, 20 * 60)
    # wash: 20..60 min
    p[20 * 60 : 60 * 60] = 150 + _RNG.normal(0, 15, 40 * 60)
    # heat 2: 60..80 min
    p[60 * 60 : 80 * 60] = 2100 + _RNG.normal(0, 30, 20 * 60)
    # rinse/dry: 80..120 min
    p[80 * 60 : 120 * 60] = 500 + _RNG.normal(0, 40, 40 * 60)
    return np.clip(p[:n_sec], 0, None)


def _washer_cycle(n_sec: int) -> np.ndarray:
    """1.5h cycle: heat, wash, rinse, spin."""
    sec = np.arange(n_sec, dtype=float)
    p = np.full(n_sec, 2.0)
    p[0 : 15 * 60] = 2100 + _RNG.normal(0, 50, 15 * 60)     # heating element
    p[15 * 60 : 60 * 60] = 120 + _RNG.normal(0, 25, 45 * 60)  # agitation
    p[60 * 60 : 75 * 60] = 90 + _RNG.normal(0, 15, 15 * 60)   # rinse pump
    p[75 * 60 : 90 * 60] = 320 + _RNG.normal(0, 40, 15 * 60)  # final spin (motor)
    return np.clip(p[:n_sec], 0, None)


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


def _generate_synthetic() -> dict[datetime, list[tuple[str, datetime]]]:
    """Generate UK-DALE-format .dat files: mains 1 Hz, dishwasher/washer 6 s.

    Returns the onset log for later use.
    """
    start = UKDALE_TRAIN_START
    end = UKDALE_TEST_END
    total_sec = int((end - start).total_seconds())

    # Hour preferences for onsets (0..23).
    # Dishwasher: strong late-evening peak (21-23), some after-lunch (14).
    dish_prefs = np.array(
        [0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.8, 1.0, 1.2,
         1.5, 1.8, 2.5, 2.2, 3.0, 2.0, 1.5, 2.0, 3.0, 3.5,
         5.0, 8.0, 6.0, 2.5]
    )
    # Washer: morning (8-10) and evening (18-20) peaks.
    wash_prefs = np.array(
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 1.0, 2.5, 5.5, 6.0,
         4.5, 3.0, 2.5, 2.0, 2.0, 2.0, 2.5, 4.0, 5.5, 4.5,
         3.0, 2.0, 1.0, 0.5]
    )

    dish_onsets = _sample_onsets(start, end, per_day_avg=1.1, hour_prefs=dish_prefs)
    wash_onsets = _sample_onsets(start, end, per_day_avg=0.9, hour_prefs=wash_prefs)

    print(f"synthetic onsets: dishwasher={len(dish_onsets)} washer={len(wash_onsets)}")

    # --- 6 s per-appliance streams (dishwasher ch6, washer ch5) -----------
    step = int(UKDALE_SUBMETER_PERIOD_S)  # 6 seconds
    n_samples = total_sec // step
    times_s = np.arange(n_samples) * step  # seconds since start

    def _render(onsets: list[datetime], cycle_fn, cycle_len_s: int) -> np.ndarray:
        sub = np.full(n_samples, 2.0)
        for on in onsets:
            t0 = int((on - start).total_seconds())
            cyc = cycle_fn(cycle_len_s)
            # downsample cycle 1Hz -> 6s by mean pooling
            pooled = cyc[: (cycle_len_s // step) * step].reshape(-1, step).mean(axis=1)
            i0 = t0 // step
            i1 = min(i0 + len(pooled), n_samples)
            sub[i0:i1] = pooled[: i1 - i0]
        return sub

    dish = _render(dish_onsets, _dishwasher_cycle, cycle_len_s=120 * 60)
    wash = _render(wash_onsets, _washer_cycle, cycle_len_s=90 * 60)

    # --- 1 Hz mains = baseload + fridge-ish cycle + dish + washer ---------
    mains_n = total_sec
    t_main = np.arange(mains_n, dtype=float)
    baseload = 180 + 20 * np.sin(2 * np.pi * t_main / 86400)              # diurnal
    fridge = 90 * (np.sin(2 * np.pi * t_main / 1800) > 0.3).astype(float)  # on/off
    # upsample 6s->1s by repetition for dish + wash, clipped to length
    dish_1s = np.repeat(dish, step)[:mains_n]
    wash_1s = np.repeat(wash, step)[:mains_n]
    mains = (baseload + fridge + dish_1s + wash_1s + _RNG.normal(0, 3, mains_n)).clip(min=0)

    # --- Write .dat files in UK-DALE format: "<unix_ts> <power>\n" --------
    start_unix = int(start.timestamp())
    ts_mains = start_unix + np.arange(mains_n, dtype=np.int64)
    ts_sub = start_unix + times_s.astype(np.int64)

    _write_dat(HOUSE_DIR / "channel_1.dat", ts_mains, mains)
    _write_dat(HOUSE_DIR / "channel_5.dat", ts_sub, wash)
    _write_dat(HOUSE_DIR / "channel_6.dat", ts_sub, dish)

    labels = "1 aggregate\n5 washing_machine\n6 dishwasher\n"
    (HOUSE_DIR / "labels.dat").write_text(labels)
    print(f"wrote {mains_n:,} 1 Hz mains + {n_samples:,} 6 s per-appliance samples")

    return {"dishwasher": dish_onsets, "washing_machine": wash_onsets}


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
    ap.add_argument("--synthetic", action="store_true",
                    help="skip network; synthesize the full window")
    ap.add_argument("--with-16khz", action="store_true",
                    help="also synthesize/download 3 days of 16 kHz FLAC")
    args = ap.parse_args()

    # Only require `requests` if we actually hit the network.
    if not args.synthetic:
        try:
            import requests  # noqa: F401
        except ImportError:
            args.synthetic = True

    source = "synthetic"
    if not args.synthetic:
        print(f"attempting real download from {UKDALE_BASE}")
        if _try_fetch_real():
            source = "real"
        else:
            print("falling back to synthetic")

    if source == "synthetic":
        _generate_synthetic()

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
        if source == "real":
            print("NOTE: real 16 kHz download not implemented; synthesizing 16 kHz slice")
        path16 = _generate_synthetic_16khz()
        files["mains_16khz_flac"] = path16
        extras["hf_source"] = "synthetic"

    write_manifest(
        UKDALE_DIR / "MANIFEST.json",
        source=source,
        url_base=UKDALE_BASE if source == "real" else None,
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
