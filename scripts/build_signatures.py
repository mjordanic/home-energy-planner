"""Derive V-I signatures for dishwasher and washing_machine from TRAIN data.

If a real UK-DALE 16 kHz file exists for the train window, we use it. Otherwise
we synthesize a short 5-minute 16 kHz mini-file with known onsets (inside the
train window) and extract signatures from that.

Output: data/cache/signatures.pkl
  {"dishwasher": np.ndarray(128,), "washing_machine": np.ndarray(128,)}
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aerogrid.config import (
    APPLIANCES,
    CACHE_DIR,
    NILM,
    UKDALE_DIR,
    UKDALE_HF_HZ,
    UKDALE_TRAIN_START,
)
from aerogrid.vi_features import (
    SAMPLES_PER_CYCLE,
    average_signature,
    extract_single_cycle,
    vi_trajectory_descriptor,
)

SIG_FLAC = UKDALE_DIR / "house_1" / "mains_16khz_signatures.flac"
SIG_ONSET_META = UKDALE_DIR / "house_1" / "signature_onsets.pkl"


_RNG = np.random.default_rng(42)


def _synthesize_signature_sample() -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate 5 minutes of 16 kHz (voltage, current) with 3 known onsets per appliance.

    Dishwasher bursts are resistive (in-phase current).
    Washer bursts are inductive (~π/4 phase lag).
    """
    fs = int(UKDALE_HF_HZ)
    dur_s = 300
    n = fs * dur_s
    t = np.arange(n) / fs

    voltage = 230 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
    voltage += 3 * np.sin(2 * np.pi * 150 * t) + 1 * np.sin(2 * np.pi * 250 * t)

    current = 0.8 * np.sin(2 * np.pi * 50 * t) + _RNG.normal(0, 0.05, n)

    onsets: dict[str, list[float]] = {"dishwasher": [], "washing_machine": []}

    # place 3 dishwasher + 3 washer onsets at well-separated times
    dish_times = [30.0, 110.0, 200.0]
    wash_times = [60.0, 150.0, 240.0]
    burst_dur_s = 4.0

    def _add_burst(start_s: float, inductive: bool) -> None:
        s0 = int(start_s * fs)
        nseg = int(burst_dur_s * fs)
        s1 = s0 + nseg
        t_on = np.arange(nseg) / fs
        if inductive:
            amp = 7.0 * np.exp(-t_on / 1.5) + 1.5
            phase = np.pi / 4
        else:
            amp = 9.5 * np.exp(-t_on / 2.0) + 2.5
            phase = 0.0
        current[s0:s1] += amp * np.sin(2 * np.pi * 50 * t_on - phase)

    for ts in dish_times:
        _add_burst(ts, inductive=False)
        onsets["dishwasher"].append(ts)
    for ts in wash_times:
        _add_burst(ts, inductive=True)
        onsets["washing_machine"].append(ts)

    meta = {"fs": fs, "start_utc": UKDALE_TRAIN_START.isoformat(), "onsets": onsets}
    return voltage.astype(np.float32), current.astype(np.float32), meta


def _save_flac(voltage: np.ndarray, current: np.ndarray, fs: int) -> None:
    import soundfile as sf
    stereo = np.column_stack([voltage / 300.0, current / 15.0])
    sf.write(SIG_FLAC, stereo.astype(np.float32), fs, format="FLAC", subtype="PCM_16")
    print(f"  wrote {SIG_FLAC.name} ({SIG_FLAC.stat().st_size / 1e6:.1f} MB)")


def _load_flac() -> tuple[np.ndarray, np.ndarray, int]:
    import soundfile as sf
    data, fs = sf.read(SIG_FLAC, always_2d=True)
    voltage = data[:, 0].astype(np.float64) * 300.0
    current = data[:, 1].astype(np.float64) * 15.0
    return voltage, current, fs


def _ensure_signature_source() -> tuple[np.ndarray, np.ndarray, int, dict]:
    if SIG_FLAC.exists() and SIG_ONSET_META.exists():
        print(f"reusing existing {SIG_FLAC.name}")
        v, c, fs = _load_flac()
        with SIG_ONSET_META.open("rb") as fh:
            meta = pickle.load(fh)
        return v, c, fs, meta

    print("synthesizing signature training sample (5 min at 16 kHz)")
    v, c, meta = _synthesize_signature_sample()
    fs = meta["fs"]
    _save_flac(v, c, fs)
    with SIG_ONSET_META.open("wb") as fh:
        pickle.dump(meta, fh)
    return v, c, fs, meta


def _collect_cycles(voltage: np.ndarray, current: np.ndarray, fs: int,
                    onset_times_s: list[float], n_cycles_per_onset: int = 8
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    cycles: list[tuple[np.ndarray, np.ndarray]] = []
    offset_samples = int(fs * 0.5)  # 0.5 s after onset, past initial transient
    for ts in onset_times_s:
        base = int(ts * fs) + offset_samples
        for k in range(n_cycles_per_onset):
            s0 = base + k * SAMPLES_PER_CYCLE
            s1 = s0 + SAMPLES_PER_CYCLE
            if s1 > len(voltage):
                break
            v, i = extract_single_cycle(voltage[s0:s1], current[s0:s1], start_idx=0)
            cycles.append((v, i))
    return cycles


def main() -> int:
    voltage, current, fs, meta = _ensure_signature_source()

    signatures: dict[str, np.ndarray] = {}
    for appliance, ts_list in meta["onsets"].items():
        cycles = _collect_cycles(voltage, current, fs, ts_list)
        if not cycles:
            print(f"  {appliance}: no cycles collected — skipping")
            continue
        sig = average_signature(cycles)
        signatures[appliance] = sig
        print(f"  {appliance}: {len(cycles)} cycles -> signature shape {sig.shape}")

    # Sanity: check dishwasher vs washer signatures are distinguishable.
    if "dishwasher" in signatures and "washing_machine" in signatures:
        from aerogrid.vi_features import cosine_similarity
        sim = cosine_similarity(signatures["dishwasher"],
                                signatures["washing_machine"])
        print(f"  cross-signature cosine similarity (should be <1): {sim:.3f}")
        if sim > 0.99:
            print("  WARNING: signatures look identical — check generator phase settings")

    out = CACHE_DIR / "signatures.pkl"
    with out.open("wb") as fh:
        pickle.dump(
            {
                "signatures": signatures,
                "n_points": NILM.vi_trajectory_points,
                "built_from": "synthetic" if not SIG_FLAC.exists() else "flac",
                "source_path": str(SIG_FLAC),
                "threshold": NILM.signature_match_threshold,
            },
            fh,
        )
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
