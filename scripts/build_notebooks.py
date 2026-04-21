"""Generate the six visualization notebooks from inline cell specs.

Running this script rebuilds notebooks/*.ipynb in-place. Each notebook is a
small driver that loads the fetched data and produces the plots described in
the plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def mk_notebook(cells: list[tuple[str, str]]) -> dict:
    """cells: list of (cell_type, source) where cell_type in {'markdown','code'}."""
    nb_cells = []
    for kind, src in cells:
        lines = src.split("\n")
        source_list = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])
        cell = {"cell_type": kind, "metadata": {}, "source": source_list}
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(name: str, cells: list[tuple[str, str]]) -> None:
    path = NB_DIR / name
    path.write_text(json.dumps(mk_notebook(cells), indent=1))
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# 01 — UK-DALE EDA                                                            #
# --------------------------------------------------------------------------- #
NB01 = [
    ("markdown", "# 01 — UK-DALE 60-day EDA\n\n"
     "Explores the synthesized or downloaded UK-DALE House 1 subset used for "
     "training (first 46 days) and the simulation window (last 14 days).\n\n"
     "Run `scripts/fetch_ukdale_subset.py` first to populate `data/ukdale/`."),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aerogrid.config import UKDALE_DIR, UKDALE_TRAIN_START, UKDALE_TEST_START, UKDALE_TEST_END

mains = pd.read_parquet(UKDALE_DIR / "house_1" / "mains_1hz.parquet")
dish = pd.read_parquet(UKDALE_DIR / "house_1" / "dishwasher_6s.parquet")
wash = pd.read_parquet(UKDALE_DIR / "house_1" / "washing_machine_6s.parquet")
onsets = pd.read_parquet(UKDALE_DIR / "house_1" / "onsets.parquet")
print(f"mains: {len(mains):,} rows | dishwasher: {len(dish):,} | washer: {len(wash):,}")
print(f"onsets: {onsets['appliance'].value_counts().to_dict()}")"""),
    ("code", """# Daily mean power (aggregate + per-appliance) with train/test split marker
fig, ax = plt.subplots(figsize=(12,4))
for df, label in [(mains, "mains"), (dish, "dishwasher"), (wash, "washing_machine")]:
    daily = df.set_index("timestamp").resample("1D")["power_w"].mean()
    ax.plot(daily.index, daily.values, label=label, linewidth=1)
ax.axvline(UKDALE_TEST_START, color="red", linestyle="--", label="train/test split")
ax.set_ylabel("power (W, daily mean)"); ax.legend(); ax.set_title("UK-DALE House 1: daily mean power")
plt.tight_layout(); plt.show()"""),
    ("code", """# Onset counts per appliance per hour-of-day, split by train/test
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, split in zip(axes, ["train", "test"]):
    sub = onsets[onsets["split"] == split]
    for app in ("dishwasher", "washing_machine"):
        hod = sub[sub["appliance"] == app]["timestamp"].dt.hour
        ax.hist(hod, bins=range(25), alpha=0.6, label=app)
    ax.set_title(f"{split}  (n={len(sub)})"); ax.set_xlabel("hour of day"); ax.legend()
axes[0].set_ylabel("# onsets"); plt.tight_layout(); plt.show()"""),
    ("code", """# Per-appliance cycle power distribution — histogram of non-zero 6 s samples
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, df, name in zip(axes, [dish, wash], ["dishwasher", "washing_machine"]):
    vals = df.loc[df["power_w"] > 50, "power_w"]
    ax.hist(vals, bins=60); ax.set_title(f"{name}: {len(vals):,} 'on' samples")
    ax.set_xlabel("power (W)"); ax.set_ylabel("count"); ax.set_yscale("log")
plt.tight_layout(); plt.show()"""),
]

# --------------------------------------------------------------------------- #
# 02 — SignalWatcher                                                          #
# --------------------------------------------------------------------------- #
NB02 = [
    ("markdown", "# 02 — SignalWatcher DWT + V-I demo\n\n"
     "Visualises an appliance onset at 16 kHz. You need to run "
     "`scripts/build_signatures.py` first (this also synthesises the sample "
     "FLAC if a real UK-DALE 16 kHz file isn't present)."),
    ("code", """import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from aerogrid.signal_watcher import SignalWatcher
from aerogrid.vi_features import SAMPLES_PER_CYCLE
from aerogrid.config import UKDALE_DIR, CACHE_DIR

flac = UKDALE_DIR / "house_1" / "mains_16khz_signatures.flac"
data, fs = sf.read(flac, always_2d=True)
voltage = data[:,0] * 300; current = data[:,1] * 15
sw = SignalWatcher.from_cache()
with (CACHE_DIR / "signatures.pkl").open("rb") as fh: pay = pickle.load(fh)
print(f"loaded {len(voltage)/fs:.1f}s of 16 kHz ({fs} Hz); signatures: {list(pay['signatures'])}")"""),
    ("code", """# Show raw current + DWT D1+D2 detail + envelope around the first few onsets
from datetime import datetime, timezone
onsets = sw.process_window(voltage, current, datetime(2014,1,1,tzinfo=timezone.utc))
print(f"detected {len(onsets)} onsets")
detail = sw.extract_transients(current)
env = np.sqrt(np.convolve(current**2, np.ones(SAMPLES_PER_CYCLE)/SAMPLES_PER_CYCLE, mode='same'))

t = np.arange(len(current))/fs
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axes[0].plot(t, current, lw=0.3); axes[0].set_ylabel("I (A)")
axes[1].plot(t, detail, lw=0.3, color="C2"); axes[1].set_ylabel("DWT D1+D2 (envelope)")
axes[2].plot(t, env, color="C1"); axes[2].set_ylabel("1-cycle RMS env (A)")
for o in onsets:
    for ax in axes:
        ax.axvline((o.timestamp - datetime(2014,1,1,tzinfo=timezone.utc)).total_seconds(),
                   color={'dishwasher':'red','washing_machine':'blue'}.get(o.appliance,'k'),
                   alpha=0.4, linestyle='--')
axes[2].set_xlabel("time (s)"); plt.tight_layout(); plt.show()"""),
    ("code", """# V-I trajectory signatures side by side
fig, axes = plt.subplots(1, len(pay["signatures"]), figsize=(4*len(pay['signatures']), 4))
if len(pay["signatures"]) == 1: axes = [axes]
for ax, (name, sig) in zip(axes, pay["signatures"].items()):
    n = pay["n_points"]; v = sig[:n]; i = sig[n:]
    ax.plot(v, i, "o-", markersize=3); ax.set_title(f"{name} V-I trajectory")
    ax.set_xlabel("voltage (norm.)"); ax.set_ylabel("current (norm.)")
    ax.grid(alpha=0.3); ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
plt.tight_layout(); plt.show()"""),
]

# --------------------------------------------------------------------------- #
# 03 — PriceOracle                                                            #
# --------------------------------------------------------------------------- #
NB03 = [
    ("markdown", "# 03 — PriceOracle evaluation on NYISO test window\n\n"
     "Compares GridFM / Chronos-2 / Seasonal-naïve forecasts on the 14-day "
     "test slice. With no GPU and no `chronos-forecasting`/`gridfm` "
     "packages installed, both fall through to the naïve baseline — the "
     "`source` label on each PriceForecast reports which code path produced "
     "the numbers."),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aerogrid.price_oracle import load_price_history, make_oracle
from aerogrid.config import NYISO_TEST_START, NYISO_TEST_END

prices = load_price_history()
print(f"loaded {len(prices):,} rows spanning {prices['timestamp'].min()} .. {prices['timestamp'].max()}")

# Train/test shapes
fig, ax = plt.subplots(figsize=(12,3))
ax.plot(prices["timestamp"], prices["lbmp"], lw=0.5, label="LBMP")
ax.axvline(NYISO_TEST_START, color="red", ls="--", label="train/test split")
ax.set_ylabel("LBMP ($/MWh)"); ax.legend(); ax.set_title("NYISO 15-min prices")
plt.tight_layout(); plt.show()"""),
    ("code", """# Rolling 24-h forecasts on each day of the test window.
from datetime import timedelta
realized = prices[(prices["timestamp"] >= NYISO_TEST_START) & (prices["timestamp"] < NYISO_TEST_END)]
impls = ["gridfm", "chronos", "naive"]
errors = {}
for impl in impls:
    oracle = make_oracle(impl)
    preds = []
    for day in pd.date_range(NYISO_TEST_START, NYISO_TEST_END - timedelta(days=1), freq="1D"):
        fc = oracle.get_15min_forecast(day, prices)
        preds.extend(fc.median)
    preds = np.array(preds[: len(realized)])
    mape = (np.abs(preds - realized["lbmp"].to_numpy()) / np.maximum(np.abs(realized["lbmp"].to_numpy()), 1)).mean() * 100
    errors[impl] = {"mape": mape, "source": fc.source}
    print(f"{impl:>10s}  source={fc.source:<30s} MAPE={mape:.1f}%")

# One-day closeup
day0 = NYISO_TEST_START
slice_real = realized[(realized["timestamp"] >= day0) & (realized["timestamp"] < day0 + timedelta(days=1))]
plt.figure(figsize=(12,4))
plt.plot(slice_real["timestamp"], slice_real["lbmp"], "k-", label="realized", lw=2)
for impl in impls:
    fc = make_oracle(impl).get_15min_forecast(day0, prices)
    t = pd.date_range(fc.slot_start, periods=len(fc.median), freq="15min", tz="UTC")
    plt.plot(t, fc.median, ls="--", label=f"{impl} ({fc.source})")
plt.ylabel("LBMP ($/MWh)"); plt.legend(); plt.title(f"Day 1 forecasts"); plt.tight_layout(); plt.show()"""),
]

# --------------------------------------------------------------------------- #
# 04 — BehavioralPredictor                                                    #
# --------------------------------------------------------------------------- #
NB04 = [
    ("markdown", "# 04 — Behavioral Predictor\n\n"
     "Fits the default HybridBehavioralPredictor on the train split and "
     "visualises P(onset) heatmaps overlaid with actually-observed test-split "
     "onsets."),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aerogrid.behavioral_predictor import make_predictor, load_onsets
from aerogrid.config import APPLIANCES, UKDALE_TEST_START, UKDALE_TEST_END

onsets = load_onsets()
pred = make_predictor().fit(onsets)
appliances = [a for a in APPLIANCES if APPLIANCES[a].cycle_slots > 0]
print(f"rates per day: {pred._daily_rate}")"""),
    ("code", """# Heatmap: predicted P(onset) as a function of (day-of-week, hour-of-day)
from datetime import datetime, timedelta, timezone
fig, axes = plt.subplots(1, len(appliances), figsize=(6*len(appliances), 4))
if len(appliances) == 1: axes = [axes]
base = datetime(2025, 1, 6, 0, tzinfo=timezone.utc)  # a Monday
for ax, app in zip(axes, appliances):
    grid = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            t = base + timedelta(days=d, hours=h)
            probs = pred.predict_onsets(app, t, horizon_slots=4)
            grid[d, h] = probs.sum()
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(0,24,2)); ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_xlabel("hour"); ax.set_title(f"P(onset) — {app}")
    plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.show()"""),
    ("code", """# Overlay test-split onsets on the predicted hour-of-day density
test = onsets[onsets["split"] == "test"]
fig, axes = plt.subplots(1, len(appliances), figsize=(6*len(appliances), 3))
if len(appliances) == 1: axes = [axes]
for ax, app in zip(axes, appliances):
    ax.plot(np.arange(96)/4, pred._density_24[app], label="trained density")
    th = test[test["appliance"] == app]["timestamp"].dt.hour + test[test["appliance"]==app]["timestamp"].dt.minute/60
    ax.plot(th, np.zeros(len(th)), "rx", markersize=8, label="test onsets")
    ax.set_title(app); ax.set_xlabel("hour"); ax.legend()
plt.tight_layout(); plt.show()"""),
]

# --------------------------------------------------------------------------- #
# 05 — Optimizer deep dive                                                    #
# --------------------------------------------------------------------------- #
NB05 = [
    ("markdown", "# 05 — Optimizer Gantt + cost analysis\n\n"
     "Solves the 96-slot MILP for the first test day, compares against the "
     "naïve baseline, and plots a Gantt + price overlay."),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aerogrid.optimizer import solve_proactive_schedule
from aerogrid.price_oracle import load_price_history, make_oracle
from aerogrid.behavioral_predictor import load_onsets, make_predictor
from aerogrid.config import NYISO_TEST_START, APPLIANCES, SLOT_MINUTES

prices_df = load_price_history()
now = NYISO_TEST_START
fc = make_oracle('naive').get_15min_forecast(now, prices_df)
prices = np.asarray(fc.median)
op = make_predictor().fit(load_onsets()).predict_all(now)
sched = solve_proactive_schedule(now, prices, op)
print(f"expected ${sched.expected_cost:.3f}  baseline ${sched.baseline_cost:.3f}  savings {sched.savings()*100:.1f}%")"""),
    ("code", """# Gantt chart of the schedule overlaid with the price forecast
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                               gridspec_kw={'height_ratios':[3,2]})
hrs = np.arange(96)/4
colors = {'dishwasher':'#1f77b4','washing_machine':'#2ca02c'}
# EV stacked bars
ax1.bar(hrs, sched.ev_power_kw, width=0.25, color='#ff7f0e', alpha=0.7, label='EV (kW)')
for t in sched.tasks:
    spec = APPLIANCES[t.appliance]
    ax1.barh(y=spec.rated_kw/2, width=t.slots*0.25, left=t.start_slot*0.25,
             height=0.8, color=colors[t.appliance], alpha=0.6, label=t.appliance)
ax1.set_ylabel("load (kW)"); ax1.legend(loc='upper right'); ax1.set_title('schedule — first test day')
ax2.plot(hrs, prices, color='k', label='LBMP forecast ($/MWh)')
ax2.set_ylabel("$/MWh"); ax2.set_xlabel("hour"); ax2.legend()
plt.tight_layout(); plt.show()"""),
]

# --------------------------------------------------------------------------- #
# 06 — End-to-end digital twin                                                #
# --------------------------------------------------------------------------- #
NB06 = [
    ("markdown", "# 06 — End-to-end digital-twin run\n\n"
     "Runs the full LangGraph loop against the 14-day test slice and plots "
     "cumulative cost, savings trajectory, and detected/planned events. "
     "Expect ~2 min for a 24-h simulation on CPU."),
    ("code", """import subprocess, sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
# Run the twin for 48 simulated hours and inspect the log.
subprocess.run([sys.executable, "-m", "aerogrid.sim.digital_twin",
                "--use-test-window", "--hours", "48"], check=True)"""),
    ("code", """import pandas as pd, matplotlib.pyplot as plt, json
from aerogrid.config import RUN_LOG_PATH
rows = [json.loads(l) for l in RUN_LOG_PATH.read_text().splitlines() if l.strip()]
log = pd.DataFrame(rows)
log["now"] = pd.to_datetime(log["now"])
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)
ax1.plot(log["now"], log["cumulative_cost"], label="optimizer")
ax1.plot(log["now"], log["cumulative_baseline_cost"], label="naive baseline", linestyle="--")
ax1.set_ylabel("$"); ax1.legend(); ax1.set_title("cumulative cost")
ax2.plot(log["now"], log["realized_price"], "k-", lw=0.7)
ax2.set_ylabel("realized LBMP ($/MWh)"); ax2.set_xlabel("time")
for reason, sub in log[log["replan_reason"].notna()].iterrows():
    ax2.axvline(sub["now"], color="red", alpha=0.3, linewidth=0.5)
plt.tight_layout(); plt.show()
print(f"n ticks: {len(log)}  replans: {log['replan_reason'].notna().sum()}  final savings: "
      f"{(1 - log['cumulative_cost'].iloc[-1] / max(log['cumulative_baseline_cost'].iloc[-1], 1e-6))*100:.1f}%")"""),
]


def main() -> int:
    write_notebook("01_explore_ukdale.ipynb", NB01)
    write_notebook("02_signal_watcher.ipynb", NB02)
    write_notebook("03_price_oracle.ipynb", NB03)
    write_notebook("04_behavioral_predictor.ipynb", NB04)
    write_notebook("05_optimizer.ipynb", NB05)
    write_notebook("06_end_to_end.ipynb", NB06)
    print("\nall notebooks generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
