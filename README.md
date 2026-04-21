# AeroGrid — Proactive Multi-Agent Home Energy Planner

AeroGrid couples a high-frequency NILM disaggregator, a behavioral
appliance-onset predictor, a physics-informed price foundation model, and a
proactive MILP scheduler into a LangGraph agent loop that minimizes
electricity cost via 15-min market arbitrage. It ships with a digital-twin
harness that replays a 14-day held-out test slice of UK-DALE + NYISO data so
the whole system is runnable offline.

## Architecture

```
             ┌──────────────────┐
             │ UK-DALE replay   │ 16 kHz mains + 6 s per-appliance
             │  (Streamer)      │
             └────────┬─────────┘
                      ▼
    ┌──────────────────────────────────────────────────┐
    │  LangGraph loop (one tick per 15-min slot)       │
    │                                                  │
    │  signal_watch ──▶ forecast_price                 │
    │       │                 │                        │
    │       ▼                 ▼                        │
    │   (NILM onsets)   (96-slot LBMP, quantiles)      │
    │       │                 │                        │
    │       └──▶ predict_behavior ──▶ optimize         │
    │                                   │              │
    │                                   ▼              │
    │                          user_confirm (HITL)     │
    │                                   │              │
    │                                   ▼              │
    │                          execute ─▶ monitor ──┐  │
    │                                               │  │
    │            replan reason? ◀───────────────────┘  │
    └──────────────────────────────────────────────────┘
                      ▲
             ┌────────┴─────────┐
             │ NYISO price feed │ (mock, injectable spike)
             │ (PriceServer)    │
             └──────────────────┘
```

Nodes:
- **`signal_watch`** — `SignalWatcher` runs DWT (db4, level 4) on the 1-cycle
  RMS envelope of the 16 kHz current, detects step onsets, and classifies via
  cosine similarity against V-I trajectory signatures.
- **`forecast_price`** — `PriceOracle` → 96-slot 15-min LBMP forecast with
  10/50/90 quantiles. `GridFMPriceOracle` (primary) → `ChronosPriceOracle`
  (alt) → `SeasonalNaiveOracle` (fallback), selected by config.
- **`predict_behavior`** — per-appliance 96-slot onset probability from the
  `HybridBehavioralPredictor` (KDE over hour-of-day × weekend multiplier).
- **`optimize`** — cvxpy MILP (HiGHS) over EV continuous power + binary cycle
  start-indicators; objective minimizes cost minus reservation-utility.
- **`user_confirm`** — `interrupt()` / `Command(resume=…)` if the MILP shifts
  a cycle >2h from the user's habitual hour.
- **`monitor`** — flags replan reasons (>25% price deviation from forecast,
  unplanned onset observed by NILM).

## Data

| source | total | train | test (= simulation) | path |
|---|---|---|---|---|
| UK-DALE House 1, 1 Hz mains + 6 s per-appliance | 60 d | 46 d | **14 d** | `data/ukdale/house_1/*.parquet` |
| UK-DALE 16 kHz stereo FLAC | 6 h inside test window | — | 6 h | `data/ukdale/house_1/mains_16khz_3day.flac` |
| NYISO RT LBMP 5-min → 15-min | 90 d | 76 d | **14 d** | `data/nyiso/nyc_15min.parquet` |
| NYISO DAM LBMP hourly | 90 d | same | same | `data/nyiso/nyc_dam.parquet` |
| ENTSO-E DE-LU (optional alt) | 30 d | 20 d | 10 d | `data/entsoe/de_lu_15min.parquet` |

All fetchers download real data and raise `FetchError` if the upstream server
is unreachable — there is no synthetic fallback. Each successful run writes
`source="real"` in `MANIFEST.json`.

Note: UK-DALE (UK residential) + NYISO (US wholesale) is explicitly a
demo artifact. Fine-tuning `GridFMPriceOracle` on ENTSO-E DE-LU so that UK
appliances run against EU prices is listed as stretch-goal future work.

## Quickstart

```bash
# 1) Python version (pyenv pins 3.12.13 via .python-version in the repo root)
pyenv install  # reads .python-version; skipped if already installed

# 2) env
uv sync --extra dev

# 3) data — fetchers require live network access; they raise on failure
.venv/bin/python scripts/fetch_ukdale_subset.py
.venv/bin/python scripts/fetch_ukdale_subset.py --with-16khz   # optional, ~800 MB
.venv/bin/python scripts/fetch_nyiso_prices.py
.venv/bin/python scripts/build_signatures.py

# 4) tests
.venv/bin/python -m pytest -q

# 5) notebooks (EDA, signal watcher, forecasts, predictor, optimizer, end-to-end)
.venv/bin/python -m jupyter lab notebooks/

# 6) full 14-day digital-twin simulation on the test slice
.venv/bin/python -m aerogrid.sim.digital_twin --use-test-window

# 6b) shorter run with a planted price spike to exercise the replan path
.venv/bin/python -m aerogrid.sim.digital_twin --use-test-window --hours 48 --inject-spike
```

Optional: to actually run the physics-informed forecaster and zero-shot
Chronos, install the forecast extras (~1.5 GB of PyTorch):

```bash
uv sync --extra forecast
# then, in aerogrid/config.py, keep PRICE_ORACLE_IMPL = "gridfm"
```

Without these, the oracle chain automatically falls back to
`SeasonalNaiveOracle` and the `source` field on every `PriceForecast` reports
exactly which code path produced each forecast.

## Repo layout

```
aerogrid/                  core library
  config.py                all pinned date windows, appliance specs
  signal_watcher.py        DWT + V-I NILM
  vi_features.py           V-I trajectory descriptor + helpers
  price_oracle.py          GridFM / Chronos-2 / Seasonal-naive
  behavioral_predictor.py  Hybrid / Chronos / Mamba stub
  optimizer.py             cvxpy + HiGHS MILP
  graph.py                 LangGraph assembly, HITL, checkpointing
  state.py                 TypedDict schema
  types.py                 ApplianceOnset / Schedule / PriceForecast dataclasses
  sim/
    streamer.py            UK-DALE test-slice replay
    price_server.py        NYISO test parquet, spike injection
    digital_twin.py        driver that ticks the graph

scripts/                   one-shot data + signature jobs
  fetch_ukdale_subset.py
  fetch_nyiso_prices.py
  fetch_entsoe_prices.py
  build_signatures.py
  build_notebooks.py

notebooks/                 six EDA / visualization notebooks (01–06)
tests/                     pytest suite
```

## What's deliberately out of scope (v1)

- Fine-tuning GridFM on ENTSO-E DE-LU prices.
- Full 16 kHz UK-DALE House 1 (the real dataset is multi-TB; the 6 h window
  is enough for the DWT demo).
- Real Mamba-3 1.5 B behavioral predictor — stubbed; the classical hybrid
  ships as the default because appliance onsets are sparse and
  periodicity-dominated, and no `mamba.cpp` analog exists as of April 2026.
- Live smart-meter / EV integration.
- Geographic coherence (UK appliance data + US prices is an explicit demo
  artifact).
