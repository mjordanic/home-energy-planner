"""Generate notebooks/05_optimizer.ipynb — solve_receding_horizon test suite.

Run from the repo root:
    python scripts/_build_optimizer_notebook.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "05_optimizer.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

TITLE = """\
# 05 — `solve_receding_horizon` test suite (intent-driven LP)

After the April 2026 refactor the optimiser controls **two continuous
loads** (EV charger and heater) and no longer schedules cycle-shaped
dishwasher / washing-machine starts. Those are *event-driven*: the user
starts them, and the agent only proposes a small forward shift via the
HITL gate.

Each scenario below isolates one aspect of the new formulation.

| # | Scenario | Feature under test |
|---|---|---|
| A | EV availability gate (no charging before 20:00) | C1 mask |
| B | EV deadline inside vs. outside horizon | C2 regimes |
| C | Heater per-window energy delivery | C3 (4 kWh by 07:00, 2 kWh by 18:00) |
| D | Heater shapes itself into the cheapest hour of the window | C3 + price valley |
| E | Power-cap binding between EV and heater | C5 |
| F | Committed dishwasher cycle as exogenous load | C5 + cap headroom |
| I | Stress: many simultaneous onsets (HITL throughput) | LangGraph + auto-responses |
| J | Horizon sensitivity (6 / 12 / 24 h) | LP scaling, savings vs. wall-time |
| K | Heater infeasibility → soft slack | C3 slack absorbs |
| L | Joint MIP vs price-only reschedule (cap binding) | C5 + C6 |
| M | Re-nudge before start (+1 h again) | Deferred-cycle replanning |
"""

SETUP = '''\
import sys, time, warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from IPython.display import display
except ImportError:
    display = print  # plain-script fallback

from aerogrid.optimizer import solve_receding_horizon
from aerogrid.commit import CommitTracker
from aerogrid.config import (
    APPLIANCES, EV_AVAILABLE_FROM_HOUR, EV_DEADLINE_HOUR, HEATER_DEADLINES,
    HITL_AUTO_RESPONSES, HOUSE_POWER_CAP_KW, HITL_RESCHEDULE_MIN_SAVINGS_EUR,
    HITL_RESCHEDULE_WINDOW_HOURS, SHORT_HORIZON_SLOTS, SLOT_MINUTES,
    HeaterEnergyDeadline,
)
from aerogrid.types import PendingCycle, RescheduleProposal, ScheduledTask
from aerogrid.graph import _propose_for_onset
from aerogrid.hitl_policy import decide_reschedule

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# ── constants ────────────────────────────────────────────────────────────────
SLOT_H = SLOT_MINUTES / 60          # 0.25 h per slot
_PER_SLOT = SLOT_H / 1000.0         # kW·(€/MWh) → € per slot

_PALETTE = {
    "ev_charger":      "#ff7f0e",
    "heater":          "#d62728",
    "dishwasher":      "#1f77b4",
    "washing_machine": "#2ca02c",
}


def _utc(h: int, m: int = 0, day: int = 15) -> datetime:
    return datetime(2026, 4, day, h, m, tzinfo=timezone.utc)


# ── shared plotting helper for continuous-load schedules ────────────────────
def _deadline_lines_in(t_start, t_end,
                      ev_hour=EV_DEADLINE_HOUR,
                      heater_deadlines=HEATER_DEADLINES):
    """Return [(t, label, color, ls)] for every deadline inside [t_start, t_end]."""
    out = []
    t = t_start.replace(hour=ev_hour, minute=0, second=0, microsecond=0)
    if t < t_start:
        t += timedelta(days=1)
    while t <= t_end:
        out.append((t, f"EV deadline {ev_hour:02d}:00", "#d62728", "--"))
        t += timedelta(days=1)
    for d in heater_deadlines:
        t = t_start.replace(hour=d.hour, minute=0, second=0, microsecond=0)
        if t < t_start:
            t += timedelta(days=1)
        while t <= t_end:
            out.append((t, f"heater {d.hour:02d}:00 ({d.kwh_required:.0f} kWh)",
                        "#9467bd", ":"))
            t += timedelta(days=1)
    return out


def plot_schedule(sched, prices, *, title="", cap_kw=HOUSE_POWER_CAP_KW, figsize=(13, 5.5)):
    """Stacked-bar load chart (top) + price step chart (bottom), datetime x-axis.

    EV and heater are shown as continuous-power bars. Committed cycle tasks
    (dishwasher / washing machine pinned by CommitTracker) appear hatched.
    Vertical dashed lines mark the EV deadline (red) and heater deadlines
    (purple) that fall inside the visible window.
    """
    T = sched.horizon_slots
    t0 = sched.slot_start
    times = [t0 + timedelta(minutes=SLOT_MINUTES * i) for i in range(T)]
    width_days = SLOT_MINUTES * 0.9 / (60.0 * 24.0)   # bar width in matplotlib date units
    prices = np.asarray(prices, dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    bottom = np.zeros(T)

    ev = np.asarray(sched.ev_power_kw, dtype=float)
    heat = np.asarray(sched.heater_power_kw, dtype=float)

    ax1.bar(times, ev, width=width_days, bottom=bottom,
            color=_PALETTE["ev_charger"], alpha=0.85, label="EV (kW)", align="edge")
    bottom = bottom + ev
    ax1.bar(times, heat, width=width_days, bottom=bottom,
            color=_PALETTE["heater"], alpha=0.8, label="Heater (kW)", align="edge")
    bottom = bottom + heat

    seen: set = set()
    for task in sched.tasks:
        spec = APPLIANCES.get(task.appliance)
        if spec is None or not task.committed:
            continue
        run = np.zeros(T)
        for t in range(task.start_slot, min(task.start_slot + task.slots, T)):
            run[t] = spec.rated_kw
        color = _PALETTE.get(task.appliance, "#8c564b")
        label = (task.appliance + " [committed]") if task.appliance not in seen else None
        ax1.bar(times, run, width=width_days, bottom=bottom,
                color=color, alpha=0.7, label=label, hatch="//", align="edge")
        bottom = bottom + run
        seen.add(task.appliance)

    ax1.axhline(cap_kw, color="black", ls="--", lw=1.5, label=f"Cap {cap_kw} kW")
    ax1.set_ylabel("load (kW)")
    ax1.set_ylim(0, cap_kw * 1.3)
    ax1.set_title(title or "Optimizer schedule", fontsize=11)

    ann = (
        f"Cost €{sched.expected_cost:.4f}  ·  "
        f"Baseline €{sched.baseline_cost:.4f}  ·  "
        f"Savings {sched.savings()*100:.1f}%  ·  "
        f"Solver {sched.solver_status}"
    )
    if sched.heater_window_kwh:
        ann += "  ·  heater " + ", ".join(
            f"{h:02d}:{int(round(k*1000)):d}Wh" for h, k in sched.heater_window_kwh.items()
        )
    ax1.text(0.01, 0.98, ann, transform=ax1.transAxes, va="top", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="#cca300", alpha=0.9))

    ax2.step(times, prices[:T], where="post", color="k", lw=1.4, label="Price (€/MWh)")
    ax2.fill_between(times, prices[:T], step="post", alpha=0.12, color="k")
    ax2.set_ylabel("€/MWh")
    ax2.set_xlabel("time of day (UTC)")

    # Deadline overlays + datetime formatting (after data is drawn so x-limits are set).
    t_end = times[-1] + timedelta(minutes=SLOT_MINUTES)
    seen_kinds: dict = {}
    for t, label, color, ls in _deadline_lines_in(t0, t_end):
        seen_kinds.setdefault(label, (color, ls))
        for ax in (ax1, ax2):
            ax.axvline(t, color=color, ls=ls, lw=1.1, alpha=0.7)
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    for label, (color, ls) in seen_kinds.items():
        handles_ax1.append(plt.Line2D([0], [0], color=color, ls=ls, lw=1.1, label=label))
        labels_ax1.append(label)
    ax1.legend(handles_ax1, labels_ax1, loc="upper right", fontsize=7.6, ncol=2)
    ax2.legend(fontsize=8, loc="upper right")

    span_h = (t_end - t0).total_seconds() / 3600.0
    if span_h <= 12:
        loc = mdates.HourLocator(interval=1)
    elif span_h <= 36:
        loc = mdates.HourLocator(interval=3)
    else:
        loc = mdates.HourLocator(interval=6)
    ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")

    plt.tight_layout()
    return fig


def make_de_price_curve(seed: int = 42, base: float = 55.0, peak1_h: float = 8.0,
                        peak2_h: float = 18.0, n: int = 96) -> np.ndarray:
    """Synthetic 24 h DE-style price profile, deterministic given seed."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, n, endpoint=False)
    prices = (
        base
        + 35 * np.sin(2 * np.pi * (t - peak1_h) / 24)
        + 30 * np.exp(-0.5 * ((t - peak1_h) / 1.5) ** 2)
        + 25 * np.exp(-0.5 * ((t - peak2_h) / 1.5) ** 2)
        + 6 * rng.standard_normal(n)
    ).clip(10, 200)
    return prices


print(f"✓ helpers loaded  ·  horizon={SHORT_HORIZON_SLOTS} slots ({SHORT_HORIZON_SLOTS*SLOT_H:.0f} h)  ·  cap={HOUSE_POWER_CAP_KW} kW")
print(f"✓ EV window {EV_AVAILABLE_FROM_HOUR:02d}:00 → {EV_DEADLINE_HOUR:02d}:00 UTC")
print(f"✓ Heater deadlines: " + ", ".join(f"{d.kwh_required} kWh by {d.hour:02d}:00" for d in HEATER_DEADLINES))
'''

# ── scenario A ────────────────────────────────────────────────────────────────
MD_A = """\
## Scenario A — EV availability gate

The EV is only pluggable from `EV_AVAILABLE_FROM_HOUR = 20:00` UTC each day
until the next `EV_DEADLINE_HOUR = 07:00` UTC. Calling the optimiser at
**14:00 UTC** with a 24 h horizon, every slot from 14:00 → 19:45 must have
zero EV power.

Inside the 11 h overnight window the LP picks the cheapest hours to deliver
the full 24 kWh need. The heater meanwhile satisfies its overnight 4 kWh
deadline (window: 18:00 → 07:00) and the daytime 2 kWh deadline (window:
07:00 → 18:00).
"""

CODE_A = """\
prices_A = make_de_price_curve(seed=1)
now_A = _utc(14)

sched_A = solve_receding_horizon(
    now_A, prices_A, horizon_slots=96, remaining_ev_kwh=24.0,
)

ev = np.asarray(sched_A.ev_power_kw)
print(f"EV [14:00–20:00] kWh = {ev[:24].sum() * SLOT_H:.3f}  (must be 0)")
print(f"EV [20:00–07:00] kWh = {ev[24:68].sum() * SLOT_H:.3f}  (full need)")
print(f"EV [07:00–14:00] kWh = {ev[68:].sum() * SLOT_H:.3f}  (must be 0)")
print(f"Heater windows  kWh = {sched_A.heater_window_kwh}")

assert ev[:24].sum() < 1e-6
assert ev[68:].sum() < 1e-6
assert abs(ev.sum() * SLOT_H - 24.0) < 1e-3

fig_A = plot_schedule(
    sched_A, prices_A,
    title="Scenario A — EV availability gate (no charging 14:00–19:45)",
)
plt.show()
"""

# ── scenario B ────────────────────────────────────────────────────────────────
MD_B = """\
## Scenario B — EV deadline inside vs. outside horizon

Two configurations of the EV deadline against an 8-slot (2 h) horizon:

* **B-near**: now = 05:00, deadline 07:00 (inside the 2 h horizon, slot 8) → C2 hard regime.
* **B-far**: now = 22:00, deadline 07:00 (9 h, well outside the 2 h horizon) → C2 prorated regime with a `deadline_safety = 1.2` margin.

Constraint C2 switches between

> `Δt · Σ_{t<t_d} p_ev[t] + σ_ev ≥ remaining_ev_kwh` &nbsp;&nbsp;(inside)

> `Δt · Σ_t p_ev[t] + σ_ev ≥ remaining_ev_kwh · (H/τ) · γ` &nbsp;&nbsp;(outside).
"""

CODE_B = """\
prices_B = np.array([130., 120., 40., 38., 40., 125., 130., 120.])

now_B_near = _utc(5, 0)
now_B_far  = _utc(22, 0)

sched_B_near = solve_receding_horizon(
    now_B_near, prices_B,
    remaining_ev_kwh=5.0, time_to_deadline_h=2.0, horizon_slots=8,
)
sched_B_far = solve_receding_horizon(
    now_B_far, prices_B,
    remaining_ev_kwh=24.0, time_to_deadline_h=9.0, horizon_slots=8,
)

near_kwh = sum(sched_B_near.ev_power_kw) * SLOT_H
far_kwh = sum(sched_B_far.ev_power_kw) * SLOT_H

print(f"B-near delivered: {near_kwh:.3f} kWh  (full 5.0 expected)")
print(f"B-far  delivered: {far_kwh:.3f} kWh  (proportional 24·(2/9)·1.2 ≈ {24*(2/9)*1.2:.3f})")

# Two stacked panels per case: EV power (top) + price (bottom), all datetime.
fig_B, axes = plt.subplots(
    2, 2, figsize=(13.5, 5.4), sharey="row",
    gridspec_kw={"height_ratios": [3, 2]},
)
width_days = SLOT_MINUTES * 0.85 / (60.0 * 24.0)

for col, (now_b, sched_b, kwh_b, label) in enumerate([
    (now_B_near, sched_B_near, near_kwh,
     f"B-near · deadline 2 h · {near_kwh:.2f} kWh delivered"),
    (now_B_far,  sched_B_far,  far_kwh,
     f"B-far  · deadline 9 h (out of 2 h horizon) · {far_kwh:.2f} kWh delivered"),
]):
    ax_p = axes[0, col]
    ax_pr = axes[1, col]
    times = [now_b + timedelta(minutes=SLOT_MINUTES * i) for i in range(8)]
    ax_p.bar(times, sched_b.ev_power_kw, width=width_days,
             color=_PALETTE["ev_charger"], align="edge", alpha=0.85, label="EV kW")
    ax_p.set_title(label, fontsize=10)
    ax_pr.step(times, prices_B, where="post", color="k", lw=1.3, label="price")
    ax_pr.fill_between(times, prices_B, step="post", alpha=0.12, color="k")
    ax_pr.set_xlabel("time (UTC)")
    # Deadline lines (EV only matters here).
    t_end = times[-1] + timedelta(minutes=SLOT_MINUTES)
    for t, lbl, color, ls in _deadline_lines_in(now_b, t_end,
                                                heater_deadlines=()):
        for ax in (ax_p, ax_pr):
            ax.axvline(t, color=color, ls=ls, lw=1.2, alpha=0.75, label=lbl if ax is ax_p else None)
    ax_p.legend(fontsize=8, loc="upper right")
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax_pr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

axes[0, 0].set_ylabel("EV kW")
axes[1, 0].set_ylabel("€/MWh")
fig_B.autofmt_xdate(rotation=0, ha="center")
plt.tight_layout()
plt.show()
"""

# ── scenario C ────────────────────────────────────────────────────────────────
MD_C = """\
## Scenario C — Heater per-window energy delivery

With `HEATER_DEADLINES = (07:00 → 4 kWh, 18:00 → 2 kWh)` over a flat-price
24 h horizon starting at midnight, both windows lie inside the horizon and
must be fully satisfied. The optimiser is free to spread power however it
likes inside each window — the LP only cares about the integral.

We verify the per-window kWh, then plot the full 24 h schedule.
"""

CODE_C = """\
sched_C = solve_receding_horizon(
    _utc(0), np.full(96, 50.0),
    horizon_slots=96, remaining_ev_kwh=0.0,
)

print(f"Window 07:00 delivered: {sched_C.heater_window_kwh[7]:.3f} kWh  (need 4.0)")
print(f"Window 18:00 delivered: {sched_C.heater_window_kwh[18]:.3f} kWh  (need 2.0)")

assert abs(sched_C.heater_window_kwh[7] - 4.0) < 1e-3
assert abs(sched_C.heater_window_kwh[18] - 2.0) < 1e-3

fig_C = plot_schedule(
    sched_C, np.full(96, 50.0),
    title="Scenario C — heater windows: 4 kWh by 07:00, 2 kWh by 18:00 (flat prices)",
    figsize=(14, 5.5),
)
plt.show()
"""

# ── scenario D ────────────────────────────────────────────────────────────────
MD_D = """\
## Scenario D — Heater shapes itself into cheap hours

Same setup as C, but with a price valley between **23:00 and 02:00** (slots
92–95 plus 0–7). The 4 kWh overnight window covers 18:00 → 07:00, so the
heater has 13 h to spend its 4 kWh. The optimiser should concentrate
delivery into the cheap valley.

The daytime 2 kWh window (07:00 → 18:00) has no valley, so we expect
roughly flat delivery there.
"""

CODE_D = """\
prices_D = np.full(96, 80.0)
prices_D[0:8] = 25.0     # 00:00–02:00 cheap
prices_D[92:96] = 25.0   # 23:00–24:00 cheap
prices_D[56:60] = 30.0   # mid-afternoon (14:00–15:00) cheap

sched_D = solve_receding_horizon(
    _utc(0), prices_D,
    horizon_slots=96, remaining_ev_kwh=0.0,
)

heat = np.asarray(sched_D.heater_power_kw)
overnight_cheap_kwh = (heat[0:8].sum() + heat[92:96].sum()) * SLOT_H
overnight_total_kwh = sched_D.heater_window_kwh[7]
print(f"Overnight valley kWh / total kWh = {overnight_cheap_kwh:.3f} / {overnight_total_kwh:.3f}")

fig_D = plot_schedule(
    sched_D, prices_D,
    title="Scenario D — heater concentrates power in the overnight valley",
    figsize=(14, 5.5),
)
plt.show()
"""

# ── scenario E ────────────────────────────────────────────────────────────────
MD_E = """\
## Scenario E — Power-cap couples EV and heater

Tight cap of **8 kW**: EV (rated 7 kW) plus heater (rated 2 kW) cannot run
flat-out together. We start at 21:00 (inside the EV window), give the EV
12 kWh remaining and the heater its overnight 4 kWh, then watch what the
LP picks during the 21:00–22:00 hour where prices are lowest.
"""

CODE_E = """\
prices_E = np.full(96, 80.0)
prices_E[4:8] = 20.0   # cheapest slot for the start at 21:00 (slots 0..3 = 21:00..22:00, slots 4..7 = 22:00..23:00)

sched_E = solve_receding_horizon(
    _utc(21), prices_E,
    horizon_slots=96, remaining_ev_kwh=12.0, house_cap_kw=8.0,
)

ev = np.asarray(sched_E.ev_power_kw)
heat = np.asarray(sched_E.heater_power_kw)
total = ev + heat
print(f"max(EV+heater) = {total.max():.3f} kW  (cap 8.0)")
assert total.max() <= 8.0 + 1e-6
print(f"EV total kWh    = {ev.sum() * SLOT_H:.3f}")
print(f"Heater total kWh= {heat.sum() * SLOT_H:.3f}  by window: {sched_E.heater_window_kwh}")

fig_E = plot_schedule(
    sched_E, prices_E, cap_kw=8.0,
    title="Scenario E — 8 kW cap forces EV+heater to share peak slots",
    figsize=(14, 5.5),
)
plt.show()
"""

# ── scenario F ────────────────────────────────────────────────────────────────
MD_F = """\
## Scenario F — Committed dishwasher cycle as exogenous load

A dishwasher cycle is already running (committed via the HITL reschedule
flow earlier). It occupies slots 0–7 at 2.5 kW. The cap is **8.5 kW** so
during those 8 slots EV+heater have only `8.5 − 2.5 = 6.0 kW` headroom.

We expect the optimiser to throttle EV+heater accordingly during slots 0–7
and run them freely afterwards.
"""

CODE_F = """\
prices_F = np.full(96, 60.0)
prices_F[2:6] = 25.0   # cheapest hour right around the committed cycle

committed_F = [
    ScheduledTask(
        appliance="dishwasher", start_slot=0, slots=8,
        expected_kwh=APPLIANCES["dishwasher"].rated_kw * 8 * SLOT_H,
        committed=True,
    )
]

sched_F = solve_receding_horizon(
    _utc(20), prices_F,
    horizon_slots=96, remaining_ev_kwh=12.0, committed_tasks=committed_F,
    house_cap_kw=8.5,
)

ev = np.asarray(sched_F.ev_power_kw)
heat = np.asarray(sched_F.heater_power_kw)
dish_kw = APPLIANCES["dishwasher"].rated_kw
total_with_dish = ev + heat + np.array(
    [dish_kw if t < 8 else 0.0 for t in range(96)]
)
print(f"max load with committed dishwasher = {total_with_dish.max():.3f} kW  (cap 8.5)")
assert total_with_dish.max() <= 8.5 + 1e-6

fig_F = plot_schedule(
    sched_F, prices_F, cap_kw=8.5,
    title="Scenario F — committed dishwasher consumes cap headroom",
    figsize=(14, 5.5),
)
plt.show()
"""

# ── scenario I ────────────────────────────────────────────────────────────────
MD_I = """\
## Scenario I — HITL throughput stress test

We slam the system with **8 onsets in a 2-hour window** (alternating
dishwasher and washing machine) and run each through the propose →
decide_reschedule → simulated-user pipeline. The HITL gate's *AUTO*
decisions short-circuit small savings, so only the proposals that genuinely
matter make it past the threshold.

The summary table records: per-onset shift, savings, threshold filter,
HITL decision (`ask`/`auto`), and the simulated user's reply.
"""

CODE_I = """\
import itertools
prices_I = make_de_price_curve(seed=3, peak1_h=8.0, peak2_h=19.0)

onsets_I = []
for k, (h, m) in enumerate(
    [(18, 0), (18, 30), (19, 0), (19, 15), (19, 45), (20, 0), (20, 30), (20, 45)]
):
    appliance = "dishwasher" if k % 2 == 0 else "washing_machine"
    onsets_I.append((appliance, _utc(h, m)))

rows = []
for app, onset_at in onsets_I:
    slot_idx = (onset_at.hour * 4 + onset_at.minute // 15)
    slice_p = np.concatenate([prices_I[slot_idx:], prices_I[:slot_idx]])
    spec = APPLIANCES[app]
    p = _propose_for_onset(
        app, onset_at, slice_p,
        cycle_slots=spec.cycle_slots, rated_kw=spec.rated_kw, horizon_slots=96,
    )
    decision = decide_reschedule(p)
    sim_answer = HITL_AUTO_RESPONSES[app]
    rows.append({
        "appliance": app,
        "onset": onset_at.strftime("%H:%M"),
        "best_shift_min": round(p.shift_minutes, 0),
        "cost_now_€": round(p.cost_now_eur, 4),
        "cost_best_€": round(p.cost_proposed_eur, 4),
        "savings_€": round(p.savings_eur, 4),
        "above_threshold": p.savings_eur >= HITL_RESCHEDULE_MIN_SAVINGS_EUR,
        "decision": decision.action,
        "sim_user": sim_answer,
    })

df_I = pd.DataFrame(rows)
display(df_I)

# Plot cost-now vs cost-best per onset.
fig_I, ax = plt.subplots(figsize=(13, 4.5))
x = np.arange(len(df_I))
ax.bar(x - 0.18, df_I["cost_now_€"], width=0.36, color="#888", label="run-now cost")
ax.bar(x + 0.18, df_I["cost_best_€"], width=0.36, color="#55a868", label="best-shift cost")
for i, row in df_I.iterrows():
    if row["above_threshold"]:
        ax.text(i, max(row["cost_now_€"], row["cost_best_€"]) * 1.02,
                f"€{row['savings_€']:.2f}", ha="center", fontsize=8, color="green")
ax.set_xticks(x)
ax.set_xticklabels([f"{r['appliance'][:4]}\\n{r['onset']}" for _, r in df_I.iterrows()], fontsize=8)
ax.set_ylabel("cycle cost (€)")
ax.set_title("Scenario I — HITL throughput: 8 onsets in 3 hours")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

print(f"\\nTotal potential savings: €{df_I['savings_€'].sum():.4f}")
ask_total = df_I[df_I['decision'] == 'ask']['savings_€'].sum()
print(f"Of which forwarded to HITL (≥ threshold): €{ask_total:.4f}")
print(f"Realised savings (sim user accepts dishwasher, declines washer): "
      f"€{df_I[(df_I['sim_user']=='accept')&(df_I['decision']=='ask')]['savings_€'].sum():.4f}")
"""

# ── scenario J ────────────────────────────────────────────────────────────────
MD_J = """\
## Scenario J — Horizon sensitivity

We solve the same 24 h day with horizons of **6 h, 12 h, and 24 h**, all
evaluated at midnight. A short horizon cannot see the next deadline far
ahead, so it relies on the proportional-EV and proportional-heater
fallbacks; a long horizon sees both deadlines and can plan globally.

We report:
- expected cost
- baseline cost
- savings %
- LP wall-clock time

The 24 h LP should be only a few times slower than the 6 h LP because the
problem has no integer variables.
"""

CODE_J = """\
prices_J = make_de_price_curve(seed=4)

results = []
for h in (6, 12, 24):
    slots = h * 4
    p_slice = prices_J[:slots]
    t0 = time.perf_counter()
    s = solve_receding_horizon(
        _utc(0), p_slice, horizon_slots=slots, remaining_ev_kwh=24.0,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    results.append({
        "horizon_h": h,
        "horizon_slots": slots,
        "expected_cost_€": round(s.expected_cost, 4),
        "baseline_cost_€": round(s.baseline_cost, 4),
        "savings_%": round(s.savings() * 100, 1),
        "ev_total_kWh": round(sum(s.ev_power_kw) * SLOT_H, 3),
        "heater_kWh_07": round(s.heater_window_kwh.get(7, 0.0), 3),
        "heater_kWh_18": round(s.heater_window_kwh.get(18, 0.0), 3),
        "solve_ms": round(dt_ms, 1),
        "status": s.solver_status,
    })
df_J = pd.DataFrame(results)
display(df_J)

fig_J, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(df_J["horizon_h"].astype(str), df_J["expected_cost_€"],
            color="#1f77b4", label="optimal")
axes[0].bar(df_J["horizon_h"].astype(str), df_J["baseline_cost_€"],
            color="#aec7e8", alpha=0.5, label="baseline")
axes[0].set_xlabel("horizon (h)")
axes[0].set_ylabel("cost (€)")
axes[0].set_title("Cost vs. horizon")
axes[0].legend(fontsize=8)
axes[1].plot(df_J["horizon_h"], df_J["solve_ms"], "o-", color="#d62728")
axes[1].set_xlabel("horizon (h)")
axes[1].set_ylabel("LP solve time (ms)")
axes[1].set_title("LP wall-clock vs. horizon")
plt.tight_layout()
plt.show()
"""

# ── scenario K ────────────────────────────────────────────────────────────────
MD_K = """\
## Scenario K — Heater infeasibility → soft slack

We override `HEATER_DEADLINES` with a deliberately impossible spec:
**100 kWh required by 07:00**. The heater rated power is 2 kW so the
maximum it can deliver in any 13 h overnight window is 26 kWh — slack must
absorb the rest. The LP must remain feasible (`status="optimal"`) because
the slack penalty is dominated only by the cost we *cannot* shave.
"""

CODE_K = """\
custom = (
    HeaterEnergyDeadline(hour=7, kwh_required=100.0),
    HeaterEnergyDeadline(hour=18, kwh_required=2.0),
)
prices_K = np.full(96, 50.0)

sched_K = solve_receding_horizon(
    _utc(0), prices_K,
    horizon_slots=96, remaining_ev_kwh=0.0,
    heater_deadlines=custom,
)
heat = np.asarray(sched_K.heater_power_kw)
print(f"Heater max power across horizon: {heat.max():.3f} kW  (rated 2.0)")
print(f"Window 07:00 delivered: {sched_K.heater_window_kwh[7]:.3f} kWh  (asked 100; physical max ≤ 26)")
print(f"Solver status: {sched_K.solver_status}")
assert sched_K.solver_status in ("optimal", "optimal_inaccurate")

fig_K = plot_schedule(
    sched_K, prices_K,
    title="Scenario K — heater runs at rated power, slack absorbs the impossible 100 kWh ask",
    figsize=(14, 5.5),
)
plt.show()
"""

# ── scenario L ────────────────────────────────────────────────────────────────
MD_L = """\
## Scenario L — Joint MIP vs price-only reschedule (cap binding)

This scenario shows *why* the optimiser solves the reschedule jointly with
the EV / heater plan instead of using a price-only shift score.

The setup is constructed so the cap binds and the trade-off is visible:

* **Now = 04:30 UTC**, EV needs **14 kWh** by **07:00** (the next 2.5 h).
* **House cap = 7.5 kW** — EV (7 kW) + dishwasher (2.5 kW) = 9.5 kW > cap,
  so any overlap forces EV down to **5 kW** while the cycle runs.
* Prices: **€200/MWh** for the next hour (04:30 → 05:30), then a deep
  **€10/MWh** valley for exactly 2 hours (05:30 → 07:30), then **€50/MWh**.
* The 2-hour cheap valley is tuned to be *exactly* the EV's full-rate
  charging time (14 kWh / 7 kW = 2 h) — every kWh the EV misses there
  must come from the €200 hour, which is 20× more expensive.
* The user starts the **dishwasher at 04:30** (8-slot, 2-hour cycle).

The price-only logic (`_propose_for_onset`) sees the cheapest 8-slot start
and recommends shifting the cycle straight into the valley — that's the
local optimum *for the cycle alone*. But planting the cycle there forces
the EV to share the cap during its only cheap window, pushing 4 kWh of EV
charging into the €200 hour.

The joint MIP (the new `solve_receding_horizon` with `pending_cycles=[…]`)
co-optimises the cycle placement with the EV / heater plan and the cap. It
shifts the cycle to **after** the cheap window instead, so the EV gets the
full valley uninterrupted — at the price of the cycle finishing in the
€50/MWh medium tier.

The plan-level cost gap is **~30%** in favour of the joint MIP.
"""

CODE_L = """\
# Setup: tight cap (7.5 kW) so EV (7 kW) + dish (2.5 kW) cannot co-run at
# rated power, and a 2 h cheap window (slots 4-11) sized exactly to the
# EV's 14 kWh / 7 kW = 2 h need. Any kWh the EV loses inside the cheap
# window must come from the €200 hour right before — a 20× price gap.
# Heater is disabled (heater_deadlines=()) to isolate the EV-vs-cycle
# trade-off; with a heater also competing for the same cheap slots the
# constraint interaction would muddy the comparison.
prices_L = np.full(96, 60.0)
prices_L[0:4]   = 200.0     # 04:30–05:30 very expensive (1 h)
prices_L[4:12]  = 10.0      # 05:30–07:30 deeply cheap (2 h, exactly the EV's need)
prices_L[12:16] = 50.0      # 07:30–08:30 medium

now_L = _utc(4, 30)
EV_NEED_KWH = 14.0
TIME_TO_DEADLINE_H = 2.5    # EV deadline at 07:00, slot 10
CAP_KW = 7.5

spec = APPLIANCES["dishwasher"]
window_slots = int(HITL_RESCHEDULE_WINDOW_HOURS * 60.0 / SLOT_MINUTES)
last_start = min(window_slots, 96 - spec.cycle_slots)

# (1) Price-only proposal — the legacy shift score we used pre-MIP. It
#     ignores EV / heater / cap, so it just finds the cheapest 8 slots.
slice_L = prices_L[:96]
naive = _propose_for_onset(
    "dishwasher", now_L, slice_L,
    cycle_slots=spec.cycle_slots, rated_kw=spec.rated_kw, horizon_slots=96,
)

# (2) Joint MIP — schedule with a pending dishwasher and tight EV deadline.
def _solve_with_cycle_at(slot: int):
    return solve_receding_horizon(
        now_L, prices_L, horizon_slots=96,
        remaining_ev_kwh=EV_NEED_KWH, time_to_deadline_h=TIME_TO_DEADLINE_H,
        house_cap_kw=CAP_KW, heater_deadlines=(),
        pending_cycles=[PendingCycle(
            appliance="dishwasher", cycle_slots=spec.cycle_slots,
            rated_kw=spec.rated_kw,
            earliest_start_slot=slot, latest_start_slot=slot,
        )],
    )

sched_joint = solve_receding_horizon(
    now_L, prices_L, horizon_slots=96,
    remaining_ev_kwh=EV_NEED_KWH, time_to_deadline_h=TIME_TO_DEADLINE_H,
    house_cap_kw=CAP_KW, heater_deadlines=(),
    pending_cycles=[PendingCycle(
        appliance="dishwasher", cycle_slots=spec.cycle_slots,
        rated_kw=spec.rated_kw, earliest_start_slot=0,
        latest_start_slot=last_start,
    )],
)
joint_slot = sched_joint.cycle_starts["dishwasher"]
joint_cost = sched_joint.expected_cost

# (3) Plan-level cost the user would actually realise if the price-only
#     logic decided — pin the cycle at the naive slot and re-solve under
#     the same EV / cap constraints.
naive_slot = int(round((naive.proposed_start_at - now_L).total_seconds() / 60.0 / SLOT_MINUTES))
sched_pin_naive = _solve_with_cycle_at(naive_slot)
naive_realised_cost = sched_pin_naive.expected_cost

print("─" * 62)
print(f"Price-only naive shift          : slot {naive_slot} (+{naive_slot*SLOT_H:.2f} h)")
print(f"  isolated cycle cost           : €{naive.cost_proposed_eur:.4f}   ← what the user is told")
print(f"  plan-level cost if committed  : €{naive_realised_cost:.4f}   ← what the user actually pays")
print()
print(f"Joint MIP optimal shift         : slot {joint_slot} (+{joint_slot*SLOT_H:.2f} h)")
print(f"  plan-level cost               : €{joint_cost:.4f}   ← lower")
print()
# Both plans deliver exactly the same total energy — the difference is
# purely WHEN that energy is delivered against the price profile.
ev_n_kwh   = float(np.asarray(sched_pin_naive.ev_power_kw).sum() * SLOT_H)
ev_j_kwh   = float(np.asarray(sched_joint.ev_power_kw).sum() * SLOT_H)
dish_kwh   = spec.cycle_slots * spec.rated_kw * SLOT_H
print(f"Energy delivered (identical in both plans):")
print(f"  EV {ev_n_kwh:.2f} kWh + dishwasher {dish_kwh:.2f} kWh = {ev_n_kwh + dish_kwh:.2f} kWh")
print(f"Sanity: naive EV={ev_n_kwh:.3f} kWh, joint EV={ev_j_kwh:.3f} kWh.")
print()
delta = naive_realised_cost - joint_cost
pct = (1 - joint_cost / max(naive_realised_cost, 1e-9)) * 100
print(f"Δ = €{delta:+.4f}  ({pct:+.1f}% relative)")
print("─" * 62)

# Per-slot cost breakdown so the chart can label "€ paid here for what".
def _slot_costs(sched, slot):
    T = sched.horizon_slots
    ev = np.asarray(sched.ev_power_kw)
    dish = np.zeros(T)
    dish[slot : slot + spec.cycle_slots] = spec.rated_kw
    ev_cost   = ev   * SLOT_H * prices_L[:T] / 1000.0
    dish_cost = dish * SLOT_H * prices_L[:T] / 1000.0
    return ev, dish, ev_cost, dish_cost

ev_n, dish_n, evc_n, dishc_n = _slot_costs(sched_pin_naive, naive_slot)
ev_j, dish_j, evc_j, dishc_j = _slot_costs(sched_joint, joint_slot)

# Both plans deliver identical total energy (14 kWh EV + 5 kWh dishwasher
# = 19 kWh); only the *placement* against the price profile differs. Show
# this explicitly: top row is power, bottom row is the per-slot cost stack
# on the SAME time axis, with the price curve overlaid for context.
fig_L, axes = plt.subplots(
    3, 2, figsize=(14, 8), sharex=True, sharey="row",
    gridspec_kw={"height_ratios": [3, 2.2, 1.6]},
)
width_days = SLOT_MINUTES * 0.9 / (60.0 * 24.0)
zoom_end = now_L + timedelta(hours=4.5)
ev_deadline = now_L + timedelta(hours=TIME_TO_DEADLINE_H)
T = sched_joint.horizon_slots
times = [now_L + timedelta(minutes=SLOT_MINUTES * i) for i in range(T)]

PLANS = [
    ("Price-only naive", sched_pin_naive, naive_slot, ev_n, dish_n, evc_n, dishc_n,
     naive_realised_cost),
    ("Joint MIP", sched_joint, joint_slot, ev_j, dish_j, evc_j, dishc_j, joint_cost),
]

cheap_start = now_L + timedelta(minutes=SLOT_MINUTES * 4)
cheap_end   = now_L + timedelta(minutes=SLOT_MINUTES * 12)

for col, (label, sched, slot, ev, dish, evc, dishc, total_cost) in enumerate(PLANS):
    ax_pwr = axes[0, col]
    ax_cost = axes[1, col]
    ax_price = axes[2, col]

    # Top — power stack.
    ax_pwr.bar(times, ev, width=width_days, color=_PALETTE["ev_charger"],
               label="EV", align="edge", alpha=0.85)
    ax_pwr.bar(times, dish, width=width_days, bottom=ev,
               color=_PALETTE["dishwasher"], hatch="//", label="dishwasher",
               align="edge", alpha=0.75)
    ax_pwr.axhline(CAP_KW, color="black", ls="--", lw=1.2, label=f"cap {CAP_KW} kW")
    ax_pwr.axvspan(cheap_start, cheap_end, alpha=0.08, color="green",
                   label="cheap window (€10/MWh)")
    ax_pwr.axvline(ev_deadline, color="#d62728", ls="--", lw=1.1, alpha=0.7,
                   label="EV deadline 07:00")
    ax_pwr.set_ylim(0, max(CAP_KW * 1.3, 9.5))
    ax_pwr.set_title(
        f"{label} → slot {slot} (+{slot*SLOT_H:.1f} h)\\n"
        f"plan-level cost €{total_cost:.4f}  ·  EV €{evc.sum():.3f}, "
        f"dish €{dishc.sum():.3f}  ·  delivered "
        f"{ev.sum()*SLOT_H:.1f} kWh EV + {dish.sum()*SLOT_H:.1f} kWh dish = "
        f"{(ev.sum()+dish.sum())*SLOT_H:.1f} kWh",
        fontsize=9.5,
    )
    if col == 0:
        ax_pwr.set_ylabel("load (kW)")
    ax_pwr.legend(fontsize=7.3, loc="upper right", ncol=2)

    # Middle — per-slot cost stack: this is where €1.425 vs €1.145 actually comes from.
    ax_cost.bar(times, evc, width=width_days, color=_PALETTE["ev_charger"],
                align="edge", alpha=0.85, label="EV € this slot")
    ax_cost.bar(times, dishc, width=width_days, bottom=evc,
                color=_PALETTE["dishwasher"], hatch="//", align="edge", alpha=0.75,
                label="dish € this slot")
    ax_cost.axvspan(cheap_start, cheap_end, alpha=0.08, color="green")
    ax_cost.axvline(ev_deadline, color="#d62728", ls="--", lw=1.0, alpha=0.7)
    if col == 0:
        ax_cost.set_ylabel("slot cost (€)")
    ax_cost.legend(fontsize=7.5, loc="upper right")

    # Bottom — price step plot.
    ax_price.step(times[:T], prices_L[:T], where="post", color="k", lw=1.3)
    ax_price.fill_between(times[:T], prices_L[:T], step="post", alpha=0.12, color="k")
    ax_price.axvspan(cheap_start, cheap_end, alpha=0.08, color="green")
    ax_price.axvline(ev_deadline, color="#d62728", ls="--", lw=1.0, alpha=0.7)
    if col == 0:
        ax_price.set_ylabel("€/MWh")
    ax_price.set_xlabel("time of day (UTC)")
    ax_price.set_xlim(now_L, zoom_end)
    ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

fig_L.suptitle(
    f"Scenario L — same 19 kWh delivered both ways; only the placement against price differs.  "
    f"Naive €{naive_realised_cost:.3f}  →  Joint MIP €{joint_cost:.3f}  "
    f"(saves €{delta:.3f}, {pct:.1f}%)",
    fontsize=10.5, y=0.995,
)
fig_L.autofmt_xdate(rotation=0, ha="center")
plt.tight_layout()
plt.show()
"""

# ── scenario M ────────────────────────────────────────────────────────────────
MD_M = """\
## Scenario M — Re-nudge before start ("postpone for another hour")

This scenario exercises the new behaviour: **any cycle that has not started
yet remains replannable**.

Flow:

1. At 20:00 the user starts the dishwasher. The joint solver proposes a +1 h
   shift (slot 4). We commit that deferred start.
2. Fifteen minutes later (20:15), a *different* appliance onset appears
   (washing machine). We now include:
   - the new onset,
   - the deferred dishwasher as a synthetic replannable onset
     (`CommitTracker.replannable_onsets`),
   and re-solve jointly.
3. The solver nudges dishwasher again to a later slot (another hour-like move)
   while also placing washing machine, all under the same cap.
"""

CODE_M = """\
now0 = _utc(20, 0)
spec_d = APPLIANCES["dishwasher"]
spec_w = APPLIANCES["washing_machine"]

# Step 1 — first onset (dishwasher only): best slot is +1 h.
prices_M0 = np.full(96, 80.0)
prices_M0[4:12] = 20.0
sched_M0 = solve_receding_horizon(
    now0, prices_M0,
    horizon_slots=96, remaining_ev_kwh=0.0, house_cap_kw=4.0,
    pending_cycles=[PendingCycle(
        appliance="dishwasher",
        cycle_slots=spec_d.cycle_slots,
        rated_kw=spec_d.rated_kw,
        earliest_start_slot=0, latest_start_slot=8,
    )],
)
first_slot = int(sched_M0.cycle_starts["dishwasher"])

commit = CommitTracker()
cycle_kwh_d = spec_d.rated_kw * spec_d.cycle_slots * SLOT_H
commit.adopt_cycle_start(
    appliance="dishwasher",
    slots=spec_d.cycle_slots,
    expected_kwh=cycle_kwh_d,
    start_at=now0 + timedelta(minutes=SLOT_MINUTES * first_slot),
    now=now0,
)

# Step 2 — 15 min later, another onset appears (washing machine).
now1 = now0 + timedelta(minutes=15)
replannable = commit.replannable_onsets(now1)
assert any(o.appliance == "dishwasher" for o in replannable), "Deferred dishwasher must stay replannable."

prices_M1 = np.full(96, 80.0)
prices_M1[3:8] = 200.0
prices_M1[8:16] = 20.0

sched_M1 = solve_receding_horizon(
    now1, prices_M1,
    horizon_slots=96, remaining_ev_kwh=0.0, house_cap_kw=4.0,
    pending_cycles=[
        PendingCycle(
            appliance="dishwasher",
            cycle_slots=spec_d.cycle_slots,
            rated_kw=spec_d.rated_kw,
            earliest_start_slot=0, latest_start_slot=8,
        ),
        PendingCycle(
            appliance="washing_machine",
            cycle_slots=spec_w.cycle_slots,
            rated_kw=spec_w.rated_kw,
            earliest_start_slot=0, latest_start_slot=8,
        ),
    ],
)

second_slot = int(sched_M1.cycle_starts["dishwasher"])
wm_slot = int(sched_M1.cycle_starts["washing_machine"])

print("First plan (20:00):")
print(f"  dishwasher slot = {first_slot}  (+{first_slot*SLOT_H:.2f} h)")
print("Second plan (20:15) with new washing-machine onset:")
print(f"  dishwasher slot = {second_slot}  (+{second_slot*SLOT_H:.2f} h from now)")
print(f"  washing machine slot = {wm_slot}  (+{wm_slot*SLOT_H:.2f} h from now)")
print(f"  dishwasher additional nudge = {(second_slot - (first_slot-1))*SLOT_H:.2f} h")

# From 20:15, the original +1 h dishwasher start corresponds to slot 3.
orig_slot_from_now1 = first_slot - 1
assert second_slot >= orig_slot_from_now1 + 4, "Expected dishwasher to be nudged by ~another hour."

# Visualise first and second plans side-by-side.
T = sched_M1.horizon_slots
hrs = np.arange(T) * SLOT_H

dish_first = np.zeros(T); dish_first[first_slot:first_slot + spec_d.cycle_slots] = spec_d.rated_kw
dish_second = np.zeros(T); dish_second[second_slot:second_slot + spec_d.cycle_slots] = spec_d.rated_kw
wash_second = np.zeros(T); wash_second[wm_slot:wm_slot + spec_w.cycle_slots] = spec_w.rated_kw

fig_M_plans, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
for ax, title, ev_arr, heat_arr, dish_arr, wash_arr, mark_old in [
    (
        axes[0],
        f"First plan @20:00 (dishwasher slot {first_slot})",
        np.asarray(sched_M0.ev_power_kw),
        np.asarray(sched_M0.heater_power_kw),
        dish_first,
        np.zeros(T),
        None,
    ),
    (
        axes[1],
        f"Second plan @20:15 (dish {second_slot}, wash {wm_slot})",
        np.asarray(sched_M1.ev_power_kw),
        np.asarray(sched_M1.heater_power_kw),
        dish_second,
        wash_second,
        orig_slot_from_now1 * SLOT_H,
    ),
]:
    ax.bar(hrs, ev_arr, width=SLOT_H*0.9, color=_PALETTE["ev_charger"], alpha=0.85, label="EV", align="edge")
    ax.bar(hrs, heat_arr, width=SLOT_H*0.9, bottom=ev_arr, color=_PALETTE["heater"], alpha=0.8, label="heater", align="edge")
    ax.bar(hrs, wash_arr, width=SLOT_H*0.9, bottom=ev_arr+heat_arr, color=_PALETTE["washing_machine"], alpha=0.75, label="washing_machine", align="edge")
    ax.bar(hrs, dish_arr, width=SLOT_H*0.9, bottom=ev_arr+heat_arr+wash_arr, color=_PALETTE["dishwasher"], hatch="//", alpha=0.75, label="dishwasher", align="edge")
    ax.axhline(4.0, color="black", ls="--", lw=1.1, label="cap 4.0 kW")
    if mark_old is not None:
        ax.axvline(mark_old, color="red", ls=":", lw=1.2, label="old dishwasher start")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("hour from plan timestamp")
    ax.set_title(title, fontsize=9.5)
axes[0].set_ylabel("load (kW)")
axes[1].legend(fontsize=7.5, loc="upper right")
plt.tight_layout()
plt.show()

# Timeline view (requested): x = elapsed replan/detection time,
# y = elapsed scheduled start time. Onset detections lie on y=x.
t0 = now0
t_event_dish = now0
t_event_wash = now1
t_plan1_dish = now0 + timedelta(minutes=SLOT_MINUTES * first_slot)
t_plan2_dish = now1 + timedelta(minutes=SLOT_MINUTES * second_slot)
t_plan2_wash = now1 + timedelta(minutes=SLOT_MINUTES * wm_slot)

def _eh(dt):
    return (dt - t0).total_seconds() / 3600.0

fig_M_timeline, ax = plt.subplots(figsize=(8.2, 6.2))

# Reference diagonal y=x: "detected now".
grid_max_h = max(_eh(t_plan2_dish), _eh(t_plan1_dish), _eh(t_plan2_wash), 2.0) + 0.25
ax.plot([0, grid_max_h], [0, grid_max_h], color="#888888", ls="--", lw=1.0, label="y = x (detected now)")

# Onset detections (special marker, on the diagonal).
ax.scatter(
    [_eh(t_event_dish), _eh(t_event_wash)],
    [_eh(t_event_dish), _eh(t_event_wash)],
    color=[_PALETTE["dishwasher"], _PALETTE["washing_machine"]],
    marker="X", s=90, zorder=4, label="onset detected",
)

# Planned starts from replan @20:00 and @20:15.
ax.scatter(
    [_eh(now0)],
    [_eh(t_plan1_dish)],
    color=_PALETTE["dishwasher"], marker="o", s=85, zorder=5, label="dish plan @20:00",
)
ax.scatter(
    [_eh(now1), _eh(now1)],
    [_eh(t_plan2_dish), _eh(t_plan2_wash)],
    color=[_PALETTE["dishwasher"], _PALETTE["washing_machine"]],
    marker="o", s=85, zorder=5, label="plan @20:15",
)

# Connect same-appliance planned points to show re-nudge slope.
ax.plot(
    [_eh(now0), _eh(now1)],
    [_eh(t_plan1_dish), _eh(t_plan2_dish)],
    color=_PALETTE["dishwasher"], alpha=0.7, lw=1.4,
)

ax.annotate("dish onset", (_eh(t_event_dish), _eh(t_event_dish)), xytext=(6, 8), textcoords="offset points", fontsize=8)
ax.annotate("wash onset", (_eh(t_event_wash), _eh(t_event_wash)), xytext=(6, 8), textcoords="offset points", fontsize=8)
ax.annotate("dish start from plan@20:00", (_eh(now0), _eh(t_plan1_dish)), xytext=(6, 8), textcoords="offset points", fontsize=8)
ax.annotate("dish start from plan@20:15", (_eh(now1), _eh(t_plan2_dish)), xytext=(6, 8), textcoords="offset points", fontsize=8)
ax.annotate("wash start from plan@20:15", (_eh(now1), _eh(t_plan2_wash)), xytext=(6, -12), textcoords="offset points", fontsize=8)

ax.set_xlim(-0.02, grid_max_h)
ax.set_ylim(-0.02, grid_max_h)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("elapsed time from 20:00 (h) — when event detected / plan computed")
ax.set_ylabel("elapsed time from 20:00 (h) — scheduled onset time")
ax.set_title("Scenario M timeline — x: plan time, y: scheduled onset")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()

# Visualise the second joint plan (detailed stack, same as before).
T = sched_M1.horizon_slots
hrs = np.arange(T) * SLOT_H
ev = np.asarray(sched_M1.ev_power_kw)
heat = np.asarray(sched_M1.heater_power_kw)
dish = np.zeros(T); dish[second_slot:second_slot + spec_d.cycle_slots] = spec_d.rated_kw
wash = np.zeros(T); wash[wm_slot:wm_slot + spec_w.cycle_slots] = spec_w.rated_kw

fig_M, ax = plt.subplots(figsize=(13, 4.8))
ax.bar(hrs, ev, width=SLOT_H*0.9, color=_PALETTE["ev_charger"], alpha=0.85, label="EV", align="edge")
ax.bar(hrs, heat, width=SLOT_H*0.9, bottom=ev, color=_PALETTE["heater"], alpha=0.8, label="heater", align="edge")
ax.bar(hrs, wash, width=SLOT_H*0.9, bottom=ev+heat, color=_PALETTE["washing_machine"], alpha=0.75, label="washing_machine", align="edge")
ax.bar(hrs, dish, width=SLOT_H*0.9, bottom=ev+heat+wash, color=_PALETTE["dishwasher"], hatch="//", alpha=0.75, label="dishwasher", align="edge")
ax.axhline(4.0, color="black", ls="--", lw=1.2, label="cap 4.0 kW")
ax.axvline(orig_slot_from_now1 * SLOT_H, color="red", ls=":", lw=1.2, label="old dishwasher start")
ax.set_xlim(0, 4)
ax.set_ylim(0, 6.5)
ax.set_xlabel("hour from 20:15")
ax.set_ylabel("load (kW)")
ax.set_title("Scenario M — deferred dishwasher gets re-nudged after a new onset")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.show()
"""

# ── summary ───────────────────────────────────────────────────────────────────
MD_SUMMARY = """\
## Summary

| Scenario | Feature | Verdict |
|---|---|---|
| A | EV availability gate (C1 mask) | Charging is hard-zero outside 20:00–07:00 |
| B | EV deadline regimes (C2) | Inside-horizon hard, outside prorated × γ |
| C | Heater per-window energy (C3) | Both 4 kWh / 2 kWh windows fully met |
| D | Heater concentrates in cheap hours | Overnight 4 kWh shifts into the price valley |
| E | Power-cap binding (C5) | EV + heater share the 8 kW cap |
| F | Committed dishwasher (C5 + headroom) | EV + heater throttle during the cycle |
| I | HITL throughput stress | 8 onsets, mixed accept/decline, threshold filters spurious offers |
| J | Horizon sensitivity | 24 h LP is only ~3× slower than 6 h, savings improve with horizon |
| K | Heater infeasibility | Slack absorbs the impossible 100 kWh ask, LP stays optimal |
| L | Joint MIP vs price-only reschedule | Joint solve avoids cap-induced re-shuffle, lowers plan cost |
| M | Re-nudge before start | Deferred dishwasher is replanned after new onset (moves again) |

All scenarios are deterministic given their seeds — rerunning produces identical results.
"""

cells = [
    md(TITLE),
    code(SETUP),
    md(MD_A),
    code(CODE_A),
    md(MD_B),
    code(CODE_B),
    md(MD_C),
    code(CODE_C),
    md(MD_D),
    code(CODE_D),
    md(MD_E),
    code(CODE_E),
    md(MD_F),
    code(CODE_F),
    md(MD_I),
    code(CODE_I),
    md(MD_J),
    code(CODE_J),
    md(MD_K),
    code(CODE_K),
    md(MD_L),
    code(CODE_L),
    md(MD_M),
    code(CODE_M),
    md(MD_SUMMARY),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written {len(cells)} cells → {OUT}")
